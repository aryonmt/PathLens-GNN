from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from pathlens_gnn.model.operators import BipartiteOperators


@dataclass(frozen=True, slots=True)
class PathLensConfig:
    embedding_dim: int = 64
    branch_dim: int = 64
    expert_hidden_dim: int = 64
    gate_hidden_dim: int = 64
    dropout: float = 0.2
    s2_weighting: str = "resource_allocation"
    adaptive_gate: bool = True
    enabled_channels: tuple[bool, bool, bool] = (True, True, True)


@dataclass(frozen=True, slots=True)
class PathLensOutput:
    logit: Tensor
    gate_weights: Tensor
    expert_logits: Tensor
    contributions: Tensor


class HopBranch(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features)


class TypedExpert(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, drug: Tensor, protein: Tensor) -> Tensor:
        pair = torch.cat((drug, protein, drug * protein), dim=-1)
        return self.layers(pair).squeeze(-1)


class PathLensGNN(nn.Module):
    def __init__(
        self,
        num_drugs: int,
        num_proteins: int,
        edge_index: Tensor,
        config: PathLensConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or PathLensConfig()
        self.num_drugs = num_drugs
        self.num_proteins = num_proteins
        self.operators = BipartiteOperators(num_drugs, num_proteins, edge_index)
        self.node_embedding = nn.Embedding(num_drugs + num_proteins, self.config.embedding_dim)
        self.self_projection = nn.Linear(self.config.embedding_dim, self.config.embedding_dim)
        self.branches = nn.ModuleList(
            HopBranch(
                self.config.embedding_dim,
                self.config.branch_dim,
                self.config.dropout,
            )
            for _ in range(3)
        )
        self.experts = nn.ModuleList(
            TypedExpert(
                self.config.branch_dim,
                self.config.expert_hidden_dim,
                self.config.dropout,
            )
            for _ in range(3)
        )
        gate_input_dim = self.config.branch_dim * 6 + 2
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, self.config.gate_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.gate_hidden_dim, 3),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.node_embedding.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self) -> tuple[Tensor, Tensor, Tensor]:
        base = self.node_embedding.weight
        one_hop = self.operators.one_hop(base) + self.self_projection(base)
        two_hop = self.operators.two_hop(base, weighting=self.config.s2_weighting)
        three_hop = self.operators.three_hop(base, weighting=self.config.s2_weighting)
        return (
            self.branches[0](one_hop),
            self.branches[1](two_hop),
            self.branches[2](three_hop),
        )

    def forward(
        self,
        drug_index: Tensor,
        protein_index: Tensor,
        structural_features: Tensor | None = None,
    ) -> PathLensOutput:
        embeddings = self.encode()
        return self.score_from_embeddings(
            embeddings, drug_index, protein_index, structural_features
        )

    def score_from_embeddings(
        self,
        embeddings: tuple[Tensor, Tensor, Tensor],
        drug_index: Tensor,
        protein_index: Tensor,
        structural_features: Tensor | None = None,
    ) -> PathLensOutput:
        protein_global = self.num_drugs + protein_index
        drug_vectors = [embedding[drug_index] for embedding in embeddings]
        protein_vectors = [embedding[protein_global] for embedding in embeddings]
        expert_logits = torch.stack(
            [
                expert(drug, protein)
                for expert, drug, protein in zip(
                    self.experts, drug_vectors, protein_vectors, strict=True
                )
            ],
            dim=-1,
        )
        if structural_features is None:
            structural_features = torch.zeros(
                drug_index.shape[0],
                2,
                dtype=expert_logits.dtype,
                device=expert_logits.device,
            )
        if structural_features.shape != (drug_index.shape[0], 2):
            raise ValueError("structural_features must have shape [num_pairs, 2]")

        enabled = torch.tensor(
            self.config.enabled_channels,
            device=expert_logits.device,
            dtype=torch.bool,
        )
        if not bool(enabled.any()):
            raise ValueError("At least one channel must be enabled")
        if self.config.adaptive_gate:
            gate_input = torch.cat(
                (*drug_vectors, *protein_vectors, torch.log1p(structural_features.clamp_min(0))),
                dim=-1,
            )
            gate_logits = self.gate(gate_input).masked_fill(~enabled, float("-inf"))
            gate_weights = torch.softmax(gate_logits, dim=-1)
        else:
            gate_weights = enabled.to(expert_logits.dtype).expand_as(expert_logits)
            gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True)
        contributions = gate_weights * expert_logits
        return PathLensOutput(
            logit=contributions.sum(dim=-1),
            gate_weights=gate_weights,
            expert_logits=expert_logits,
            contributions=contributions,
        )
