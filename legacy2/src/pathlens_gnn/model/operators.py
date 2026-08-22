from __future__ import annotations

import torch
from torch import Tensor, nn


class BipartiteOperators(nn.Module):
    """Hop-separated operators without materializing A² or A³."""

    def __init__(
        self,
        num_drugs: int,
        num_proteins: int,
        edge_index: Tensor,
    ) -> None:
        super().__init__()
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        self.num_drugs = num_drugs
        self.num_proteins = num_proteins
        self.register_buffer("drug_edges", edge_index[0].long())
        self.register_buffer("protein_edges", edge_index[1].long())

        drug_degree = torch.bincount(edge_index[0], minlength=num_drugs).float().clamp_min(1)
        protein_degree = torch.bincount(edge_index[1], minlength=num_proteins).float().clamp_min(1)
        self.register_buffer("drug_degree", drug_degree)
        self.register_buffer("protein_degree", protein_degree)

    def one_hop(self, features: Tensor) -> Tensor:
        drug_x, protein_x = self._split(features)
        norm = torch.rsqrt(
            self.drug_degree[self.drug_edges] * self.protein_degree[self.protein_edges]
        )
        drug_out = torch.zeros_like(drug_x)
        protein_out = torch.zeros_like(protein_x)
        drug_out.index_add_(0, self.drug_edges, protein_x[self.protein_edges] * norm.unsqueeze(1))
        protein_out.index_add_(0, self.protein_edges, drug_x[self.drug_edges] * norm.unsqueeze(1))
        return torch.cat((drug_out, protein_out), dim=0)

    def two_hop(self, features: Tensor, *, weighting: str = "resource_allocation") -> Tensor:
        if weighting not in {"resource_allocation", "count"}:
            raise ValueError(f"Unsupported S2 weighting: {weighting}")
        drug_x, protein_x = self._split(features)
        return torch.cat(
            (
                self._project_drugs(drug_x, weighting),
                self._project_proteins(protein_x, weighting),
            ),
            dim=0,
        )

    def resource_allocation_two_hop(self, features: Tensor) -> Tensor:
        return self.two_hop(features, weighting="resource_allocation")

    def three_hop(self, features: Tensor, *, weighting: str = "resource_allocation") -> Tensor:
        """Propagate through opposite-type bridge paths without two-hop return walks."""
        return self.one_hop(self.two_hop(features, weighting=weighting))

    def _project_drugs(self, features: Tensor, weighting: str) -> Tensor:
        sqrt_degree = torch.sqrt(self.drug_degree).unsqueeze(1)
        normalized = features / sqrt_degree
        at_proteins = torch.zeros(
            self.num_proteins,
            features.shape[1],
            device=features.device,
            dtype=features.dtype,
        )
        at_proteins.index_add_(0, self.protein_edges, normalized[self.drug_edges])
        if weighting == "resource_allocation":
            at_proteins = at_proteins / self.protein_degree.unsqueeze(1)
        projected = torch.zeros_like(features)
        projected.index_add_(0, self.drug_edges, at_proteins[self.protein_edges])

        diagonal = torch.zeros(self.num_drugs, device=features.device, dtype=features.dtype)
        edge_weight = (
            torch.reciprocal(self.protein_degree[self.protein_edges])
            if weighting == "resource_allocation"
            else torch.ones_like(self.protein_edges, dtype=features.dtype)
        )
        diagonal.index_add_(0, self.drug_edges, edge_weight)
        projected = projected - diagonal.unsqueeze(1) * normalized
        return projected / sqrt_degree

    def _project_proteins(self, features: Tensor, weighting: str) -> Tensor:
        sqrt_degree = torch.sqrt(self.protein_degree).unsqueeze(1)
        normalized = features / sqrt_degree
        at_drugs = torch.zeros(
            self.num_drugs,
            features.shape[1],
            device=features.device,
            dtype=features.dtype,
        )
        at_drugs.index_add_(0, self.drug_edges, normalized[self.protein_edges])
        if weighting == "resource_allocation":
            at_drugs = at_drugs / self.drug_degree.unsqueeze(1)
        projected = torch.zeros_like(features)
        projected.index_add_(0, self.protein_edges, at_drugs[self.drug_edges])

        diagonal = torch.zeros(self.num_proteins, device=features.device, dtype=features.dtype)
        edge_weight = (
            torch.reciprocal(self.drug_degree[self.drug_edges])
            if weighting == "resource_allocation"
            else torch.ones_like(self.drug_edges, dtype=features.dtype)
        )
        diagonal.index_add_(0, self.protein_edges, edge_weight)
        projected = projected - diagonal.unsqueeze(1) * normalized
        return projected / sqrt_degree

    def _split(self, features: Tensor) -> tuple[Tensor, Tensor]:
        expected = self.num_drugs + self.num_proteins
        if features.ndim != 2 or features.shape[0] != expected:
            raise ValueError(f"features must have shape [{expected}, hidden_dim]")
        return features[: self.num_drugs], features[self.num_drugs :]
