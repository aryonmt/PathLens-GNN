from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from pathlens_gnn.graph.index import BipartiteIndex
from pathlens_gnn.model.pathlens import PathLensGNN

DEFAULT_DRUG_CHUNK = 16


@dataclass(frozen=True, slots=True)
class DevicePairFeatures:
    drug_mass: Tensor
    protein_mass: Tensor
    bridge: Tensor

    @classmethod
    def from_index(cls, graph: BipartiteIndex, device: torch.device) -> DevicePairFeatures:
        drug_mass, protein_mass, bridge = graph.pair_feature_tables()
        return cls(
            drug_mass=torch.as_tensor(drug_mass, device=device),
            protein_mass=torch.as_tensor(protein_mass, device=device),
            bridge=torch.as_tensor(bridge, device=device),
        )

    def lookup(self, drugs: Tensor, proteins: Tensor) -> Tensor:
        return torch.stack(
            (
                self.drug_mass[drugs] + self.protein_mass[proteins],
                self.bridge[drugs, proteins],
            ),
            dim=-1,
        )


def score_all_proteins(
    model: PathLensGNN,
    embeddings: tuple[Tensor, Tensor, Tensor],
    features: DevicePairFeatures,
    drugs: Tensor,
    *,
    chunk_size: int = DEFAULT_DRUG_CHUNK,
) -> Tensor:
    """Score every protein for each drug. Returns `[len(drugs), num_proteins]` logits."""
    num_proteins = model.num_proteins
    protein_index = torch.arange(num_proteins, device=drugs.device)
    rows: list[Tensor] = []
    for start in range(0, int(drugs.shape[0]), chunk_size):
        chunk = drugs[start : start + chunk_size]
        drug_index = chunk.repeat_interleave(num_proteins)
        protein_ids = protein_index.repeat(int(chunk.shape[0]))
        logits = model.score_from_embeddings(
            embeddings,
            drug_index,
            protein_ids,
            features.lookup(drug_index, protein_ids),
        ).logit
        rows.append(logits.view(int(chunk.shape[0]), num_proteins))
    return torch.cat(rows, dim=0)
