from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def score_from_embeddings(
    model: Any,
    embeddings: Tensor,
    *,
    num_drugs: int,
    num_proteins: int,
    drug_batch: int = 64,
) -> Tensor:
    drug = embeddings[:num_drugs]
    protein = embeddings[num_drugs : num_drugs + num_proteins]
    rows: list[Tensor] = []
    for start in range(0, num_drugs, drug_batch):
        block = drug[start : start + drug_batch]
        feat = torch.cat(
            (
                block[:, None, :].expand(-1, num_proteins, -1),
                protein[None, :, :].expand(block.size(0), -1, -1),
            ),
            dim=-1,
        )
        rows.append(model.decoder2(model.decoder1(feat)).squeeze(-1))
    return torch.cat(rows, dim=0)
