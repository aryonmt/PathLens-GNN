# Copyright (c) 2020 Kexin Huang
# Copyright (c) 2026 Amirreza Nemati
# SPDX-License-Identifier: BSD-3-Clause
#
# Encoder/decoder algebra from Huang et al., SkipGNN (legacy1/SkipGNN/models.py).
# Decoder stays a two-layer linear map with no hidden activation.

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pathlens.model.layers import GraphConvolution


class SkipGNN(nn.Module):
    def __init__(
        self,
        nfeat: int,
        nhid1: int,
        nhid2: int,
        nhid_decode1: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.o_gc1 = GraphConvolution(nfeat, nhid1)
        self.o_gc2 = GraphConvolution(nhid1, nhid2)
        self.o_gc1_s = GraphConvolution(nhid1, nhid1)
        self.s_gc1 = GraphConvolution(nfeat, nhid1)
        self.s_gc1_o = GraphConvolution(nfeat, nhid1)
        self.s_gc2_o = GraphConvolution(nhid1, nhid2)
        self.dropout = dropout
        self.decoder1 = nn.Linear(nhid2 * 2, nhid_decode1)
        self.decoder2 = nn.Linear(nhid_decode1, 1)

    def encode(self, features: Tensor, o_adj: Tensor, s_adj: Tensor) -> Tensor:
        original = F.relu(self.o_gc1(features, o_adj) + self.s_gc1_o(features, s_adj))
        skip = F.relu(self.s_gc1(features, s_adj) + self.o_gc1_s(original, o_adj))
        original = F.dropout(original, self.dropout, training=self.training)
        skip = F.dropout(skip, self.dropout, training=self.training)
        return self.o_gc2(original, o_adj) + self.s_gc2_o(skip, s_adj)

    def decode(self, embeddings: Tensor, left: Tensor, right: Tensor) -> Tensor:
        feat = torch.cat((embeddings[left], embeddings[right]), dim=1)
        return self.decoder2(self.decoder1(feat))

    def forward(
        self,
        features: Tensor,
        o_adj: Tensor,
        s_adj: Tensor,
        idx: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        embeddings = self.encode(features, o_adj, s_adj)
        return self.decode(embeddings, idx[0], idx[1]), embeddings


def scipy_to_torch_sparse(matrix: Any, device: torch.device | str) -> Tensor:
    sparse = matrix.tocoo().astype(np.float32)
    indices = torch.tensor(
        np.vstack((sparse.row, sparse.col)), dtype=torch.int64, device=device
    )
    values = torch.tensor(sparse.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(
        indices, values, tuple(sparse.shape), device=device
    ).coalesce()


def score_all_pairs(
    model: SkipGNN,
    features: Tensor,
    o_adj: Tensor,
    s_adj: Tensor,
    *,
    num_drugs: int,
    num_proteins: int,
    drug_batch: int = 64,
) -> Tensor:
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        embeddings = model.encode(features, o_adj, s_adj)
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
        scores = torch.cat(rows, dim=0)
    if was_training:
        model.train()
    return scores
