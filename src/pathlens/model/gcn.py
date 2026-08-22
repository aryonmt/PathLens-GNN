# Copyright (c) 2020 Kexin Huang
# Copyright (c) 2026 Amirreza Nemati
# SPDX-License-Identifier: BSD-3-Clause
#
# Two-layer GCN on the original SkipGNN adjacency (A + I, then D^{-1/2} A D^{-1/2}).
# This is the paper's GCN baseline: same decoder as SkipGNN, no skip graph.

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pathlens.model.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(
        self,
        nfeat: int,
        nhid1: int,
        nhid2: int,
        nhid_decode1: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.dropout = dropout
        self.decoder1 = nn.Linear(nhid2 * 2, nhid_decode1)
        self.decoder2 = nn.Linear(nhid_decode1, 1)

    def encode(self, features: Tensor, adj: Tensor) -> Tensor:
        hidden = F.relu(self.gc1(features, adj))
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        return self.gc2(hidden, adj)

    def decode(self, embeddings: Tensor, left: Tensor, right: Tensor) -> Tensor:
        feat = torch.cat((embeddings[left], embeddings[right]), dim=1)
        return self.decoder2(self.decoder1(feat))

    def forward(
        self,
        features: Tensor,
        adj: Tensor,
        idx: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        embeddings = self.encode(features, adj)
        return self.decode(embeddings, idx[0], idx[1]), embeddings
