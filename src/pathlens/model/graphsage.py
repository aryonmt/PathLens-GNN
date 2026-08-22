# Copyright (c) 2026 Amirreza Nemati
# SPDX-License-Identifier: BSD-3-Clause
#
# Hamilton et al. GraphSAGE-mean. Not a Huang et al. paper baseline (that table
# is GCN/GIN/JK-Net/MixHop). Same two-layer sizes and decoder as our GCN card.

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parameter import Parameter


class SAGEConv(nn.Module):
    """h' = W_self h + W_neigh mean({h_u : u in N(v)}). Self is not in N(v)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_self = Parameter(torch.empty(in_features, out_features))
        self.weight_neigh = Parameter(torch.empty(in_features, out_features))
        self.bias = Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight_self.data.uniform_(-stdv, stdv)
        self.weight_neigh.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, features: Tensor, adj: Tensor) -> Tensor:
        self_h = _linear(features, self.weight_self)
        neighborhood = _propagate(adj, features)
        neigh_h = _linear(neighborhood, self.weight_neigh)
        degree = _row_sum(adj).clamp_min(1.0).unsqueeze(1)
        return self_h + neigh_h / degree + self.bias


class GraphSAGE(nn.Module):
    def __init__(
        self,
        nfeat: int,
        nhid1: int,
        nhid2: int,
        nhid_decode1: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.sage1 = SAGEConv(nfeat, nhid1)
        self.sage2 = SAGEConv(nhid1, nhid2)
        self.dropout = dropout
        self.decoder1 = nn.Linear(nhid2 * 2, nhid_decode1)
        self.decoder2 = nn.Linear(nhid_decode1, 1)

    def encode(self, features: Tensor, adj: Tensor) -> Tensor:
        hidden = F.relu(self.sage1(features, adj))
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        return self.sage2(hidden, adj)

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


def _linear(features: Tensor, weight: Tensor) -> Tensor:
    if features.is_sparse:
        return torch.sparse.mm(features, weight)
    return features @ weight


def _propagate(adj: Tensor, features: Tensor) -> Tensor:
    if adj.is_sparse:
        return torch.sparse.mm(adj, features)
    if features.is_sparse:
        return adj @ features.to_dense()
    return adj @ features


def _row_sum(adj: Tensor) -> Tensor:
    if adj.is_sparse:
        return torch.sparse.sum(adj, dim=1).to_dense()
    return adj.sum(dim=1)
