# Copyright (c) 2020 Kexin Huang
# Copyright (c) 2026 Amirreza Nemati
# SPDX-License-Identifier: BSD-3-Clause
#
# GCN layer from Huang et al., SkipGNN (legacy1/SkipGNN/layers.py).

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """Dense-or-sparse GCN layer, same algebra as Kipf & Welling / SkipGNN."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(in_features, out_features))
        self.bias = Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs: Tensor, adj: Tensor) -> Tensor:
        support = torch.sparse.mm(inputs, self.weight) if inputs.is_sparse else inputs @ self.weight
        output = torch.sparse.mm(adj, support) if adj.is_sparse else adj @ support
        if self.bias is not None:
            return output + self.bias
        return output

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features})"
