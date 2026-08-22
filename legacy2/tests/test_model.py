from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pathlens_gnn.model.operators import BipartiteOperators  # noqa: E402
from pathlens_gnn.model.pathlens import PathLensConfig, PathLensGNN  # noqa: E402


def test_sparse_operators_and_model_gradient_flow() -> None:
    edges = torch.tensor([[0, 1, 1, 2], [0, 0, 1, 1]])
    operators = BipartiteOperators(3, 2, edges)
    features = torch.randn(5, 8, requires_grad=True)
    assert operators.one_hop(features).shape == features.shape
    assert operators.resource_allocation_two_hop(features).shape == features.shape
    assert operators.three_hop(features).shape == features.shape

    model = PathLensGNN(
        3,
        2,
        edges,
        PathLensConfig(embedding_dim=8, branch_dim=8, expert_hidden_dim=8, gate_hidden_dim=8),
    )
    output = model(
        torch.tensor([0, 2]),
        torch.tensor([1, 0]),
        torch.tensor([[1.0, 2.0], [0.5, 1.0]]),
    )
    assert output.logit.shape == (2,)
    assert torch.allclose(output.contributions.sum(dim=1), output.logit)
    assert torch.allclose(output.gate_weights.sum(dim=1), torch.ones(2))
    output.logit.sum().backward()
    assert model.node_embedding.weight.grad is not None


def test_three_hop_operator_excludes_immediate_return_walks() -> None:
    operators = BipartiteOperators(1, 1, torch.tensor([[0], [0]]))
    features = torch.ones(2, 3)
    assert torch.count_nonzero(operators.three_hop(features)) == 0
