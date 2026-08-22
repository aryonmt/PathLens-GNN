from __future__ import annotations

import pytest


def test_residual_forward_adds_the_frozen_three_hop() -> None:
    torch = pytest.importorskip("torch")
    from pathlens.model.residual import ResidualThreeHop

    model = ResidualThreeHop(num_drugs=3, num_proteins=4, dim=8, hidden=16, dropout=0.0)
    drugs = torch.tensor([0, 1])
    proteins = torch.tensor([1, 2])
    hop = torch.tensor([3.0, -1.5])
    residual = model.residual(drugs, proteins)
    scores = model.forward(drugs, proteins, hop)
    assert torch.allclose(scores, hop + residual)


def test_residual_score_matrix_matches_pair_forward() -> None:
    torch = pytest.importorskip("torch")
    from pathlens.model.residual import ResidualThreeHop

    model = ResidualThreeHop(num_drugs=2, num_proteins=3, dim=4, hidden=8, dropout=0.0)
    hop = torch.zeros((2, 3))
    hop[0, 1] = 2.0
    matrix = model.score_matrix(hop)
    drugs = torch.tensor([0, 0, 1])
    proteins = torch.tensor([1, 2, 0])
    pairs = model.forward(drugs, proteins, hop[drugs, proteins])
    assert torch.allclose(matrix[drugs, proteins], pairs, atol=1e-5)
