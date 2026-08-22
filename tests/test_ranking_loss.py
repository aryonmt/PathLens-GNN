from __future__ import annotations

import numpy as np
import pytest

from pathlens.training.ranking import sample_ranked_negatives, sampled_softmax_loss


def test_ranked_negatives_never_include_known_positives_and_are_deterministic() -> None:
    positives = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    known_by_drug = {0: frozenset({1, 3}), 1: frozenset({0, 2})}
    protein_degrees = np.asarray([4, 1, 4, 1], dtype=np.int64)
    first = sample_ranked_negatives(
        positives,
        num_proteins=4,
        known_by_drug=known_by_drug,
        protein_degrees=protein_degrees,
        num_negatives=3,
        hard_fraction=0.0,
        seed=7,
    )
    second = sample_ranked_negatives(
        positives,
        num_proteins=4,
        known_by_drug=known_by_drug,
        protein_degrees=protein_degrees,
        num_negatives=3,
        hard_fraction=0.0,
        seed=7,
    )
    assert first.shape == (2, 3)
    assert np.array_equal(first, second)
    assert 1 not in first[0] and 3 not in first[0]
    assert 0 not in first[1] and 2 not in first[1]


def test_sampled_softmax_loss_is_cross_entropy_with_positive_as_class_zero() -> None:
    torch = pytest.importorskip("torch")
    positive = torch.tensor([2.0, -1.0])
    negatives = torch.tensor([[0.5, -0.5], [0.0, 1.0]])
    loss = sampled_softmax_loss(positive, negatives, temperature=1.0)
    expected = torch.nn.functional.cross_entropy(
        torch.cat((positive.unsqueeze(1), negatives), dim=1),
        torch.zeros(2, dtype=torch.long),
    )
    assert torch.allclose(loss, expected)
