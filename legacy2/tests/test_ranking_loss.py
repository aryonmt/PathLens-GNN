from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from pathlens_gnn.training.ranking import sample_ranked_negatives, sampled_softmax_loss


def test_sampled_softmax_loss_is_cross_entropy_with_positive_as_class_zero() -> None:
    positive = torch.tensor([2.0, -1.0])
    negatives = torch.tensor([[0.5, -0.5], [0.0, 1.0]])
    loss = sampled_softmax_loss(positive, negatives, temperature=1.0)
    expected = F.cross_entropy(
        torch.cat((positive.unsqueeze(1), negatives), dim=1),
        torch.zeros(2, dtype=torch.long),
    )
    assert torch.allclose(loss, expected)


def test_sampled_softmax_loss_scales_logits_by_temperature() -> None:
    positive = torch.tensor([4.0])
    negatives = torch.tensor([[0.0, 2.0]])
    actual = sampled_softmax_loss(positive, negatives, temperature=2.0)
    expected = F.cross_entropy(
        torch.tensor([[2.0, 0.0, 1.0]]),
        torch.zeros(1, dtype=torch.long),
    )
    assert torch.allclose(actual, expected)


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


def test_ranked_negatives_reuse_allowed_pool_per_drug() -> None:
    positives = np.asarray([[0, 1], [0, 2], [1, 0]], dtype=np.int64)
    known_by_drug = {0: frozenset({1, 2}), 1: frozenset({0, 3})}
    protein_degrees = np.asarray([2, 2, 1, 1], dtype=np.int64)
    sampled = sample_ranked_negatives(
        positives,
        num_proteins=4,
        known_by_drug=known_by_drug,
        protein_degrees=protein_degrees,
        num_negatives=4,
        hard_fraction=0.0,
        seed=3,
    )
    assert sampled.shape == (3, 4)
    assert not np.isin(sampled[0], [1, 2]).any()
    assert not np.isin(sampled[1], [1, 2]).any()
    assert not np.isin(sampled[2], [0, 3]).any()


def test_hard_negative_mix_prefers_same_degree_quantile() -> None:
    positives = np.asarray([[0, 0]], dtype=np.int64)
    known_by_drug = {0: frozenset({0})}
    # Proteins 0 and 1 are high-degree; 2 and 3 are low-degree.
    protein_degrees = np.asarray([8, 7, 1, 1], dtype=np.int64)
    sampled = sample_ranked_negatives(
        positives,
        num_proteins=4,
        known_by_drug=known_by_drug,
        protein_degrees=protein_degrees,
        num_negatives=4,
        hard_fraction=1.0,
        seed=11,
        quantiles=2,
    )
    assert sampled.shape == (1, 4)
    assert set(sampled[0].tolist()).issubset({1, 2, 3})
    assert (sampled[0] == 1).sum() >= 1
