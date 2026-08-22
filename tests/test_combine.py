from __future__ import annotations

import numpy as np
import pytest

from pathlens.graph.combine import (
    blend_scores,
    mean_rank_scores,
    ranks_from_scores,
    reciprocal_rank_fusion,
    row_zscore,
    select_blend_alpha,
)


def test_row_zscore_centers_each_drug() -> None:
    matrix = np.asarray([[1.0, 3.0, 5.0], [10.0, 10.0, 10.0]], dtype=np.float64)
    zscored = row_zscore(matrix)
    assert zscored[0].mean() == pytest.approx(0.0)
    assert zscored[1].tolist() == [0.0, 0.0, 0.0]


def test_blend_alpha_one_keeps_left_ranking() -> None:
    left = np.asarray([[3.0, 1.0, 2.0]], dtype=np.float64)
    right = np.asarray([[1.0, 9.0, 2.0]], dtype=np.float64)
    blended = blend_scores(left, right, alpha=1.0)
    assert int(np.argmax(blended[0])) == int(np.argmax(left[0]))


def test_blend_alpha_zero_keeps_right_ranking() -> None:
    left = np.asarray([[3.0, 1.0, 2.0]], dtype=np.float64)
    right = np.asarray([[1.0, 9.0, 2.0]], dtype=np.float64)
    blended = blend_scores(left, right, alpha=0.0)
    assert int(np.argmax(blended[0])) == int(np.argmax(right[0]))


def test_ranks_give_one_to_the_highest_score() -> None:
    scores = np.asarray([[0.1, 0.9, 0.3]], dtype=np.float64)
    ranks = ranks_from_scores(scores)
    assert ranks[0, 1] == 1
    assert ranks[0, 0] == 3


def test_reciprocal_rank_fusion_keeps_a_shared_winner() -> None:
    left = np.asarray([[1.0, 2.0, 9.0]], dtype=np.float64)
    right = np.asarray([[3.0, 1.0, 8.0]], dtype=np.float64)
    fused = reciprocal_rank_fusion(left, right)
    assert int(np.argmax(fused[0])) == 2


def test_mean_rank_scores_match_negative_average_rank() -> None:
    left = np.asarray([[3.0, 1.0]], dtype=np.float64)
    right = np.asarray([[1.0, 3.0]], dtype=np.float64)
    fused = mean_rank_scores(left, right)
    ranks = (ranks_from_scores(left) + ranks_from_scores(right)) / 2.0
    assert np.allclose(fused, -ranks)


def test_select_blend_alpha_picks_the_matrix_that_ranks_the_positive() -> None:
    left = np.asarray([[5.0, 4.0, 0.0], [0.0, 4.0, 5.0]], dtype=np.float64)
    right = np.asarray([[0.0, 4.0, 5.0], [5.0, 4.0, 0.0]], dtype=np.float64)
    positives = np.asarray([[0, 0], [1, 2]], dtype=np.int64)
    alpha, sweep = select_blend_alpha(
        left,
        right,
        positives,
        alphas=(0.0, 0.5, 1.0),
    )
    assert alpha == pytest.approx(1.0)
    by_alpha = {row["alpha"]: row["mrr"] for row in sweep}
    assert by_alpha[1.0] > by_alpha[0.0]
