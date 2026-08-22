from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pathlens.evaluation.ranking import (
    filtered_per_drug_ranking_from_scores,
    known_proteins_by_drug,
)

BLEND_ALPHAS: tuple[float, ...] = tuple(round(index / 10, 1) for index in range(11))
RRF_K = 60


def row_zscore(matrix: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(matrix, dtype=np.float64)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    safe = np.where(std > 0, std, 1.0)
    zscored = (values - mean) / safe
    zscored = np.where(std > 0, zscored, 0.0)
    return zscored


def blend_scores(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    alpha: float,
) -> NDArray[np.float64]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return alpha * row_zscore(left) + (1.0 - alpha) * row_zscore(right)


def ranks_from_scores(matrix: NDArray[np.floating]) -> NDArray[np.int64]:
    values = np.asarray(matrix, dtype=np.float64)
    order = np.argsort(-values, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    positions = np.arange(1, values.shape[1] + 1, dtype=np.int64)
    for row, ranked in enumerate(order):
        ranks[row, ranked] = positions
    return ranks


def reciprocal_rank_fusion(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    *,
    k: int = RRF_K,
) -> NDArray[np.float64]:
    if k < 0:
        raise ValueError("k must be non-negative")
    left_ranks = ranks_from_scores(left).astype(np.float64)
    right_ranks = ranks_from_scores(right).astype(np.float64)
    return 1.0 / (k + left_ranks) + 1.0 / (k + right_ranks)


def mean_rank_scores(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
) -> NDArray[np.float64]:
    return -(ranks_from_scores(left) + ranks_from_scores(right)).astype(np.float64) / 2.0


def select_blend_alpha(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    positives: NDArray[np.int64],
    *,
    known_by_drug: dict[int, frozenset[int]] | None = None,
    alphas: tuple[float, ...] = BLEND_ALPHAS,
) -> tuple[float, list[dict[str, Any]]]:
    if not alphas:
        raise ValueError("alphas must not be empty")
    positives = np.asarray(positives, dtype=np.int64)
    if known_by_drug is None:
        known_by_drug = known_proteins_by_drug(positives)
    sweep: list[dict[str, Any]] = []
    best_alpha = alphas[0]
    best_mrr = -1.0
    for alpha in alphas:
        scores = blend_scores(left, right, alpha)
        report = filtered_per_drug_ranking_from_scores(
            positives,
            scores=scores,
            known_by_drug=known_by_drug,
        )
        sweep.append({"alpha": float(alpha), "mrr": report.mrr})
        if report.mrr > best_mrr:
            best_mrr = report.mrr
            best_alpha = float(alpha)
    return best_alpha, sweep
