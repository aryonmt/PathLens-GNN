from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RankingReport:
    mrr: float
    hits_at_10: float
    hits_at_50: float
    queries: int


def filtered_per_drug_ranking(
    positive_pairs: NDArray[np.int64],
    *,
    num_proteins: int,
    known_by_drug: dict[int, frozenset[int]],
    score_pairs: Callable[[NDArray[np.int64]], NDArray[np.floating]],
) -> RankingReport:
    reciprocal_ranks: list[float] = []
    for drug, target in np.asarray(positive_pairs, dtype=np.int64):
        candidates = np.arange(num_proteins, dtype=np.int64)
        filtered = known_by_drug.get(int(drug), frozenset()) - {int(target)}
        mask = np.ones(num_proteins, dtype=bool)
        if filtered:
            mask[np.fromiter(filtered, dtype=np.int64)] = False
        candidates = candidates[mask]
        pairs = np.column_stack((np.full(len(candidates), int(drug), dtype=np.int64), candidates))
        scores = np.asarray(score_pairs(pairs), dtype=np.float64)
        target_score = scores[candidates == target][0]
        rank = 1 + int(np.sum(scores > target_score))
        reciprocal_ranks.append(1.0 / rank)
    values = np.asarray(reciprocal_ranks, dtype=np.float64)
    return RankingReport(
        mrr=float(values.mean()),
        hits_at_10=float(np.mean(values >= 1 / 10)),
        hits_at_50=float(np.mean(values >= 1 / 50)),
        queries=len(values),
    )
