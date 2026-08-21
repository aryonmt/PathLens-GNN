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
    positives = np.asarray(positive_pairs, dtype=np.int64)
    unique_drugs = np.unique(positives[:, 0])
    rows = np.empty((len(unique_drugs), num_proteins), dtype=np.float64)
    proteins = np.arange(num_proteins, dtype=np.int64)
    for row, drug in enumerate(unique_drugs):
        pairs = np.column_stack(
            (np.full(num_proteins, int(drug), dtype=np.int64), proteins)
        )
        rows[row] = np.asarray(score_pairs(pairs), dtype=np.float64)
    return filtered_per_drug_ranking_from_scores(
        positives,
        scores=rows,
        known_by_drug=known_by_drug,
        drug_ids=unique_drugs,
    )


def filtered_per_drug_ranking_from_scores(
    positive_pairs: NDArray[np.int64],
    *,
    scores: NDArray[np.floating],
    known_by_drug: dict[int, frozenset[int]],
    drug_ids: NDArray[np.int64] | None = None,
) -> RankingReport:
    scores = np.asarray(scores, dtype=np.float64)
    if drug_ids is None:
        row_for_drug = None
    else:
        row_for_drug = {int(drug): row for row, drug in enumerate(np.asarray(drug_ids))}
    reciprocal_ranks: list[float] = []
    for drug, target in np.asarray(positive_pairs, dtype=np.int64):
        row = scores[int(drug)] if row_for_drug is None else scores[row_for_drug[int(drug)]]
        filtered = known_by_drug.get(int(drug), frozenset()) - {int(target)}
        mask = np.ones(row.shape[0], dtype=bool)
        if filtered:
            mask[np.fromiter(filtered, dtype=np.int64)] = False
        target_score = row[int(target)]
        rank = 1 + int(np.sum(row[mask] > target_score))
        reciprocal_ranks.append(1.0 / rank)
    values = np.asarray(reciprocal_ranks, dtype=np.float64)
    return RankingReport(
        mrr=float(values.mean()),
        hits_at_10=float(np.mean(values >= 1 / 10)),
        hits_at_50=float(np.mean(values >= 1 / 50)),
        queries=len(values),
    )
