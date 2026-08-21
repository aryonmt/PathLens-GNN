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
    ranks = query_ranks_from_scores(
        positive_pairs,
        scores=scores,
        known_by_drug=known_by_drug,
        drug_ids=drug_ids,
    )
    return ranking_report_from_ranks(ranks)


def query_ranks_from_scores(
    positive_pairs: NDArray[np.int64],
    *,
    scores: NDArray[np.floating],
    known_by_drug: dict[int, frozenset[int]],
    drug_ids: NDArray[np.int64] | None = None,
) -> NDArray[np.int64]:
    scores = np.asarray(scores, dtype=np.float64)
    if drug_ids is None:
        row_for_drug = None
    else:
        row_for_drug = {int(drug): row for row, drug in enumerate(np.asarray(drug_ids))}
    ranks = np.empty(len(positive_pairs), dtype=np.int64)
    for index, (drug, target) in enumerate(np.asarray(positive_pairs, dtype=np.int64)):
        row = scores[int(drug)] if row_for_drug is None else scores[row_for_drug[int(drug)]]
        filtered = known_by_drug.get(int(drug), frozenset()) - {int(target)}
        mask = np.ones(row.shape[0], dtype=bool)
        if filtered:
            mask[np.fromiter(filtered, dtype=np.int64)] = False
        target_score = row[int(target)]
        ranks[index] = 1 + int(np.sum(row[mask] > target_score))
    return ranks


def ranking_report_from_ranks(ranks: NDArray[np.integer]) -> RankingReport:
    ranks = np.asarray(ranks, dtype=np.int64)
    if ranks.size == 0:
        return RankingReport(mrr=0.0, hits_at_10=0.0, hits_at_50=0.0, queries=0)
    values = 1.0 / ranks.astype(np.float64)
    return RankingReport(
        mrr=float(values.mean()),
        hits_at_10=float(np.mean(ranks <= 10)),
        hits_at_50=float(np.mean(ranks <= 50)),
        queries=int(len(ranks)),
    )


def hits_curve(
    ranks: NDArray[np.integer],
    *,
    ks: tuple[int, ...] = (1, 3, 5, 10, 20, 50),
) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.int64)
    if ranks.size == 0:
        return {f"hits_at_{k}": 0.0 for k in ks}
    return {f"hits_at_{k}": float(np.mean(ranks <= k)) for k in ks}
