from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

HITS_KS = (1, 3, 5, 10, 20, 50)
NDCG_KS = (10, 50)
TIE_MODES = ("strict_gt", "average", "random")
RANDOM_TIE_SEED = 41


@dataclass(frozen=True, slots=True)
class RankingReport:
    mrr: float
    hits_at_1: float
    hits_at_3: float
    hits_at_5: float
    hits_at_10: float
    hits_at_20: float
    hits_at_50: float
    ndcg_at_10: float
    ndcg_at_50: float
    queries: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "mrr": self.mrr,
            "hits_at_1": self.hits_at_1,
            "hits_at_3": self.hits_at_3,
            "hits_at_5": self.hits_at_5,
            "hits_at_10": self.hits_at_10,
            "hits_at_20": self.hits_at_20,
            "hits_at_50": self.hits_at_50,
            "ndcg_at_10": self.ndcg_at_10,
            "ndcg_at_50": self.ndcg_at_50,
            "queries": self.queries,
        }


def known_proteins_by_drug(pairs: NDArray[np.int64]) -> dict[int, frozenset[int]]:
    grouped: dict[int, set[int]] = defaultdict(set)
    for drug, protein in np.asarray(pairs, dtype=np.int64):
        grouped[int(drug)].add(int(protein))
    return {drug: frozenset(proteins) for drug, proteins in grouped.items()}


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
        pairs = np.column_stack((np.full(num_proteins, int(drug), dtype=np.int64), proteins))
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
    tie_mode: str = "strict_gt",
    seed: int = RANDOM_TIE_SEED,
) -> RankingReport:
    ranks = query_ranks_from_scores(
        positive_pairs,
        scores=scores,
        known_by_drug=known_by_drug,
        drug_ids=drug_ids,
        tie_mode=tie_mode,
        seed=seed,
    )
    return ranking_report_from_ranks(ranks)


def query_ranks_from_scores(
    positive_pairs: NDArray[np.int64],
    *,
    scores: NDArray[np.floating],
    known_by_drug: dict[int, frozenset[int]],
    drug_ids: NDArray[np.int64] | None = None,
    tie_mode: str = "strict_gt",
    seed: int = RANDOM_TIE_SEED,
) -> NDArray[np.float64]:
    if tie_mode not in TIE_MODES:
        raise ValueError(f"Unknown tie_mode {tie_mode!r}. Expected one of {TIE_MODES}")
    scores = np.asarray(scores, dtype=np.float64)
    if drug_ids is None:
        row_for_drug = None
    else:
        row_for_drug = {int(drug): row for row, drug in enumerate(np.asarray(drug_ids))}
    ranks = np.empty(len(positive_pairs), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for index, (drug, target) in enumerate(np.asarray(positive_pairs, dtype=np.int64)):
        row = scores[int(drug)] if row_for_drug is None else scores[row_for_drug[int(drug)]]
        n_better, n_tied = _tie_counts(row, int(target), known_by_drug.get(int(drug), frozenset()))
        ranks[index] = _rank_from_ties(n_better, n_tied, tie_mode=tie_mode, rng=rng)
    return ranks


def tie_diagnostics(
    positive_pairs: NDArray[np.int64],
    *,
    scores: NDArray[np.floating],
    known_by_drug: dict[int, frozenset[int]],
    drug_ids: NDArray[np.int64] | None = None,
) -> dict[str, float | int]:
    scores = np.asarray(scores, dtype=np.float64)
    positives = np.asarray(positive_pairs, dtype=np.int64)
    if drug_ids is None:
        row_for_drug = None
    else:
        row_for_drug = {int(drug): row for row, drug in enumerate(np.asarray(drug_ids))}
    n_tied_others: list[int] = []
    n_better_list: list[int] = []
    zero_target = 0
    in_tie = 0
    for drug, target in positives:
        row = scores[int(drug)] if row_for_drug is None else scores[row_for_drug[int(drug)]]
        n_better, n_tied = _tie_counts(row, int(target), known_by_drug.get(int(drug), frozenset()))
        n_tied_others.append(n_tied)
        n_better_list.append(n_better)
        if row[int(target)] == 0.0:
            zero_target += 1
        if n_tied:
            in_tie += 1
    queries = len(positives)
    tied = np.asarray(n_tied_others, dtype=np.float64)
    return {
        "queries": queries,
        "fraction_zero_target": 0.0 if queries == 0 else zero_target / queries,
        "mean_tied_others": 0.0 if queries == 0 else float(tied.mean()),
        "median_tied_others": 0.0 if queries == 0 else float(np.median(tied)),
        "mean_tie_group": 0.0 if queries == 0 else float((tied + 1.0).mean()),
        "median_tie_group": 0.0 if queries == 0 else float(np.median(tied + 1.0)),
        "mean_better": 0.0 if queries == 0 else float(np.mean(n_better_list)),
        "fraction_in_tie": 0.0 if queries == 0 else in_tie / queries,
    }


def mrr_by_drug_degree_tertile(
    positive_pairs: NDArray[np.int64],
    ranks: NDArray[np.floating],
    drug_degree: NDArray[np.floating],
) -> dict[str, dict[str, float | int]]:
    positives = np.asarray(positive_pairs, dtype=np.int64)
    ranks = np.asarray(ranks, dtype=np.float64)
    degrees = np.asarray(drug_degree, dtype=np.float64)[positives[:, 0]]
    lower, upper = np.quantile(degrees, [1 / 3, 2 / 3])
    masks = {
        "low": degrees <= lower,
        "mid": (degrees > lower) & (degrees <= upper),
        "high": degrees > upper,
    }
    sliced: dict[str, dict[str, float | int]] = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        selected = ranks[mask]
        sliced[name] = {
            "queries": int(mask.sum()),
            "mrr": float((1.0 / selected).mean()),
            "hits_at_10": float(np.mean(selected <= 10)),
        }
    return sliced


def _tie_counts(
    row: NDArray[np.floating],
    target: int,
    known: frozenset[int],
) -> tuple[int, int]:
    others = np.ones(row.shape[0], dtype=bool)
    filtered = known - {target}
    if filtered:
        others[np.fromiter(filtered, dtype=np.int64)] = False
    others[target] = False
    target_score = row[target]
    n_better = int(np.sum(row[others] > target_score))
    n_tied = int(np.sum(row[others] == target_score))
    return n_better, n_tied


def _rank_from_ties(
    n_better: int,
    n_tied: int,
    *,
    tie_mode: str,
    rng: np.random.Generator,
) -> float:
    if tie_mode == "strict_gt":
        return float(1 + n_better)
    if tie_mode == "average":
        return 1.0 + n_better + (n_tied / 2.0)
    return float(1 + n_better + int(rng.integers(0, n_tied + 1)))


def ranking_report_from_ranks(ranks: NDArray[np.floating] | NDArray[np.integer]) -> RankingReport:
    ranks = np.asarray(ranks, dtype=np.float64)
    hits = hits_curve(ranks)
    mrr = 0.0 if ranks.size == 0 else float((1.0 / ranks).mean())
    return RankingReport(
        mrr=mrr,
        hits_at_1=hits["hits_at_1"],
        hits_at_3=hits["hits_at_3"],
        hits_at_5=hits["hits_at_5"],
        hits_at_10=hits["hits_at_10"],
        hits_at_20=hits["hits_at_20"],
        hits_at_50=hits["hits_at_50"],
        ndcg_at_10=_ndcg_at_k(ranks, 10) if ranks.size else 0.0,
        ndcg_at_50=_ndcg_at_k(ranks, 50) if ranks.size else 0.0,
        queries=int(ranks.size),
    )


def hits_curve(
    ranks: NDArray[np.floating] | NDArray[np.integer],
    *,
    ks: tuple[int, ...] = HITS_KS,
) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.float64)
    if ranks.size == 0:
        return {f"hits_at_{k}": 0.0 for k in ks}
    return {f"hits_at_{k}": float(np.mean(ranks <= k)) for k in ks}


def _ndcg_at_k(ranks: NDArray[np.floating], k: int) -> float:
    dcg = np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)
    return float(dcg.mean())
