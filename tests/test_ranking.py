from __future__ import annotations

import numpy as np

from pathlens.evaluation.metrics import classification_report, select_f1_threshold, sigmoid
from pathlens.evaluation.ranking import (
    filtered_per_drug_ranking,
    filtered_per_drug_ranking_from_scores,
    hits_curve,
    query_ranks_from_scores,
    ranking_report_from_ranks,
)


def test_filtered_ranking_from_score_matrix_matches_pair_scorer() -> None:
    positives = np.asarray([[0, 1], [0, 2], [1, 0]], dtype=np.int64)
    scores = np.asarray(
        [
            [0.1, 0.9, 0.4],
            [0.8, 0.2, 0.3],
        ],
        dtype=np.float64,
    )
    known_by_drug = {0: frozenset({1, 2}), 1: frozenset({0})}

    def score_pairs(pairs: np.ndarray) -> np.ndarray:
        return np.asarray([scores[int(drug), int(protein)] for drug, protein in pairs])

    from_pairs = filtered_per_drug_ranking(
        positives,
        num_proteins=3,
        known_by_drug=known_by_drug,
        score_pairs=score_pairs,
    )
    from_matrix = filtered_per_drug_ranking_from_scores(
        positives,
        scores=scores,
        known_by_drug=known_by_drug,
    )
    assert from_matrix.queries == 3
    assert from_matrix.mrr == from_pairs.mrr
    assert from_matrix.hits_at_10 == from_pairs.hits_at_10
    assert from_matrix.hits_at_50 == from_pairs.hits_at_50
    assert from_matrix.ndcg_at_10 > 0


def test_query_ranks_and_hits_curve_match_filtered_report() -> None:
    positives = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
    scores = np.asarray(
        [
            [0.2, 0.9, 0.1],
            [0.8, 0.3, 0.4],
        ],
        dtype=np.float64,
    )
    known_by_drug = {0: frozenset({1}), 1: frozenset({0})}
    ranks = query_ranks_from_scores(
        positives,
        scores=scores,
        known_by_drug=known_by_drug,
    )
    report = ranking_report_from_ranks(ranks)
    from_scores = filtered_per_drug_ranking_from_scores(
        positives,
        scores=scores,
        known_by_drug=known_by_drug,
    )
    assert ranks.tolist() == [1, 1]
    assert report.mrr == from_scores.mrr
    assert hits_curve(ranks)["hits_at_1"] == 1.0
    assert hits_curve(ranks)["hits_at_10"] == 1.0
    assert report.ndcg_at_10 == 1.0


def test_metrics_use_validation_selected_probability_threshold() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    logits = np.asarray([-4.0, -1.0, 1.0, 4.0])
    report = classification_report(labels, logits)
    assert report.auroc == 1.0
    assert report.auprc == 1.0
    assert report.f1 == 1.0
    assert 0.0 <= report.ece <= 1.0
    assert select_f1_threshold(labels, sigmoid(logits)) == report.threshold
