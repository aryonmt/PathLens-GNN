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


def test_strict_gt_ignores_ties_average_splits_them() -> None:
    positives = np.asarray([[0, 1]], dtype=np.int64)
    scores = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    known_by_drug = {0: frozenset({1})}
    strict = query_ranks_from_scores(
        positives,
        scores=scores,
        known_by_drug=known_by_drug,
        tie_mode="strict_gt",
    )
    average = query_ranks_from_scores(
        positives,
        scores=scores,
        known_by_drug=known_by_drug,
        tie_mode="average",
    )
    assert strict.tolist() == [2]
    assert average.tolist() == [3.0]


def test_random_tie_rank_is_seeded_and_in_the_tied_block() -> None:
    positives = np.asarray([[0, 1]], dtype=np.int64)
    scores = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    known_by_drug = {0: frozenset({1})}
    first = query_ranks_from_scores(
        positives,
        scores=scores,
        known_by_drug=known_by_drug,
        tie_mode="random",
        seed=41,
    )
    second = query_ranks_from_scores(
        positives,
        scores=scores,
        known_by_drug=known_by_drug,
        tie_mode="random",
        seed=41,
    )
    other = query_ranks_from_scores(
        positives,
        scores=scores,
        known_by_drug=known_by_drug,
        tie_mode="random",
        seed=7,
    )
    assert first.tolist() == second.tolist()
    assert first[0] in {2, 3, 4}
    assert other[0] in {2, 3, 4}


def test_unknown_tie_mode_is_rejected() -> None:
    positives = np.asarray([[0, 0]], dtype=np.int64)
    scores = np.asarray([[1.0, 0.0]], dtype=np.float64)
    try:
        query_ranks_from_scores(
            positives,
            scores=scores,
            known_by_drug={0: frozenset({0})},
            tie_mode="optimistic",
        )
    except ValueError as error:
        assert "tie_mode" in str(error)
    else:
        raise AssertionError("expected ValueError")


def test_filtering_test_positives_improves_rank_when_they_outscore_the_target() -> None:
    from pathlens.evaluation.ranking import known_proteins_by_drug

    positives = np.asarray([[0, 1]], dtype=np.int64)
    scores = np.asarray([[0.5, 0.4, 0.9]], dtype=np.float64)
    all_positive = np.asarray([[0, 1], [0, 2]], dtype=np.int64)
    visible = np.asarray([[0, 1]], dtype=np.int64)
    with_test = query_ranks_from_scores(
        positives,
        scores=scores,
        known_by_drug=known_proteins_by_drug(all_positive),
    )
    without_test = query_ranks_from_scores(
        positives,
        scores=scores,
        known_by_drug=known_proteins_by_drug(visible),
    )
    assert with_test.tolist() == [2]
    assert without_test.tolist() == [3]


def test_tie_diagnostics_count_zero_targets_and_tied_others() -> None:
    from pathlens.evaluation.ranking import tie_diagnostics

    positives = np.asarray([[0, 1], [0, 0]], dtype=np.int64)
    scores = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    known_by_drug = {0: frozenset({0, 1})}
    payload = tie_diagnostics(
        positives,
        scores=scores,
        known_by_drug=known_by_drug,
    )
    assert payload["queries"] == 2
    assert payload["fraction_zero_target"] == 0.5
    assert payload["mean_tied_others"] == 1.0
    assert payload["fraction_in_tie"] == 0.5


def test_mrr_by_drug_degree_tertile_splits_queries() -> None:
    from pathlens.evaluation.ranking import mrr_by_drug_degree_tertile

    positives = np.asarray([[0, 0], [1, 0], [2, 0]], dtype=np.int64)
    ranks = np.asarray([1.0, 2.0, 10.0], dtype=np.float64)
    drug_degree = np.asarray([1.0, 5.0, 20.0], dtype=np.float64)
    sliced = mrr_by_drug_degree_tertile(positives, ranks, drug_degree)
    assert set(sliced) == {"low", "mid", "high"}
    assert sliced["low"]["queries"] == 1
    assert sliced["low"]["mrr"] == 1.0
    assert sliced["high"]["mrr"] == 0.1


def test_metrics_use_validation_selected_probability_threshold() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    logits = np.asarray([-4.0, -1.0, 1.0, 4.0])
    report = classification_report(labels, logits)
    assert report.auroc == 1.0
    assert report.auprc == 1.0
    assert report.f1 == 1.0
    assert 0.0 <= report.ece <= 1.0
    assert select_f1_threshold(labels, sigmoid(logits)) == report.threshold
