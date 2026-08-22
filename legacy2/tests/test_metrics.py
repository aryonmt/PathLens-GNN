from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from pathlens_gnn.evaluation.metrics import (
    bootstrap_auprc,
    classification_report,
    select_f1_threshold,
    sigmoid,
)


def test_metrics_use_validation_selected_probability_threshold() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    logits = np.asarray([-4.0, -1.0, 1.0, 4.0])
    report = classification_report(labels, logits)
    assert report.auroc == 1.0
    assert report.auprc == 1.0
    assert report.f1 == 1.0
    assert 0.0 <= report.ece <= 1.0
    assert select_f1_threshold(labels, 1 / (1 + np.exp(-logits))) == report.threshold


def test_classification_curves_cover_perfect_separator() -> None:
    from pathlens_gnn.evaluation.metrics import classification_curves

    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    logits = np.asarray([-4.0, -1.0, 1.0, 4.0])
    curves = classification_curves(labels, logits)
    assert curves["recall"][0] == 1.0 or curves["precision"][-1] == 1.0
    assert curves["tpr"][-1] == 1.0
    assert curves["fpr"][0] == 0.0


def test_select_f1_threshold_matches_sklearn_on_unique_scores() -> None:
    generator = np.random.default_rng(13)
    labels = generator.integers(0, 2, 64)
    labels[0] = 0
    labels[1] = 1
    probabilities = generator.random(64)
    chosen = select_f1_threshold(labels, probabilities)
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    expected = max(
        candidates,
        key=lambda threshold: f1_score(labels, probabilities >= threshold, zero_division=0),
    )
    assert chosen == float(expected)


def test_bootstrap_auprc_covers_point_estimate_and_stays_fast() -> None:
    generator = np.random.default_rng(41)
    labels = np.concatenate((np.ones(1500, dtype=np.int64), np.zeros(1500, dtype=np.int64)))
    logits = np.concatenate((generator.normal(2.0, 1.0, 1500), generator.normal(-1.0, 1.0, 1500)))
    started = time.perf_counter()
    lower, upper = bootstrap_auprc(labels, logits)
    elapsed = time.perf_counter() - started
    point = float(average_precision_score(labels, sigmoid(logits)))
    assert lower <= point <= upper
    assert elapsed < 2.0
