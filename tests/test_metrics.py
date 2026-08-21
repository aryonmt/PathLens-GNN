from __future__ import annotations

import numpy as np

from pathlens_gnn.evaluation.metrics import classification_report, select_f1_threshold


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
