from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True, slots=True)
class ClassificationReport:
    auroc: float
    auprc: float
    f1: float
    threshold: float
    brier: float
    ece: float
    samples: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def sigmoid(logits: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(logits, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def select_f1_threshold(labels: NDArray[np.integer], probabilities: NDArray[np.floating]) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    candidates = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    predicted = probabilities[None, :] >= candidates[:, None]
    true_positive = np.count_nonzero(predicted & (labels[None, :] == 1), axis=1)
    false_positive = np.count_nonzero(predicted & (labels[None, :] == 0), axis=1)
    false_negative = int((labels == 1).sum()) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    scores = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros(len(candidates), dtype=np.float64),
        where=denominator > 0,
    )
    return float(candidates[int(np.argmax(scores))])


def expected_calibration_error(
    labels: NDArray[np.integer],
    probabilities: NDArray[np.floating],
    *,
    bins: int = 10,
) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = max(1, len(labels))
    error = 0.0
    for index in range(bins):
        lower, upper = boundaries[index : index + 2]
        selected = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if selected.any():
            error += (
                selected.sum()
                / total
                * abs(probabilities[selected].mean() - labels[selected].mean())
            )
    return float(error)


def classification_report(
    labels: NDArray[np.integer],
    logits: NDArray[np.floating],
    *,
    threshold: float | None = None,
) -> ClassificationReport:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = sigmoid(np.asarray(logits))
    chosen_threshold = (
        threshold if threshold is not None else select_f1_threshold(labels, probabilities)
    )
    return ClassificationReport(
        auroc=float(roc_auc_score(labels, probabilities)),
        auprc=float(average_precision_score(labels, probabilities)),
        f1=float(f1_score(labels, probabilities >= chosen_threshold, zero_division=0)),
        threshold=float(chosen_threshold),
        brier=float(brier_score_loss(labels, probabilities)),
        ece=expected_calibration_error(labels, probabilities),
        samples=len(labels),
    )


def bootstrap_auprc(
    labels: NDArray[np.int64],
    logits: NDArray[np.float64],
    *,
    samples: int = 400,
    seed: int = 20260809,
) -> list[float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = sigmoid(np.asarray(logits))
    generator = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        selected = generator.integers(0, len(labels), len(labels))
        if len(np.unique(labels[selected])) == 2:
            values.append(float(average_precision_score(labels[selected], probabilities[selected])))
    if not values:
        return [float("nan"), float("nan")]
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def classification_curves(
    labels: NDArray[np.integer],
    logits: NDArray[np.floating],
) -> dict[str, NDArray[np.float64]]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = sigmoid(np.asarray(logits))
    precision, recall, _pr_thresholds = precision_recall_curve(labels, probabilities)
    fpr, tpr, _roc_thresholds = roc_curve(labels, probabilities)
    return {
        "precision": np.asarray(precision, dtype=np.float64),
        "recall": np.asarray(recall, dtype=np.float64),
        "fpr": np.asarray(fpr, dtype=np.float64),
        "tpr": np.asarray(tpr, dtype=np.float64),
    }
