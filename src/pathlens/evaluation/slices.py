from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pathlens.evaluation.metrics import classification_report


def degree_slices(
    labels: NDArray[np.int64],
    logits: NDArray[np.float64],
    pairs: NDArray[np.int64],
    drug_degree: NDArray[np.floating],
    protein_degree: NDArray[np.floating],
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    degrees = np.minimum(drug_degree[pairs[:, 0]], protein_degree[pairs[:, 1]])
    lower, upper = np.quantile(degrees, [1 / 3, 2 / 3])
    masks = {
        "low": degrees <= lower,
        "mid": (degrees > lower) & (degrees <= upper),
        "high": degrees > upper,
    }
    return {
        name: classification_report(labels[mask], logits[mask], threshold=threshold).to_dict()
        for name, mask in masks.items()
        if mask.any() and len(np.unique(labels[mask])) == 2
    }
