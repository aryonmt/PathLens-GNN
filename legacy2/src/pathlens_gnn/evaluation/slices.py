from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pathlens_gnn.evaluation.metrics import classification_report
from pathlens_gnn.graph.index import BipartiteIndex


def degree_slices(
    labels: NDArray[np.int64],
    logits: NDArray[np.float64],
    pairs: NDArray[np.int64],
    graph: BipartiteIndex,
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    degrees = np.asarray(
        [
            min(len(graph.drug_neighbors[int(drug)]), len(graph.protein_neighbors[int(protein)]))
            for drug, protein in pairs
        ]
    )
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
