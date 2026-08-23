from __future__ import annotations

import numpy as np

from pathlens.data.processed import ProcessedSplit
from pathlens.evaluation.eda import hop_reachability, summarize_split


def _split() -> ProcessedSplit:
    return ProcessedSplit(
        num_drugs=4,
        num_proteins=4,
        seed=13,
        manifest={"seed": 13},
        all_positive=np.asarray(
            [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 1], [2, 3], [3, 0], [3, 2]],
            dtype=np.int64,
        ),
        context=np.asarray(
            [[0, 0], [0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]],
            dtype=np.int64,
        ),
        train_positive=np.asarray([[3, 0]], dtype=np.int64),
        train_uniform=np.asarray([[0, 3]], dtype=np.int64),
        validation_positive=np.asarray([[1, 1]], dtype=np.int64),
        validation_uniform=np.asarray([[2, 0]], dtype=np.int64),
        validation_hard=np.asarray([[2, 0]], dtype=np.int64),
        _test=None,
    )


def test_eda_summary_counts_context_degrees() -> None:
    summary = summarize_split(_split())
    assert summary["num_drugs"] == 4
    assert summary["edges"]["context"] == 7
    assert summary["degree"]["drug"]["zeros"] == 0
    assert summary["coverage"]["validation_drugs_with_context"] == 1.0


def test_hop_reachability_reports_validation_zero_mass() -> None:
    reach = hop_reachability(_split(), device="cpu")
    assert reach["validation_queries"] == 1
    assert 0.0 <= reach["three_hop_zero_fraction"] <= 1.0
