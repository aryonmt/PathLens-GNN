from __future__ import annotations

import numpy as np
import pytest

from pathlens.data.processed import ProcessedSplit, SealedTestError
from pathlens.evaluation.report import evaluate_for_stage, evaluate_score_matrix


def _split(*, allow_test: bool) -> ProcessedSplit:
    test = (
        {
            "test_positive": np.asarray([[0, 2]], dtype=np.int64),
            "test_uniform": np.asarray([[3, 1]], dtype=np.int64),
            "test_hard": np.asarray([[3, 1]], dtype=np.int64),
        }
        if allow_test
        else None
    )
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
        _test=test,
    )


def test_evaluate_score_matrix_can_target_test() -> None:
    split = _split(allow_test=True)
    scores = np.zeros((4, 4), dtype=np.float64)
    scores[1, 1] = 9.0
    scores[0, 2] = 8.0
    validation = evaluate_score_matrix(
        scores, split, split_name="validation", include_bootstrap=False
    )
    test = evaluate_score_matrix(scores, split, split_name="test", include_bootstrap=False)
    assert validation["filtered_ranking"]["queries"] == 1
    assert test["filtered_ranking"]["queries"] == 1
    assert test["filtered_ranking"]["mrr"] == pytest.approx(1.0)


def test_test_split_stays_sealed_without_arrays() -> None:
    split = _split(allow_test=False)
    scores = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(SealedTestError):
        evaluate_score_matrix(scores, split, split_name="test", include_bootstrap=False)


def test_evaluate_for_stage_adds_test_only_on_final() -> None:
    split = _split(allow_test=True)
    scores = np.eye(4, dtype=np.float64)
    eval_payload = evaluate_for_stage(scores, split, "eval")
    assert "test" not in eval_payload
    assert "curves" in eval_payload["classification"]["hard"]
    final_payload = evaluate_for_stage(scores, split, "final")
    assert "filtered_ranking" in final_payload["test"]
    smoke = evaluate_for_stage(scores, split, "smoke")
    assert "curves" not in smoke["classification"]["hard"]
    assert "test" not in smoke
