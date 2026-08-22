from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pathlens_gnn.model.pathlens import PathLensConfig  # noqa: E402
from pathlens_gnn.training.report import write_validation_report  # noqa: E402
from pathlens_gnn.training.runner import TrainingConfig, train_experiment  # noqa: E402


def test_validation_report_writes_curves_without_test_metrics(tmp_path: Path) -> None:
    processed = _mini_processed(tmp_path)
    run = tmp_path / "run"
    train_experiment(
        processed,
        run,
        TrainingConfig(
            seed=13,
            max_epochs=1,
            patience=1,
            device="cpu",
            loss="sampled_softmax",
            num_negatives=4,
            model=PathLensConfig(
                embedding_dim=8,
                branch_dim=8,
                expert_hidden_dim=8,
                gate_hidden_dim=8,
            ),
        ),
    )
    output = tmp_path / "report"
    payload = write_validation_report(
        processed,
        output,
        checkpoint=str(run / "checkpoint.pt"),
    )
    assert payload["split"] == "validation"
    assert "test_positive" not in payload
    ranking = payload["models"]["full_adaptive_ranking"]
    assert "curves" in ranking["classification"]["hard"]
    assert "hits_at_1" in ranking["filtered_ranking"]
    npz = np.load(output / "validation-report.npz")
    assert "full_adaptive_ranking__ranks" in npz.files
    assert "full_adaptive_ranking__hard_logits" in npz.files
    report = json.loads((output / "validation-report.json").read_text(encoding="utf-8"))
    assert report["split"] == "validation"
    assert "normalized_three_hop" in report["models"]


def _mini_processed(root: Path) -> Path:
    processed = root / "processed"
    processed.mkdir()
    (processed / "entities.json").write_text(
        json.dumps(
            {
                "drugs": ["DB00001", "DB00002", "DB00003"],
                "proteins": ["P00001", "P00002", "P00003", "P00004", "P00005", "P00006"],
            }
        ),
        encoding="utf-8",
    )
    context = np.asarray([[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [2, 3]], dtype=np.int64)
    train_positive = np.asarray([[0, 2], [1, 3]], dtype=np.int64)
    validation_positive = np.asarray([[2, 0]], dtype=np.int64)
    test_positive = np.asarray([[0, 3]], dtype=np.int64)
    all_positive = np.concatenate((context, train_positive, validation_positive, test_positive))
    np.savez(
        processed / "splits.npz",
        all_positive=all_positive,
        context=context,
        train_positive=train_positive,
        validation_positive=validation_positive,
        test_positive=test_positive,
        train_uniform=np.asarray([[0, 3], [1, 0]], dtype=np.int64),
        validation_uniform=np.asarray([[2, 1]], dtype=np.int64),
        test_uniform=np.asarray([[2, 1]], dtype=np.int64),
        validation_hard=np.asarray([[2, 1]], dtype=np.int64),
        test_hard=np.asarray([[2, 1]], dtype=np.int64),
    )
    return processed
