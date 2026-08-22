from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pathlens_gnn.model.pathlens import PathLensConfig  # noqa: E402
from pathlens_gnn.training.runner import TrainingConfig, train_experiment  # noqa: E402


def test_ranking_trainer_reports_filtered_mrr(tmp_path: Path) -> None:
    processed = _mini_processed(tmp_path)
    output = tmp_path / "run"
    summary = train_experiment(
        processed,
        output,
        TrainingConfig(
            seed=13,
            max_epochs=2,
            patience=2,
            device="cpu",
            loss="sampled_softmax",
            num_negatives=4,
            hard_negative_fraction=0.5,
            model=PathLensConfig(
                embedding_dim=8,
                branch_dim=8,
                expert_hidden_dim=8,
                gate_hidden_dim=8,
            ),
        ),
    )
    assert 0.0 <= summary["validation_filtered_mrr"] <= 1.0
    assert 0.0 <= summary["validation_hard_auprc"] <= 1.0
    assert (output / "checkpoint.pt").exists()
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["loss"] == "sampled_softmax"


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
    all_positive = np.concatenate(
        (context, train_positive, validation_positive, test_positive)
    )
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
