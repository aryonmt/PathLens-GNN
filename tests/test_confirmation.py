from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pathlens_gnn.training.confirmation import run_confirmation  # noqa: E402
from pathlens_gnn.training.runner import training_config_from_yaml  # noqa: E402


def test_training_config_from_yaml_overrides_seed(tmp_path: Path) -> None:
    path = tmp_path / "pathlens_ranking.yaml"
    path.write_text(_ranking_yaml(max_epochs=2), encoding="utf-8")
    config = training_config_from_yaml(path, seed=29)
    assert config.seed == 29
    assert config.loss == "sampled_softmax"
    assert config.num_negatives == 4
    assert config.model.embedding_dim == 8


def test_confirmation_from_yaml_writes_selected_seed_13_checkpoint(tmp_path: Path) -> None:
    processed = _mini_processed(tmp_path)
    config = tmp_path / "pathlens_ranking.yaml"
    config.write_text(_ranking_yaml(max_epochs=1), encoding="utf-8")
    output = tmp_path / "confirmation"
    payload = run_confirmation(processed, output, config_path=config)
    selected = Path(payload["selected"]["seed_13_checkpoint"])
    assert selected.exists()
    assert payload["seeds"] == [13, 29, 71]
    assert len(payload["candidates"]) == 1
    metrics = json.loads((output / "candidate-0" / "seed-13" / "metrics.json").read_text())
    assert metrics["loss"] == "sampled_softmax"
    assert "validation_filtered_mrr" in metrics


def _ranking_yaml(*, max_epochs: int) -> str:
    return f"""
seed: 13
learning_rate: 0.001
weight_decay: 0.0
max_epochs: {max_epochs}
patience: {max_epochs}
device: cpu
loss: sampled_softmax
num_negatives: 4
hard_negative_fraction: 0.5
softmax_temperature: 1.0
model:
  embedding_dim: 8
  branch_dim: 8
  expert_hidden_dim: 8
  gate_hidden_dim: 8
  dropout: 0.0
  s2_weighting: count
  adaptive_gate: true
  enabled_channels: [true, true, true]
"""


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
