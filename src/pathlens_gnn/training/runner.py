from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from pathlens_gnn.evaluation.metrics import classification_report
from pathlens_gnn.graph.index import BipartiteIndex
from pathlens_gnn.model.pathlens import PathLensConfig, PathLensGNN


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 13
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 300
    patience: int = 30
    device: str = "auto"
    model: PathLensConfig = PathLensConfig()


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_experiment(
    processed_dir: str | Path,
    output_dir: str | Path,
    config: TrainingConfig,
) -> dict[str, Any]:
    set_deterministic_seed(config.seed)
    processed = Path(processed_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    entities = json.loads((processed / "entities.json").read_text(encoding="utf-8"))
    arrays = np.load(processed / "splits.npz")
    num_drugs = len(entities["drugs"])
    num_proteins = len(entities["proteins"])
    context = arrays["context"].astype(np.int64)
    graph_index = BipartiteIndex(num_drugs, num_proteins, context)
    device = _resolve_device(config.device)
    print(
        f"[train] seed={config.seed} device={device} "
        f"max_epochs={config.max_epochs} patience={config.patience}",
        flush=True,
    )

    edge_index = torch.as_tensor(context.T, dtype=torch.long, device=device)
    model = PathLensGNN(num_drugs, num_proteins, edge_index, config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    preprocessing_start = time.perf_counter()
    train_pairs, train_labels, train_features = _build_candidates(
        arrays["train_positive"], arrays["train_uniform"], graph_index
    )
    validation_pairs, validation_labels, validation_features = _build_candidates(
        arrays["validation_positive"], arrays["validation_hard"], graph_index
    )
    train_tensors = _to_tensors(train_pairs, train_labels, train_features, device)
    validation_tensors = _to_tensors(
        validation_pairs, validation_labels, validation_features, device
    )
    print(
        f"[train] Prepared {len(train_pairs)} train and {len(validation_pairs)} "
        f"validation pairs on CPU in {time.perf_counter() - preprocessing_start:.1f}s; "
        f"training tensors moved to {device}",
        flush=True,
    )

    best_auprc = float("-inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()

    for epoch in range(config.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_output = model(train_tensors[0], train_tensors[1], train_tensors[3])
        loss = criterion(train_output.logit, train_tensors[2])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            validation_output = model(
                validation_tensors[0], validation_tensors[1], validation_tensors[3]
            )
        report = classification_report(
            validation_labels.astype(np.int64),
            validation_output.logit.detach().cpu().numpy(),
        )
        history.append(
            {"epoch": epoch, "train_loss": float(loss.item()), "validation_auprc": report.auprc}
        )
        if report.auprc > best_auprc + 1e-8:
            best_auprc = report.auprc
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(
                f"[train] epoch={epoch + 1}/{config.max_epochs} "
                f"loss={loss.item():.4f} validation_hard_auprc={report.auprc:.4f} "
                f"best={best_auprc:.4f} stale={stale_epochs}/{config.patience}",
                flush=True,
            )
        if stale_epochs >= config.patience:
            print(f"[train] Early stopping at epoch {epoch + 1}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)
    checkpoint = {
        "state_dict": best_state,
        "training_config": asdict(config),
        "num_drugs": num_drugs,
        "num_proteins": num_proteins,
        "best_epoch": best_epoch,
        "validation_hard_auprc": best_auprc,
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    summary: dict[str, Any] = {
        "best_epoch": best_epoch,
        "validation_hard_auprc": best_auprc,
        "training_seconds": elapsed,
        "device": str(device),
        "epochs_run": len(history),
        "config": asdict(config),
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(
        f"[train] Completed {len(history)} epochs in {elapsed:.1f}s on {device}; "
        f"best_epoch={best_epoch + 1} best_validation_hard_auprc={best_auprc:.4f}",
        flush=True,
    )
    return summary


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _build_candidates(
    positives: NDArray[np.int64],
    negatives: NDArray[np.int64],
    graph_index: BipartiteIndex,
) -> tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.float32]]:
    pairs = np.concatenate((positives, negatives)).astype(np.int64)
    labels = np.concatenate(
        (np.ones(len(positives), dtype=np.float32), np.zeros(len(negatives), dtype=np.float32))
    )
    features = graph_index.structural_features(pairs)
    return pairs, labels, features


def _to_tensors(
    pairs: NDArray[np.int64],
    labels: NDArray[np.float32],
    features: NDArray[np.float32],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        torch.as_tensor(pairs[:, 0], dtype=torch.long, device=device),
        torch.as_tensor(pairs[:, 1], dtype=torch.long, device=device),
        torch.as_tensor(labels, dtype=torch.float32, device=device),
        torch.as_tensor(features, dtype=torch.float32, device=device),
    )
