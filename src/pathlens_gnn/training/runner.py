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
from pathlens_gnn.evaluation.ranking import filtered_per_drug_ranking
from pathlens_gnn.graph.index import BipartiteIndex
from pathlens_gnn.model.pathlens import PathLensConfig, PathLensGNN
from pathlens_gnn.training.ranking import (
    known_proteins_by_drug,
    sample_ranked_negatives,
    sampled_softmax_loss,
)

RANKING_LOSS = "sampled_softmax"
BCE_LOSS = "bce"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 13
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 300
    patience: int = 30
    device: str = "auto"
    loss: str = BCE_LOSS
    num_negatives: int = 64
    hard_negative_fraction: float = 0.25
    softmax_temperature: float = 1.0
    model: PathLensConfig = PathLensConfig()


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def training_config_from_trial(
    spec: dict[str, Any],
    parameters: dict[str, Any],
    *,
    seed: int,
) -> TrainingConfig:
    fixed = spec["fixed"]
    model = PathLensConfig(
        embedding_dim=parameters["embedding_dim"],
        branch_dim=fixed["branch_dim"],
        expert_hidden_dim=fixed["expert_hidden_dim"],
        gate_hidden_dim=parameters["gate_hidden_dim"],
        dropout=parameters["dropout"],
        s2_weighting=parameters["s2_weighting"],
    )
    return TrainingConfig(
        seed=seed,
        learning_rate=parameters["learning_rate"],
        weight_decay=parameters["weight_decay"],
        max_epochs=int(fixed["max_epochs"]),
        patience=int(fixed["patience"]),
        loss=str(fixed.get("loss", BCE_LOSS)),
        num_negatives=int(fixed.get("num_negatives", 64)),
        hard_negative_fraction=float(fixed.get("hard_negative_fraction", 0.25)),
        softmax_temperature=float(fixed.get("softmax_temperature", 1.0)),
        model=model,
    )


def train_experiment(
    processed_dir: str | Path,
    output_dir: str | Path,
    config: TrainingConfig,
) -> dict[str, Any]:
    if config.loss not in {BCE_LOSS, RANKING_LOSS}:
        raise ValueError(f"Unsupported training loss: {config.loss}")
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
    ranking_loss = config.loss == RANKING_LOSS
    print(
        f"[train] seed={config.seed} device={device} loss={config.loss} "
        f"max_epochs={config.max_epochs} patience={config.patience}",
        flush=True,
    )

    edge_index = torch.as_tensor(context.T, dtype=torch.long, device=device)
    model = PathLensGNN(num_drugs, num_proteins, edge_index, config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    bce_criterion = nn.BCEWithLogitsLoss()

    preprocessing_start = time.perf_counter()
    train_positive = arrays["train_positive"].astype(np.int64)
    known_by_drug = known_proteins_by_drug(arrays["all_positive"])
    protein_degrees = np.bincount(context[:, 1], minlength=num_proteins).astype(np.int64)
    train_pairs, train_labels, train_features = _build_candidates(
        train_positive, arrays["train_uniform"], graph_index
    )
    validation_pairs, validation_labels, validation_features = _build_candidates(
        arrays["validation_positive"], arrays["validation_hard"], graph_index
    )
    train_tensors = _to_tensors(train_pairs, train_labels, train_features, device)
    validation_tensors = _to_tensors(
        validation_pairs, validation_labels, validation_features, device
    )
    train_positive_features = graph_index.structural_features(train_positive)
    print(
        f"[train] Prepared {len(train_positive)} train positives and "
        f"{len(validation_pairs)} validation pairs on CPU in "
        f"{time.perf_counter() - preprocessing_start:.1f}s; "
        f"training tensors moved to {device}",
        flush=True,
    )

    best_auprc = float("-inf")
    best_mrr = float("-inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()

    for epoch in range(config.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if ranking_loss:
            loss = _ranking_epoch_loss(
                model,
                graph_index,
                train_positive,
                train_positive_features,
                known_by_drug,
                protein_degrees,
                config,
                epoch,
                device,
            )
        else:
            train_output = model(train_tensors[0], train_tensors[1], train_tensors[3])
            loss = bce_criterion(train_output.logit, train_tensors[2])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            embeddings = model.encode()
            validation_output = model.score_from_embeddings(
                embeddings,
                validation_tensors[0],
                validation_tensors[1],
                validation_tensors[3],
            )
            report = classification_report(
                validation_labels.astype(np.int64),
                validation_output.logit.detach().cpu().numpy(),
            )
            mrr = (
                _validation_mrr(
                    model,
                    embeddings,
                    graph_index,
                    arrays["validation_positive"].astype(np.int64),
                    known_by_drug,
                    num_proteins,
                    device,
                )
                if ranking_loss
                else float("nan")
            )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "validation_auprc": report.auprc,
        }
        if ranking_loss:
            record["validation_mrr"] = mrr
        history.append(record)
        improved = _selection_improved(
            report.auprc, mrr, best_auprc, best_mrr, ranking=ranking_loss
        )
        if improved:
            best_auprc = report.auprc
            best_mrr = mrr if ranking_loss else best_mrr
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            extra = f" validation_mrr={mrr:.4f}" if ranking_loss else ""
            print(
                f"[train] epoch={epoch + 1}/{config.max_epochs} "
                f"loss={loss.item():.4f} validation_hard_auprc={report.auprc:.4f}"
                f"{extra} best_auprc={best_auprc:.4f} stale={stale_epochs}/{config.patience}",
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
        "validation_filtered_mrr": None if not ranking_loss else best_mrr,
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    summary: dict[str, Any] = {
        "best_epoch": best_epoch,
        "validation_hard_auprc": best_auprc,
        "training_seconds": elapsed,
        "device": str(device),
        "epochs_run": len(history),
        "loss": config.loss,
        "config": asdict(config),
    }
    if ranking_loss:
        summary["validation_filtered_mrr"] = best_mrr
    (output / "metrics.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    extra = f" best_validation_filtered_mrr={best_mrr:.4f}" if ranking_loss else ""
    print(
        f"[train] Completed {len(history)} epochs in {elapsed:.1f}s on {device}; "
        f"best_epoch={best_epoch + 1} best_validation_hard_auprc={best_auprc:.4f}{extra}",
        flush=True,
    )
    return summary


def _ranking_epoch_loss(
    model: PathLensGNN,
    graph_index: BipartiteIndex,
    train_positive: NDArray[np.int64],
    train_positive_features: NDArray[np.float32],
    known_by_drug: dict[int, frozenset[int]],
    protein_degrees: NDArray[np.int64],
    config: TrainingConfig,
    epoch: int,
    device: torch.device,
) -> Tensor:
    negatives = sample_ranked_negatives(
        train_positive,
        num_proteins=model.num_proteins,
        known_by_drug=known_by_drug,
        protein_degrees=protein_degrees,
        num_negatives=config.num_negatives,
        hard_fraction=config.hard_negative_fraction,
        seed=config.seed + epoch * 1_000_003,
    )
    embeddings = model.encode()
    positive_drugs, positive_proteins, positive_features = _pair_tensors(
        train_positive, train_positive_features, device
    )
    positive_logits = model.score_from_embeddings(
        embeddings, positive_drugs, positive_proteins, positive_features
    ).logit
    flat_pairs = np.column_stack(
        (
            np.repeat(train_positive[:, 0], config.num_negatives),
            negatives.reshape(-1),
        )
    )
    negative_features = graph_index.structural_features(flat_pairs)
    negative_drugs, negative_proteins, negative_feature_tensor = _pair_tensors(
        flat_pairs, negative_features, device
    )
    negative_logits = model.score_from_embeddings(
        embeddings, negative_drugs, negative_proteins, negative_feature_tensor
    ).logit.view(len(train_positive), config.num_negatives)
    return sampled_softmax_loss(
        positive_logits,
        negative_logits,
        temperature=config.softmax_temperature,
    )


def _validation_mrr(
    model: PathLensGNN,
    embeddings: tuple[Tensor, Tensor, Tensor],
    graph_index: BipartiteIndex,
    validation_positive: NDArray[np.int64],
    known_by_drug: dict[int, frozenset[int]],
    num_proteins: int,
    device: torch.device,
) -> float:
    def score_pairs(pairs: NDArray[np.int64]) -> NDArray[np.floating]:
        features = graph_index.structural_features(pairs)
        drugs, proteins, feature_tensor = _pair_tensors(pairs, features, device)
        logits = model.score_from_embeddings(embeddings, drugs, proteins, feature_tensor).logit
        return logits.detach().cpu().numpy()

    report = filtered_per_drug_ranking(
        validation_positive,
        num_proteins=num_proteins,
        known_by_drug=known_by_drug,
        score_pairs=score_pairs,
    )
    return report.mrr


def _selection_improved(
    auprc: float,
    mrr: float,
    best_auprc: float,
    best_mrr: float,
    *,
    ranking: bool,
) -> bool:
    if not ranking:
        return auprc > best_auprc + 1e-8
    if mrr > best_mrr + 1e-8:
        return True
    return abs(mrr - best_mrr) <= 1e-8 and auprc > best_auprc + 1e-8


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


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
    drugs, proteins, feature_tensor = _pair_tensors(pairs, features, device)
    return (
        drugs,
        proteins,
        torch.as_tensor(labels, dtype=torch.float32, device=device),
        feature_tensor,
    )


def _pair_tensors(
    pairs: NDArray[np.int64],
    features: NDArray[np.float32],
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    pairs = np.asarray(pairs, dtype=np.int64)
    return (
        torch.as_tensor(pairs[:, 0], dtype=torch.long, device=device),
        torch.as_tensor(pairs[:, 1], dtype=torch.long, device=device),
        torch.as_tensor(features, dtype=torch.float32, device=device),
    )
