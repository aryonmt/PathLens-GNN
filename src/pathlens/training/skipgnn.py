# Copyright (c) 2020 Kexin Huang
# Copyright (c) 2026 Amirreza Nemati
# SPDX-License-Identifier: BSD-3-Clause
#
# Training loop follows Huang et al. SkipGNN (legacy1/SkipGNN/train.py):
# Adam, BCE on 1:1 pairs, keep the best validation AUROC checkpoint.
# Repairs: typed negatives, no drop_last, pair indices on device, detached
# history, inference_mode, BCE-with-logits, official PathLens scoring after
# selection. The encoder/skip-graph equations are unchanged.

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from pathlens.data.processed import ProcessedSplit
from pathlens.evaluation.report import evaluate_score_matrix
from pathlens.graph.skip import (
    build_skipgnn_adjacencies,
    labeled_entity_pairs,
    visible_training_edges,
)
from pathlens.runtime.device import as_numpy


def run_skipgnn(
    split: ProcessedSplit,
    config: dict[str, Any],
    *,
    device: str,
    stage: str,
    run_dir: str | Path | None = None,
) -> dict[str, Any]:
    torch = _require_torch()
    hyper = _hyperparams(config)
    epochs = 2 if stage == "smoke" else hyper["max_epochs"]
    _seed_everything(torch, hyper["seed"])

    visible = visible_training_edges(split.context, split.train_positive)
    original, skip = build_skipgnn_adjacencies(split.num_drugs, split.num_proteins, visible)
    num_entities = split.num_drugs + split.num_proteins
    features = _sparse_identity(torch, num_entities, device)
    o_adj = _to_torch_sparse(torch, original, device)
    s_adj = _to_torch_sparse(torch, skip, device)

    from pathlens.model.skipgnn import SkipGNN, score_all_pairs

    model = SkipGNN(
        nfeat=num_entities,
        nhid1=hyper["hidden1"],
        nhid2=hyper["hidden2"],
        nhid_decode1=hyper["hidden_decode1"],
        dropout=hyper["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyper["learning_rate"],
        weight_decay=hyper["weight_decay"],
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loader = _pair_loader(
        torch,
        split.train_positive,
        split.train_uniform,
        split.num_drugs,
        batch_size=hyper["batch_size"],
        device=device,
        shuffle=True,
    )

    started = time.perf_counter()
    best_auroc = -1.0
    best_epoch = 0
    best_state = {
        name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
    }
    last_loss = 0.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for left, right, labels in loader:
            optimizer.zero_grad()
            logits, _ = model(features, o_adj, s_adj, (left, right))
            loss = loss_fn(logits.squeeze(-1), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        last_loss = epoch_loss / max(1, batches)
        val_auroc = _validation_auroc(
            torch,
            model,
            features,
            o_adj,
            s_adj,
            split,
            device,
        )
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
        history.append({"epoch": epoch, "loss": last_loss, "val_auroc": val_auroc})
        print(
            f"[skipgnn] epoch={epoch}/{epochs} loss={last_loss:.4f} "
            f"val_auroc={val_auroc:.4f} best={best_auroc:.4f}",
            flush=True,
        )

    model.load_state_dict(best_state)
    model.to(device)
    scores = score_all_pairs(
        model,
        features,
        o_adj,
        s_adj,
        num_drugs=split.num_drugs,
        num_proteins=split.num_proteins,
    )
    full = stage == "eval"
    payload = evaluate_score_matrix(
        as_numpy(scores),
        split,
        include_curves=full,
        include_bootstrap=full,
        include_slices=full,
    )
    payload["score_seconds"] = time.perf_counter() - started
    payload["training"] = {
        "seed": hyper["seed"],
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_auroc": best_auroc,
        "final_loss": last_loss,
        "selection": "validation_auroc_1to1",
        "history": history,
        "hyperparameters": hyper,
    }
    if run_dir is not None and stage != "smoke":
        destination = Path(run_dir) / "model.pt"
        torch.save(
            {
                "state_dict": best_state,
                "method": "skipgnn",
                "best_val_auroc": best_auroc,
                "best_epoch": best_epoch,
                "hyperparameters": hyper,
            },
            destination,
        )
        payload["checkpoint"] = str(destination)
    return payload


def _hyperparams(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(config.get("seed", 13)),
        "learning_rate": float(config.get("learning_rate", 5e-4)),
        "weight_decay": float(config.get("weight_decay", 5e-4)),
        "max_epochs": int(config.get("max_epochs", 15)),
        "batch_size": int(config.get("batch_size", 256)),
        "dropout": float(config.get("dropout", 0.5)),
        "hidden1": int(config.get("hidden1", 64)),
        "hidden2": int(config.get("hidden2", 16)),
        "hidden_decode1": int(config.get("hidden_decode1", 512)),
    }


def _pair_loader(
    torch: Any,
    positives: np.ndarray,
    negatives: np.ndarray,
    num_drugs: int,
    *,
    batch_size: int,
    device: str,
    shuffle: bool,
) -> Any:
    pairs, labels = labeled_entity_pairs(positives, negatives, num_drugs)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(pairs[:, 0], dtype=torch.long, device=device),
        torch.as_tensor(pairs[:, 1], dtype=torch.long, device=device),
        torch.as_tensor(labels, dtype=torch.float32, device=device),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, len(dataset))),
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )


def _validation_auroc(
    torch: Any,
    model: Any,
    features: Any,
    o_adj: Any,
    s_adj: Any,
    split: ProcessedSplit,
    device: str,
) -> float:
    pairs, labels = labeled_entity_pairs(
        split.validation_positive,
        split.validation_uniform,
        split.num_drugs,
    )
    left = torch.as_tensor(pairs[:, 0], dtype=torch.long, device=device)
    right = torch.as_tensor(pairs[:, 1], dtype=torch.long, device=device)
    model.eval()
    with torch.inference_mode():
        logits, _ = model(features, o_adj, s_adj, (left, right))
    scores = logits.detach().cpu().numpy().reshape(-1)
    if len(np.unique(labels)) < 2:
        return 0.0
    return float(roc_auc_score(labels, scores))


def _sparse_identity(torch: Any, size: int, device: str) -> Any:
    index = torch.arange(size, device=device)
    return torch.sparse_coo_tensor(
        torch.stack((index, index)),
        torch.ones(size, dtype=torch.float32, device=device),
        (size, size),
    ).coalesce()


def _to_torch_sparse(torch: Any, matrix: Any, device: str) -> Any:
    from pathlens.model.skipgnn import scipy_to_torch_sparse

    return scipy_to_torch_sparse(matrix, device)


def _seed_everything(torch: Any, seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "skipgnn training requires PyTorch. Kaggle images already have it. "
            "Locally install the train extra only if you intend to run the trainer here."
        ) from error
    return torch
