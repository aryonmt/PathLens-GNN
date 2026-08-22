# Copyright (c) 2026 Amirreza Nemati
# SPDX-License-Identifier: BSD-3-Clause
#
# Same repaired BCE loop as GCN/SkipGNN. Encoder is Hamilton mean-SAGE on
# unnormalized A without self-loops.

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pathlens.data.processed import ProcessedSplit
from pathlens.evaluation.report import evaluate_score_matrix
from pathlens.graph.skip import build_symmetric_adjacency, visible_training_edges
from pathlens.runtime.device import as_numpy
from pathlens.training.gcn import _gcn_validation_auroc
from pathlens.training.skipgnn import (
    _hyperparams,
    _pair_loader,
    _require_torch,
    _seed_everything,
    _sparse_identity,
    _to_torch_sparse,
)


def run_graphsage(
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
    adjacency = build_symmetric_adjacency(split.num_drugs, split.num_proteins, visible)
    num_entities = split.num_drugs + split.num_proteins
    features = _sparse_identity(torch, num_entities, device)
    adj = _to_torch_sparse(torch, adjacency, device)

    from pathlens.model.decode import score_from_embeddings
    from pathlens.model.graphsage import GraphSAGE

    model = GraphSAGE(
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
            logits, _ = model(features, adj, (left, right))
            loss = loss_fn(logits.squeeze(-1), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1
        last_loss = epoch_loss / max(1, batches)
        val_auroc = _gcn_validation_auroc(torch, model, features, adj, split, device)
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
        history.append({"epoch": epoch, "loss": last_loss, "val_auroc": val_auroc})
        print(
            f"[graphsage] epoch={epoch}/{epochs} loss={last_loss:.4f} "
            f"val_auroc={val_auroc:.4f} best={best_auroc:.4f}",
            flush=True,
        )

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        embeddings = model.encode(features, adj)
        scores = score_from_embeddings(
            model,
            embeddings,
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
                "method": "graphsage",
                "best_val_auroc": best_auroc,
                "best_epoch": best_epoch,
                "hyperparameters": hyper,
            },
            destination,
        )
        payload["checkpoint"] = str(destination)
    return payload
