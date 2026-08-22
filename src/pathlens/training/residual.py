from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from pathlens.data.processed import ProcessedSplit
from pathlens.evaluation.ranking import (
    filtered_per_drug_ranking_from_scores,
    known_proteins_by_drug,
)
from pathlens.evaluation.report import evaluate_score_matrix
from pathlens.graph.scoring import build_adjacency, score_three_hop
from pathlens.runtime.device import as_numpy
from pathlens.training.ranking import sample_ranked_negatives, sampled_softmax_loss
from pathlens.training.skipgnn import _require_torch, _seed_everything


def run_residual_three_hop(
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

    hop = as_numpy(
        score_three_hop(build_adjacency(split.num_drugs, split.num_proteins, split.context, device))
    ).astype(np.float32, copy=False)
    hop_tensor = torch.as_tensor(hop, dtype=torch.float32, device=device)

    from pathlens.model.residual import ResidualThreeHop

    model = ResidualThreeHop(
        split.num_drugs,
        split.num_proteins,
        dim=hyper["dim"],
        hidden=hyper["hidden"],
        dropout=hyper["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hyper["learning_rate"],
        weight_decay=hyper["weight_decay"],
    )
    known = known_proteins_by_drug(split.all_positive)
    protein_degrees = np.bincount(split.context[:, 1], minlength=split.num_proteins).astype(
        np.int64
    )
    train_positive = np.asarray(split.train_positive, dtype=np.int64)
    drugs = torch.as_tensor(train_positive[:, 0], dtype=torch.long, device=device)
    proteins = torch.as_tensor(train_positive[:, 1], dtype=torch.long, device=device)

    started = time.perf_counter()
    best_mrr = -1.0
    best_epoch = 0
    best_state = _clone_state(model)
    last_loss = 0.0
    history: list[dict[str, float | int]] = []
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        negatives = sample_ranked_negatives(
            train_positive,
            num_proteins=split.num_proteins,
            known_by_drug=known,
            protein_degrees=protein_degrees,
            num_negatives=hyper["num_negatives"],
            hard_fraction=hyper["hard_fraction"],
            seed=hyper["seed"] + epoch * 1_000_003,
        )
        neg_proteins = torch.as_tensor(negatives, dtype=torch.long, device=device)
        pos_hop = hop_tensor[drugs, proteins]
        neg_hop = hop_tensor[drugs.unsqueeze(1), neg_proteins]
        positive_logits = model.forward(drugs, proteins, pos_hop)
        negative_logits = model.forward(
            drugs.repeat_interleave(hyper["num_negatives"]),
            neg_proteins.reshape(-1),
            neg_hop.reshape(-1),
        ).view(len(train_positive), hyper["num_negatives"])
        loss = sampled_softmax_loss(
            positive_logits,
            negative_logits,
            temperature=hyper["temperature"],
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        val_mrr = _validation_mrr(model, hop_tensor, split, known)
        if val_mrr > best_mrr:
            best_mrr = val_mrr
            best_epoch = epoch
            best_state = _clone_state(model)
            stale = 0
        else:
            stale += 1
        history.append({"epoch": epoch, "loss": last_loss, "val_mrr": val_mrr})
        print(
            f"[residual_three_hop] epoch={epoch}/{epochs} loss={last_loss:.4f} "
            f"val_mrr={val_mrr:.4f} best={best_mrr:.4f}",
            flush=True,
        )
        if stage != "smoke" and stale >= hyper["patience"]:
            break

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    scores = as_numpy(model.score_matrix(hop_tensor))
    full = stage == "eval"
    payload = evaluate_score_matrix(
        scores,
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
        "best_val_mrr": best_mrr,
        "final_loss": last_loss,
        "selection": "validation_filtered_mrr",
        "history": history,
        "hyperparameters": hyper,
        "base": "three_hop",
    }
    if run_dir is not None and stage != "smoke":
        destination = Path(run_dir) / "model.pt"
        torch.save(
            {
                "state_dict": best_state,
                "method": "residual_three_hop",
                "best_val_mrr": best_mrr,
                "best_epoch": best_epoch,
                "hyperparameters": hyper,
            },
            destination,
        )
        payload["checkpoint"] = str(destination)
    return payload


def _validation_mrr(
    model: Any,
    hop_tensor: Any,
    split: ProcessedSplit,
    known: dict[int, frozenset[int]],
) -> float:
    model.eval()
    torch = _require_torch()
    with torch.inference_mode():
        scores = as_numpy(model.score_matrix(hop_tensor))
    return filtered_per_drug_ranking_from_scores(
        split.validation_positive,
        scores=scores,
        known_by_drug=known,
    ).mrr


def _clone_state(model: Any) -> dict[str, Any]:
    return {
        key: {name: tensor.detach().cpu().clone() for name, tensor in block.items()}
        for key, block in model.state_dict().items()
    }


def _hyperparams(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(config.get("seed", 13)),
        "learning_rate": float(config.get("learning_rate", 1e-3)),
        "weight_decay": float(config.get("weight_decay", 1e-4)),
        "max_epochs": int(config.get("max_epochs", 40)),
        "patience": int(config.get("patience", 8)),
        "dim": int(config.get("dim", 32)),
        "hidden": int(config.get("hidden", 64)),
        "dropout": float(config.get("dropout", 0.1)),
        "num_negatives": int(config.get("num_negatives", 64)),
        "hard_fraction": float(config.get("hard_fraction", 0.25)),
        "temperature": float(config.get("temperature", 1.0)),
    }
