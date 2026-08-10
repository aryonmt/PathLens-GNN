from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from pathlens_gnn.evaluation.metrics import classification_report, select_f1_threshold, sigmoid
from pathlens_gnn.evaluation.ranking import filtered_per_drug_ranking
from pathlens_gnn.graph.index import BipartiteIndex
from pathlens_gnn.model.pathlens import PathLensConfig, PathLensGNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the one-time sealed PathLens evaluation.")
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--freeze-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-sealed-test", action="store_true")
    args = parser.parse_args()
    if not args.confirm_sealed_test:
        raise SystemExit("Refusing to open test data without --confirm-sealed-test")
    freeze = json.loads(args.freeze_record.read_text(encoding="utf-8"))
    if freeze.get("checkpoint_sha256") != _sha256(args.checkpoint):
        raise SystemExit("Freeze record does not match the selected checkpoint")
    result = evaluate(args.processed, args.checkpoint)
    result["freeze_record"] = freeze
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def evaluate(processed: Path, checkpoint_path: Path) -> dict[str, Any]:
    entities = json.loads((processed / "entities.json").read_text(encoding="utf-8"))
    arrays = np.load(processed / "splits.npz")
    context = arrays["context"].astype(np.int64)
    num_drugs = len(entities["drugs"])
    num_proteins = len(entities["proteins"])
    graph = BipartiteIndex(num_drugs, num_proteins, context)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = PathLensConfig(**checkpoint["training_config"]["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PathLensGNN(
        num_drugs,
        num_proteins,
        torch.as_tensor(context.T, dtype=torch.long, device=device),
        config,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        embeddings = model.encode()

    projection_drug = np.asarray(
        [graph.projection_mass_drug(index) for index in range(num_drugs)], dtype=np.float32
    )
    projection_protein = np.asarray(
        [graph.projection_mass_protein(index) for index in range(num_proteins)],
        dtype=np.float32,
    )

    def score_pairs(pairs: NDArray[np.int64]) -> NDArray[np.float64]:
        pairs = np.asarray(pairs, dtype=np.int64)
        result = np.empty(len(pairs), dtype=np.float64)
        for drug in np.unique(pairs[:, 0]):
            selected = np.flatnonzero(pairs[:, 0] == drug)
            proteins = pairs[selected, 1]
            bridge = graph.bridge_scores_for_drug(int(drug))[proteins]
            features = np.column_stack(
                (projection_drug[int(drug)] + projection_protein[proteins], bridge)
            )
            with torch.inference_mode():
                output = model.score_from_embeddings(
                    embeddings,
                    torch.full((len(selected),), int(drug), dtype=torch.long, device=device),
                    torch.as_tensor(proteins, dtype=torch.long, device=device),
                    torch.as_tensor(features, dtype=torch.float32, device=device),
                )
            result[selected] = output.logit.detach().cpu().numpy()
        return result

    validation_pairs, validation_labels = _candidates(
        arrays["validation_positive"], arrays["validation_hard"]
    )
    validation_logits = score_pairs(validation_pairs)
    threshold = select_f1_threshold(validation_labels, sigmoid(validation_logits))
    reports: dict[str, Any] = {}
    for negative_name in ("test_uniform", "test_hard"):
        pairs, labels = _candidates(arrays["test_positive"], arrays[negative_name])
        logits = score_pairs(pairs)
        report = classification_report(labels, logits, threshold=threshold)
        reports[negative_name] = {
            **report.to_dict(),
            "auprc_bootstrap_95_ci": _bootstrap_auprc(labels, logits),
            "degree_slices": _degree_slices(labels, logits, pairs, graph, threshold),
        }

    known_by_drug: dict[int, frozenset[int]] = {}
    for drug in range(num_drugs):
        known_by_drug[drug] = frozenset(
            int(protein) for source, protein in arrays["all_positive"] if int(source) == drug
        )
    ranking_started = time.perf_counter()
    ranking = filtered_per_drug_ranking(
        arrays["test_positive"],
        num_proteins=num_proteins,
        known_by_drug=known_by_drug,
        score_pairs=score_pairs,
    )
    sample_pairs = np.column_stack(
        (np.zeros(min(512, num_proteins), dtype=np.int64), np.arange(min(512, num_proteins)))
    )
    score_pairs(sample_pairs)
    latency_started = time.perf_counter()
    score_pairs(sample_pairs)
    latency_ms = (time.perf_counter() - latency_started) * 1000 / len(sample_pairs)
    return {
        "checkpoint_sha256": _sha256(checkpoint_path),
        "seed": checkpoint["training_config"]["seed"],
        "validation_selected_threshold": threshold,
        "classification": reports,
        "filtered_ranking": asdict(ranking),
        "ranking_seconds": time.perf_counter() - ranking_started,
        "inference_latency_ms_per_pair": latency_ms,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "device": str(device),
    }


def _candidates(
    positives: NDArray[np.int64], negatives: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    return (
        np.concatenate((positives, negatives)).astype(np.int64),
        np.concatenate((np.ones(len(positives)), np.zeros(len(negatives)))).astype(np.int64),
    )


def _bootstrap_auprc(
    labels: NDArray[np.int64], logits: NDArray[np.float64], *, samples: int = 400
) -> list[float]:
    generator = np.random.default_rng(20260809)
    values: list[float] = []
    for _ in range(samples):
        selected = generator.integers(0, len(labels), len(labels))
        if len(np.unique(labels[selected])) == 2:
            values.append(classification_report(labels[selected], logits[selected]).auprc)
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def _degree_slices(
    labels: NDArray[np.int64],
    logits: NDArray[np.float64],
    pairs: NDArray[np.int64],
    graph: BipartiteIndex,
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    degrees = np.asarray(
        [
            min(len(graph.drug_neighbors[int(drug)]), len(graph.protein_neighbors[int(protein)]))
            for drug, protein in pairs
        ]
    )
    lower, upper = np.quantile(degrees, [1 / 3, 2 / 3])
    masks = {
        "low": degrees <= lower,
        "mid": (degrees > lower) & (degrees <= upper),
        "high": degrees > upper,
    }
    return {
        name: classification_report(labels[mask], logits[mask], threshold=threshold).to_dict()
        for name, mask in masks.items()
        if mask.any() and len(np.unique(labels[mask])) == 2
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
