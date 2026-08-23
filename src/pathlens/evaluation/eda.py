from __future__ import annotations

from typing import Any

import numpy as np

from pathlens.data.processed import ProcessedSplit
from pathlens.graph.scoring import build_adjacency, score_three_hop
from pathlens.runtime.device import as_numpy


def summarize_split(split: ProcessedSplit) -> dict[str, Any]:
    context = np.asarray(split.context, dtype=np.int64)
    all_positive = np.asarray(split.all_positive, dtype=np.int64)
    drug_degree = np.bincount(context[:, 0], minlength=split.num_drugs)
    protein_degree = np.bincount(context[:, 1], minlength=split.num_proteins)
    return {
        "num_drugs": split.num_drugs,
        "num_proteins": split.num_proteins,
        "seed": split.seed,
        "edges": {
            "all_positive": int(len(all_positive)),
            "context": int(len(context)),
            "train_positive": int(len(split.train_positive)),
            "validation_positive": int(len(split.validation_positive)),
        },
        "degree": {
            "drug": _degree_stats(drug_degree),
            "protein": _degree_stats(protein_degree),
        },
        "coverage": {
            "drugs_with_context": int((drug_degree > 0).sum()),
            "proteins_with_context": int((protein_degree > 0).sum()),
            "validation_drugs_with_context": _covered_fraction(
                split.validation_positive[:, 0], drug_degree
            ),
        },
    }


def hop_reachability(
    split: ProcessedSplit,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    adjacency = build_adjacency(split.num_drugs, split.num_proteins, split.context, device)
    hop = as_numpy(score_three_hop(adjacency))
    pairs = np.asarray(split.validation_positive, dtype=np.int64)
    scores = hop[pairs[:, 0], pairs[:, 1]]
    zero = scores <= 0
    return {
        "validation_queries": int(len(pairs)),
        "three_hop_zero": int(zero.sum()),
        "three_hop_zero_fraction": float(zero.mean()) if len(pairs) else 0.0,
        "three_hop_positive_mean": float(scores[~zero].mean()) if (~zero).any() else 0.0,
    }


def write_eda_figures(
    split: ProcessedSplit,
    output: str | Any,
    *,
    device: str = "cpu",
) -> list[Any]:
    from pathlib import Path

    from pathlens.evaluation.figures import _pyplot

    plt = _pyplot()
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    context = np.asarray(split.context, dtype=np.int64)
    drug_degree = np.bincount(context[:, 0], minlength=split.num_drugs)
    protein_degree = np.bincount(context[:, 1], minlength=split.num_proteins)
    written = [
        _degree_hist(plt, drug_degree, output_dir / "degree_hist_drugs.png", "Drug context degree"),
        _degree_hist(
            plt,
            protein_degree,
            output_dir / "degree_hist_proteins.png",
            "Protein context degree",
        ),
        _split_sizes(plt, split, output_dir / "split_sizes.png"),
    ]
    reach = hop_reachability(split, device=device)
    written.append(_zero_mass(plt, reach, output_dir / "three_hop_zero_mass.png"))
    return written


def _degree_stats(values: np.ndarray) -> dict[str, float | int]:
    positive = values[values > 0]
    return {
        "mean": float(values.mean()),
        "mean_nonzero": float(positive.mean()) if len(positive) else 0.0,
        "median_nonzero": float(np.median(positive)) if len(positive) else 0.0,
        "max": int(values.max()) if len(values) else 0,
        "zeros": int((values == 0).sum()),
    }


def _covered_fraction(node_ids: np.ndarray, degrees: np.ndarray) -> float:
    if len(node_ids) == 0:
        return 1.0
    return float((degrees[np.asarray(node_ids, dtype=np.int64)] > 0).mean())


def _degree_hist(plt: Any, degrees: np.ndarray, path: Any, title: str) -> Any:
    figure, axes = plt.subplots(figsize=(6.2, 4.0))
    positive = degrees[degrees > 0]
    bins = min(40, max(5, int(np.sqrt(max(1, len(positive))))))
    axes.hist(positive, bins=bins, color="#1f4e79", alpha=0.85)
    axes.set_xlabel("Context degree")
    axes.set_ylabel("Count")
    axes.set_title(title)
    axes.grid(True, axis="y", alpha=0.3)
    from pathlens.evaluation.figures import _save

    return _save(plt, figure, path)


def _split_sizes(plt: Any, split: ProcessedSplit, path: Any) -> Any:
    from pathlens.evaluation.figures import _save

    labels = ["Context", "Train +", "Val +", "All +"]
    values = [
        len(split.context),
        len(split.train_positive),
        len(split.validation_positive),
        len(split.all_positive),
    ]
    figure, axes = plt.subplots(figsize=(6.0, 4.0))
    axes.bar(labels, values, color="#1f4e79")
    axes.set_ylabel("Edges")
    axes.set_title("Coverage-preserving split sizes")
    axes.grid(True, axis="y", alpha=0.3)
    return _save(plt, figure, path)


def _zero_mass(plt: Any, reach: dict[str, Any], path: Any) -> Any:
    from pathlens.evaluation.figures import _save

    figure, axes = plt.subplots(figsize=(5.4, 4.0))
    zero = reach["three_hop_zero_fraction"]
    axes.bar(["3-hop = 0", "3-hop > 0"], [zero, 1.0 - zero], color=["#8c2d04", "#1f4e79"])
    axes.set_ylim(0, 1)
    axes.set_ylabel("Fraction of validation positives")
    axes.set_title("Zero-mass 3-hop targets (no L3 witness in context)")
    axes.grid(True, axis="y", alpha=0.3)
    return _save(plt, figure, path)
