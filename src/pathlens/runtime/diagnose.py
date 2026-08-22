from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pathlens.data.processed import ProcessedSplit
from pathlens.evaluation.ranking import (
    RANDOM_TIE_SEED,
    TIE_MODES,
    filtered_per_drug_ranking_from_scores,
    known_proteins_by_drug,
    mrr_by_drug_degree_tertile,
    query_ranks_from_scores,
    tie_diagnostics,
)
from pathlens.graph.scoring import build_adjacency, score_method
from pathlens.runtime.device import as_numpy, describe_device

DIAGNOSTIC_METHOD = "ranking_diagnostics"
SCORED_HEURISTICS = ("degree", "resource_allocation", "three_hop")
FILTERS = ("all_positive", "visible")


def run_ranking_diagnostics(
    split: ProcessedSplit,
    *,
    device: str,
    stage: str,
    run_dir: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    if stage == "train":
        raise ValueError("ranking_diagnostics is a diagnostic; use STAGE=smoke or STAGE=eval")
    started = time.perf_counter()
    include_pathlens = stage == "eval"
    adjacency = build_adjacency(split.num_drugs, split.num_proteins, split.context, device)
    scored: dict[str, NDArray[np.floating]] = {}
    for method_id in SCORED_HEURISTICS:
        scored[method_id] = as_numpy(score_method(method_id, adjacency)).astype(
            np.float64, copy=False
        )
    skipped: dict[str, str] = {}
    if include_pathlens:
        try:
            from pathlens.runtime.pathlens_infer import score_frozen_pathlens

            scored["pathlens_ranking"] = score_frozen_pathlens(
                split,
                device=device,
                repo_root=repo_root,
            )
        except (RuntimeError, FileNotFoundError, ImportError, ValueError) as error:
            skipped["pathlens_ranking"] = f"{type(error).__name__}: {error}"

    filters = {
        "all_positive": known_proteins_by_drug(split.all_positive),
        "visible": known_proteins_by_drug(visible_positive_pairs(split)),
    }
    drug_degree = np.bincount(split.context[:, 0], minlength=split.num_drugs).astype(np.float64)
    methods: dict[str, Any] = {}
    for method_id, matrix in scored.items():
        methods[method_id] = diagnose_score_matrix(
            matrix,
            split,
            filters=filters,
            drug_degree=drug_degree,
        )
    payload: dict[str, Any] = {
        "diagnostic": True,
        "filters": list(FILTERS),
        "tie_modes": list(TIE_MODES),
        "random_tie_seed": RANDOM_TIE_SEED,
        "methods": methods,
        "skipped": skipped,
        "score_seconds": time.perf_counter() - started,
        "environment": describe_device(device),
        "notes": (
            "Filed cards keep strict_gt + all_positive. These columns are diagnostics. "
            "visible = context ∪ train ∪ validation (no test array is read). "
            "all_positive is the canonical edge list and therefore includes test edges "
            "in the filter mask only."
        ),
    }
    if run_dir is not None and stage == "eval":
        written = write_diagnostic_figures(payload, Path(run_dir) / "figures")
        payload["figures"] = [str(path) for path in written]
    return payload


def diagnose_score_matrix(
    scores: NDArray[np.floating],
    split: ProcessedSplit,
    *,
    filters: dict[str, dict[int, frozenset[int]]],
    drug_degree: NDArray[np.floating],
) -> dict[str, Any]:
    positives = split.validation_positive
    by_filter: dict[str, Any] = {}
    for filter_name, known in filters.items():
        ranking: dict[str, Any] = {}
        for tie_mode in TIE_MODES:
            report = filtered_per_drug_ranking_from_scores(
                positives,
                scores=scores,
                known_by_drug=known,
                tie_mode=tie_mode,
                seed=RANDOM_TIE_SEED,
            )
            ranking[tie_mode] = report.to_dict()
        strict_ranks = query_ranks_from_scores(
            positives,
            scores=scores,
            known_by_drug=known,
            tie_mode="strict_gt",
        )
        by_filter[filter_name] = {
            "ranking": ranking,
            "ties": tie_diagnostics(positives, scores=scores, known_by_drug=known),
            "mrr_by_drug_degree": mrr_by_drug_degree_tertile(
                positives, strict_ranks, drug_degree
            ),
        }
    return by_filter


def visible_positive_pairs(split: ProcessedSplit) -> NDArray[np.int64]:
    stacked = np.concatenate(
        (split.context, split.train_positive, split.validation_positive),
        axis=0,
    )
    return np.unique(stacked, axis=0)


def write_diagnostic_figures(payload: dict[str, Any], directory: Path) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    directory.mkdir(parents=True, exist_ok=True)
    methods = [method_id for method_id in payload["methods"]]
    modes = list(TIE_MODES)
    values = [
        [
            payload["methods"][method_id]["all_positive"]["ranking"][mode]["mrr"]
            for mode in modes
        ]
        for method_id in methods
    ]
    figure, axis = plt.subplots(figsize=(10, 4.5))
    index = np.arange(len(methods))
    width = 0.25
    for offset, mode in enumerate(modes):
        axis.bar(index + (offset - 1) * width, [row[offset] for row in values], width, label=mode)
    axis.set_xticks(index)
    axis.set_xticklabels(methods, rotation=15, ha="right")
    axis.set_ylabel("validation MRR")
    axis.set_title("Tie-break diagnostics (all_positive filter)")
    axis.legend()
    axis.set_ylim(0, max(0.5, max(max(row) for row in values) * 1.15))
    figure.tight_layout()
    path = directory / "mrr_tie_break.png"
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return [path]
