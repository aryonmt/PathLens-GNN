from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from pathlens.data.processed import ProcessedSplit
from pathlens.evaluation.metrics import (
    bootstrap_auprc,
    classification_curves,
    classification_report,
)
from pathlens.evaluation.ranking import (
    filtered_per_drug_ranking_from_scores,
    known_proteins_by_drug,
)
from pathlens.evaluation.slices import degree_slices
from pathlens.graph.scoring import lookup_pairs
from pathlens.runtime.device import as_numpy

SplitName = Literal["validation", "test"]
FULL_STAGES = frozenset({"eval", "final"})


def evaluate_score_matrix(
    scores: Any,
    processed: ProcessedSplit,
    *,
    include_curves: bool = True,
    include_bootstrap: bool = True,
    include_slices: bool = True,
    split_name: SplitName = "validation",
) -> dict[str, Any]:
    matrix = as_numpy(scores).astype(np.float64, copy=False)
    positives, hard_negatives, uniform_negatives = _query_banks(processed, split_name)
    known = known_proteins_by_drug(processed.all_positive)
    ranking = filtered_per_drug_ranking_from_scores(
        positives,
        scores=matrix,
        known_by_drug=known,
    )
    hard = _bank_report(
        matrix,
        positives,
        hard_negatives,
        include_curves=include_curves,
        include_bootstrap=include_bootstrap,
    )
    uniform = _bank_report(
        matrix,
        positives,
        uniform_negatives,
        include_curves=include_curves,
        include_bootstrap=include_bootstrap,
    )
    if include_slices:
        pairs, labels, logits = _bank_arrays(matrix, positives, hard_negatives)
        drug_degree = np.bincount(processed.context[:, 0], minlength=processed.num_drugs)
        protein_degree = np.bincount(processed.context[:, 1], minlength=processed.num_proteins)
        hard["degree_slices"] = degree_slices(
            labels,
            logits,
            pairs,
            drug_degree.astype(np.float64),
            protein_degree.astype(np.float64),
            threshold=float(hard["threshold"]),
        )
    return {
        "split": split_name,
        "filtered_ranking": ranking.to_dict(),
        "classification": {"hard": hard, "uniform": uniform},
    }


def evaluate_for_stage(scores: Any, processed: ProcessedSplit, stage: str) -> dict[str, Any]:
    full = stage in FULL_STAGES
    kwargs = {
        "include_curves": full,
        "include_bootstrap": full,
        "include_slices": full,
    }
    payload = evaluate_score_matrix(scores, processed, split_name="validation", **kwargs)
    if stage == "final":
        payload["test"] = evaluate_score_matrix(scores, processed, split_name="test", **kwargs)
    return payload


def _query_banks(
    processed: ProcessedSplit, split_name: SplitName
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    if split_name == "validation":
        return (
            processed.validation_positive,
            processed.validation_hard,
            processed.validation_uniform,
        )
    if split_name == "test":
        return (
            processed.test_array("test_positive"),
            processed.test_array("test_hard"),
            processed.test_array("test_uniform"),
        )
    raise ValueError(f"Unknown split {split_name!r}")


def _bank_report(
    scores: NDArray[np.float64],
    positives: NDArray[np.int64],
    negatives: NDArray[np.int64],
    *,
    include_curves: bool,
    include_bootstrap: bool,
) -> dict[str, Any]:
    _pairs, labels, logits = _bank_arrays(scores, positives, negatives)
    report = classification_report(labels, logits).to_dict()
    payload: dict[str, Any] = dict(report)
    if include_curves:
        payload["curves"] = {
            name: np.asarray(values).tolist()
            for name, values in classification_curves(labels, logits).items()
        }
    if include_bootstrap:
        payload["auprc_ci"] = bootstrap_auprc(labels, logits)
    return payload


def _bank_arrays(
    scores: NDArray[np.float64] | Any,
    positives: NDArray[np.int64],
    negatives: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    pairs = np.concatenate((positives, negatives)).astype(np.int64)
    labels = np.concatenate((np.ones(len(positives)), np.zeros(len(negatives)))).astype(np.int64)
    if isinstance(scores, np.ndarray) and scores.ndim == 2:
        logits = scores[pairs[:, 0], pairs[:, 1]]
    else:
        logits = lookup_pairs(scores, pairs)
    return pairs, labels, np.asarray(logits, dtype=np.float64)
