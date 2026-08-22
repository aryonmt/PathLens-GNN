from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from pathlens_gnn.evaluation.metrics import (
    bootstrap_auprc,
    classification_curves,
    classification_report,
    select_f1_threshold,
    sigmoid,
)
from pathlens_gnn.evaluation.ranking import (
    hits_curve,
    query_ranks_from_scores,
    ranking_report_from_ranks,
)
from pathlens_gnn.evaluation.slices import degree_slices
from pathlens_gnn.graph.index import BipartiteIndex
from pathlens_gnn.model.pathlens import PathLensConfig, PathLensGNN
from pathlens_gnn.training.ranking import known_proteins_by_drug
from pathlens_gnn.training.scoring import DevicePairFeatures, score_all_proteins

TEST_KEYS = ("test_positive", "test_uniform", "test_hard")
ScoreFn = Callable[[NDArray[np.int64]], tuple[NDArray[np.float64], NDArray[np.float64] | None]]
MatrixFn = Callable[[NDArray[np.int64]], NDArray[np.float64]]


def write_validation_report(
    processed: Path,
    output: Path,
    *,
    checkpoint: Path,
    freeze_record: Path | None = None,
    registered: Path | None = None,
) -> dict[str, Any]:
    payload = _evaluate_split(
        processed,
        checkpoint,
        split="validation",
        freeze_record=freeze_record,
        registered=registered,
    )
    leaked = [key for key in TEST_KEYS if key in payload]
    if leaked:
        raise RuntimeError(f"Validation report must not include {leaked}")
    output.mkdir(parents=True, exist_ok=True)
    arrays = payload.pop("arrays")
    (output / "validation-report.json").write_text(
        json.dumps(_jsonable(payload), indent=2), encoding="utf-8"
    )
    np.savez_compressed(output / "validation-report.npz", **arrays)
    print(json.dumps(_jsonable(_summary(payload)), indent=2), flush=True)
    payload["arrays"] = arrays
    return payload


def write_final_artifacts(
    processed: Path,
    checkpoint: Path,
    output_json: Path,
    *,
    freeze_record: Path,
) -> dict[str, Any]:
    payload = _evaluate_split(
        processed,
        checkpoint,
        split="test",
        freeze_record=freeze_record,
        registered=None,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    arrays = payload.pop("arrays")
    output_json.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    np.savez_compressed(output_json.with_name("final-artifacts.npz"), **arrays)
    payload["arrays"] = arrays
    return payload


def _evaluate_split(
    processed: Path,
    checkpoint: Path,
    *,
    split: str,
    freeze_record: Path | None,
    registered: Path | None,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    processed = Path(processed)
    checkpoint = Path(checkpoint)
    entities = json.loads((processed / "entities.json").read_text(encoding="utf-8"))
    arrays = np.load(processed / "splits.npz")
    graph = BipartiteIndex(len(entities["drugs"]), len(entities["proteins"]), arrays["context"])
    known = known_proteins_by_drug(arrays["all_positive"].astype(np.int64))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[report] split={split} device={device}", flush=True)
    print("[report] building pair-feature tables...", flush=True)
    pair_features = DevicePairFeatures.from_index(graph, device)
    freeze = None
    if freeze_record is not None:
        freeze = json.loads(Path(freeze_record).read_text(encoding="utf-8"))
        if freeze.get("checkpoint_sha256") != _sha256(checkpoint):
            raise SystemExit("Freeze record does not match the selected checkpoint")

    models: dict[str, Any] = {}
    stored: dict[str, NDArray[np.floating]] = {}
    models["full_adaptive_ranking"], ranking_arrays = _evaluate_named(
        "full_adaptive_ranking",
        lambda: _evaluate_checkpoint(
            Path(checkpoint), arrays, graph, pair_features, known, device, split=split
        ),
    )
    stored.update(_prefix_arrays("full_adaptive_ranking", ranking_arrays))
    for name, score_pairs, score_matrix in _heuristic_scorers(graph):
        summary, values = _evaluate_named(
            name,
            lambda score_pairs=score_pairs, score_matrix=score_matrix: _evaluate_score_fns(
                score_pairs, score_matrix, arrays, graph, known, split=split
            ),
        )
        models[name] = summary
        stored.update(_prefix_arrays(name, values))
    if registered is not None:
        registry_path = Path("configs/experiments/registered.yaml")
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        for experiment in registry["models"]:
            name = str(experiment["name"])
            if name == "full_adaptive_ranking":
                continue
            ckpt = Path(registered) / name / "checkpoint.pt"
            if not ckpt.exists():
                print(f"[report] skip {name}: missing {ckpt}", flush=True)
                continue
            models[name], values = _evaluate_named(
                name,
                lambda ckpt=ckpt: _evaluate_checkpoint(
                    ckpt, arrays, graph, pair_features, known, device, split=split
                ),
            )
            stored.update(_prefix_arrays(name, values))
    else:
        print("[report] registered baselines not attached", flush=True)
    return {
        "split": split,
        "device": str(device),
        "checkpoint_sha256": _sha256(Path(checkpoint)),
        "freeze_record": freeze,
        "models": models,
        "arrays": stored,
    }


def _evaluate_checkpoint(
    checkpoint_path: Path,
    arrays: np.lib.npyio.NpzFile,
    graph: BipartiteIndex,
    pair_features: DevicePairFeatures,
    known: dict[int, frozenset[int]],
    device: torch.device,
    *,
    split: str,
) -> tuple[dict[str, Any], dict[str, NDArray[np.floating]]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = PathLensConfig(**checkpoint["training_config"]["model"])
    model = PathLensGNN(
        graph.num_drugs,
        graph.num_proteins,
        torch.as_tensor(arrays["context"].T, dtype=torch.long, device=device),
        config,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        embeddings = model.encode()

    def score_pairs(pairs: NDArray[np.int64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        pairs = np.asarray(pairs, dtype=np.int64)
        drugs = torch.as_tensor(pairs[:, 0], dtype=torch.long, device=device)
        proteins = torch.as_tensor(pairs[:, 1], dtype=torch.long, device=device)
        with torch.inference_mode():
            output = model.score_from_embeddings(
                embeddings,
                drugs,
                proteins,
                pair_features.lookup(drugs, proteins),
            )
        return (
            output.logit.detach().cpu().numpy().astype(np.float64, copy=False),
            output.gate_weights.detach().cpu().numpy().astype(np.float64, copy=False),
        )

    def score_matrix(drugs: NDArray[np.int64]) -> NDArray[np.float64]:
        drug_tensor = torch.as_tensor(drugs, dtype=torch.long, device=device)
        with torch.inference_mode():
            matrix = score_all_proteins(model, embeddings, pair_features, drug_tensor)
        return matrix.detach().cpu().numpy().astype(np.float64, copy=False)

    summary, values = _evaluate_score_fns(
        score_pairs, score_matrix, arrays, graph, known, split=split
    )
    summary["checkpoint"] = str(checkpoint_path)
    return summary, values


def _evaluate_named(
    name: str,
    run: Callable[[], tuple[dict[str, Any], dict[str, NDArray[np.floating]]]],
) -> tuple[dict[str, Any], dict[str, NDArray[np.floating]]]:
    print(f"[report] scoring {name}...", flush=True)
    summary, values = run()
    ranking = summary["filtered_ranking"]
    hard = summary["classification"]["hard"]
    print(
        f"[report] {name} hard_auprc={hard['auprc']:.4f} mrr={ranking['mrr']:.4f}",
        flush=True,
    )
    return summary, values


def _heuristic_scorers(
    graph: BipartiteIndex,
) -> list[tuple[str, ScoreFn, MatrixFn]]:
    drug_degree = np.asarray(
        [len(graph.drug_neighbors[index]) for index in range(graph.num_drugs)],
        dtype=np.float64,
    )
    protein_degree = np.asarray(
        [len(graph.protein_neighbors[index]) for index in range(graph.num_proteins)],
        dtype=np.float64,
    )
    _drug_mass, _protein_mass, bridge = graph.pair_feature_tables()

    def degree_pairs(pairs: NDArray[np.int64]) -> tuple[NDArray[np.float64], None]:
        pairs = np.asarray(pairs, dtype=np.int64)
        return np.log1p(drug_degree[pairs[:, 0]] * protein_degree[pairs[:, 1]]), None

    def degree_matrix(drugs: NDArray[np.int64]) -> NDArray[np.float64]:
        drugs = np.asarray(drugs, dtype=np.int64)
        return np.log1p(drug_degree[drugs][:, None] * protein_degree[None, :])

    def three_hop_pairs(pairs: NDArray[np.int64]) -> tuple[NDArray[np.float64], None]:
        pairs = np.asarray(pairs, dtype=np.int64)
        return np.asarray(bridge[pairs[:, 0], pairs[:, 1]], dtype=np.float64), None

    def three_hop_matrix(drugs: NDArray[np.int64]) -> NDArray[np.float64]:
        return np.asarray(bridge[np.asarray(drugs, dtype=np.int64)], dtype=np.float64)

    return [
        ("degree_type_shortcut", degree_pairs, degree_matrix),
        ("normalized_three_hop", three_hop_pairs, three_hop_matrix),
    ]


def _evaluate_score_fns(
    score_pairs: ScoreFn,
    score_matrix: MatrixFn,
    arrays: np.lib.npyio.NpzFile,
    graph: BipartiteIndex,
    known: dict[int, frozenset[int]],
    *,
    split: str,
) -> tuple[dict[str, Any], dict[str, NDArray[np.floating]]]:
    threshold_pairs, threshold_labels = _candidates(
        arrays["validation_positive"], arrays["validation_hard"]
    )
    threshold_logits, _gate = score_pairs(threshold_pairs)
    threshold = select_f1_threshold(threshold_labels, sigmoid(threshold_logits))
    classification: dict[str, Any] = {}
    stored: dict[str, NDArray[np.floating]] = {}
    for negative in ("uniform", "hard"):
        pairs, labels = _candidates(arrays[f"{split}_positive"], arrays[f"{split}_{negative}"])
        logits, gate = score_pairs(pairs)
        curves = classification_curves(labels, logits)
        classification[negative] = {
            **classification_report(labels, logits, threshold=threshold).to_dict(),
            "auprc_bootstrap_95_ci": bootstrap_auprc(labels, logits),
            "degree_slices": degree_slices(labels, logits, pairs, graph, threshold),
            "curves": {key: value.tolist() for key, value in curves.items()},
        }
        stored[f"{negative}_logits"] = logits
        stored[f"{negative}_labels"] = labels.astype(np.float64)
        if gate is not None:
            stored[f"{negative}_gate_weights"] = gate
            positives = len(arrays[f"{split}_positive"])
            classification["gate_mean_on_positives"] = gate[:positives].mean(axis=0).tolist()
    positives = arrays[f"{split}_positive"].astype(np.int64)
    unique_drugs = np.unique(positives[:, 0])
    matrix = score_matrix(unique_drugs)
    ranks = query_ranks_from_scores(
        positives, scores=matrix, known_by_drug=known, drug_ids=unique_drugs
    )
    stored["ranks"] = ranks.astype(np.float64)
    ranking = ranking_report_from_ranks(ranks)
    return (
        {
            "validation_selected_threshold": threshold,
            "classification": classification,
            "filtered_ranking": {**asdict(ranking), **hits_curve(ranks)},
        },
        stored,
    )


def _candidates(
    positives: NDArray[np.int64], negatives: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    return (
        np.concatenate((positives, negatives)).astype(np.int64),
        np.concatenate((np.ones(len(positives)), np.zeros(len(negatives)))).astype(np.int64),
    )


def _prefix_arrays(
    name: str, values: dict[str, NDArray[np.floating]]
) -> dict[str, NDArray[np.floating]]:
    return {f"{name}__{key}": value for key, value in values.items()}


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    models = {}
    for name, model in payload["models"].items():
        ranking = model["filtered_ranking"]
        hard = model["classification"]["hard"]
        models[name] = {
            "hard_auprc": hard["auprc"],
            "uniform_auprc": model["classification"]["uniform"]["auprc"],
            "mrr": ranking["mrr"],
            "hits_at_10": ranking["hits_at_10"],
            "hits_at_50": ranking["hits_at_50"],
        }
    return {"split": payload["split"], "device": payload["device"], "models": models}


def _sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating | np.integer):
        return value.item()
    return value
