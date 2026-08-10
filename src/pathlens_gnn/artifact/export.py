from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pathlens_gnn.artifact.runtime import ArtifactRuntime, sha256_file
from pathlens_gnn.constants import SCHEMA_VERSION
from pathlens_gnn.graph.index import BipartiteIndex
from pathlens_gnn.model.pathlens import PathLensConfig, PathLensGNN


def export_inference_artifact(
    processed_dir: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    model_version: str,
    top_k: int = 100,
    explanation_k: int = 25,
) -> dict[str, Any]:
    processed = Path(processed_dir)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Artifact output must be empty: {output}")
    (output / "embeddings").mkdir(parents=True, exist_ok=True)
    (output / "graph").mkdir(parents=True, exist_ok=True)

    entities = json.loads((processed / "entities.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((processed / "manifest.json").read_text(encoding="utf-8"))
    arrays = np.load(processed / "splits.npz")
    context = arrays["context"].astype(np.int64)
    all_positive = arrays["all_positive"].astype(np.int64)
    num_drugs = len(entities["drugs"])
    num_proteins = len(entities["proteins"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_config = PathLensConfig(**checkpoint["training_config"]["model"])
    model = PathLensGNN(
        num_drugs,
        num_proteins,
        torch.as_tensor(context.T, dtype=torch.long),
        model_config,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        encoded = model.encode()
    embedding_arrays = [value.detach().cpu().numpy().astype(np.float32) for value in encoded]
    for hop, values in enumerate(embedding_arrays, start=1):
        np.save(output / f"embeddings/s{hop}.npy", values)
    np.save(output / "graph/context_edges.npy", context)
    np.save(output / "graph/known_edges.npy", all_positive)

    state = model.state_dict()
    export_keys = [key for key in state if key.startswith("experts.") or key.startswith("gate.")]
    np.savez(
        output / "model_weights.npz",
        **{key: state[key].detach().cpu().numpy() for key in export_keys},
    )

    graph_index = BipartiteIndex(num_drugs, num_proteins, context)
    database = _create_database(output / "pathlens.db")
    _insert_entities_and_edges(database, entities, context, all_positive)
    scores_path = output / "scores.tmp"
    score_matrix = np.memmap(
        scores_path,
        dtype=np.float32,
        mode="w+",
        shape=(num_drugs, num_proteins),
    )
    known_by_drug: dict[int, set[int]] = {index: set() for index in range(num_drugs)}
    for drug, protein in all_positive:
        known_by_drug[int(drug)].add(int(protein))
    protein_projection = np.asarray(
        [graph_index.projection_mass_protein(index) for index in range(num_proteins)],
        dtype=np.float32,
    )
    torch_embeddings: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (
        torch.as_tensor(embedding_arrays[0]),
        torch.as_tensor(embedding_arrays[1]),
        torch.as_tensor(embedding_arrays[2]),
    )
    first_parity_pair: tuple[int, int, tuple[float, float]] | None = None

    for drug in range(num_drugs):
        bridge_scores = graph_index.bridge_scores_for_drug(drug)
        projection = graph_index.projection_mass_drug(drug) + protein_projection
        features = np.column_stack((projection, bridge_scores)).astype(np.float32)
        with torch.inference_mode():
            output_score = model.score_from_embeddings(
                torch_embeddings,
                torch.full((num_proteins,), drug, dtype=torch.long),
                torch.arange(num_proteins),
                torch.as_tensor(features),
            )
        logits = output_score.logit.numpy()
        scores = (100.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))).astype(np.float32)
        if known_by_drug[drug]:
            scores[np.fromiter(known_by_drug[drug], dtype=np.int64)] = -np.inf
        score_matrix[drug] = scores
        selected = _top_indices(scores, min(top_k, num_proteins - len(known_by_drug[drug])))
        _insert_rankings(
            database,
            entities["drugs"][drug],
            [entities["proteins"][int(index)] for index in selected],
            scores[selected],
            bridge_scores[selected],
        )
        for rank, protein in enumerate(selected[:explanation_k]):
            raw_features = (float(projection[protein]), float(bridge_scores[protein]))
            _insert_pair_evidence(
                database,
                entities,
                graph_index,
                (drug, int(protein)),
                raw_features,
            )
            if first_parity_pair is None and rank == 0:
                first_parity_pair = (drug, int(protein), raw_features)

    score_matrix.flush()
    for protein in range(num_proteins):
        scores = np.asarray(score_matrix[:, protein])
        selected = _top_indices(scores, min(top_k, num_drugs))
        _insert_rankings(
            database,
            entities["proteins"][protein],
            [entities["drugs"][int(index)] for index in selected],
            scores[selected],
            np.zeros(len(selected), dtype=np.float32),
        )
        for drug in selected[:explanation_k]:
            bridge_score = float(
                sum(
                    path.weight for path in graph_index.bridge_paths(int(drug), protein, limit=None)
                )
            )
            raw_features = (
                graph_index.projection_mass_drug(int(drug)) + float(protein_projection[protein]),
                bridge_score,
            )
            _insert_pair_evidence(
                database,
                entities,
                graph_index,
                (int(drug), protein),
                raw_features,
            )
    database.commit()
    database.close()
    del score_matrix
    scores_path.unlink(missing_ok=True)

    evaluation_summary = {
        "checkpoint_validation_hard_auprc": checkpoint.get("validation_hard_auprc"),
        "best_epoch": checkpoint.get("best_epoch"),
        "research_results_pending_final_sealed_test": True,
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps(evaluation_summary, indent=2), encoding="utf-8", newline="\n"
    )
    (output / "model_card.md").write_text(
        "# PathLens-GNN Model Card\n\n"
        "Transductive BioSNAP DTI research-prioritization model. Scores are not clinical "
        "probabilities or validated interactions. Unknown non-edges may include "
        "undiscovered positives.\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = _write_manifest(
        output,
        source_manifest,
        model_version,
        num_drugs,
        num_proteins,
        model_config,
        checkpoint_path=Path(checkpoint_path),
        training_config=checkpoint["training_config"],
    )
    if first_parity_pair is not None:
        _assert_runtime_parity(
            output,
            model,
            torch_embeddings,
            first_parity_pair,
        )
    return manifest


def _assert_runtime_parity(
    output: Path,
    model: PathLensGNN,
    embeddings: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    pair: tuple[int, int, tuple[float, float]],
) -> None:
    drug, protein, structural = pair
    runtime = ArtifactRuntime(output)
    numpy_score = runtime.score_pair(drug, protein, structural)
    with torch.inference_mode():
        torch_score = model.score_from_embeddings(
            embeddings,
            torch.tensor([drug]),
            torch.tensor([protein]),
            torch.tensor([structural], dtype=torch.float32),
        )
    difference = abs(numpy_score.logit - float(torch_score.logit.item()))
    runtime.close()
    if difference > 1e-5:
        raise RuntimeError(f"PyTorch/NumPy parity failed: absolute error {difference}")


def _create_database(path: Path) -> sqlite3.Connection:
    database = sqlite3.connect(path)
    database.executescript(
        """
        CREATE TABLE entities (entity_id TEXT PRIMARY KEY, entity_type TEXT NOT NULL,
          display_name TEXT NOT NULL, local_index INTEGER NOT NULL, degree INTEGER NOT NULL);
        CREATE INDEX entities_search ON entities(display_name, entity_id);
        CREATE TABLE known_edges (drug_id TEXT NOT NULL, protein_id TEXT NOT NULL,
          PRIMARY KEY(drug_id, protein_id));
        CREATE INDEX known_edges_protein ON known_edges(protein_id);
        CREATE TABLE recommendations (source_id TEXT NOT NULL, target_id TEXT NOT NULL,
          rank INTEGER NOT NULL, pathlens_score REAL NOT NULL, bridge_count INTEGER NOT NULL,
          PRIMARY KEY(source_id, rank));
        CREATE TABLE pair_features (drug_id TEXT NOT NULL, protein_id TEXT NOT NULL,
          projection_mass REAL NOT NULL, bridge_score REAL NOT NULL,
          PRIMARY KEY(drug_id, protein_id));
        CREATE TABLE explanations (drug_id TEXT NOT NULL, protein_id TEXT NOT NULL,
          payload_json TEXT NOT NULL, truncated INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY(drug_id, protein_id));
        """
    )
    return database


def _insert_entities_and_edges(
    database: sqlite3.Connection,
    entities: dict[str, list[str]],
    context: np.ndarray[Any, Any],
    all_positive: np.ndarray[Any, Any],
) -> None:
    drug_degree = np.bincount(context[:, 0], minlength=len(entities["drugs"]))
    protein_degree = np.bincount(context[:, 1], minlength=len(entities["proteins"]))
    database.executemany(
        "INSERT INTO entities VALUES (?, 'drug', ?, ?, ?)",
        [
            (entity_id, entity_id, index, int(drug_degree[index]))
            for index, entity_id in enumerate(entities["drugs"])
        ],
    )
    database.executemany(
        "INSERT INTO entities VALUES (?, 'protein', ?, ?, ?)",
        [
            (entity_id, entity_id, index, int(protein_degree[index]))
            for index, entity_id in enumerate(entities["proteins"])
        ],
    )
    database.executemany(
        "INSERT INTO known_edges VALUES (?, ?)",
        [
            (entities["drugs"][int(drug)], entities["proteins"][int(protein)])
            for drug, protein in all_positive
        ],
    )


def _insert_rankings(
    database: sqlite3.Connection,
    source_id: str,
    targets: list[str],
    scores: np.ndarray[Any, Any],
    bridges: np.ndarray[Any, Any],
) -> None:
    database.executemany(
        "INSERT INTO recommendations VALUES (?, ?, ?, ?, ?)",
        [
            (source_id, target, rank + 1, float(scores[rank]), int(bridges[rank] > 0))
            for rank, target in enumerate(targets)
        ],
    )


def _insert_pair_evidence(
    database: sqlite3.Connection,
    entities: dict[str, list[str]],
    graph: BipartiteIndex,
    pair: tuple[int, int],
    structural: tuple[float, float],
) -> None:
    drug, protein = pair
    drug_id = entities["drugs"][drug]
    protein_id = entities["proteins"][protein]
    database.execute(
        "INSERT OR REPLACE INTO pair_features VALUES (?, ?, ?, ?)",
        (drug_id, protein_id, *structural),
    )
    payload = {
        "projection_context": {
            "similar_drugs": [
                {"entity_id": entities["drugs"][index], "score": score}
                for index, score in graph.top_similar_drugs(drug)
            ],
            "similar_proteins": [
                {"entity_id": entities["proteins"][index], "score": score}
                for index, score in graph.top_similar_proteins(protein)
            ],
        },
        "bridge_paths": [
            {
                "nodes": [
                    drug_id,
                    entities["proteins"][path.intermediate_protein],
                    entities["drugs"][path.intermediate_drug],
                    protein_id,
                ],
                "weight": path.weight,
            }
            for path in graph.bridge_paths(drug, protein)
        ],
    }
    database.execute(
        "INSERT OR REPLACE INTO explanations VALUES (?, ?, ?, 0)",
        (drug_id, protein_id, json.dumps(payload)),
    )


def _top_indices(scores: np.ndarray[Any, Any], count: int) -> np.ndarray[Any, Any]:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    count = min(count, int(np.isfinite(scores).sum()))
    partition = np.argpartition(scores, -count)[-count:]
    return partition[np.argsort(scores[partition])[::-1]]


def _write_manifest(
    output: Path,
    source_manifest: dict[str, Any],
    model_version: str,
    num_drugs: int,
    num_proteins: int,
    model_config: PathLensConfig,
    checkpoint_path: Path,
    training_config: dict[str, Any],
) -> dict[str, Any]:
    payloads = [
        "model_weights.npz",
        "embeddings/s1.npy",
        "embeddings/s2.npy",
        "embeddings/s3.npy",
        "graph/context_edges.npy",
        "graph/known_edges.npy",
        "pathlens.db",
        "evaluation_summary.json",
        "model_card.md",
    ]
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "dataset_version": source_manifest["dataset_version"],
        "created_at": datetime.now(UTC).isoformat(),
        "name": "PathLens-GNN",
        "summary": "Transductive path-aware BioSNAP DTI prioritization.",
        "limitations": [
            "Known graph entities only",
            "Unknown non-edges are not verified negatives",
            "Research prioritization only; not clinical",
        ],
        "dimensions": {
            "num_drugs": num_drugs,
            "num_proteins": num_proteins,
            "embedding_dim": model_config.branch_dim,
        },
        "model": {
            "enabled_channels": list(model_config.enabled_channels),
            "adaptive_gate": model_config.adaptive_gate,
            "gelu": "tanh",
        },
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_config": training_config,
        "source_manifest": source_manifest,
        "files": {relative: sha256_file(output / relative) for relative in payloads},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    return manifest
