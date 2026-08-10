from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pathlens_gnn.artifact.runtime import ArtifactRuntime, sha256_file
from pathlens_gnn.constants import SCHEMA_VERSION

ROOT = Path("artifacts/fixtures/demo-v1")


def main() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    (ROOT / "embeddings").mkdir(parents=True)
    (ROOT / "graph").mkdir(parents=True)
    rng = np.random.default_rng(13)
    num_drugs, num_proteins, embedding_dim = 3, 3, 4
    for hop in (1, 2, 3):
        np.save(
            ROOT / f"embeddings/s{hop}.npy",
            rng.normal(0, 0.25, (num_drugs + num_proteins, embedding_dim)).astype(np.float32),
        )

    context_edges = np.asarray([[0, 0], [1, 0], [1, 1], [2, 1]], dtype=np.int64)
    np.save(ROOT / "graph/context_edges.npy", context_edges)
    np.save(ROOT / "graph/known_edges.npy", context_edges)

    weights: dict[str, np.ndarray] = {}
    for index in range(3):
        weights[f"experts.{index}.layers.0.weight"] = rng.normal(
            0, 0.2, (5, embedding_dim * 3)
        ).astype(np.float32)
        weights[f"experts.{index}.layers.0.bias"] = np.zeros(5, dtype=np.float32)
        weights[f"experts.{index}.layers.3.weight"] = rng.normal(0, 0.2, (1, 5)).astype(np.float32)
        weights[f"experts.{index}.layers.3.bias"] = np.zeros(1, dtype=np.float32)
    weights["gate.0.weight"] = rng.normal(0, 0.15, (6, embedding_dim * 6 + 2)).astype(np.float32)
    weights["gate.0.bias"] = np.zeros(6, dtype=np.float32)
    weights["gate.3.weight"] = rng.normal(0, 0.15, (3, 6)).astype(np.float32)
    weights["gate.3.bias"] = np.zeros(3, dtype=np.float32)
    np.savez(ROOT / "model_weights.npz", **weights)

    build_database(ROOT / "pathlens.db")
    (ROOT / "evaluation_summary.json").write_text(
        json.dumps(
            {"fixture": True, "warning": "Synthetic artifact; not a research result."},
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    (ROOT / "model_card.md").write_text(
        "# Fixture Model Card\n\nSynthetic artifact for API/UI tests only.\n",
        encoding="utf-8",
        newline="\n",
    )
    write_manifest(num_drugs, num_proteins, embedding_dim)

    runtime = ArtifactRuntime(ROOT)
    database = sqlite3.connect(ROOT / "pathlens.db")
    for source, target in (("DB00001", "P00002"), ("DB00003", "P00001")):
        drug_index = int(source[-1]) - 1
        protein_index = int(target[-1]) - 1
        score = runtime.score_pair(drug_index, protein_index, (1.0, 0.5))
        database.execute(
            "UPDATE recommendations SET pathlens_score = ? WHERE source_id = ? AND target_id = ?",
            (score.pathlens_score, source, target),
        )
    database.commit()
    database.close()
    runtime.close()
    write_manifest(num_drugs, num_proteins, embedding_dim)


def build_database(path: Path) -> None:
    database = sqlite3.connect(path)
    database.executescript(
        """
        CREATE TABLE entities (
          entity_id TEXT PRIMARY KEY,
          entity_type TEXT NOT NULL CHECK(entity_type IN ('drug', 'protein')),
          display_name TEXT NOT NULL,
          local_index INTEGER NOT NULL,
          degree INTEGER NOT NULL
        );
        CREATE INDEX entities_search ON entities(display_name, entity_id);
        CREATE TABLE known_edges (
          drug_id TEXT NOT NULL,
          protein_id TEXT NOT NULL,
          PRIMARY KEY(drug_id, protein_id)
        );
        CREATE INDEX known_edges_protein ON known_edges(protein_id);
        CREATE TABLE recommendations (
          source_id TEXT NOT NULL,
          target_id TEXT NOT NULL,
          rank INTEGER NOT NULL,
          pathlens_score REAL NOT NULL,
          bridge_count INTEGER NOT NULL,
          PRIMARY KEY(source_id, rank)
        );
        CREATE TABLE pair_features (
          drug_id TEXT NOT NULL,
          protein_id TEXT NOT NULL,
          projection_mass REAL NOT NULL,
          bridge_score REAL NOT NULL,
          PRIMARY KEY(drug_id, protein_id)
        );
        CREATE TABLE explanations (
          drug_id TEXT NOT NULL,
          protein_id TEXT NOT NULL,
          payload_json TEXT NOT NULL,
          truncated INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY(drug_id, protein_id)
        );
        """
    )
    entities = [
        ("DB00001", "drug", "Fixture Drug Alpha", 0, 1),
        ("DB00002", "drug", "Fixture Drug Beta", 1, 2),
        ("DB00003", "drug", "Fixture Drug Gamma", 2, 1),
        ("P00001", "protein", "Fixture Protein One", 0, 2),
        ("P00002", "protein", "Fixture Protein Two", 1, 2),
        ("P00003", "protein", "Fixture Protein Three", 2, 0),
    ]
    database.executemany("INSERT INTO entities VALUES (?, ?, ?, ?, ?)", entities)
    edges = [
        ("DB00001", "P00001"),
        ("DB00002", "P00001"),
        ("DB00002", "P00002"),
        ("DB00003", "P00002"),
    ]
    database.executemany("INSERT INTO known_edges VALUES (?, ?)", edges)
    database.executemany(
        "INSERT INTO recommendations VALUES (?, ?, ?, ?, ?)",
        [
            ("DB00001", "P00002", 1, 50.0, 1),
            ("DB00003", "P00001", 1, 50.0, 1),
        ],
    )
    database.executemany(
        "INSERT INTO pair_features VALUES (?, ?, ?, ?)",
        [
            ("DB00001", "P00002", 1.0, 0.5),
            ("DB00003", "P00001", 1.0, 0.5),
        ],
    )
    explanation = {
        "projection_context": {
            "similar_drugs": [{"entity_id": "DB00002", "score": 0.5}],
            "similar_proteins": [{"entity_id": "P00001", "score": 0.5}],
        },
        "bridge_paths": [
            {
                "nodes": ["DB00001", "P00001", "DB00002", "P00002"],
                "weight": 0.25,
            }
        ],
    }
    database.execute(
        "INSERT INTO explanations VALUES (?, ?, ?, ?)",
        ("DB00001", "P00002", json.dumps(explanation), 0),
    )
    database.commit()
    database.close()


def write_manifest(num_drugs: int, num_proteins: int, embedding_dim: int) -> None:
    payloads = [
        "model_weights.npz",
        "embeddings/s1.npy",
        "embeddings/s2.npy",
        "embeddings/s3.npy",
        "pathlens.db",
        "graph/context_edges.npy",
        "graph/known_edges.npy",
        "evaluation_summary.json",
        "model_card.md",
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "model_version": "fixture-v1",
        "dataset_version": "fixture-dti-v1",
        "created_at": datetime.now(UTC).isoformat(),
        "name": "PathLens-GNN Fixture",
        "summary": "Synthetic artifact for contract and visualization tests.",
        "limitations": ["Synthetic data", "Not a research result", "Not clinical"],
        "dimensions": {
            "num_drugs": num_drugs,
            "num_proteins": num_proteins,
            "embedding_dim": embedding_dim,
        },
        "model": {"enabled_channels": [True, True, True], "gelu": "tanh"},
        "files": {relative: sha256_file(ROOT / relative) for relative in payloads},
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
