from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from pathlens_gnn.graph.index import BipartiteIndex


class ArtifactValidationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class PairScore:
    logit: float
    pathlens_score: float
    gate_weights: tuple[float, float, float]
    expert_logits: tuple[float, float, float]
    contributions: tuple[float, float, float]


class ArtifactRuntime:
    def __init__(self, artifact_dir: str | Path, *, verify_hashes: bool = True) -> None:
        self.root = Path(artifact_dir)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise ArtifactValidationError(f"Missing artifact manifest: {manifest_path}")
        self.manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest(verify_hashes)
        self.embeddings = tuple(
            np.load(self.root / f"embeddings/s{hop}.npy", mmap_mode="r") for hop in (1, 2, 3)
        )
        self.weights = np.load(self.root / "model_weights.npz")
        database_uri = f"file:{(self.root / 'pathlens.db').resolve().as_posix()}?mode=ro"
        self.database = sqlite3.connect(database_uri, uri=True, check_same_thread=False)
        self.database.row_factory = sqlite3.Row
        self._database_lock = RLock()
        self.num_drugs = int(self.manifest["dimensions"]["num_drugs"])
        self.num_proteins = int(self.manifest["dimensions"]["num_proteins"])
        self.enabled_channels = np.asarray(
            self.manifest["model"].get("enabled_channels", [True, True, True]), dtype=bool
        )
        context_path = self.root / "graph/context_edges.npy"
        self.graph = (
            BipartiteIndex(
                self.num_drugs,
                self.num_proteins,
                np.load(context_path, mmap_mode="r"),
            )
            if context_path.is_file()
            else None
        )
        self.drug_ids = tuple(
            str(row["entity_id"])
            for row in self.fetchall(
                "SELECT entity_id FROM entities WHERE entity_type = 'drug' ORDER BY local_index"
            )
        )
        self.protein_ids = tuple(
            str(row["entity_id"])
            for row in self.fetchall(
                "SELECT entity_id FROM entities WHERE entity_type = 'protein' ORDER BY local_index"
            )
        )

    @property
    def model_version(self) -> str:
        return str(self.manifest["model_version"])

    @property
    def dataset_version(self) -> str:
        return str(self.manifest["dataset_version"])

    def close(self) -> None:
        with self._database_lock:
            self.database.close()
        self.weights.close()

    def fetchone(self, statement: str, parameters: Sequence[object] = ()) -> sqlite3.Row | None:
        """Run one read query safely across FastAPI's worker threads."""
        with self._database_lock:
            return cast(
                sqlite3.Row | None,
                self.database.execute(statement, parameters).fetchone(),
            )

    def fetchall(self, statement: str, parameters: Sequence[object] = ()) -> list[sqlite3.Row]:
        """Run one read query safely across FastAPI's worker threads."""
        with self._database_lock:
            return self.database.execute(statement, parameters).fetchall()

    def score_pair(
        self,
        drug_index: int,
        protein_index: int,
        structural_features: tuple[float, float] = (0.0, 0.0),
    ) -> PairScore:
        if not 0 <= drug_index < self.num_drugs:
            raise IndexError("drug index is out of range")
        if not 0 <= protein_index < self.num_proteins:
            raise IndexError("protein index is out of range")
        drug_vectors = [embedding[drug_index] for embedding in self.embeddings]
        protein_global = self.num_drugs + protein_index
        protein_vectors = [embedding[protein_global] for embedding in self.embeddings]
        expert_logits = np.asarray(
            [
                self._expert(index, drug, protein)
                for index, (drug, protein) in enumerate(
                    zip(drug_vectors, protein_vectors, strict=True)
                )
            ],
            dtype=np.float64,
        )
        gate_input = np.concatenate(
            (
                *drug_vectors,
                *protein_vectors,
                np.log1p(np.maximum(0.0, np.asarray(structural_features))),
            )
        )
        gate_hidden = _gelu_tanh(self._linear(gate_input, "gate.0.weight", "gate.0.bias"))
        gate_logits = self._linear(gate_hidden, "gate.3.weight", "gate.3.bias")
        gate_logits = np.where(self.enabled_channels, gate_logits, -np.inf)
        gate_weights = _softmax(gate_logits)
        contributions = gate_weights * expert_logits
        logit = float(contributions.sum())
        return PairScore(
            logit=logit,
            pathlens_score=float(100.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))),
            gate_weights=_triple(gate_weights),
            expert_logits=_triple(expert_logits),
            contributions=_triple(contributions),
        )

    def _expert(self, index: int, drug: NDArray[Any], protein: NDArray[Any]) -> float:
        pair = np.concatenate((drug, protein, drug * protein))
        hidden = _gelu_tanh(
            self._linear(
                pair,
                f"experts.{index}.layers.0.weight",
                f"experts.{index}.layers.0.bias",
            )
        )
        output = self._linear(
            hidden,
            f"experts.{index}.layers.3.weight",
            f"experts.{index}.layers.3.bias",
        )
        return float(np.asarray(output).reshape(-1)[0])

    def _linear(self, values: NDArray[Any], weight_key: str, bias_key: str) -> NDArray[np.float64]:
        weight = np.asarray(self.weights[weight_key], dtype=np.float64)
        bias = np.asarray(self.weights[bias_key], dtype=np.float64)
        return np.asarray(values, dtype=np.float64) @ weight.T + bias

    def _validate_manifest(self, verify_hashes: bool) -> None:
        required = {
            "schema_version",
            "model_version",
            "dataset_version",
            "dimensions",
            "model",
            "files",
        }
        missing = required - self.manifest.keys()
        if missing:
            raise ArtifactValidationError(f"Manifest is missing fields: {sorted(missing)}")
        for relative, expected_hash in self.manifest["files"].items():
            path = self.root / relative
            if not path.is_file():
                raise ArtifactValidationError(f"Missing artifact payload: {relative}")
            if verify_hashes and sha256_file(path) != expected_hash:
                raise ArtifactValidationError(f"Artifact checksum mismatch: {relative}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _gelu_tanh(values: NDArray[Any]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    coefficient = np.sqrt(2.0 / np.pi)
    return np.asarray(
        0.5 * array * (1.0 + np.tanh(coefficient * (array + 0.044715 * array**3))),
        dtype=np.float64,
    )


def _softmax(values: NDArray[Any]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    shifted = array - np.max(array)
    exponent = np.exp(shifted)
    return np.asarray(exponent / exponent.sum(), dtype=np.float64)


def _triple(values: NDArray[Any]) -> tuple[float, float, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.shape != (3,):
        raise ArtifactValidationError(f"Expected three channel values, got {flat.shape}")
    return float(flat[0]), float(flat[1]), float(flat[2])
