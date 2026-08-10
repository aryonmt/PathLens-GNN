from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pathlens_gnn.constants import DEFAULT_DATASET_VERSION, SCHEMA_VERSION
from pathlens_gnn.data.canonical import load_biosnap_tsv
from pathlens_gnn.data.negative import (
    sample_degree_matched_negatives,
    sample_uniform_negatives,
)
from pathlens_gnn.data.schema import Edge
from pathlens_gnn.data.split import coverage_preserving_split


def prepare_biosnap_dataset(
    source: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 13,
) -> dict[str, object]:
    dataset = load_biosnap_tsv(source)
    split = coverage_preserving_split(dataset.edges, seed=seed)
    known = frozenset(dataset.edges)

    negative_sets = {
        "train_uniform": sample_uniform_negatives(
            split.train,
            drugs=dataset.drugs,
            proteins=dataset.proteins,
            known_positives=known,
            seed=seed + 101,
        ),
        "validation_uniform": sample_uniform_negatives(
            split.validation,
            drugs=dataset.drugs,
            proteins=dataset.proteins,
            known_positives=known,
            seed=seed + 102,
        ),
        "test_uniform": sample_uniform_negatives(
            split.test,
            drugs=dataset.drugs,
            proteins=dataset.proteins,
            known_positives=known,
            seed=seed + 103,
        ),
        "validation_hard": sample_degree_matched_negatives(
            split.validation,
            drugs=dataset.drugs,
            proteins=dataset.proteins,
            known_positives=known,
            context_edges=split.context,
            seed=seed + 202,
        ),
        "test_hard": sample_degree_matched_negatives(
            split.test,
            drugs=dataset.drugs,
            proteins=dataset.proteins,
            known_positives=known,
            context_edges=split.context,
            seed=seed + 203,
        ),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    drug_index = {entity_id: index for index, entity_id in enumerate(dataset.drugs)}
    protein_index = {entity_id: index for index, entity_id in enumerate(dataset.proteins)}

    (output / "entities.json").write_text(
        json.dumps(
            {
                "drugs": list(dataset.drugs),
                "proteins": list(dataset.proteins),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    arrays: dict[str, np.ndarray] = {
        "all_positive": _edge_array(dataset.edges, drug_index, protein_index),
        "context": _edge_array(split.context, drug_index, protein_index),
        "train_positive": _edge_array(split.train, drug_index, protein_index),
        "validation_positive": _edge_array(split.validation, drug_index, protein_index),
        "test_positive": _edge_array(split.test, drug_index, protein_index),
    }
    arrays.update(
        {
            name: _edge_array(result.edges, drug_index, protein_index)
            for name, result in negative_sets.items()
        }
    )
    np.savez(str(output / "splits.npz"), **arrays)  # type: ignore[arg-type]

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DEFAULT_DATASET_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "source": {
            "path_hint": Path(source).name,
            "sha256": dataset.source_sha256,
        },
        "counts": {
            "edges": len(dataset.edges),
            "drugs": len(dataset.drugs),
            "proteins": len(dataset.proteins),
            "entities": dataset.entity_count,
            "context": len(split.context),
            "train": len(split.train),
            "validation": len(split.validation),
            "test": len(split.test),
        },
        "requested_holdout": split.requested_holdout,
        "eligible_fraction": split.eligible_fraction,
        "negative_fallbacks": {
            name: result.fallback_count for name, result in negative_sets.items()
        },
        "files": {"entities": "entities.json", "splits": "splits.npz"},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _edge_array(
    edges: tuple[Edge, ...],
    drug_index: dict[str, int],
    protein_index: dict[str, int],
) -> np.ndarray:
    return np.asarray(
        [(drug_index[edge.drug_id], protein_index[edge.protein_id]) for edge in edges],
        dtype=np.int64,
    ).reshape(-1, 2)
