from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from pathlens_gnn.data.schema import CanonicalDataset, Edge


class CanonicalDataError(ValueError):
    """Raised when a source file cannot satisfy the canonical DTI contract."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_biosnap_tsv(path: str | Path, *, strict: bool = True) -> CanonicalDataset:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    valid_edges: set[Edge] = set()
    rejected: list[tuple[int, tuple[str, ...]]] = []

    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            normalized = tuple(cell.strip() for cell in row)
            if normalized in {("#Drug", "Gene"), ("Drug", "Gene")}:
                continue
            if len(normalized) != 2:
                rejected.append((line_number, normalized))
                continue
            try:
                valid_edges.add(Edge(normalized[0], normalized[1]))
            except ValueError:
                rejected.append((line_number, normalized))

    if strict and rejected:
        preview = ", ".join(f"line {line}: {row!r}" for line, row in rejected[:5])
        raise CanonicalDataError(f"Rejected {len(rejected)} malformed rows ({preview})")
    if not valid_edges:
        raise CanonicalDataError("No valid DTI edges found")

    edges = tuple(sorted(valid_edges))
    drugs = tuple(sorted({edge.drug_id for edge in edges}))
    proteins = tuple(sorted({edge.protein_id for edge in edges}))
    return CanonicalDataset(
        edges=edges,
        drugs=drugs,
        proteins=proteins,
        source_sha256=sha256_file(source),
    )
