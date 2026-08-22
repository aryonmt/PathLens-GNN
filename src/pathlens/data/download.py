from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

from pathlens.constants import BIOSNAP_URL, EXPECTED_SOURCE_SHA256
from pathlens.data.canonical import sha256_file


def ensure_biosnap_tsv(
    destination: str | Path,
    *,
    url: str = BIOSNAP_URL,
    expected_sha256: str | None = EXPECTED_SOURCE_SHA256,
) -> Path:
    path = Path(destination)
    if path.is_file():
        if expected_sha256 is not None and sha256_file(path) != expected_sha256:
            raise ValueError(f"Existing file {path} does not match the locked BioSNAP SHA-256")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = path.with_suffix(path.suffix + ".gz")
    request = urllib.request.Request(url, headers={"User-Agent": "PathLens-GNN/0.2"})
    with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as handle:
        handle.write(response.read())
    with gzip.open(archive, "rb") as compressed, path.open("wb") as raw:
        raw.write(compressed.read())
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("Downloaded BioSNAP TSV does not match the locked SHA-256")
    return path
