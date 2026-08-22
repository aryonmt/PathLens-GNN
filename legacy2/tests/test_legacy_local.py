from __future__ import annotations

from pathlib import Path

import pytest

from pathlens_gnn.data.canonical import load_biosnap_tsv

LEGACY_SOURCE = Path("legacy/data/DTI/biosnap.tsv")


@pytest.mark.skipif(
    not LEGACY_SOURCE.exists(), reason="local ignored legacy archive is unavailable"
)
def test_local_legacy_source_matches_official_canonical_counts() -> None:
    dataset = load_biosnap_tsv(LEGACY_SOURCE)
    assert len(dataset.edges) == 15_138
    assert len(dataset.drugs) == 5_017
    assert len(dataset.proteins) == 2_324
    assert dataset.entity_count == 7_341
