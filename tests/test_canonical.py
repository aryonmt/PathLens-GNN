from __future__ import annotations

from pathlib import Path

import pytest

from pathlens_gnn.data.canonical import CanonicalDataError, load_biosnap_tsv
from pathlens_gnn.data.schema import Edge

FIXTURE = Path(__file__).parent / "fixtures" / "biosnap_header_bug.tsv"


def test_header_is_not_an_entity_or_positive_edge() -> None:
    dataset = load_biosnap_tsv(FIXTURE)
    assert len(dataset.edges) == 6
    assert "#Drug" not in dataset.drugs
    assert "Gene" not in dataset.proteins
    assert dataset.entity_count == 6


def test_edge_rejects_wrong_types() -> None:
    with pytest.raises(ValueError, match="Invalid drug"):
        Edge("P12345", "P00001")
    with pytest.raises(ValueError, match="Invalid protein"):
        Edge("DB00001", "Gene")


def test_strict_loader_reports_malformed_rows(tmp_path: Path) -> None:
    source = tmp_path / "bad.tsv"
    source.write_text("#Drug\tGene\nDB00001\tP00001\nnot-a-drug\tP00002\n", encoding="utf-8")
    with pytest.raises(CanonicalDataError, match="Rejected 1 malformed"):
        load_biosnap_tsv(source)
