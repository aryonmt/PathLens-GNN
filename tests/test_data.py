from __future__ import annotations

from pathlib import Path

import pytest

from pathlens.constants import DEFAULT_SPLIT_SEED
from pathlens.data.canonical import CanonicalDataError, load_biosnap_tsv
from pathlens.data.negative import sample_degree_matched_negatives, sample_uniform_negatives
from pathlens.data.prepare import prepare_biosnap_dataset
from pathlens.data.processed import SealedTestError, load_processed
from pathlens.data.schema import Edge
from pathlens.data.split import coverage_preserving_split

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


def test_split_is_deterministic_disjoint_and_transductive() -> None:
    dataset = load_biosnap_tsv(FIXTURE)
    first = coverage_preserving_split(dataset.edges, seed=13)
    second = coverage_preserving_split(dataset.edges, seed=13)
    assert first == second
    assert set(first.context).isdisjoint(first.train)
    assert set(first.context).isdisjoint(first.validation)
    assert set(first.context).isdisjoint(first.test)


def test_split_seed_changes_holdout_assignment() -> None:
    dataset = load_biosnap_tsv(FIXTURE)
    first = coverage_preserving_split(dataset.edges, seed=13)
    second = coverage_preserving_split(dataset.edges, seed=DEFAULT_SPLIT_SEED)
    assert DEFAULT_SPLIT_SEED != 13
    assert first.train != second.train


def test_negative_samplers_never_emit_known_edges() -> None:
    dataset = load_biosnap_tsv(FIXTURE)
    split = coverage_preserving_split(dataset.edges, seed=13)
    known = frozenset(dataset.edges)
    uniform = sample_uniform_negatives(
        split.train,
        drugs=dataset.drugs,
        proteins=dataset.proteins,
        known_positives=known,
        seed=7,
    )
    hard = sample_degree_matched_negatives(
        split.train,
        drugs=dataset.drugs,
        proteins=dataset.proteins,
        known_positives=known,
        context_edges=split.context,
        seed=7,
    )
    assert set(uniform.edges).isdisjoint(known)
    assert set(hard.edges).isdisjoint(known)
    assert len(uniform.edges) == len(split.train)
    assert len(hard.edges) == len(split.train)


def test_processed_split_keeps_test_sealed(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    prepare_biosnap_dataset(FIXTURE, processed, seed=13, strict_identity=False)
    split = load_processed(processed, allow_test=False)
    assert split.train_uniform.shape == split.train_positive.shape
    with pytest.raises(SealedTestError):
        split.test_array("test_positive")
    opened = load_processed(processed, allow_test=True)
    assert opened.test_array("test_positive").ndim == 2
