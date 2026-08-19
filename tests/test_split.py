from __future__ import annotations

from pathlib import Path

from pathlens_gnn.constants import DEFAULT_SPLIT_SEED
from pathlens_gnn.data.canonical import load_biosnap_tsv
from pathlens_gnn.data.negative import (
    sample_degree_matched_negatives,
    sample_uniform_negatives,
)
from pathlens_gnn.data.split import coverage_preserving_split

FIXTURE = Path(__file__).parent / "fixtures" / "biosnap_header_bug.tsv"


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
