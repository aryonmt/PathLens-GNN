from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from pathlens.data.schema import Edge


@dataclass(frozen=True, slots=True)
class NegativeSampleResult:
    edges: tuple[Edge, ...]
    fallback_count: int = 0


def sample_uniform_negatives(
    positives: Iterable[Edge],
    *,
    drugs: tuple[str, ...],
    proteins: tuple[str, ...],
    known_positives: frozenset[Edge],
    seed: int,
    ratio: float = 1.0,
) -> NegativeSampleResult:
    target = round(len(tuple(positives)) * ratio)
    capacity = len(drugs) * len(proteins) - len(known_positives)
    if target > capacity:
        raise ValueError("Requested more unique negatives than available non-edges")
    rng = np.random.default_rng(seed)
    selected: set[Edge] = set()
    max_attempts = max(10_000, target * 100)
    attempts = 0
    while len(selected) < target and attempts < max_attempts:
        candidate = Edge(
            drugs[int(rng.integers(len(drugs)))],
            proteins[int(rng.integers(len(proteins)))],
        )
        if candidate not in known_positives:
            selected.add(candidate)
        attempts += 1
    if len(selected) != target:
        raise RuntimeError("Uniform negative sampler exhausted its deterministic attempt budget")
    return NegativeSampleResult(tuple(sorted(selected)))


def sample_degree_matched_negatives(
    positives: Iterable[Edge],
    *,
    drugs: tuple[str, ...],
    proteins: tuple[str, ...],
    known_positives: frozenset[Edge],
    context_edges: Iterable[Edge],
    seed: int,
    quantiles: int = 4,
) -> NegativeSampleResult:
    context = tuple(context_edges)
    drug_degree: defaultdict[str, int] = defaultdict(int)
    protein_degree: defaultdict[str, int] = defaultdict(int)
    for edge in context:
        drug_degree[edge.drug_id] += 1
        protein_degree[edge.protein_id] += 1

    drug_bins = _degree_bins(drugs, drug_degree, quantiles)
    protein_bins = _degree_bins(proteins, protein_degree, quantiles)
    drugs_by_bin = _invert_bins(drug_bins)
    proteins_by_bin = _invert_bins(protein_bins)
    rng = np.random.default_rng(seed)
    selected: set[Edge] = set()
    fallback_count = 0

    for positive in positives:
        desired_d = drug_bins[positive.drug_id]
        desired_p = protein_bins[positive.protein_id]
        found: Edge | None = None
        for radius in range(quantiles):
            candidate_drugs = _nearby_items(drugs_by_bin, desired_d, radius)
            candidate_proteins = _nearby_items(proteins_by_bin, desired_p, radius)
            if not candidate_drugs or not candidate_proteins:
                continue
            for _ in range(500):
                candidate = Edge(
                    candidate_drugs[int(rng.integers(len(candidate_drugs)))],
                    candidate_proteins[int(rng.integers(len(candidate_proteins)))],
                )
                if candidate not in known_positives and candidate not in selected:
                    found = candidate
                    if radius:
                        fallback_count += 1
                    break
            if found is not None:
                break
        if found is None:
            fallback = sample_uniform_negatives(
                [positive],
                drugs=drugs,
                proteins=proteins,
                known_positives=frozenset((*known_positives, *selected)),
                seed=seed + len(selected) + 1,
            )
            found = fallback.edges[0]
            fallback_count += 1
        selected.add(found)

    return NegativeSampleResult(tuple(sorted(selected)), fallback_count)


def _degree_bins(items: tuple[str, ...], degrees: dict[str, int], quantiles: int) -> dict[str, int]:
    ordered = sorted(items, key=lambda item: (degrees.get(item, 0), item))
    return {
        item: min(quantiles - 1, index * quantiles // max(1, len(ordered)))
        for index, item in enumerate(ordered)
    }


def _invert_bins(mapping: dict[str, int]) -> dict[int, tuple[str, ...]]:
    result: dict[int, list[str]] = defaultdict(list)
    for item, bin_id in mapping.items():
        result[bin_id].append(item)
    return {key: tuple(values) for key, values in result.items()}


def _nearby_items(groups: dict[int, tuple[str, ...]], center: int, radius: int) -> tuple[str, ...]:
    return tuple(
        item for bin_id, items in groups.items() if abs(bin_id - center) <= radius for item in items
    )
