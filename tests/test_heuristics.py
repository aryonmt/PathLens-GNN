from __future__ import annotations

import numpy as np
import pytest

from pathlens.constants import HEURISTIC_METHODS
from pathlens.graph.scoring import (
    SCORERS,
    build_adjacency,
    lookup_pairs,
    score_degree,
    score_resource_allocation,
    score_three_hop,
)
from pathlens.runtime.device import as_numpy


def _toy_edges() -> np.ndarray:
    # d0-p0-d1-p1 is a three-hop bridge; d0 and d1 share p0 as two-hop context.
    return np.asarray([[0, 0], [1, 0], [1, 1], [2, 1]], dtype=np.int64)


def _legacy_three_hop(
    num_drugs: int, num_proteins: int, edges: np.ndarray, drug: int
) -> np.ndarray:
    drug_neighbors: dict[int, list[int]] = {index: [] for index in range(num_drugs)}
    protein_neighbors: dict[int, list[int]] = {index: [] for index in range(num_proteins)}
    for source, target in edges:
        drug_neighbors[int(source)].append(int(target))
        protein_neighbors[int(target)].append(int(source))
    scores = np.zeros(num_proteins, dtype=np.float32)
    for intermediate_protein in drug_neighbors[drug]:
        protein_degree = max(1, len(protein_neighbors[intermediate_protein]))
        for intermediate_drug in protein_neighbors[intermediate_protein]:
            if intermediate_drug == drug:
                continue
            drug_degree = max(1, len(drug_neighbors[intermediate_drug]))
            weight = 1.0 / (protein_degree * drug_degree)
            for target_protein in drug_neighbors[intermediate_drug]:
                scores[target_protein] += weight
    return scores


def test_three_hop_matches_normalized_bridge_walk() -> None:
    edges = _toy_edges()
    adjacency = build_adjacency(3, 2, edges, "cpu")
    scores = as_numpy(score_three_hop(adjacency))
    for drug in range(3):
        assert np.allclose(scores[drug], _legacy_three_hop(3, 2, edges, drug))
    pairs = np.asarray([[0, 1]], dtype=np.int64)
    assert lookup_pairs(scores, pairs)[0] == pytest.approx(0.25)


def test_degree_and_resource_allocation_on_toy_graph() -> None:
    edges = _toy_edges()
    adjacency = build_adjacency(3, 2, edges, "cpu")
    degree = as_numpy(score_degree(adjacency))
    allocation = as_numpy(score_resource_allocation(adjacency))
    # deg(d0)=1, deg(p0)=2, deg(p1)=2
    assert np.isclose(degree[0, 0], np.log1p(2.0))
    assert np.isclose(degree[0, 1], np.log1p(2.0))
    # W[0,1] = 1/deg(p0) = 0.5; RA[0, p] = 0.5 * B[1, p]
    assert np.allclose(allocation[0], [0.5, 0.5])
    assert allocation[0, 0] != as_numpy(score_three_hop(adjacency))[0, 0]


def test_scorers_match_heuristic_method_ids() -> None:
    assert set(SCORERS) == set(HEURISTIC_METHODS)
