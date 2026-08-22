from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pathlens.graph.skip import (
    bipartite_to_entity_pairs,
    build_skipgnn_adjacencies,
    build_symmetric_adjacency,
    labeled_entity_pairs,
    visible_training_edges,
)


def _legacy_normalize_adj(adj: sp.spmatrix) -> sp.coo_matrix:
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def _legacy_graphs(
    num_entities: int, entity_edges: np.ndarray
) -> tuple[sp.coo_matrix, sp.coo_matrix]:
    edges = np.asarray(entity_edges, dtype=np.int64)
    adj = sp.coo_matrix(
        (np.ones(len(edges), dtype=np.float32), (edges[:, 0], edges[:, 1])),
        shape=(num_entities, num_entities),
        dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # noqa: SIM300
    skip = adj.dot(adj).sign()
    skip = _legacy_normalize_adj(skip)
    original = adj + sp.eye(adj.shape[0])
    original = _legacy_normalize_adj(original)
    return original.tocoo(), skip.tocoo()


def test_skip_and_original_graphs_match_legacy1_utils() -> None:
    num_drugs, num_proteins = 4, 3
    bipartite = np.asarray([[0, 0], [0, 1], [1, 0], [2, 2], [3, 1]], dtype=np.int64)
    entity_edges = np.column_stack((bipartite[:, 0], bipartite[:, 1] + num_drugs))
    expected_original, expected_skip = _legacy_graphs(num_drugs + num_proteins, entity_edges)
    original, skip = build_skipgnn_adjacencies(num_drugs, num_proteins, bipartite)
    assert np.allclose(original.toarray(), expected_original.toarray())
    assert np.allclose(skip.toarray(), expected_skip.toarray())


def test_skip_graph_has_no_opposite_type_edges() -> None:
    num_drugs, num_proteins = 3, 2
    bipartite = np.asarray([[0, 0], [1, 0], [1, 1], [2, 1]], dtype=np.int64)
    _original, skip = build_skipgnn_adjacencies(num_drugs, num_proteins, bipartite)
    dense = np.abs(skip.toarray())
    assert np.allclose(dense[:num_drugs, num_drugs:], 0.0)
    assert np.allclose(dense[num_drugs:, :num_drugs], 0.0)


def test_bipartite_pairs_map_protein_index_after_drugs() -> None:
    pairs = np.asarray([[0, 1], [2, 0]], dtype=np.int64)
    assert np.array_equal(
        bipartite_to_entity_pairs(pairs, num_drugs=5),
        np.asarray([[0, 6], [2, 5]], dtype=np.int64),
    )


def test_labeled_pairs_are_one_to_one_and_typed() -> None:
    positives = np.asarray([[0, 1]], dtype=np.int64)
    negatives = np.asarray([[2, 0]], dtype=np.int64)
    pairs, labels = labeled_entity_pairs(positives, negatives, num_drugs=4)
    assert np.array_equal(pairs, np.asarray([[0, 5], [2, 4]], dtype=np.int64))
    assert np.array_equal(labels, np.asarray([1.0, 0.0], dtype=np.float32))
    assert np.all(pairs[:, 1] >= 4)


def test_labeled_pairs_reject_unbalanced_banks() -> None:
    with pytest.raises(ValueError, match="1:1"):
        labeled_entity_pairs(
            np.asarray([[0, 0], [1, 0]], dtype=np.int64),
            np.asarray([[2, 1]], dtype=np.int64),
            num_drugs=3,
        )


def test_symmetric_adjacency_has_no_self_loops() -> None:
    bipartite = np.asarray([[0, 0], [1, 0]], dtype=np.int64)
    adj = build_symmetric_adjacency(2, 2, bipartite)
    dense = adj.toarray()
    assert np.allclose(np.diag(dense), 0.0)
    assert np.allclose(dense, dense.T)
    assert dense[0, 2] == 1.0
    assert dense[2, 0] == 1.0


def test_visible_training_edges_exclude_validation() -> None:
    context = np.asarray([[0, 0], [1, 1]], dtype=np.int64)
    train = np.asarray([[2, 0]], dtype=np.int64)
    validation = np.asarray([[0, 1]], dtype=np.int64)
    visible = visible_training_edges(context, train)
    assert {tuple(row) for row in visible} == {(0, 0), (1, 1), (2, 0)}
    assert (0, 1) not in {tuple(row) for row in visible}
    assert not any(np.array_equal(row, validation[0]) for row in visible)
