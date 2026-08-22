"""Skip-graph construction from Huang et al. SkipGNN (legacy1).

The original graph is the symmetrized training adjacency plus self-loops,
then D^{-1/2} A D^{-1/2}. The skip graph is sign(A @ A) without an extra
identity, then the same normalization. Equations match `legacy1/SkipGNN/utils.py`.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

SparseAdj = sp.coo_matrix


def bipartite_to_entity_pairs(
    pairs: NDArray[np.int64],
    num_drugs: int,
) -> NDArray[np.int64]:
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return np.column_stack((pairs[:, 0], pairs[:, 1] + num_drugs))


def labeled_entity_pairs(
    positives: NDArray[np.int64],
    negatives: NDArray[np.int64],
    num_drugs: int,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    positive_pairs = bipartite_to_entity_pairs(positives, num_drugs)
    negative_pairs = bipartite_to_entity_pairs(negatives, num_drugs)
    if len(positive_pairs) != len(negative_pairs):
        raise ValueError("SkipGNN training requires a 1:1 positive/negative ratio")
    pairs = np.concatenate((positive_pairs, negative_pairs), axis=0)
    labels = np.concatenate(
        (
            np.ones(len(positive_pairs), dtype=np.float32),
            np.zeros(len(negative_pairs), dtype=np.float32),
        )
    )
    return pairs, labels


def visible_training_edges(
    context: NDArray[np.int64],
    train_positive: NDArray[np.int64],
) -> NDArray[np.int64]:
    stacked = np.concatenate(
        (
            np.asarray(context, dtype=np.int64).reshape(-1, 2),
            np.asarray(train_positive, dtype=np.int64).reshape(-1, 2),
        ),
        axis=0,
    )
    if stacked.size == 0:
        return stacked.reshape(0, 2)
    ordered = stacked[np.lexsort((stacked[:, 1], stacked[:, 0]))]
    unique_mask = np.ones(len(ordered), dtype=bool)
    unique_mask[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    return ordered[unique_mask]


def build_skipgnn_adjacencies(
    num_drugs: int,
    num_proteins: int,
    bipartite_edges: NDArray[np.int64],
) -> tuple[SparseAdj, SparseAdj]:
    num_entities = num_drugs + num_proteins
    pairs = np.asarray(bipartite_edges, dtype=np.int64).reshape(-1, 2)
    if len(pairs) == 0:
        empty = sp.coo_matrix((num_entities, num_entities), dtype=np.float32)
        original = _normalize_adj(empty + sp.eye(num_entities, dtype=np.float32))
        skip = _normalize_adj(empty)
        return original.tocoo(), skip.tocoo()

    entity_src = pairs[:, 0]
    entity_dst = pairs[:, 1] + num_drugs
    adj = sp.coo_matrix(
        (np.ones(len(pairs), dtype=np.float32), (entity_src, entity_dst)),
        shape=(num_entities, num_entities),
        dtype=np.float32,
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # noqa: SIM300
    skip = _normalize_adj(adj.dot(adj).sign())
    original = _normalize_adj(adj + sp.eye(num_entities, dtype=np.float32))
    return original.tocoo(), skip.tocoo()


def _normalize_adj(adj: sp.spmatrix) -> sp.coo_matrix:
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    scale = sp.diags(d_inv_sqrt)
    return adj.dot(scale).transpose().dot(scale).tocoo()
