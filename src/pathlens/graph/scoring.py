from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pathlens.runtime.device import as_numpy, is_torch

Adjacency = Any
Scorer = Any


def build_adjacency(
    num_drugs: int,
    num_proteins: int,
    edges: NDArray[np.int64],
    device: str,
) -> Adjacency:
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if not device.startswith("cuda"):
        adjacency = np.zeros((num_drugs, num_proteins), dtype=np.float32)
        if len(edges):
            adjacency[edges[:, 0], edges[:, 1]] = 1.0
        return adjacency
    torch = _torch()
    adjacency = torch.zeros((num_drugs, num_proteins), dtype=torch.float32, device=device)
    if len(edges):
        index = torch.as_tensor(edges, device=device)
        adjacency[index[:, 0], index[:, 1]] = 1.0
    return adjacency


def score_degree(adjacency: Adjacency) -> Adjacency:
    drug_degree, protein_degree = _degrees(adjacency)
    return _log1p(drug_degree[:, None] * protein_degree[None, :])


def score_resource_allocation(adjacency: Adjacency) -> Adjacency:
    """Same-type RA/AA projection, then one hop to the opposite part.

    W[d, d'] = sum_p B[d, p] B[d', p] / deg(p). Score(d, p) = sum_{d' != d} W[d, d'] B[d', p].
    """
    return _zero_diagonal(_drug_projection(adjacency)) @ adjacency


def score_three_hop(adjacency: Adjacency) -> Adjacency:
    """Normalized three-hop bridge used in campaign v2 (legacy `normalized_three_hop`).

    Same walk as resource allocation, with an extra 1/deg(d') on the intermediate drug.
    """
    drug_degree, _protein_degree = _degrees(adjacency)
    inverse_drug = 1.0 / _clamp_min(drug_degree, 1.0)
    weighted = _zero_diagonal(_drug_projection(adjacency)) * inverse_drug[None, :]
    return weighted @ adjacency


SCORERS: dict[str, Scorer] = {
    "degree": score_degree,
    "resource_allocation": score_resource_allocation,
    "three_hop": score_three_hop,
}


def score_method(method_id: str, adjacency: Adjacency) -> Adjacency:
    try:
        scorer = SCORERS[method_id]
    except KeyError as error:
        raise KeyError(f"No heuristic scorer for {method_id!r}") from error
    return scorer(adjacency)


def lookup_pairs(scores: Adjacency, pairs: NDArray[np.int64]) -> NDArray[np.float64]:
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if is_torch(scores):
        index = _torch().as_tensor(pairs, device=scores.device)
        return as_numpy(scores[index[:, 0], index[:, 1]]).astype(np.float64, copy=False)
    return np.asarray(scores[pairs[:, 0], pairs[:, 1]], dtype=np.float64)


def _drug_projection(adjacency: Adjacency) -> Adjacency:
    _, protein_degree = _degrees(adjacency)
    inverse_protein = 1.0 / _clamp_min(protein_degree, 1.0)
    return (adjacency * inverse_protein) @ _transpose(adjacency)


def _degrees(adjacency: Adjacency) -> tuple[Adjacency, Adjacency]:
    if isinstance(adjacency, np.ndarray):
        return adjacency.sum(axis=1), adjacency.sum(axis=0)
    return adjacency.sum(dim=1), adjacency.sum(dim=0)


def _transpose(adjacency: Adjacency) -> Adjacency:
    if isinstance(adjacency, np.ndarray):
        return adjacency.T
    return adjacency.T


def _zero_diagonal(matrix: Adjacency) -> Adjacency:
    if isinstance(matrix, np.ndarray):
        out = matrix.copy()
        np.fill_diagonal(out, 0.0)
        return out
    out = matrix.clone()
    out.fill_diagonal_(0.0)
    return out


def _log1p(values: Adjacency) -> Adjacency:
    if isinstance(values, np.ndarray):
        return np.log1p(values)
    return values.log1p()


def _clamp_min(values: Adjacency, minimum: float) -> Adjacency:
    if isinstance(values, np.ndarray):
        return np.maximum(values, minimum)
    return values.clamp_min(minimum)


def _torch() -> Any:
    import torch

    return torch
