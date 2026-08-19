from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor


def known_proteins_by_drug(pairs: NDArray[np.int64]) -> dict[int, frozenset[int]]:
    grouped: dict[int, set[int]] = defaultdict(set)
    for drug, protein in np.asarray(pairs, dtype=np.int64):
        grouped[int(drug)].add(int(protein))
    return {drug: frozenset(proteins) for drug, proteins in grouped.items()}


def sampled_softmax_loss(
    positive_logits: Tensor,
    negative_logits: Tensor,
    *,
    temperature: float = 1.0,
) -> Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = torch.cat((positive_logits.unsqueeze(1), negative_logits), dim=1) / temperature
    targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return torch.nn.functional.cross_entropy(logits, targets)


def sample_ranked_negatives(
    positives: NDArray[np.int64],
    *,
    num_proteins: int,
    known_by_drug: dict[int, frozenset[int]],
    protein_degrees: NDArray[np.int64],
    num_negatives: int,
    hard_fraction: float,
    seed: int,
    quantiles: int = 4,
) -> NDArray[np.int64]:
    if num_negatives < 1:
        raise ValueError("num_negatives must be at least 1")
    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be in [0, 1]")
    positives = np.asarray(positives, dtype=np.int64).reshape(-1, 2)
    degrees = np.asarray(protein_degrees, dtype=np.int64)
    bins = _degree_bins(degrees, quantiles)
    rng = np.random.default_rng(seed)
    num_hard = int(round(num_negatives * hard_fraction))
    sampled = np.empty((len(positives), num_negatives), dtype=np.int64)
    all_proteins = np.arange(num_proteins, dtype=np.int64)
    for row, (drug, protein) in enumerate(positives):
        forbidden = known_by_drug.get(int(drug), frozenset())
        allowed = all_proteins[np.isin(all_proteins, list(forbidden), invert=True)]
        if allowed.size == 0:
            raise RuntimeError(f"No typed non-edges remain for drug {int(drug)}")
        hard_pool = _hard_pool(int(protein), allowed, bins, quantiles)
        uniform_count = num_negatives - num_hard
        chosen: list[int] = []
        if num_hard:
            chosen.extend(rng.choice(hard_pool, size=num_hard, replace=True).tolist())
        if uniform_count:
            chosen.extend(rng.choice(allowed, size=uniform_count, replace=True).tolist())
        sampled[row] = np.asarray(chosen, dtype=np.int64)
    return sampled


def _degree_bins(degrees: NDArray[np.int64], quantiles: int) -> NDArray[np.int64]:
    order = np.lexsort((np.arange(len(degrees)), degrees))
    bins = np.empty(len(degrees), dtype=np.int64)
    count = max(1, len(degrees))
    for rank, protein in enumerate(order):
        bins[int(protein)] = min(quantiles - 1, rank * quantiles // count)
    return bins


def _hard_pool(
    protein: int,
    allowed: NDArray[np.int64],
    bins: NDArray[np.int64],
    quantiles: int,
) -> NDArray[np.int64]:
    target = int(bins[protein])
    for radius in range(quantiles):
        selected = allowed[np.abs(bins[allowed] - target) <= radius]
        if selected.size:
            return selected
    return allowed
