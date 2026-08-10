from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from pathlens_gnn.data.schema import Edge


@dataclass(frozen=True, slots=True)
class EdgeSplit:
    context: tuple[Edge, ...]
    train: tuple[Edge, ...]
    validation: tuple[Edge, ...]
    test: tuple[Edge, ...]
    seed: int
    requested_holdout: int

    @property
    def eligible_fraction(self) -> float:
        total = sum(map(len, (self.context, self.train, self.validation, self.test)))
        return (len(self.train) + len(self.validation) + len(self.test)) / total


def coverage_preserving_split(
    edges: tuple[Edge, ...] | list[Edge],
    *,
    seed: int = 13,
    context_ratio: float = 0.60,
    train_ratio: float = 0.20,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> EdgeSplit:
    ratios = np.array([context_ratio, train_ratio, validation_ratio, test_ratio])
    if not np.isclose(ratios.sum(), 1.0) or np.any(ratios <= 0):
        raise ValueError("Split ratios must be positive and sum to one")

    unique_edges = tuple(sorted(set(edges)))
    drug_degree = Counter(edge.drug_id for edge in unique_edges)
    protein_degree = Counter(edge.protein_id for edge in unique_edges)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique_edges))
    target_holdout = round(len(unique_edges) * (1 - context_ratio))

    context = set(unique_edges)
    holdout: list[Edge] = []
    for index in order:
        if len(holdout) >= target_holdout:
            break
        edge = unique_edges[int(index)]
        if drug_degree[edge.drug_id] <= 1 or protein_degree[edge.protein_id] <= 1:
            continue
        context.remove(edge)
        holdout.append(edge)
        drug_degree[edge.drug_id] -= 1
        protein_degree[edge.protein_id] -= 1

    holdout = [holdout[int(i)] for i in rng.permutation(len(holdout))]
    supervised_ratio = train_ratio + validation_ratio + test_ratio
    train_end = round(len(holdout) * train_ratio / supervised_ratio)
    validation_end = train_end + round(len(holdout) * validation_ratio / supervised_ratio)

    split = EdgeSplit(
        context=tuple(sorted(context)),
        train=tuple(sorted(holdout[:train_end])),
        validation=tuple(sorted(holdout[train_end:validation_end])),
        test=tuple(sorted(holdout[validation_end:])),
        seed=seed,
        requested_holdout=target_holdout,
    )
    assert_split_integrity(split)
    return split


def assert_split_integrity(split: EdgeSplit) -> None:
    groups = [set(split.context), set(split.train), set(split.validation), set(split.test)]
    for index, left in enumerate(groups):
        for right in groups[index + 1 :]:
            if overlap := left & right:
                raise AssertionError(f"Split leakage detected: {next(iter(overlap))}")

    context_drugs = {edge.drug_id for edge in split.context}
    context_proteins = {edge.protein_id for edge in split.context}
    for edge in (*split.train, *split.validation, *split.test):
        if edge.drug_id not in context_drugs or edge.protein_id not in context_proteins:
            raise AssertionError(f"Cold-start edge leaked into transductive split: {edge}")
