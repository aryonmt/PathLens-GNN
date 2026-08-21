from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class BridgePath:
    drug_index: int
    intermediate_protein: int
    intermediate_drug: int
    protein_index: int
    weight: float


class BipartiteIndex:
    def __init__(
        self,
        num_drugs: int,
        num_proteins: int,
        edges: NDArray[np.int64],
    ) -> None:
        self.num_drugs = num_drugs
        self.num_proteins = num_proteins
        drug_neighbors: dict[int, set[int]] = defaultdict(set)
        protein_neighbors: dict[int, set[int]] = defaultdict(set)
        for drug, protein in np.asarray(edges, dtype=np.int64):
            drug_neighbors[int(drug)].add(int(protein))
            protein_neighbors[int(protein)].add(int(drug))
        self.drug_neighbors = {
            index: tuple(sorted(drug_neighbors.get(index, set()))) for index in range(num_drugs)
        }
        self.protein_neighbors = {
            index: tuple(sorted(protein_neighbors.get(index, set())))
            for index in range(num_proteins)
        }
        self._pair_feature_tables: (
            tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]] | None
        ) = None

    def pair_feature_tables(
        self,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
        if self._pair_feature_tables is None:
            drug_mass = np.asarray(
                [self.projection_mass_drug(index) for index in range(self.num_drugs)],
                dtype=np.float32,
            )
            protein_mass = np.asarray(
                [self.projection_mass_protein(index) for index in range(self.num_proteins)],
                dtype=np.float32,
            )
            bridge = np.empty((self.num_drugs, self.num_proteins), dtype=np.float32)
            for drug in range(self.num_drugs):
                bridge[drug] = self.bridge_scores_for_drug(drug)
            self._pair_feature_tables = (drug_mass, protein_mass, bridge)
        return self._pair_feature_tables

    def structural_features(self, pairs: NDArray[np.int64]) -> NDArray[np.float32]:
        pairs = np.asarray(pairs, dtype=np.int64)
        drug_mass, protein_mass, bridge = self.pair_feature_tables()
        features = np.empty((len(pairs), 2), dtype=np.float32)
        features[:, 0] = drug_mass[pairs[:, 0]] + protein_mass[pairs[:, 1]]
        features[:, 1] = bridge[pairs[:, 0], pairs[:, 1]]
        return features

    def bridge_scores_for_drug(self, drug: int) -> NDArray[np.float32]:
        scores = np.zeros(self.num_proteins, dtype=np.float32)
        for intermediate_protein in self.drug_neighbors[drug]:
            protein_degree = max(1, len(self.protein_neighbors[intermediate_protein]))
            for intermediate_drug in self.protein_neighbors[intermediate_protein]:
                if intermediate_drug == drug:
                    continue
                drug_degree = max(1, len(self.drug_neighbors[intermediate_drug]))
                weight = 1.0 / (protein_degree * drug_degree)
                for target_protein in self.drug_neighbors[intermediate_drug]:
                    scores[target_protein] += weight
        return scores

    def bridge_paths(
        self, drug: int, protein: int, *, limit: int | None = 5
    ) -> tuple[BridgePath, ...]:
        paths: list[BridgePath] = []
        target_drugs = set(self.protein_neighbors[protein])
        for intermediate_protein in self.drug_neighbors[drug]:
            shared_drugs = target_drugs.intersection(self.protein_neighbors[intermediate_protein])
            for intermediate_drug in shared_drugs:
                if intermediate_drug == drug:
                    continue
                weight = 1.0 / max(
                    1,
                    len(self.protein_neighbors[intermediate_protein])
                    * len(self.drug_neighbors[intermediate_drug]),
                )
                paths.append(
                    BridgePath(
                        drug,
                        intermediate_protein,
                        intermediate_drug,
                        protein,
                        weight,
                    )
                )
        paths.sort(
            key=lambda path: (
                -path.weight,
                path.intermediate_protein,
                path.intermediate_drug,
            )
        )
        return tuple(paths if limit is None else paths[:limit])

    def projection_mass_drug(self, drug: int) -> float:
        return sum(
            max(0, len(self.protein_neighbors[protein]) - 1)
            / max(1, len(self.protein_neighbors[protein]))
            for protein in self.drug_neighbors[drug]
        )

    def projection_mass_protein(self, protein: int) -> float:
        return sum(
            max(0, len(self.drug_neighbors[drug]) - 1) / max(1, len(self.drug_neighbors[drug]))
            for drug in self.protein_neighbors[protein]
        )

    def top_similar_drugs(self, drug: int, *, limit: int = 5) -> tuple[tuple[int, float], ...]:
        scores: dict[int, float] = defaultdict(float)
        for protein in self.drug_neighbors[drug]:
            degree = max(1, len(self.protein_neighbors[protein]))
            for candidate in self.protein_neighbors[protein]:
                if candidate != drug:
                    scores[candidate] += 1.0 / degree
        return tuple(sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit])

    def top_similar_proteins(
        self, protein: int, *, limit: int = 5
    ) -> tuple[tuple[int, float], ...]:
        scores: dict[int, float] = defaultdict(float)
        for drug in self.protein_neighbors[protein]:
            degree = max(1, len(self.drug_neighbors[drug]))
            for candidate in self.drug_neighbors[drug]:
                if candidate != protein:
                    scores[candidate] += 1.0 / degree
        return tuple(sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit])
