from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from pathlens_gnn.constants import PSEUDO_ENTITIES

DRUG_ID_PATTERN = re.compile(r"^DB\d{5}$")
PROTEIN_ID_PATTERN = re.compile(r"^[A-Z0-9]{6,10}$")


class EntityType(StrEnum):
    DRUG = "drug"
    PROTEIN = "protein"


@dataclass(frozen=True, slots=True, order=True)
class Edge:
    drug_id: str
    protein_id: str

    def __post_init__(self) -> None:
        if not is_valid_drug_id(self.drug_id):
            raise ValueError(f"Invalid drug identifier: {self.drug_id!r}")
        if not is_valid_protein_id(self.protein_id):
            raise ValueError(f"Invalid protein identifier: {self.protein_id!r}")


@dataclass(frozen=True, slots=True)
class Entity:
    entity_id: str
    entity_type: EntityType
    display_name: str


@dataclass(frozen=True, slots=True)
class CanonicalDataset:
    edges: tuple[Edge, ...]
    drugs: tuple[str, ...]
    proteins: tuple[str, ...]
    source_sha256: str

    @property
    def entity_count(self) -> int:
        return len(self.drugs) + len(self.proteins)


def is_valid_drug_id(value: str) -> bool:
    return value not in PSEUDO_ENTITIES and bool(DRUG_ID_PATTERN.fullmatch(value))


def is_valid_protein_id(value: str) -> bool:
    return value not in PSEUDO_ENTITIES and bool(PROTEIN_ID_PATTERN.fullmatch(value))
