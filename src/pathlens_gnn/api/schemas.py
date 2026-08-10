from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EntityKind = Literal["drug", "protein"]


class VersionedModel(BaseModel):
    model_version: str
    dataset_version: str


class EntityResponse(VersionedModel):
    entity_id: str
    entity_type: EntityKind
    display_name: str
    degree: int


class EntitySearchResponse(VersionedModel):
    items: list[EntityResponse]
    next_cursor: int | None = None


class RecommendationItem(BaseModel):
    source_id: str
    target_id: str
    target_type: EntityKind
    target_name: str
    rank: int
    pathlens_score: float
    bridge_count: int


class RecommendationResponse(VersionedModel):
    source: EntityResponse
    items: list[RecommendationItem]


class PredictRequest(BaseModel):
    source_id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)


class PairScoreResponse(VersionedModel):
    source_id: str
    target_id: str
    drug_id: str
    protein_id: str
    known: bool
    pathlens_score: float
    logit: float
    gate_weights: list[float]
    expert_logits: list[float]
    contributions: list[float]


class GraphNode(BaseModel):
    id: str
    label: str
    entity_type: EntityKind
    degree: int
    role: str = "context"


class GraphLink(BaseModel):
    source: str
    target: str
    kind: str
    weight: float = 1.0
    hop: int = 1


class GraphResponse(VersionedModel):
    nodes: list[GraphNode]
    links: list[GraphLink]
    truncated: bool


class ExplanationResponse(VersionedModel):
    source_id: str
    target_id: str
    projection_context: dict[str, list[dict[str, str | float]]]
    bridge_paths: list[dict[str, object]]
    channel_contributions: list[float]
    truncated: bool
