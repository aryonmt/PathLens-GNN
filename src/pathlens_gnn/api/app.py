from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from pathlens_gnn.api.schemas import (
    EntityResponse,
    EntitySearchResponse,
    ExplanationResponse,
    GraphLink,
    GraphNode,
    GraphResponse,
    PairScoreResponse,
    PredictRequest,
    RecommendationItem,
    RecommendationResponse,
)
from pathlens_gnn.artifact.runtime import ArtifactRuntime
from pathlens_gnn.constants import (
    MAX_EGO_LINKS,
    MAX_EGO_NODES,
    MAX_RECOMMENDATIONS,
    MAX_SEARCH_RESULTS,
)


def create_app(artifact_dir: str | Path | None = None) -> FastAPI:
    artifact_value: str | Path = (
        artifact_dir
        if artifact_dir is not None
        else os.environ.get("PATHLENS_ARTIFACT_DIR", "artifacts/fixtures/demo-v1")
    )
    resolved_artifact = Path(artifact_value)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            app.state.runtime = ArtifactRuntime(resolved_artifact)
            app.state.load_error = None
        except Exception as error:  # health endpoint must explain unavailable artifacts
            app.state.runtime = None
            app.state.load_error = str(error)
        yield
        if app.state.runtime is not None:
            app.state.runtime.close()

    app = FastAPI(
        title="PathLens-GNN API",
        version="1.0.0",
        description="Inference-only research prioritization for known BioSNAP DTI entities.",
        lifespan=lifespan,
    )
    origins = os.getenv("PATHLENS_CORS_ORIGINS", "http://localhost:5173").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in origins if origin.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def runtime(request: Request) -> ArtifactRuntime:
        loaded = cast(ArtifactRuntime | None, request.app.state.runtime)
        if loaded is None:
            raise HTTPException(status_code=503, detail=request.app.state.load_error)
        return loaded

    @app.get("/api/v1/health")
    def health(request: Request) -> dict[str, Any]:
        loaded = request.app.state.runtime
        if loaded is None:
            return {"ready": False, "error": request.app.state.load_error}
        return {
            "ready": True,
            "model_version": loaded.model_version,
            "dataset_version": loaded.dataset_version,
        }

    @app.get("/api/v1/model")
    def model_info(loaded: ArtifactRuntime = Depends(runtime)) -> dict[str, Any]:
        return {
            "model_version": loaded.model_version,
            "dataset_version": loaded.dataset_version,
            "name": loaded.manifest.get("name", "PathLens-GNN"),
            "summary": loaded.manifest.get("summary", "Path-aware DTI research prioritization"),
            "limitations": loaded.manifest.get("limitations", []),
        }

    @app.get("/api/v1/entities", response_model=EntitySearchResponse)
    def search_entities(
        q: str = "",
        entity_type: str | None = Query(None, alias="type"),
        limit: int = Query(25, ge=1, le=MAX_SEARCH_RESULTS),
        cursor: int = Query(0, ge=0),
        loaded: ArtifactRuntime = Depends(runtime),
    ) -> EntitySearchResponse:
        filters = ["(entity_id LIKE ? OR display_name LIKE ?)"]
        term = f"%{q}%"
        parameters: list[Any] = [term, term]
        if entity_type:
            if entity_type not in {"drug", "protein"}:
                raise HTTPException(status_code=422, detail="type must be drug or protein")
            filters.append("entity_type = ?")
            parameters.append(entity_type)
        parameters.extend((limit + 1, cursor))
        rows = loaded.fetchall(
            f"SELECT * FROM entities WHERE {' AND '.join(filters)} "
            "ORDER BY display_name, entity_id LIMIT ? OFFSET ?",
            parameters,
        )
        items = [_entity_response(row, loaded) for row in rows[:limit]]
        return EntitySearchResponse(
            model_version=loaded.model_version,
            dataset_version=loaded.dataset_version,
            items=items,
            next_cursor=cursor + limit if len(rows) > limit else None,
        )

    @app.get("/api/v1/entities/{entity_id}", response_model=EntityResponse)
    def get_entity(entity_id: str, loaded: ArtifactRuntime = Depends(runtime)) -> EntityResponse:
        return _entity_response(_require_entity(loaded, entity_id), loaded)

    @app.get("/api/v1/recommendations/{entity_id}", response_model=RecommendationResponse)
    def recommendations(
        entity_id: str,
        limit: int = Query(25, ge=1, le=MAX_RECOMMENDATIONS),
        min_score: float = Query(0.0, ge=0.0, le=100.0),
        loaded: ArtifactRuntime = Depends(runtime),
    ) -> RecommendationResponse:
        source = _require_entity(loaded, entity_id)
        rows = loaded.fetchall(
            "SELECT r.*, e.entity_type AS target_type, e.display_name AS target_name "
            "FROM recommendations r JOIN entities e ON e.entity_id = r.target_id "
            "WHERE r.source_id = ? AND r.pathlens_score >= ? ORDER BY r.rank LIMIT ?",
            (entity_id, min_score, limit),
        )
        return RecommendationResponse(
            model_version=loaded.model_version,
            dataset_version=loaded.dataset_version,
            source=_entity_response(source, loaded),
            items=[RecommendationItem(**dict(row)) for row in rows],
        )

    @app.post("/api/v1/predict", response_model=PairScoreResponse)
    def predict(
        payload: PredictRequest, loaded: ArtifactRuntime = Depends(runtime)
    ) -> PairScoreResponse:
        return _predict_pair(loaded, payload.source_id, payload.target_id)

    @app.get("/api/v1/explanations/{source_id}/{target_id}", response_model=ExplanationResponse)
    def explanation(
        source_id: str,
        target_id: str,
        loaded: ArtifactRuntime = Depends(runtime),
    ) -> ExplanationResponse:
        score = _predict_pair(loaded, source_id, target_id)
        row = loaded.fetchone(
            "SELECT payload_json, truncated FROM explanations WHERE drug_id = ? AND protein_id = ?",
            (score.drug_id, score.protein_id),
        )
        payload: dict[str, Any]
        truncated = False
        if row:
            payload = json.loads(row["payload_json"])
            truncated = bool(row["truncated"])
        else:
            payload, truncated = _compute_explanation(loaded, score.drug_id, score.protein_id)
        return ExplanationResponse(
            model_version=loaded.model_version,
            dataset_version=loaded.dataset_version,
            source_id=source_id,
            target_id=target_id,
            projection_context=cast(
                dict[str, list[dict[str, str | float]]], payload["projection_context"]
            ),
            bridge_paths=cast(list[dict[str, object]], payload["bridge_paths"]),
            channel_contributions=score.contributions,
            truncated=truncated,
        )

    @app.get("/api/v1/graph/ego/{entity_id}", response_model=GraphResponse)
    def ego_graph(
        entity_id: str,
        hops: int = Query(2, ge=1, le=3),
        max_nodes: int = Query(MAX_EGO_NODES, ge=2, le=MAX_EGO_NODES),
        loaded: ArtifactRuntime = Depends(runtime),
    ) -> GraphResponse:
        _require_entity(loaded, entity_id)
        return _build_ego_graph(loaded, entity_id, hops, max_nodes)

    @app.get("/api/v1/graph/pair", response_model=GraphResponse)
    def pair_graph(
        source_id: str,
        target_id: str,
        loaded: ArtifactRuntime = Depends(runtime),
    ) -> GraphResponse:
        explanation_value = explanation(source_id, target_id, loaded)
        return _graph_from_explanation(loaded, explanation_value)

    @app.get("/api/v1/exports/recommendations/{entity_id}.csv")
    def export_recommendations(
        entity_id: str, loaded: ArtifactRuntime = Depends(runtime)
    ) -> StreamingResponse:
        _require_entity(loaded, entity_id)
        rows = loaded.fetchall(
            "SELECT source_id, target_id, rank, pathlens_score, bridge_count "
            "FROM recommendations WHERE source_id = ? ORDER BY rank",
            (entity_id,),
        )
        stream = io.StringIO()
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("source_id", "target_id", "rank", "pathlens_score", "bridge_count"))
        writer.writerows(tuple(row) for row in rows)
        return StreamingResponse(
            iter([stream.getvalue()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{entity_id}-recommendations.csv"'
            },
        )

    return app


def _require_entity(loaded: ArtifactRuntime, entity_id: str) -> sqlite3.Row:
    row = loaded.fetchone("SELECT * FROM entities WHERE entity_id = ?", (entity_id,))
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown entity: {entity_id}")
    return row


def _entity_response(row: sqlite3.Row, loaded: ArtifactRuntime) -> EntityResponse:
    return EntityResponse(
        model_version=loaded.model_version,
        dataset_version=loaded.dataset_version,
        entity_id=row["entity_id"],
        entity_type=row["entity_type"],
        display_name=row["display_name"],
        degree=row["degree"],
    )


def _predict_pair(loaded: ArtifactRuntime, source_id: str, target_id: str) -> PairScoreResponse:
    source = _require_entity(loaded, source_id)
    target = _require_entity(loaded, target_id)
    if source["entity_type"] == target["entity_type"]:
        raise HTTPException(
            status_code=422,
            detail="DTI prediction requires one drug and one protein",
        )
    drug = source if source["entity_type"] == "drug" else target
    protein = target if target["entity_type"] == "protein" else source
    evidence = loaded.fetchone(
        "SELECT projection_mass, bridge_score FROM pair_features "
        "WHERE drug_id = ? AND protein_id = ?",
        (drug["entity_id"], protein["entity_id"]),
    )
    structural = (
        (float(evidence["projection_mass"]), float(evidence["bridge_score"]))
        if evidence
        else _compute_structural_features(
            loaded,
            int(drug["local_index"]),
            int(protein["local_index"]),
        )
    )
    score = loaded.score_pair(drug["local_index"], protein["local_index"], structural)
    known = (
        loaded.fetchone(
            "SELECT 1 FROM known_edges WHERE drug_id = ? AND protein_id = ?",
            (drug["entity_id"], protein["entity_id"]),
        )
        is not None
    )
    return PairScoreResponse(
        model_version=loaded.model_version,
        dataset_version=loaded.dataset_version,
        source_id=source_id,
        target_id=target_id,
        drug_id=drug["entity_id"],
        protein_id=protein["entity_id"],
        known=known,
        pathlens_score=score.pathlens_score,
        logit=score.logit,
        gate_weights=list(score.gate_weights),
        expert_logits=list(score.expert_logits),
        contributions=list(score.contributions),
    )


def _compute_structural_features(
    loaded: ArtifactRuntime, drug_index: int, protein_index: int
) -> tuple[float, float]:
    if loaded.graph is None:
        return 0.0, 0.0
    graph = loaded.graph
    projection = graph.projection_mass_drug(drug_index) + graph.projection_mass_protein(
        protein_index
    )
    bridge = sum(path.weight for path in graph.bridge_paths(drug_index, protein_index, limit=None))
    return float(projection), float(bridge)


def _compute_explanation(
    loaded: ArtifactRuntime, drug_id: str, protein_id: str
) -> tuple[dict[str, Any], bool]:
    if loaded.graph is None:
        return {
            "projection_context": {"similar_drugs": [], "similar_proteins": []},
            "bridge_paths": [],
        }, False
    drug = _require_entity(loaded, drug_id)
    protein = _require_entity(loaded, protein_id)
    drug_index = int(drug["local_index"])
    protein_index = int(protein["local_index"])
    graph = loaded.graph
    paths = graph.bridge_paths(drug_index, protein_index, limit=6)
    return {
        "projection_context": {
            "similar_drugs": [
                {"entity_id": loaded.drug_ids[index], "score": score}
                for index, score in graph.top_similar_drugs(drug_index)
            ],
            "similar_proteins": [
                {"entity_id": loaded.protein_ids[index], "score": score}
                for index, score in graph.top_similar_proteins(protein_index)
            ],
        },
        "bridge_paths": [
            {
                "nodes": [
                    drug_id,
                    loaded.protein_ids[path.intermediate_protein],
                    loaded.drug_ids[path.intermediate_drug],
                    protein_id,
                ],
                "weight": path.weight,
            }
            for path in paths[:5]
        ],
    }, len(paths) > 5


def _build_ego_graph(
    loaded: ArtifactRuntime, entity_id: str, hops: int, max_nodes: int
) -> GraphResponse:
    visited = {entity_id}
    queue: deque[tuple[str, int]] = deque([(entity_id, 0)])
    links: list[GraphLink] = []
    truncated = False
    while queue and len(links) < MAX_EGO_LINKS:
        current, depth = queue.popleft()
        if depth >= hops:
            continue
        rows = loaded.fetchall(
            "SELECT drug_id, protein_id FROM known_edges WHERE drug_id = ? OR protein_id = ?",
            (current, current),
        )
        for row in rows:
            neighbor = row["protein_id"] if row["drug_id"] == current else row["drug_id"]
            if neighbor not in visited:
                if len(visited) >= max_nodes:
                    truncated = True
                    continue
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
            links.append(GraphLink(source=row["drug_id"], target=row["protein_id"], kind="known"))
            if len(links) >= MAX_EGO_LINKS:
                truncated = True
                break
    nodes = [_graph_node(_require_entity(loaded, node_id)) for node_id in sorted(visited)]
    return GraphResponse(
        model_version=loaded.model_version,
        dataset_version=loaded.dataset_version,
        nodes=nodes,
        links=links,
        truncated=truncated,
    )


def _graph_from_explanation(
    loaded: ArtifactRuntime, explanation: ExplanationResponse
) -> GraphResponse:
    node_ids = {explanation.source_id, explanation.target_id}
    links: list[GraphLink] = [
        GraphLink(
            source=explanation.source_id,
            target=explanation.target_id,
            kind="prediction",
            weight=max(abs(value) for value in explanation.channel_contributions),
            hop=0,
        )
    ]
    for path in explanation.bridge_paths:
        ids = [str(value) for value in cast(list[object], path["nodes"])]
        node_ids.update(ids)
        for left, right in zip(ids, ids[1:], strict=False):
            links.append(GraphLink(source=left, target=right, kind="support", hop=3))
    nodes = []
    for node_id in sorted(node_ids):
        row = _require_entity(loaded, node_id)
        role = (
            "candidate" if node_id in {explanation.source_id, explanation.target_id} else "support"
        )
        nodes.append(_graph_node(row, role=role))
    return GraphResponse(
        model_version=loaded.model_version,
        dataset_version=loaded.dataset_version,
        nodes=nodes,
        links=links,
        truncated=explanation.truncated,
    )


def _graph_node(row: sqlite3.Row, *, role: str = "context") -> GraphNode:
    return GraphNode(
        id=row["entity_id"],
        label=row["display_name"],
        entity_type=row["entity_type"],
        degree=row["degree"],
        role=role,
    )


app = create_app()
