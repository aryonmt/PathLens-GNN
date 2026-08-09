from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from pathlens_gnn.api.app import create_app

FIXTURE = Path("artifacts/fixtures/demo-v1")


def test_discovery_pair_graph_and_export_flow() -> None:
    with TestClient(create_app(FIXTURE)) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200
        assert health.json()["ready"] is True

        search = client.get("/api/v1/entities", params={"q": "Alpha"})
        assert search.status_code == 200
        assert search.json()["items"][0]["entity_id"] == "DB00001"

        recommendations = client.get("/api/v1/recommendations/DB00001")
        assert recommendations.status_code == 200
        assert recommendations.json()["items"][0]["target_id"] == "P00002"

        score = client.post("/api/v1/predict", json={"source_id": "DB00001", "target_id": "P00002"})
        assert score.status_code == 200
        assert score.json()["known"] is False

        explanation = client.get("/api/v1/explanations/DB00001/P00002")
        assert len(explanation.json()["bridge_paths"]) == 1

        on_demand = client.get("/api/v1/explanations/DB00003/P00001")
        assert on_demand.status_code == 200
        assert len(on_demand.json()["bridge_paths"]) == 1
        assert on_demand.json()["truncated"] is False

        graph = client.get(
            "/api/v1/graph/pair",
            params={"source_id": "DB00001", "target_id": "P00002"},
        )
        assert graph.status_code == 200
        assert any(link["kind"] == "prediction" for link in graph.json()["links"])

        export = client.get("/api/v1/exports/recommendations/DB00001.csv")
        assert export.status_code == 200
        assert "source_id,target_id" in export.text


def test_api_rejects_same_type_and_unknown_entities() -> None:
    with TestClient(create_app(FIXTURE)) as client:
        same_type = client.post(
            "/api/v1/predict", json={"source_id": "DB00001", "target_id": "DB00002"}
        )
        assert same_type.status_code == 422
        assert client.get("/api/v1/entities/UNKNOWN").status_code == 404
