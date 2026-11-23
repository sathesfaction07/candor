import json

import pytest

from sme_bridge.app import app as flask_app
from sme_bridge.core.engine import Engine
from sme_bridge.core.gate import ThresholdGate
from sme_bridge.core.ledger import InMemoryLedger
from sme_bridge.models import Chunk
from sme_bridge.retrieval.keyword import KeywordBackend


@pytest.fixture
def app():
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def setup_engine(monkeypatch):
    backend = KeywordBackend()
    backend.index(
        [
            Chunk(
                chunk_id="c1",
                doc_id="safety.pdf",
                text=(
                    "Button-cell batteries must be secured in compartments that require a "
                    "tool to open."
                ),
            )
        ]
    )
    ledger = InMemoryLedger()
    gate = ThresholdGate(threshold=0.90)
    engine = Engine(retrieval=backend, ledger=ledger, gate=gate)

    monkeypatch.setattr("sme_bridge.app.engine", engine)
    monkeypatch.setattr("sme_bridge.app.ledger", ledger)
    monkeypatch.setattr("sme_bridge.app.retrieval", backend)
    return engine


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert data["retrieval_backend"] == "keyword"
    assert data["gate_calibrated"] is False


def test_query_endpoint_returns_answer(client, setup_engine):
    response = client.post(
        "/api/query",
        json={"query": "What are the battery requirements?"},
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "answered" in data
    assert "confidence" in data
    assert "citations" in data
    assert "retrieval_support" in data["confidence"]
    assert "composite" in data["confidence"]
    assert data["confidence"]["gate_reason"]


def test_query_endpoint_empty_query(client):
    response = client.post("/api/query", json={"query": ""}, content_type="application/json")
    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data


def test_query_endpoint_missing_query(client):
    response = client.post("/api/query", json={}, content_type="application/json")
    assert response.status_code == 400


def test_query_endpoint_abstains_with_low_confidence(client, setup_engine, monkeypatch):
    gate = ThresholdGate(threshold=0.99)
    monkeypatch.setattr("sme_bridge.app.engine.gate", gate)

    response = client.post(
        "/api/query",
        json={"query": "What are the battery requirements?"},
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["answered"] is False
    assert "gap_id" in data
    assert "sme_question" in data


def test_gaps_endpoint_returns_open_gaps(client, setup_engine, monkeypatch):
    gate = ThresholdGate(threshold=0.99)
    monkeypatch.setattr("sme_bridge.app.engine.gate", gate)

    client.post(
        "/api/query",
        json={"query": "What are the battery requirements?"},
        content_type="application/json",
    )

    response = client.get("/api/gaps")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "gaps" in data
    assert data["count"] >= 1
    assert "gap_id" in data["gaps"][0]
    assert "query" in data["gaps"][0]
    assert "sme_question" in data["gaps"][0]


def test_answer_gap_endpoint(client, setup_engine, monkeypatch):
    gate = ThresholdGate(threshold=0.99)
    monkeypatch.setattr("sme_bridge.app.engine.gate", gate)

    query_response = client.post(
        "/api/query",
        json={"query": "What are the battery requirements?"},
        content_type="application/json",
    )
    gap_id = json.loads(query_response.data)["gap_id"]

    response = client.post(
        f"/api/gaps/{gap_id}/answer",
        json={"answer": "Battery compartments must be secured with screws."},
        content_type="application/json",
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["gap_id"] == gap_id
    assert data["indexed"] is True


def test_answer_gap_not_found(client):
    response = client.post(
        "/api/gaps/nonexistent-id/answer",
        json={"answer": "Some answer"},
        content_type="application/json",
    )
    assert response.status_code == 404


def test_answer_gap_already_answered(client, setup_engine, monkeypatch):
    gate = ThresholdGate(threshold=0.99)
    monkeypatch.setattr("sme_bridge.app.engine.gate", gate)

    query_response = client.post(
        "/api/query",
        json={"query": "What are the battery requirements?"},
        content_type="application/json",
    )
    gap_id = json.loads(query_response.data)["gap_id"]

    client.post(
        f"/api/gaps/{gap_id}/answer",
        json={"answer": "Battery compartments must be secured with screws."},
        content_type="application/json",
    )

    response = client.post(
        f"/api/gaps/{gap_id}/answer",
        json={"answer": "Another answer"},
        content_type="application/json",
    )
    assert response.status_code == 400


def test_answer_gap_empty_answer(client, setup_engine, monkeypatch):
    gate = ThresholdGate(threshold=0.99)
    monkeypatch.setattr("sme_bridge.app.engine.gate", gate)

    query_response = client.post(
        "/api/query",
        json={"query": "What are the battery requirements?"},
        content_type="application/json",
    )
    gap_id = json.loads(query_response.data)["gap_id"]

    response = client.post(
        f"/api/gaps/{gap_id}/answer",
        json={"answer": ""},
        content_type="application/json",
    )
    assert response.status_code == 400


def test_calibration_endpoint_uncalibrated(client):
    response = client.get("/api/calibration")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["calibrated"] is False
    assert "message" in data
    assert "UNCALIBRATED" in data["message"]
    assert "0.90" in data["message"]
