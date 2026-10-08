"""test_api.py

Integration tests for FastAPI endpoints (/health, /codelist, /evaluate).
"""

from fastapi.testclient import TestClient
from codelist_rag.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["default_model"] == "openai:gpt-5.5"
    assert "version" in data
    assert "terminology_release" in data


def test_evaluate_endpoint():
    payload = {
        "generated_codes": [{"code": "99900001", "term": "Concept 1"}],
        "gold_codes": ["99900001", "99900002"],
        "retrieval_type": "rag",
        "retrieved_codes": ["99900001", "99900002"],
    }
    response = client.post("/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["precision"] == 1.0
    assert data["recall"] == 0.5
    assert data["true_positives"] == 1
    assert data["false_negatives"] == 1


def test_codelist_without_index_returns_503():
    response = client.post("/codelist", json={"condition": "Asthma", "use_rag": True})
    assert response.status_code == 503


def test_codelist_without_api_key_returns_503(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.post("/codelist", json={"condition": "Asthma", "use_rag": False})
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]
