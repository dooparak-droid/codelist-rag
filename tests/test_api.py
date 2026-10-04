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
