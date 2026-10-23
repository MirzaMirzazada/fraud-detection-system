from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_endpoint():
    payload = {
        "Time": 1000.0,
        "Amount": 50.0,
        "features": {
            f"V{i}": 0.0 for i in range(1, 29)
        },
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()
    assert 0 <= result["fraud_probability"] <= 1
    assert 0 <= result["threshold"] <= 1
    assert isinstance(result["is_suspicious"], bool)


def test_missing_features_are_rejected():
    payload = {
        "Time": 1000.0,
        "Amount": 50.0,
        "features": {"V1": 0.0},
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422