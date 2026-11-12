
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_features():
    response = client.get("/features")

    assert response.status_code == 200
    assert len(response.json()["required_features"]) == 14


def test_missing_features():
    response = client.post(
        "/predict",
        json={"features": {"age": 51}}
    )

    assert response.status_code == 422


def test_known_prediction():
    features = {
        "dis_seeing": 0.0,
        "dis_loss_sensation": 0.0,
        "dis_understanding": 0.0,
        "dis_hearing": 0.0,
        "uses_public_transport": 1.0,
        "multiple_disability": 0.0,
        "principal_activity_status": 95.0,
        "dis_deformity": 0.0,
        "dis_anxiety_mood": 0.0,
        "relation_to_head": 1.0,
        "age": 51.0,
        "education_level": 1.0,
        "accesses_public_building": 0.0,
        "log_monthly_exp": 9.093918908958932
    }

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["probability"] == pytest.approx(
        0.3059,
        abs=0.0001
    )

    assert result["predicted_class"] == 0
