
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ---------------------------------------
# 1. Test API health
# ---------------------------------------

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# ---------------------------------------
# 2. Test required features
# ---------------------------------------

def test_features():
    response = client.get("/features")

    assert response.status_code == 200
    assert len(response.json()["required_features"]) == 14


# ---------------------------------------
# 3. Test missing features
# ---------------------------------------

def test_missing_features():
    response = client.post(
        "/predict",
        json={"features": {"age": 51}}
    )

    assert response.status_code == 422


# ---------------------------------------
# 4. Test corrected prediction
# ---------------------------------------

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

    # Corrected training-aligned probability
    assert result["probability"] == pytest.approx(
        0.5849,
        abs=0.0001
    )

    # Probability above 50% gives class 1
    assert result["predicted_class"] == 1

    # Updated API version
    assert result["model_version"] == "1.0.1"


# ---------------------------------------
# 5. Reject invalid categorical values
# ---------------------------------------

def test_reject_fractional_nominal_code():
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
        "education_level": 1.5,
        "accesses_public_building": 0.0,
        "log_monthly_exp": 9.093918908958932
    }

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 422
