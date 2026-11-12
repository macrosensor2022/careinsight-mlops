
from pathlib import Path
from typing import Dict
import math

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]

model = joblib.load(
    ROOT / "models" / "caregiver_model.joblib"
)

features = joblib.load(
    ROOT / "models" / "selected_features.joblib"
)

# These features were represented as string categories
# in the original training notebook.
NOMINAL_FEATURES = [
    "education_level",
    "relation_to_head",
    "principal_activity_status"
]

app = FastAPI(
    title="CareInsight ML API",
    version="1.0.1"
)


class PredictionInput(BaseModel):
    features: Dict[str, float]


@app.get("/")
def home():
    return {
        "message": "CareInsight ML API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.get("/features")
def get_features():
    return {
        "required_features": features
    }


@app.post("/predict")
def predict(data: PredictionInput):
    values = data.features

    # Verify that exactly the required 14 features exist.
    if set(values) != set(features):
        raise HTTPException(
            status_code=422,
            detail="Input features do not match model requirements"
        )

    # Reject NaN and infinite numeric inputs.
    if not all(math.isfinite(v) for v in values.values()):
        raise HTTPException(
            status_code=422,
            detail="All values must be finite"
        )

    # Nominal survey codes must be whole numbers.
    for name in NOMINAL_FEATURES:
        if name in values and not float(values[name]).is_integer():
            raise HTTPException(
                status_code=422,
                detail=f"{name} must be an integer survey code"
            )

    # Preserve the selected feature order.
    input_df = pd.DataFrame(
        [{name: values[name] for name in features}]
    )

    # Match categorical data representation
    # used during model training.
    for name in NOMINAL_FEATURES:
        input_df[name] = input_df[name].map(
            lambda value: str(int(value))
        )

    # Run inference with the saved pipeline.
    probability = float(
        model.predict_proba(input_df)[0, 1]
    )

    return {
        "probability": round(probability, 4),
        "predicted_class": int(probability >= 0.5),
        "model_version": "1.0.1"
    }
