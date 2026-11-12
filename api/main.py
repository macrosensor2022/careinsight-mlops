
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

app = FastAPI(
    title="CareInsight ML API",
    version="1.0.0"
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

    if set(values) != set(features):
        raise HTTPException(
            status_code=422,
            detail="Input features do not match model requirements"
        )

    if not all(math.isfinite(v) for v in values.values()):
        raise HTTPException(
            status_code=422,
            detail="All values must be finite"
        )

    input_df = pd.DataFrame(
        [{name: values[name] for name in features}]
    )

    probability = float(
        model.predict_proba(input_df)[0, 1]
    )

    return {
        "probability": round(probability, 4),
        "predicted_class": int(probability >= 0.5),
        "model_version": "1.0.0"
    }
