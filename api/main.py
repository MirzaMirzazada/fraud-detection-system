from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "models" / "fraud_model.joblib"

app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    description="Transaction risk scoring demonstration.",
)

artifact = joblib.load(ARTIFACT_PATH)
model = artifact["model"]
features = artifact["features"]
threshold = artifact["threshold"]


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Time: float = Field(ge=0)
    Amount: float = Field(ge=0)

    # Supply all V1–V28 values as one object.
    # They are required because the trained model expects 30 features.
    features: dict[str, float]


class Prediction(BaseModel):
    fraud_probability: float
    threshold: float
    is_suspicious: bool
    decision: str


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction):
    expected_pca = {f"V{i}" for i in range(1, 29)}

    if set(transaction.features) != expected_pca:
        raise HTTPException(
            status_code=422,
            detail="features must contain exactly V1 through V28",
        )

    values = {
        "Time": transaction.Time,
        "Amount": transaction.Amount,
        **transaction.features,
    }

    row = pd.DataFrame([values], columns=features)
    probability = float(model.predict_proba(row)[0, 1])
    suspicious = probability >= threshold

    return Prediction(
        fraud_probability=probability,
        threshold=threshold,
        is_suspicious=suspicious,
        decision="flag_for_review" if suspicious else "lower_risk",
    )