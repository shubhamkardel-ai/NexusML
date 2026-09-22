from fastapi import FastAPI
from pydantic import BaseModel, Field

from services.model_service import load_model


app = FastAPI(
    title="NexusML Prediction API",
    description="Production-style ML prediction service powered by the MLflow Model Registry.",
    version="1.0.0",
)


model = load_model()


class ChurnRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    tenure_months: int = Field(..., ge=0)
    monthly_charges: float = Field(..., ge=0)
    support_tickets: int = Field(..., ge=0)
    usage_hours: float = Field(..., ge=0)
    contract_length: int = Field(..., ge=0)


@app.get("/health")
def health():
    """Return API health status."""

    return {
        "status": "healthy",
        "model": "NexusML-Churn-Model",
        "model_version": "1",
    }


@app.post("/predict")
def predict(request: ChurnRequest):
    """Generate a churn prediction."""

    features = [[
        request.age,
        request.tenure_months,
        request.monthly_charges,
        request.support_tickets,
        request.usage_hours,
        request.contract_length,
    ]]

    prediction = int(model.predict(features)[0])

    probability = float(model.predict_proba(features)[0][1])

    return {
        "prediction": prediction,
        "churn_probability": round(probability, 4),
        "model": "NexusML-Churn-Model",
        "model_version": "1",
    }