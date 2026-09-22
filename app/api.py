from fastapi import FastAPI
from pydantic import BaseModel, Field

from services.model_service import load_model
from services.prediction_logger import initialize_database, log_prediction


MODEL_NAME = "NexusML-Churn-Model"
MODEL_VERSION = "1"


app = FastAPI(
    title="NexusML Prediction API",
    description=(
        "Production-style ML prediction service powered by "
        "the MLflow Model Registry."
    ),
    version="1.0.0",
)


model = load_model()

# Initialize the prediction logging database when the API starts.
initialize_database()


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
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict")
def predict(request: ChurnRequest):
    """Generate and log a churn prediction."""

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

    log_prediction(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        age=request.age,
        tenure_months=request.tenure_months,
        monthly_charges=request.monthly_charges,
        support_tickets=request.support_tickets,
        usage_hours=request.usage_hours,
        contract_length=request.contract_length,
        prediction=prediction,
        churn_probability=probability,
    )

    return {
        "prediction": prediction,
        "churn_probability": round(probability, 4),
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }