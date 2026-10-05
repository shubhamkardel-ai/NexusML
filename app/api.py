import time

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from services.logging_config import get_logger

from services.model_service import (
    load_model,
    get_production_model_version,
)
from services.prediction_logger import initialize_database, log_prediction

from services.monitoring_service import (
    get_model_metrics,
    get_prediction_metrics,
)

from services.drift_detection import (
    detect_drift,
    get_production_data,
)

MODEL_NAME = "NexusML-Churn-Model"
PRODUCTION_ALIAS = "champion"

logger = get_logger("NexusML.API")

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
    production_version = get_production_model_version()

    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_alias": PRODUCTION_ALIAS,
        "model_version": production_version,
    }

@app.get("/ready")
def readiness():
    """
    Check whether NexusML is ready to serve predictions.
    """

    try:
        production_version = get_production_model_version()

        return {
            "status": "ready",
            "model": MODEL_NAME,
            "model_alias": PRODUCTION_ALIAS,
            "model_version": production_version,
        }

    except Exception as error:
        logger.exception("Readiness check failed")

        return {
            "status": "not_ready",
            "reason": str(error),
        }

@app.get("/monitoring/metrics")
def monitoring_metrics():
    """Return production prediction metrics."""

    return get_prediction_metrics()

@app.get("/monitoring/models")
def monitoring_models():
    """Return production metrics grouped by model version."""

    return get_model_metrics()

@app.get("/monitoring/drift")
def monitoring_drift():
    """Return the current production feature drift report."""

    production_df = get_production_data()

    if production_df.empty:
        return {
            "status": "insufficient_data",
            "production_samples": 0,
        }

    return detect_drift(production_df)

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

    logger.info(
        "Prediction generated | model=%s | version=%s | prediction=%s | probability=%.4f",
        MODEL_NAME,
        get_production_model_version(),
        prediction,
        probability,
    )

    log_prediction(
        model_name=MODEL_NAME,
        model_version=get_production_model_version(),
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
        "model_alias": PRODUCTION_ALIAS,
        "model_version": get_production_model_version(),
    }