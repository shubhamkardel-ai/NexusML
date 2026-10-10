
from pathlib import Path

import mlflow
from mlflow import MlflowClient

MODEL_NAME = "NexusML-Churn-Model"
PRODUCTION_ALIAS = "champion"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


def get_mlflow_client():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def get_production_model_version():
    client = get_mlflow_client()
    model_version = client.get_model_version_by_alias(
        MODEL_NAME, PRODUCTION_ALIAS
    )
    return str(model_version.version)


def load_model():
    client = get_mlflow_client()

    model_version = client.get_model_version_by_alias(
        MODEL_NAME, PRODUCTION_ALIAS
    )

    # MLflow 3 may register a model using a model ID instead
    # of a portable filesystem path.
    model_id = model_version.source.rsplit("/", 1)[-1]

    run = client.get_run(model_version.run_id)
    experiment_id = run.info.experiment_id

    artifact_path = (
        PROJECT_ROOT
        / "mlruns"
        / str(experiment_id)
        / "models"
        / model_id
        / "artifacts"
    )

    if not (artifact_path / "MLmodel").is_file():
        raise FileNotFoundError(
            f"MLflow model artifacts not found: {artifact_path}"
        )

    return mlflow.sklearn.load_model(str(artifact_path))
