import mlflow


MODEL_NAME = "NexusML-Churn-Model"
PRODUCTION_ALIAS = "champion"
MLFLOW_TRACKING_URI = "sqlite:///./mlflow.db"


def load_model():
    """
    Load the production model using the MLflow champion alias.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}"

    model = mlflow.sklearn.load_model(model_uri)

    return model


def get_production_model_version():
    """
    Get the version currently assigned to the production alias.
    """
    from mlflow import MlflowClient

    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    model_version = client.get_model_version_by_alias(
        MODEL_NAME,
        PRODUCTION_ALIAS,
    )

    return str(model_version.version)