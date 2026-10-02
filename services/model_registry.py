import mlflow
from mlflow import MlflowClient


MLFLOW_TRACKING_URI = "sqlite:///./mlflow.db"
MODEL_NAME = "NexusML-Churn-Model"
PRODUCTION_ALIAS = "champion"


def get_client():
    """Return an MLflow client connected to the NexusML tracking database."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def get_production_model_version():
    """
    Get the model version currently assigned to the production alias.
    """
    client = get_client()

    model_version = client.get_model_version_by_alias(
        MODEL_NAME,
        PRODUCTION_ALIAS,
    )

    return model_version.version


def promote_model(version):
    """
    Promote a registered model version to the production alias.
    """
    client = get_client()

    client.set_registered_model_alias(
        MODEL_NAME,
        PRODUCTION_ALIAS,
        str(version),
    )

    return {
        "model_name": MODEL_NAME,
        "model_version": str(version),
        "alias": PRODUCTION_ALIAS,
        "status": "promoted",
    }


if __name__ == "__main__":
    print("=" * 60)
    print("NexusML — Model Registry")
    print("=" * 60)

    print(f"\nModel: {MODEL_NAME}")
    print(f"Production alias: {PRODUCTION_ALIAS}")

    try:
        version = get_production_model_version()

        print(f"Current production version: {version}")

    except Exception as error:
        print("\nNo production alias is configured yet.")
        print(f"Reason: {error}")

    print("=" * 60)