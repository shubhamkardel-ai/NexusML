import mlflow


MODEL_NAME = "NexusML-Churn-Model"
MODEL_VERSION = "1"

MLFLOW_TRACKING_URI = "sqlite:///./mlflow.db"


def load_model():
    """Load the registered production model from MLflow."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

    model = mlflow.sklearn.load_model(model_uri)

    return model


if __name__ == "__main__":
    print("=" * 60)
    print("NexusML — Model Loading Test")
    print("=" * 60)

    model = load_model()

    print(f"Model: {MODEL_NAME}")
    print(f"Version: {MODEL_VERSION}")
    print(f"Model type: {type(model).__name__}")
    print("Status: Model loaded successfully")
    print("=" * 60)