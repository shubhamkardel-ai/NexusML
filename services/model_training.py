import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from services.data_validation import validate_dataset


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

FEATURE_COLUMNS = [
    "age",
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "usage_hours",
    "contract_length",
]

EXPERIMENT_NAME = "NexusML-Customer-Churn"


def train_model():
    """Train, evaluate, and track the customer churn model."""

    data = validate_dataset()

    X = data[FEATURE_COLUMNS]
    y = data["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    mlflow.set_tracking_uri("sqlite:///./mlflow.db")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        metrics = {
            "accuracy": round(
                accuracy_score(y_test, predictions),
                4,
            ),
            "precision": round(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                4,
            ),
            "recall": round(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                4,
            ),
            "f1_score": round(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                4,
            ),
        }

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))

        mlflow.log_metrics(metrics)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, MODEL_PATH)

        with METRICS_PATH.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=4)

        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        mlflow.log_artifact(
            str(METRICS_PATH),
            artifact_path="evaluation",
        )

        run_id = mlflow.active_run().info.run_id

    return metrics, run_id


def main():
    metrics, run_id = train_model()

    print("=" * 60)
    print("NexusML — Model Training with MLflow")
    print("=" * 60)

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    print(f"Model saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")
    print(f"MLflow run ID: {run_id}")

    print("=" * 60)


if __name__ == "__main__":
    main()