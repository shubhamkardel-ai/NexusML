import json
from pathlib import Path

import joblib
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


def train_model():
    """Train and evaluate the customer churn model."""

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

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, predictions), 4),
        "precision": round(
            precision_score(y_test, predictions, zero_division=0),
            4,
        ),
        "recall": round(
            recall_score(y_test, predictions, zero_division=0),
            4,
        ),
        "f1_score": round(
            f1_score(y_test, predictions, zero_division=0),
            4,
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    return metrics


def main():
    metrics = train_model()

    print("=" * 60)
    print("NexusML — Model Training")
    print("=" * 60)

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    print(f"Model saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")

    print("=" * 60)


if __name__ == "__main__":
    main()