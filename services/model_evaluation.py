from pathlib import Path

import mlflow
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = Path("data/raw/customer_churn.csv")
CURRENT_MODEL_PATH = Path("models/churn_model.joblib")

FEATURE_COLUMNS = [
    "age",
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "usage_hours",
    "contract_length",
]

TARGET_COLUMN = "churn"

# Minimum improvement required for promotion.
MIN_F1_IMPROVEMENT = 0.01

MLFLOW_TRACKING_URI = "sqlite:///./mlflow.db"
MLFLOW_EXPERIMENT_NAME = "NexusML-Customer-Churn"


def load_evaluation_data():
    """Load and split the churn dataset for model evaluation."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def calculate_metrics(model, X_test, y_test):
    """Calculate classification metrics for a model."""

    predictions = model.predict(X_test)

    return {
        "accuracy": round(
            float(accuracy_score(y_test, predictions)),
            4,
        ),
        "precision": round(
            float(
                precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "recall": round(
            float(
                recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
        "f1_score": round(
            float(
                f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                )
            ),
            4,
        ),
    }


def evaluate_current_model():
    """Evaluate the currently deployed model."""

    if not CURRENT_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {CURRENT_MODEL_PATH}"
        )

    model = joblib.load(CURRENT_MODEL_PATH)

    _, X_test, _, y_test = load_evaluation_data()

    return calculate_metrics(
        model,
        X_test,
        y_test,
    )


def train_candidate_model(
    X_train,
    y_train,
):
    """Train a candidate model using the training dataset."""

    candidate_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    candidate_model.fit(
        X_train,
        y_train,
    )

    return candidate_model


def evaluate_candidate_model():
    """Train and evaluate a new candidate model."""

    X_train, X_test, y_train, y_test = load_evaluation_data()

    candidate_model = train_candidate_model(
        X_train,
        y_train,
    )

    metrics = calculate_metrics(
        candidate_model,
        X_test,
        y_test,
    )

    return candidate_model, metrics


def validate_candidate(
    current_metrics,
    candidate_metrics,
):
    """
    Determine whether the candidate model satisfies
    the promotion criteria.
    """

    f1_improvement = (
        candidate_metrics["f1_score"]
        - current_metrics["f1_score"]
    )

    accuracy_improvement = (
        candidate_metrics["accuracy"]
        - current_metrics["accuracy"]
    )

    approved = (
        f1_improvement >= MIN_F1_IMPROVEMENT
        and candidate_metrics["accuracy"]
        >= current_metrics["accuracy"]
    )

    return {
        "approved": bool(approved),
        "f1_improvement": round(
            float(f1_improvement),
            4,
        ),
        "accuracy_improvement": round(
            float(accuracy_improvement),
            4,
        ),
        "minimum_f1_improvement": MIN_F1_IMPROVEMENT,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("NexusML — Model Evaluation & Validation")
    print("=" * 60)

    current_metrics = evaluate_current_model()

    print("\nCurrent Model:")
    for metric, value in current_metrics.items():
        print(f"{metric}: {value}")

    candidate_model, candidate_metrics = (
        evaluate_candidate_model()
    )

    print("\nCandidate Model:")
    for metric, value in candidate_metrics.items():
        print(f"{metric}: {value}")

    validation = validate_candidate(
        current_metrics,
        candidate_metrics,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="candidate-model-validation"):

        mlflow.log_params({
            "model_type": "LogisticRegression",
            "validation_threshold_f1": MIN_F1_IMPROVEMENT,
        })

        mlflow.log_metrics({
            "current_accuracy": current_metrics["accuracy"],
            "current_precision": current_metrics["precision"],
            "current_recall": current_metrics["recall"],
            "current_f1_score": current_metrics["f1_score"],
            "candidate_accuracy": candidate_metrics["accuracy"],
            "candidate_precision": candidate_metrics["precision"],
            "candidate_recall": candidate_metrics["recall"],
            "candidate_f1_score": candidate_metrics["f1_score"],
            "f1_improvement": validation["f1_improvement"],
            "accuracy_improvement": validation["accuracy_improvement"],
        })

        mlflow.set_tag(
            "promotion_approved",
            str(validation["approved"]),
        )

    validation_report = {
        "current_model": {
            "metrics": current_metrics,
        },
        "candidate_model": {
            "metrics": candidate_metrics,
        },
        "validation": validation,
    }

    print("\nValidation Report:")

    print(
        f"F1 improvement: "
        f"{validation_report['validation']['f1_improvement']}"
    )

    print(
        f"Accuracy improvement: "
        f"{validation_report['validation']['accuracy_improvement']}"
    )

    print(
        f"Minimum required F1 improvement: "
        f"{validation_report['validation']['minimum_f1_improvement']}"
    )

    print(
        f"Promotion approved: "
        f"{validation_report['validation']['approved']}"
    )

    print("=" * 60)