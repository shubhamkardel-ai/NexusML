from pathlib import Path

import json
import mlflow
import mlflow.sklearn
import pandas as pd

from services.model_evaluation import (
    evaluate_current_model,
    evaluate_candidate_model,
    validate_candidate,
)

from services.model_registry import promote_model

DATA_PATH = Path("data/raw/customer_churn.csv")

MLFLOW_TRACKING_URI = "sqlite:///./mlflow.db"
MLFLOW_EXPERIMENT_NAME = "NexusML-Customer-Churn"
REGISTERED_MODEL_NAME = "NexusML-Churn-Model"


def check_retraining_required():
    """
    Check whether production drift requires model retraining.
    """

    from services.drift_detection import (
        detect_drift,
        get_production_data,
    )

    production_df = get_production_data()

    if production_df.empty:
        return {
            "retraining_required": False,
            "reason": "No production data available",
        }

    drift_result = detect_drift(production_df)

    retraining_required = bool(
        drift_result.get("drift_detected", False)
    )

    return {
        "retraining_required": retraining_required,
        "reason": (
            "Feature drift detected"
            if retraining_required
            else "No significant drift detected"
        ),
        "drift_result": drift_result,
    }


def run_retraining():
    """
    Train, evaluate, and validate a candidate model.
    """

    print("=" * 60)
    print("NexusML — Automated Retraining")
    print("=" * 60)

    print("\nChecking whether retraining is required...")

    retraining_check = check_retraining_required()

    print(
        f"Retraining required: "
        f"{retraining_check['retraining_required']}"
    )

    print(
        f"Reason: "
        f"{retraining_check['reason']}"
    )

    if not retraining_check["retraining_required"]:
        print("\nRetraining skipped.")

        return {
            "status": "skipped",
            "reason": retraining_check["reason"],
        }

    print("\nDrift detected.")
    print("Starting candidate model training...")

    current_metrics = evaluate_current_model()

    candidate_model, candidate_metrics = (
        evaluate_candidate_model()
    )

    validation = validate_candidate(
        current_metrics,
        candidate_metrics,
    )

    print("\nCurrent Model Metrics:")

    for metric, value in current_metrics.items():
        print(f"{metric}: {value}")

    print("\nCandidate Model Metrics:")

    for metric, value in candidate_metrics.items():
        print(f"{metric}: {value}")

    print("\nValidation Result:")

    print(
        f"F1 improvement: "
        f"{validation['f1_improvement']}"
    )

    print(
        f"Accuracy improvement: "
        f"{validation['accuracy_improvement']}"
    )

    print(
        f"Promotion approved: "
        f"{validation['approved']}"
    )

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name="automated-retraining"
    ):

        mlflow.log_params({
            "model_type": "LogisticRegression",
            "retraining_trigger": "feature_drift",
        })

        mlflow.log_metrics({
            "current_f1_score": current_metrics["f1_score"],
            "candidate_f1_score": candidate_metrics["f1_score"],
            "current_accuracy": current_metrics["accuracy"],
            "candidate_accuracy": candidate_metrics["accuracy"],
            "f1_improvement": validation["f1_improvement"],
            "accuracy_improvement": validation[
                "accuracy_improvement"
            ],
        })

        validation_report = {
            "current_model": {
                "metrics": current_metrics,
            },
            "candidate_model": {
                "metrics": candidate_metrics,
            },
            "validation": validation,
            "retraining_trigger": "feature_drift",
        }

        with open(
                "validation_report.json",
                "w",
                encoding="utf-8",
        ) as report_file:
            json.dump(
                validation_report,
                report_file,
                indent=4,
            )

        mlflow.log_artifact(
            "validation_report.json",
        )

        mlflow.set_tags({
            "promotion_approved": str(
                validation["approved"]
            ),
            "retraining_triggered": "true",
        })

        if validation["approved"]:

            print("\nCandidate approved.")
            print("Registering candidate model...")

            model_info = mlflow.sklearn.log_model(
                sk_model=candidate_model,
                name="candidate_model",
                registered_model_name=REGISTERED_MODEL_NAME,
            )

            registered_version = model_info.registered_model_version

            print(
                f"Candidate registered as version "
                f"{registered_version}."
            )

            print("Promoting candidate to champion...")

            promotion_result = promote_model(
                registered_version
            )

            print(
                f"Production model updated: "
                f"version {registered_version}"
            )

            status = "approved"

        else:

            print("\nCandidate rejected.")
            print("Production model remains unchanged.")

            promotion_result = None

            status = "rejected"

    print("\nRetraining Status:")
    print(status)

    print("=" * 60)

    return {
        "status": status,
        "current_metrics": current_metrics,
        "candidate_metrics": candidate_metrics,
        "validation": validation,
        "promotion": promotion_result,
    }


if __name__ == "__main__":
    run_retraining()