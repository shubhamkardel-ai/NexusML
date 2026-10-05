import json
import mlflow

from mlflow import MlflowClient

from services.model_registry import (
    MODEL_NAME,
    PRODUCTION_ALIAS,
    get_production_model_version,
)

MLFLOW_TRACKING_URI = "sqlite:///./mlflow.db"
EXPERIMENT_NAME = "NexusML-Customer-Churn"


def get_mlflow_client():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def get_experiment():
    client = get_mlflow_client()

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        return None

    return experiment


def get_experiment_runs(limit=10):
    client = get_mlflow_client()
    experiment = get_experiment()

    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=limit,
    )

    results = []

    for run in runs:
        results.append(
            {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
            }
        )

    return results


def get_latest_retraining_result(runs):
    for run in runs:
        metrics = run.get("metrics", {})
        params = run.get("params", {})

        if "candidate_f1_score" not in metrics:
            continue

        current_f1 = metrics.get("current_f1_score")
        candidate_f1 = metrics.get("candidate_f1_score")
        f1_improvement = metrics.get("f1_improvement")

        current_accuracy = metrics.get(
            "current_accuracy"
        )
        candidate_accuracy = metrics.get(
            "candidate_accuracy"
        )
        accuracy_improvement = metrics.get(
            "accuracy_improvement"
        )

        if (
            candidate_f1 is not None
            and current_f1 is not None
            and candidate_f1 > current_f1
        ):
            decision = "candidate_outperformed_production"
        else:
            decision = "candidate_did_not_outperform_production"

        return {
            "run_id": run["run_id"],
            "trigger": params.get(
                "retraining_trigger",
                "not specified",
            ),
            "current_f1": current_f1,
            "candidate_f1": candidate_f1,
            "f1_improvement": f1_improvement,
            "current_accuracy": current_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "accuracy_improvement": accuracy_improvement,
            "decision": decision,
        }

    return None


def get_registered_model_info():
    client = get_mlflow_client()

    versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    model_versions = []

    for version in versions:
        model_versions.append(
            {
                "version": version.version,
                "run_id": version.run_id,
                "status": version.status,
                "aliases": list(version.aliases),
            }
        )

    return {
        "model_name": MODEL_NAME,
        "production_alias": PRODUCTION_ALIAS,
        "production_version": get_production_model_version(),
        "versions": model_versions,
    }


def build_copilot_context():
    recent_runs = get_experiment_runs()

    latest_retraining = get_latest_retraining_result(
        recent_runs
    )

    context = {
        "project": {
            "name": "NexusML",
            "description": (
                "Production ML Lifecycle & "
                "Model Reliability Platform"
            ),
        },

        "production": {
            "model": get_registered_model_info(),
        },

        "latest_retraining": latest_retraining,

        "experiment": {
            "name": EXPERIMENT_NAME,
            "recent_runs": recent_runs,
        },
    }

    return context


def get_context_json():
    context = build_copilot_context()

    return json.dumps(
        context,
        indent=2,
        default=str,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("NexusML — AI MLOps Copilot Context")
    print("=" * 70)

    print(get_context_json())

    print("=" * 70)