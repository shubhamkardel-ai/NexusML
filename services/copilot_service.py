from services.copilot_context import build_copilot_context


def get_latest_retraining_result(context):
    runs = context["experiment"]["recent_runs"]

    for run in runs:
        metrics = run.get("metrics", {})
        params = run.get("params", {})

        if "candidate_f1_score" in metrics:
            return {
                "run_id": run["run_id"],
                "trigger": params.get("retraining_trigger"),
                "current_f1": metrics.get("current_f1_score"),
                "candidate_f1": metrics.get("candidate_f1_score"),
                "f1_improvement": metrics.get("f1_improvement"),
                "current_accuracy": metrics.get("current_accuracy"),
                "candidate_accuracy": metrics.get("candidate_accuracy"),
                "accuracy_improvement": metrics.get(
                    "accuracy_improvement"
                ),
            }

    return None


def answer_question(question: str):
    context = build_copilot_context()

    question_lower = question.lower()

    model_info = context["model"]
    latest_retraining = get_latest_retraining_result(context)

    if (
        "production model" in question_lower
        or "current model" in question_lower
        or "champion" in question_lower
    ):
        return (
            f"The current production model is "
            f"{model_info['model_name']} version "
            f"{model_info['production_version']}, "
            f"using the '{model_info['production_alias']}' alias."
        )

    if (
        "retrain" in question_lower
        or "retraining" in question_lower
    ):
        if latest_retraining is None:
            return "No retraining result was found in the MLflow history."

        return (
            f"The latest retraining was triggered by "
            f"{latest_retraining['trigger']}. "
            f"The candidate F1 score was "
            f"{latest_retraining['candidate_f1']}, compared with "
            f"{latest_retraining['current_f1']} for the production model. "
            f"The F1 improvement was "
            f"{latest_retraining['f1_improvement']}. "
            f"The candidate therefore did not improve production "
            f"performance."
        )

    if (
        "performance" in question_lower
        or "f1" in question_lower
        or "accuracy" in question_lower
    ):
        if latest_retraining is None:
            return "No recent candidate evaluation was found."

        return (
            f"Current production F1: "
            f"{latest_retraining['current_f1']}. "
            f"Candidate F1: "
            f"{latest_retraining['candidate_f1']}. "
            f"Current production accuracy: "
            f"{latest_retraining['current_accuracy']}. "
            f"Candidate accuracy: "
            f"{latest_retraining['candidate_accuracy']}."
        )

    if (
        "experiment" in question_lower
        or "mlflow" in question_lower
    ):
        runs = context["experiment"]["recent_runs"]

        return (
            f"The MLflow experiment "
            f"'{context['experiment']['name']}' contains "
            f"{len(runs)} recent runs available to the Copilot."
        )

    return (
        "I can currently answer questions about the production model, "
        "retraining decisions, model performance, and MLflow experiments."
    )


if __name__ == "__main__":
    print("=" * 70)
    print("NexusML — AI MLOps Copilot")
    print("=" * 70)

    questions = [
        "What is the current production model?",
        "Why did NexusML retrain the model?",
        "Did the latest candidate improve performance?",
        "How many MLflow experiment runs are available?",
    ]

    for question in questions:
        print(f"\nUser: {question}")
        print(f"Copilot: {answer_question(question)}")

    print("\n" + "=" * 70)