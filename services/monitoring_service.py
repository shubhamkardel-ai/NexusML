from sqlalchemy import func

from services.prediction_logger import PredictionLog, SessionLocal


def get_prediction_metrics() -> dict:
    """Calculate production prediction metrics."""

    db = SessionLocal()

    try:
        total_predictions = (
            db.query(func.count(PredictionLog.id)).scalar()
        )

        average_probability = (
            db.query(func.avg(PredictionLog.churn_probability)).scalar()
        )

        churn_predictions = (
            db.query(func.count(PredictionLog.id))
            .filter(PredictionLog.prediction == 1)
            .scalar()
        )

        non_churn_predictions = (
            db.query(func.count(PredictionLog.id))
            .filter(PredictionLog.prediction == 0)
            .scalar()
        )

        return {
            "total_predictions": total_predictions or 0,
            "churn_predictions": churn_predictions or 0,
            "non_churn_predictions": non_churn_predictions or 0,
            "average_churn_probability": round(
                float(average_probability or 0),
                4,
            ),
        }

    finally:
        db.close()


def get_model_metrics() -> dict:
    """Return production metrics grouped by model version."""

    db = SessionLocal()

    try:
        rows = (
            db.query(
                PredictionLog.model_name,
                PredictionLog.model_version,
                func.count(PredictionLog.id).label("prediction_count"),
                func.avg(
                    PredictionLog.churn_probability
                ).label("average_probability"),
            )
            .group_by(
                PredictionLog.model_name,
                PredictionLog.model_version,
            )
            .all()
        )

        models = []

        for row in rows:
            models.append(
                {
                    "model": row.model_name,
                    "model_version": row.model_version,
                    "prediction_count": row.prediction_count,
                    "average_churn_probability": round(
                        float(row.average_probability or 0),
                        4,
                    ),
                }
            )

        return {
            "models": models,
        }

    finally:
        db.close()


if __name__ == "__main__":
    print("=" * 60)
    print("NexusML — Production Monitoring")
    print("=" * 60)

    metrics = get_prediction_metrics()

    print("\nOverall Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    model_metrics = get_model_metrics()

    print("\nModel Metrics:")
    for model in model_metrics["models"]:
        print(model)

    print("=" * 60)