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


if __name__ == "__main__":
    print("=" * 60)
    print("NexusML — Production Monitoring")
    print("=" * 60)

    metrics = get_prediction_metrics()

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("=" * 60)