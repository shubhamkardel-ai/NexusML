from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

from services.prediction_logger import PredictionLog, SessionLocal

MIN_PRODUCTION_SAMPLES = 30

REFERENCE_PATH = Path("data/processed/reference_data.csv")

FEATURE_COLUMNS = [
    "age",
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "usage_hours",
    "contract_length",
]


def create_reference_dataset() -> pd.DataFrame:
    """Create the reference dataset used for production drift detection."""

    source_path = Path("data/raw/customer_churn.csv")

    if not source_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {source_path}"
        )

    df = pd.read_csv(source_path)

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required feature columns: {missing_columns}"
        )

    reference_df = df[FEATURE_COLUMNS].copy()

    REFERENCE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_df.to_csv(
        REFERENCE_PATH,
        index=False,
    )

    return reference_df


def calculate_feature_drift(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    p_value_threshold: float = 0.05,
) -> dict:
    """
    Compare reference and production feature distributions.

    Drift is detected when the KS test p-value is below
    the configured significance threshold.
    """

    results = {}

    for feature in FEATURE_COLUMNS:
        reference_values = reference_df[feature].dropna()
        production_values = production_df[feature].dropna()

        reference_mean = float(reference_values.mean())
        production_mean = float(production_values.mean())

        if reference_mean == 0:
            percentage_change = 0.0
        else:
            percentage_change = (
                abs(production_mean - reference_mean)
                / abs(reference_mean)
            ) * 100

        ks_statistic, p_value = ks_2samp(
            reference_values,
            production_values,
        )

    results[feature] = {
        "reference_mean": float(round(reference_mean, 4)),
        "production_mean": float(round(production_mean, 4)),
        "percentage_change": float(round(percentage_change, 2)),
        "ks_statistic": float(round(float(ks_statistic), 4)),
        "p_value": float(round(float(p_value), 4)),
        "drift_detected": bool(p_value < p_value_threshold),
    }

    return results

def get_production_data() -> pd.DataFrame:
    """Retrieve production feature data from prediction logs."""

    db = SessionLocal()

    try:
        rows = (
            db.query(
                PredictionLog.age,
                PredictionLog.tenure_months,
                PredictionLog.monthly_charges,
                PredictionLog.support_tickets,
                PredictionLog.usage_hours,
                PredictionLog.contract_length,
            )
            .all()
        )

        if not rows:
            return pd.DataFrame(columns=FEATURE_COLUMNS)

        production_df = pd.DataFrame(
            rows,
            columns=FEATURE_COLUMNS,
        )

        return production_df

    finally:
        db.close()


def detect_drift(
    production_df: pd.DataFrame,
    threshold: float = 20.0,
) -> dict:
    """
    Detect feature drift against the reference dataset.

    A feature is considered drifted when its mean changes by more
    than the configured percentage threshold.
    """

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"Reference dataset not found: {REFERENCE_PATH}"
        )

    reference_df = pd.read_csv(REFERENCE_PATH)

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in production_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Production data is missing features: {missing_columns}"
        )

    if len(production_df) < MIN_PRODUCTION_SAMPLES:
        return {
            "drift_detected": False,
            "status": "insufficient_data",
            "production_samples": len(production_df),
            "minimum_required_samples": MIN_PRODUCTION_SAMPLES,
            "drift_threshold_percent": threshold,
            "p_value_threshold": 0.05,
            "drifted_features": [],
            "feature_results": {},
        }

    feature_results = calculate_feature_drift(
        reference_df,
        production_df,
    )

    drifted_features = [
        feature
        for feature, result in feature_results.items()
        if result["drift_detected"]
    ]

    return {
        "drift_detected": len(drifted_features) > 0,
        "status": (
            "drift_detected"
            if drifted_features
            else "no_drift"
        ),
        "production_samples": len(production_df),
        "minimum_required_samples": MIN_PRODUCTION_SAMPLES,
        "drift_threshold_percent": threshold,
        "drifted_features": drifted_features,
        "feature_results": feature_results,
    }

if __name__ == "__main__":
    print("=" * 60)
    print("NexusML — Feature Drift Detection")
    print("=" * 60)

    reference_df = create_reference_dataset()

    print(f"Reference rows: {len(reference_df)}")
    print(f"Reference columns: {len(reference_df.columns)}")

    production_df = get_production_data()

    print(f"Production rows: {len(production_df)}")

    if production_df.empty:
        print("\nNo production prediction data available.")
        print("Generate predictions through the API first.")
        print("=" * 60)
        raise SystemExit

    result = detect_drift(production_df)

    print(f"Status: {result['status']}")
    print(
        f"Production samples: "
        f"{result['production_samples']}"
    )

    if result["status"] == "insufficient_data":
        print(
            "\nNot enough production data for reliable "
            "drift detection."
        )
        print(
            f"Minimum required: "
            f"{result['minimum_required_samples']}"
        )
        print("=" * 60)
        raise SystemExit

    print("\nDrift Status:")
    print(f"Drift detected: {result['drift_detected']}")
    print(
        f"Threshold: "
        f"{result['drift_threshold_percent']}%"
    )

    print("\nFeature Analysis:")

    for feature, metrics in result["feature_results"].items():
        print(
            f"{feature}: "
            f"reference={metrics['reference_mean']}, "
            f"production={metrics['production_mean']}, "
            f"change={metrics['percentage_change']}%, "
            f"KS={metrics['ks_statistic']}, "
            f"p-value={metrics['p_value']}, "
            f"drift={metrics['drift_detected']}"
        )

    print("\nDrifted Features:")

    if result["drifted_features"]:
        for feature in result["drifted_features"]:
            print(f"- {feature}")
    else:
        print("- None")

    print("=" * 60)