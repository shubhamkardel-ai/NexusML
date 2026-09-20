from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/customer_churn.csv")

REQUIRED_COLUMNS = {
    "customer_id",
    "age",
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "usage_hours",
    "contract_length",
    "churn",
}


def validate_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load and validate the customer churn dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("Dataset is empty.")

    if data["customer_id"].duplicated().any():
        raise ValueError("Duplicate customer IDs detected.")

    if data["churn"].isna().any():
        raise ValueError("Missing values found in churn target.")

    if not data["churn"].isin([0, 1]).all():
        raise ValueError("Churn target must contain only 0 or 1.")

    numeric_columns = [
        "age",
        "tenure_months",
        "monthly_charges",
        "support_tickets",
        "usage_hours",
        "contract_length",
    ]

    if data[numeric_columns].isna().any().any():
        raise ValueError("Missing values found in numeric features.")

    return data


def main() -> None:
    data = validate_dataset()

    print("=" * 60)
    print("NexusML — Data Validation")
    print("=" * 60)
    print("Status: PASSED")
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Target distribution:")
    print(data["churn"].value_counts().sort_index())
    print("=" * 60)


if __name__ == "__main__":
    main()