from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUM_CUSTOMERS = 2000


def generate_customer_churn_data() -> pd.DataFrame:
    """Generate a reproducible synthetic customer churn dataset."""

    rng = np.random.default_rng(RANDOM_SEED)

    data = pd.DataFrame(
        {
            "customer_id": [
                f"CUST_{i:05d}" for i in range(1, NUM_CUSTOMERS + 1)
            ],
            "age": rng.integers(18, 70, NUM_CUSTOMERS),
            "tenure_months": rng.integers(1, 72, NUM_CUSTOMERS),
            "monthly_charges": rng.uniform(20, 150, NUM_CUSTOMERS).round(2),
            "support_tickets": rng.integers(0, 10, NUM_CUSTOMERS),
            "usage_hours": rng.uniform(1, 100, NUM_CUSTOMERS).round(2),
            "contract_length": rng.choice(
                [1, 12, 24],
                NUM_CUSTOMERS,
                p=[0.45, 0.35, 0.20],
            ),
        }
    )

    # Create a realistic synthetic churn signal.
    churn_score = (
        0.035 * data["monthly_charges"]
        - 0.025 * data["tenure_months"]
        + 0.20 * data["support_tickets"]
        - 0.015 * data["usage_hours"]
        - 0.04 * data["contract_length"]
    )

    probability = 1 / (1 + np.exp(-churn_score / 5))

    data["churn"] = (
        rng.random(NUM_CUSTOMERS) < probability
    ).astype(int)

    return data


def main() -> None:
    output_path = Path("data/raw/customer_churn.csv")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_customer_churn_data()
    data.to_csv(output_path, index=False)

    print("=" * 60)
    print("NexusML — Customer Churn Dataset")
    print("=" * 60)
    print(f"Rows: {len(data)}")
    print(f"Columns: {len(data.columns)}")
    print(f"Output: {output_path}")
    print(f"Churn rate: {data['churn'].mean():.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()