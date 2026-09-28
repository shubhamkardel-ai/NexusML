import random

import requests


API_URL = "http://127.0.0.1:8000/predict"


def generate_prediction_data(count: int = 30) -> None:
    """Generate normal production prediction requests."""

    random.seed(42)

    successful = 0

    for i in range(count):
        payload = {
            "age": random.randint(25, 65),
            "tenure_months": random.randint(6, 60),
            "monthly_charges": round(
                random.uniform(50, 120),
                2,
            ),
            "support_tickets": random.randint(1, 8),
            "usage_hours": round(
                random.uniform(25, 80),
                2,
            ),
            "contract_length": random.choice(
                [6, 12, 24]
            ),
        }

        response = requests.post(
            API_URL,
            json=payload,
            timeout=10,
        )

        if response.status_code == 200:
            successful += 1
        else:
            print(
                f"Request {i + 1} failed: "
                f"{response.status_code}"
            )

    print("=" * 60)
    print("NexusML — Production Data Generator")
    print("=" * 60)
    print(f"Requests sent: {count}")
    print(f"Successful predictions: {successful}")
    print("Status: Production data generated successfully")
    print("=" * 60)


if __name__ == "__main__":
    generate_prediction_data()