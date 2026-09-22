from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker


DATABASE_URL = "sqlite:///./nexusml.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


class PredictionLog(Base):
    """Database record for a production model prediction."""

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)

    timestamp = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)

    age = Column(Integer, nullable=False)
    tenure_months = Column(Integer, nullable=False)
    monthly_charges = Column(Float, nullable=False)
    support_tickets = Column(Integer, nullable=False)
    usage_hours = Column(Float, nullable=False)
    contract_length = Column(Integer, nullable=False)

    prediction = Column(Integer, nullable=False)
    churn_probability = Column(Float, nullable=False)


def initialize_database() -> None:
    """Create database tables if they do not already exist."""

    Base.metadata.create_all(bind=engine)


def log_prediction(
    *,
    model_name: str,
    model_version: str,
    age: int,
    tenure_months: int,
    monthly_charges: float,
    support_tickets: int,
    usage_hours: float,
    contract_length: int,
    prediction: int,
    churn_probability: float,
) -> PredictionLog:
    """Store a model prediction in the database."""

    db = SessionLocal()

    try:
        record = PredictionLog(
            model_name=model_name,
            model_version=model_version,
            age=age,
            tenure_months=tenure_months,
            monthly_charges=monthly_charges,
            support_tickets=support_tickets,
            usage_hours=usage_hours,
            contract_length=contract_length,
            prediction=prediction,
            churn_probability=churn_probability,
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return record

    finally:
        db.close()


if __name__ == "__main__":
    initialize_database()

    print("=" * 60)
    print("NexusML — Prediction Logging Database")
    print("=" * 60)
    print("Database: nexusml.db")
    print("Table: prediction_logs")
    print("Status: Database initialized successfully")
    print("=" * 60)