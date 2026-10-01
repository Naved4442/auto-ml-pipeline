from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import joblib
from airflow.decorators import dag, task
from sqlalchemy import text

from src.cleaning import clean_data
from src.database import get_engine
from src.evaluation import compute_metrics
from src.training import train_and_evaluate_models
from src.validation import validate_data

EXPECTED_COLUMNS = [
    "PassengerId",
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked",
    "Survived",
]
TARGET_COLUMN = "Survived"


def get_project_root() -> Path:
    if Path("/opt/airflow").exists():
        return Path("/opt/airflow")
    return Path(__file__).resolve().parents[1]


@dag(
    dag_id="auto_ml_pipeline",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=3),
        "owner": "data-team",
    },
    tags=["ml", "automl", "starter"],
    description="Automated ML pipeline for Titanic-style classification data.",
)
def auto_ml_pipeline_dag():
    @task(retries=2, retry_delay=timedelta(minutes=3))
    def collect() -> str:
        dataset_url = os.getenv("DATASET_URL")
        if not dataset_url:
            raise ValueError("Missing required environment variable: DATASET_URL")

        project_root = get_project_root()
        raw_dir = project_root / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        csv_path = raw_dir / "titanic_raw.csv"

        # Download the latest raw dataset so the remaining pipeline can work from a local copy.
        with urlopen(dataset_url) as response, open(csv_path, "wb") as file:
            file.write(response.read())

        return str(csv_path)

    @task(retries=2, retry_delay=timedelta(minutes=3))
    def clean(file_path: str) -> str:
        df = pd.read_csv(file_path)
        cleaned_df = clean_data(df)
        cleaned_file = Path(file_path).with_name("titanic_cleaned.csv")
        cleaned_df.to_csv(cleaned_file, index=False)
        return str(cleaned_file)

    @task(retries=2, retry_delay=timedelta(minutes=3))
    def validate(file_path: str) -> str:
        df = pd.read_csv(file_path)
        validate_data(df, EXPECTED_COLUMNS, TARGET_COLUMN)
        return file_path

    @task(retries=2, retry_delay=timedelta(minutes=3))
    def load(file_path: str) -> None:
        df = pd.read_csv(file_path)
        engine = get_engine()
        with engine.begin() as connection:
            df.to_sql("cleaned_data", con=connection, if_exists="replace", index=False)

    @task(retries=2, retry_delay=timedelta(minutes=3))
    def train() -> dict[str, object]:
        engine = get_engine()
        with engine.connect() as connection:
            df = pd.read_sql_table("cleaned_data", con=connection)
        return train_and_evaluate_models(df, TARGET_COLUMN)

    @task(retries=2, retry_delay=timedelta(minutes=3))
    def evaluate(train_result: dict[str, object]) -> dict[str, dict[str, float | str]]:
        baseline_metrics = compute_metrics(
            train_result["y_test"],
            train_result["baseline_pipeline"].predict(train_result["X_test"]),
            "Baseline",
        )
        real_metrics = compute_metrics(
            train_result["y_test"],
            train_result["real_pipeline"].predict(train_result["X_test"]),
            "Logistic Regression",
        )
        return {"baseline": baseline_metrics, "real_model": real_metrics}

    @task(retries=2, retry_delay=timedelta(minutes=3))
    def save(train_result: dict[str, object], evaluation_result: dict[str, dict[str, float | str]]) -> str:
        project_root = get_project_root()
        model_dir = project_root / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "auto_ml_pipeline_model.joblib"
        joblib.dump(train_result["real_pipeline"], model_path)

        engine = get_engine()
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS model_runs (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        model_name VARCHAR(100) NOT NULL,
                        accuracy DOUBLE PRECISION,
                        precision DOUBLE PRECISION,
                        recall DOUBLE PRECISION,
                        f1 DOUBLE PRECISION
                    )
                    """
                )
            )

        rows = []
        for model_name, metrics in evaluation_result.items():
            rows.append(
                {
                    "timestamp": datetime.utcnow(),
                    "model_name": metrics["model_name"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                }
            )

        with engine.begin() as connection:
            pd.DataFrame(rows).to_sql("model_runs", con=connection, if_exists="append", index=False)
        return str(model_path)

    collect_task = collect()
    clean_task = clean(collect_task)
    validate_task = validate(clean_task)
    load_task = load(validate_task)
    train_task = train()
    evaluate_task = evaluate(train_task)
    save_task = save(train_task, evaluate_task)

    load_task >> train_task
    train_task >> evaluate_task >> save_task


auto_ml_pipeline_dag()
