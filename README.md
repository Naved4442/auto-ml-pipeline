# auto-ml-pipeline

A beginner-friendly, interview-ready automated machine learning pipeline using Apache Airflow, PostgreSQL, pandas, and scikit-learn.

The project replaces a manual workflow with a reproducible DAG:

`collect -> clean -> validate -> load -> train -> evaluate -> save`

## What It Does

- Downloads a Titanic classification dataset.
- Cleans names, types, missing values, invalid rows, and duplicates.
- Validates the dataset before training.
- Loads cleaned data into PostgreSQL.
- Splits data before preprocessing to avoid test-set leakage.
- Compares a `DummyClassifier` baseline with Logistic Regression.
- Stores model metrics in PostgreSQL.
- Saves the trained scikit-learn pipeline as a joblib artifact.

## Project Structure

```text
.
├── dags/                  # Airflow DAG
├── docker/                # Airflow image definition
├── src/                   # Cleaning, validation, preprocessing, training, evaluation
├── tests/                 # Unit tests for cleaning and validation
├── docker-compose.yml     # PostgreSQL + Airflow services
├── requirements.txt       # Pinned Python dependencies
├── .env.example           # Configuration template
└── README.md
```

Generated data, models, Airflow logs, local configuration, and Python caches are ignored by Git.

## Architecture And Data Flow

1. Airflow runs the `auto_ml_pipeline` DAG on a daily schedule.
2. `collect` downloads the raw CSV into `data/raw/`.
3. `clean` produces a normalized, null-free dataset.
4. `validate` checks required columns, target classes, and remaining nulls.
5. `load` writes `cleaned_data` to PostgreSQL.
6. `train` reads the table, splits the data, fits preprocessing and models, and returns the test set.
7. `evaluate` calculates accuracy, precision, recall, and F1.
8. `save` writes `model_runs` and saves `models/auto_ml_pipeline_model.joblib`.

## Run With Docker Compose

Prerequisites: Docker Engine, Docker Compose, and Git.

1. Create local configuration:

   ```bash
   cp .env.example .env
   ```

2. Replace the placeholder values in `.env`. The PostgreSQL password in `DATABASE_URL` and `AIRFLOW__DATABASE__SQL_ALCHEMY_CONN` must match `POSTGRES_PASSWORD`.

3. Build and start the services:

   ```bash
   docker compose up -d --build
   ```

4. Open Airflow at <http://localhost:8080>.

   The default local username is `admin`; use the value configured in `AIRFLOW_ADMIN_PASSWORD`.

5. Trigger the DAG from the Airflow UI, or use the CLI:

   ```bash
   docker compose exec airflow-scheduler airflow dags unpause auto_ml_pipeline
   docker compose exec airflow-scheduler airflow dags trigger auto_ml_pipeline
   ```

6. Inspect service status and logs:

   ```bash
   docker compose ps
   docker compose logs -f airflow-scheduler
   ```

PostgreSQL is available to the host on port `5433` and to Docker services at `postgres:5432`.

## Run Tests Locally

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Verified Results

The pipeline was run with the public Titanic CSV, a fixed `random_state=42`, a 20% stratified test split, and the Docker PostgreSQL service. The cleaned dataset contained 891 rows and PostgreSQL contained 891 rows in `cleaned_data`.

| Model | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Dummy baseline | 0.6145 | 0.0000 | 0.0000 | 0.0000 |
| Logistic Regression | 0.8268 | 0.8065 | 0.7246 | 0.7634 |

These values are measured outputs from the current code, not target values or estimates.

## Interview Walkthrough

- **Architecture:** Airflow orchestrates small Python tasks; PostgreSQL stores cleaned data and run metrics; joblib stores the final pipeline.
- **Data quality:** Cleaning happens before validation, and invalid data stops downstream tasks.
- **Leakage prevention:** The train/test split happens before the fitted preprocessing pipeline.
- **Baseline comparison:** The dummy model shows whether Logistic Regression adds useful signal.
- **Reproducibility:** Dependency pins, fixed random state, environment configuration, and Docker Compose make the run repeatable.
- **Testing:** Unit tests cover cleaning behavior and validation failures; Docker verifies service integration.

## Stop The Stack

```bash
docker compose down
```

Use `docker compose down -v` only when you intentionally want to remove the PostgreSQL volume and its stored tables.
