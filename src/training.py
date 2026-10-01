from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.evaluation import compute_metrics
from src.preprocessing import build_preprocessor


def train_and_evaluate_models(df: pd.DataFrame, target_column: str) -> dict[str, object]:
    """Train both the baseline and real model using a leakage-safe workflow."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)

    # Split first so the preprocessing steps never learn from the test data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Data leakage means a model sees information from the test set during fitting.
    # ColumnTransformer is useful because it applies different preprocessing rules to different feature types.
    baseline_preprocessor = build_preprocessor(X_train, target_column)
    real_preprocessor = build_preprocessor(X_train, target_column)

    baseline_pipeline = Pipeline(
        steps=[
            ("preprocessor", baseline_preprocessor),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )

    real_pipeline = Pipeline(
        steps=[
            ("preprocessor", real_preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    baseline_pipeline.fit(X_train, y_train)
    real_pipeline.fit(X_train, y_train)

    # We compare against the baseline to check whether the real model is doing useful work.
    baseline_metrics = compute_metrics(y_test, baseline_pipeline.predict(X_test), "Baseline")
    real_metrics = compute_metrics(y_test, real_pipeline.predict(X_test), "Logistic Regression")

    return {
        "X_test": X_test,
        "y_test": y_test,
        "baseline_pipeline": baseline_pipeline,
        "real_pipeline": real_pipeline,
        "baseline_metrics": baseline_metrics,
        "real_metrics": real_metrics,
    }
