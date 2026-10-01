from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(X: pd.DataFrame, target_column: str) -> ColumnTransformer:
    """Build a leakage-safe preprocessor that handles numeric and categorical columns."""
    numeric_features = [
        column for column in X.select_dtypes(include=["number"]).columns if column != target_column
    ]
    categorical_features = [
        column for column in X.columns if column not in numeric_features and column != target_column
    ]

    transformers: list[tuple[str, Pipeline, Sequence[str]]] = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No valid feature columns were found for preprocessing.")

    return ColumnTransformer(transformers=transformers)
