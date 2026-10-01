from __future__ import annotations

import pandas as pd


def validate_data(df: pd.DataFrame, expected_columns: list[str], target_column: str) -> bool:
    """Validate that the cleaned dataset is ready for training."""
    if df is None or df.empty:
        raise ValueError("Dataset is empty. The cleaned dataset must contain at least one row.")

    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        if target_column in missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}. Target column '{target_column}' is missing from the dataset."
            )
        raise ValueError(f"Missing required columns: {missing_columns}")

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the dataset.")

    if df[target_column].isnull().all():
        raise ValueError(f"Target column '{target_column}' contains no usable values.")

    unique_classes = df[target_column].dropna().unique()
    if len(unique_classes) < 2:
        raise ValueError(f"Target column '{target_column}' must contain at least 2 classes for classification.")

    if df.isnull().any().any():
        raise ValueError("Dataset still contains missing values after cleaning. Please inspect the cleaning step.")

    if "PassengerId" in df.columns and df["PassengerId"].isna().any():
        raise ValueError("PassengerId column contains invalid null values.")

    return True
