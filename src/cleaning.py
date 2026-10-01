from __future__ import annotations

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw Titanic-style dataset before validation and model training."""
    if df is None:
        raise ValueError("Input dataframe is missing.")

    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    # Convert obvious numeric columns to numbers so the downstream checks are consistent.
    for column in ["PassengerId", "Survived", "Pclass", "Age", "Fare"]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    # Normalise text values so the same category names are treated consistently.
    for column in ["Sex", "Embarked"]:
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype("string").str.strip().str.lower()

    # Keep only rows that still look physically valid for a passenger record.
    if "Pclass" in cleaned.columns:
        cleaned = cleaned[cleaned["Pclass"].between(1, 3, inclusive="both")].copy()
    if "Age" in cleaned.columns:
        cleaned = cleaned[cleaned["Age"].fillna(0) >= 0].copy()
    if "Fare" in cleaned.columns:
        cleaned = cleaned[cleaned["Fare"].fillna(0) >= 0].copy()

    # Fill missing values with sensible defaults instead of dropping large parts of the dataset.
    if "Age" in cleaned.columns:
        cleaned["Age"] = cleaned["Age"].fillna(cleaned["Age"].median())
    if "Fare" in cleaned.columns:
        cleaned["Fare"] = cleaned["Fare"].fillna(cleaned["Fare"].median())
    if "Embarked" in cleaned.columns:
        mode_value = cleaned["Embarked"].mode(dropna=True)
        cleaned["Embarked"] = cleaned["Embarked"].fillna(mode_value.iloc[0] if not mode_value.empty else "s")
    if "Sex" in cleaned.columns:
        mode_value = cleaned["Sex"].mode(dropna=True)
        cleaned["Sex"] = cleaned["Sex"].fillna(mode_value.iloc[0] if not mode_value.empty else "male")

    if "Survived" in cleaned.columns:
        cleaned["Survived"] = pd.to_numeric(cleaned["Survived"], errors="coerce")
        cleaned = cleaned.dropna(subset=["Survived"]).copy()
        cleaned["Survived"] = cleaned["Survived"].astype(int)

    for column in cleaned.select_dtypes(include=["number"]).columns:
        cleaned[column] = cleaned[column].fillna(cleaned[column].median())
    for column in cleaned.select_dtypes(exclude=["number"]).columns:
        mode_value = cleaned[column].mode(dropna=True)
        cleaned[column] = cleaned[column].fillna(mode_value.iloc[0] if not mode_value.empty else "unknown")

    # Remove duplicate passenger records after the values are filled so the cleaner record wins.
    if "PassengerId" in cleaned.columns:
        cleaned = cleaned.drop_duplicates(subset=["PassengerId"], keep="first").reset_index(drop=True)
    else:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned.reset_index(drop=True)
