import pandas as pd

import pytest

from src.validation import validate_data


VALID_COLUMNS = ["PassengerId", "Pclass", "Sex", "Age", "Fare", "Embarked", "Survived"]


def test_validate_data_accepts_valid_dataset():
    df = pd.DataFrame(
        {
            "PassengerId": [1, 2],
            "Pclass": [1, 2],
            "Sex": ["female", "male"],
            "Age": [29.0, 35.0],
            "Fare": [100.0, 40.0],
            "Embarked": ["S", "C"],
            "Survived": [1, 0],
        }
    )

    validate_data(df, VALID_COLUMNS, "Survived")


def test_validate_data_fails_for_missing_columns():
    df = pd.DataFrame({"PassengerId": [1], "Pclass": [1], "Sex": ["male"]})

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(df, VALID_COLUMNS, "Survived")


def test_validate_data_fails_for_empty_dataframe():
    df = pd.DataFrame(columns=VALID_COLUMNS)

    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_data(df, VALID_COLUMNS, "Survived")


def test_validate_data_fails_for_missing_target():
    df = pd.DataFrame({"PassengerId": [1], "Pclass": [1], "Sex": ["male"], "Age": [30], "Fare": [50], "Embarked": ["S"]})

    with pytest.raises(ValueError, match="Target column 'Survived' is missing"):
        validate_data(df, VALID_COLUMNS, "Survived")


def test_validate_data_fails_for_single_class_target():
    df = pd.DataFrame(
        {
            "PassengerId": [1, 2],
            "Pclass": [1, 2],
            "Sex": ["female", "male"],
            "Age": [29.0, 35.0],
            "Fare": [100.0, 40.0],
            "Embarked": ["S", "C"],
            "Survived": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="Target column 'Survived' must contain at least 2 classes"):
        validate_data(df, VALID_COLUMNS, "Survived")
