import pandas as pd

from src.cleaning import clean_data


def test_clean_data_handles_normal_input():
    df = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Survived": [1, 0, 1],
            "Pclass": [1, 2, 3],
            "Sex": ["female", "male", "female"],
            "Age": [29.0, 30.0, 25.0],
            "Fare": [100.0, 50.0, 75.0],
            "Embarked": ["S", "C", "Q"],
        }
    )

    result = clean_data(df)

    assert list(result.columns) == list(df.columns)
    assert len(result) == 3
    assert result["Survived"].isin([0, 1]).all()


def test_clean_data_fills_missing_values_and_removes_duplicates():
    df = pd.DataFrame(
        {
            "PassengerId": [1, 1, 2],
            "Survived": [1, 1, 0],
            "Pclass": [1, 1, 2],
            "Sex": ["female", "female", "male"],
            "Age": [None, 29.0, None],
            "Fare": [100.0, 100.0, None],
            "Embarked": [None, "S", "C"],
        }
    )

    result = clean_data(df)

    assert len(result) == 2
    assert not result.duplicated().any()
    assert pd.notna(result["Age"]).all()
    assert pd.notna(result["Fare"]).all()
    assert pd.notna(result["Embarked"]).all()


def test_clean_data_removes_invalid_rows():
    df = pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Survived": [1, 0, 1],
            "Pclass": [1, 0, 3],
            "Sex": ["female", "male", "female"],
            "Age": [29.0, -5.0, 40.0],
            "Fare": [100.0, 200.0, 0.0],
            "Embarked": ["S", "C", "Q"],
        }
    )

    result = clean_data(df)

    assert len(result) == 2
    assert set(result["PassengerId"]) == {1, 3}


def test_clean_data_keeps_expected_structure():
    df = pd.DataFrame(
        {
            "PassengerId": [10],
            "Survived": [0],
            "Pclass": [3],
            "Sex": ["male"],
            "Age": [20],
            "Fare": [10.5],
            "Embarked": ["S"],
        }
    )

    result = clean_data(df)

    assert list(result.columns) == [
        "PassengerId",
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "Embarked",
    ]
