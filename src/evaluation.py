from __future__ import annotations

from typing import Any

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true: Any, y_pred: Any, model_name: str) -> dict[str, float | str]:
    """Return a simple metrics dictionary for the classification task."""
    return {
        "model_name": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }
