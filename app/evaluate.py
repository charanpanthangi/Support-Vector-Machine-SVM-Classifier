"""Evaluation metrics for classification models."""

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


Metrics = Dict[str, float | np.ndarray]


def evaluate_model(y_true, y_pred) -> Metrics:
    """Compute common classification metrics and confusion matrix."""

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


__all__ = ["evaluate_model", "Metrics"]
