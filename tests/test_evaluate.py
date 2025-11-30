import numpy as np

from app.evaluate import evaluate_model


def test_evaluate_model_returns_metrics():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])

    metrics = evaluate_model(y_true, y_pred)

    expected_keys = {"accuracy", "precision", "recall", "f1", "confusion_matrix"}
    assert expected_keys.issubset(metrics.keys())
    assert metrics["confusion_matrix"].shape == (2, 2)
