"""Data loading utilities for the SVM classifier tutorial."""

from typing import Tuple

import pandas as pd
from sklearn.datasets import load_iris


def load_iris_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the iris dataset as pandas structures.

    Returns:
        Tuple containing the feature DataFrame ``X`` and target ``y`` as a Series.
    """

    iris = load_iris(as_frame=True)
    X: pd.DataFrame = iris.data
    y: pd.Series = iris.target
    return X, y


__all__ = ["load_iris_data"]
