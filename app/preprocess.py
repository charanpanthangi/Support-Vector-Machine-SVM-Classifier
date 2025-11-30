"""Preprocessing helpers for splitting and scaling data.

Scaling is critical for SVMs because the algorithm measures distances
between points. Features with large ranges can dominate the distance
calculation, so we standardize every feature before training.
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into train and test sets.

    Args:
        X: Feature DataFrame.
        y: Target labels.
        test_size: Proportion reserved for testing.
        random_state: Seed for reproducibility.

    Returns:
        Train/test split of features and labels.
    """

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def get_scaler() -> StandardScaler:
    """Create a StandardScaler for zero-mean, unit-variance scaling."""

    return StandardScaler()


__all__ = ["split_data", "get_scaler"]
