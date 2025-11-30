"""Model creation utilities using scikit-learn's SVC."""

from typing import Literal

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

KernelType = Literal["linear", "poly", "rbf", "sigmoid"]


def build_svm_classifier(
    kernel: KernelType = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
    degree: int = 3,
    random_state: int | None = 42,
) -> Pipeline:
    """Create a pipeline that scales features then trains an SVM classifier.

    The pipeline first standardizes features because SVMs rely on distance
    calculations. Kernels map data into higher-dimensional spaces to find a
    maximum-margin hyperplane; the default RBF kernel bends the space so that
    non-linear boundaries can be learned.
    """

    svm = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        probability=True,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", svm),
        ]
    )
    return pipeline


__all__ = ["build_svm_classifier"]
