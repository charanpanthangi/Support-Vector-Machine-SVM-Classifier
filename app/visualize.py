"""Visualization utilities for model results."""

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA


plt.style.use("seaborn-v0_8")


def plot_confusion_matrix(cm: np.ndarray, class_names: Iterable[str], output_path: str) -> None:
    """Plot and save a confusion matrix heatmap as SVG."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output, format="svg")
    plt.close()


def plot_pca_decision_regions(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    output_path: str,
    title: str = "PCA decision regions",
) -> None:
    """Project data to 2D with PCA and plot decision boundaries."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_original_space = pca.inverse_transform(grid)
    preds = model.predict(grid_original_space)
    preds = preds.reshape(xx.shape)

    cmap_light = ListedColormap(["#a6cee3", "#b2df8a", "#fb9a99"])
    cmap_bold = ListedColormap(["#1f78b4", "#33a02c", "#e31a1c"])

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, preds, cmap=cmap_light, alpha=0.6)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=60)
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.title(title)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.tight_layout()
    plt.savefig(output, format="svg")
    plt.close()


__all__ = ["plot_confusion_matrix", "plot_pca_decision_regions"]
