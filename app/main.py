"""Main execution script for training and evaluating an SVM classifier."""

from pathlib import Path

from app.data import load_iris_data
from app.evaluate import evaluate_model
from app.model import build_svm_classifier
from app.preprocess import split_data
from app.visualize import plot_confusion_matrix, plot_pca_decision_regions


OUTPUT_DIR = Path("artifacts")


def run() -> None:
    """Run the full pipeline: load, split, train, evaluate, and visualize."""

    X, y = load_iris_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_svm_classifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)

    print("SVM classification metrics:")
    for name in ["accuracy", "precision", "recall", "f1"]:
        print(f"{name.title()}: {metrics[name]:.3f}")

    # Visualizations
    OUTPUT_DIR.mkdir(exist_ok=True)
    cm_path = OUTPUT_DIR / "confusion_matrix.svg"
    decision_path = OUTPUT_DIR / "pca_decision_regions.svg"

    plot_confusion_matrix(metrics["confusion_matrix"], class_names=y.unique(), output_path=str(cm_path))
    plot_pca_decision_regions(X, y, model, output_path=str(decision_path))

    print(f"Saved confusion matrix to {cm_path}")
    print(f"Saved PCA decision regions to {decision_path}")


if __name__ == "__main__":
    run()
