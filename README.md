# Support Vector Machine (SVM) Classifier

Beginner-friendly template and tutorial for building a Support Vector Machine classifier using scikit-learn and the classic iris dataset.

## What is an SVM?
- An SVM finds the **maximum-margin hyperplane** that separates classes with the widest possible gap.
- Points that touch the margin are the **support vectors**; they define the boundary.
- The **kernel trick** lets us operate in a high-dimensional feature space without explicitly computing the coordinates. Kernels such as RBF (default), linear, or polynomial let SVMs learn flexible decision boundaries.
- Feature scaling is critical because SVMs rely on distance calculations—always standardize inputs before training.
- SVMs shine in high-dimensional spaces and when you need strong margins between classes.

## Dataset
We use scikit-learn's built-in **iris** dataset (150 samples, 4 features, 3 classes). It is small, clean, and perfect for trying out SVMs.

## Project structure
```
app/
  data.py          # Load the iris dataset
  preprocess.py    # Train/test split and scaling helpers
  model.py         # Build an SVM pipeline (RBF kernel by default)
  evaluate.py      # Accuracy, precision, recall, F1, confusion matrix
  visualize.py     # Confusion matrix heatmap and PCA decision regions
  main.py          # End-to-end workflow
notebooks/
  demo_svm_classification.ipynb
examples/
  README_examples.md
```

## How it works (workflow)
1. **Load** the iris dataset.
2. **Split** into train and test sets.
3. **Scale** features with `StandardScaler`.
4. **Train** an SVM classifier (RBF kernel, `C=1.0`, `gamma='scale'`).
5. **Evaluate** with accuracy, precision, recall, F1, and confusion matrix.
6. **Visualize** the confusion matrix and PCA-based decision regions (saved as SVG).

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/main.py
```

To explore interactively, launch Jupyter:
```bash
jupyter notebook notebooks/demo_svm_classification.ipynb
```

## Running tests
```bash
pytest
```

## Docker
Build and run inside a lightweight container:
```bash
docker build -t svm-classifier .
docker run --rm svm-classifier
```

## Future improvements
- Hyperparameter tuning with grid search or randomized search.
- Experiment with the polynomial kernel or adjust `C` and `gamma` for different margins.
- Compare against `LinearSVC` for high-dimensional sparse features.

## License
MIT License. See [LICENSE](LICENSE) for details.
