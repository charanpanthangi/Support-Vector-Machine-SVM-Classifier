import numpy as np

from app.data import load_iris_data
from app.model import build_svm_classifier
from app.preprocess import split_data


def test_model_fit_and_predict():
    X, y = load_iris_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25, random_state=0)

    model = build_svm_classifier()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert preds.shape == y_test.shape
    assert set(np.unique(preds)).issubset(set(y.unique()))
