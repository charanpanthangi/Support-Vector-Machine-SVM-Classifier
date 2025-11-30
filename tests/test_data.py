import pandas as pd

from app.data import load_iris_data


def test_load_iris_data_shapes():
    X, y = load_iris_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == len(y)
    # Iris has 150 samples and 4 features
    assert X.shape == (150, 4)
    assert y.nunique() == 3
