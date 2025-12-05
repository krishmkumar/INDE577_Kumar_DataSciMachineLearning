import numpy as np
from rice2025.utils.preprocess import normalize, train_test_split

def test_normalize_mean_std():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    Xn = normalize(X)
    assert np.allclose(np.mean(Xn, axis=0), 0, atol=1e-7)
    assert np.allclose(np.std(Xn, axis=0), 1, atol=1e-7)

def test_train_test_split_shapes():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    assert len(X_test) == 2
    assert len(X_train) == 8
