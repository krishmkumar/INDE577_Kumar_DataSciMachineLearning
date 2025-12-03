import numpy as np
import pytest
from rice2025.supervised_learning.regression_tree import RegressionTree


def simple_train_test_split(X, y, test_ratio=0.3, seed=42):
    """
    Minimal train-test splitter to avoid sklearn.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    test_size = int(len(X) * test_ratio)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def test_fit_runs():
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])

    model = RegressionTree(max_depth=2)
    model.fit(X, y)

    assert model.tree is not None, "Tree should be built after fit()"


def test_predict_shape():
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])

    model = RegressionTree(max_depth=2)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape, "predict() output shape should match y"


def test_predict_valid_classes():
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1, 0])

    model = RegressionTree()
    model.fit(X, y)
    preds = model.predict(X)

    for p in preds:
        assert p in model.classes_, "Predictions must belong to the learned classes"


def test_predict_proba_sums_to_one():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 1, 0, 1])

    model = RegressionTree()
    model.fit(X, y)
    probs = model.predict_proba(X)

    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Each row of predict_proba must sum to 1"


def test_simple_learning_separable_data():
    """
    Perfectly separable dataset:
    x < 0 → class 0
    x > 0 → class 1
    """
    X = np.array([[-3], [-1], [-2], [2], [3], [4]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = RegressionTree(max_depth=2)
    model.fit(X, y)
    preds = model.predict(X)

    assert np.array_equal(preds, y), "Tree should perfectly learn separable dataset"


def test_random_dataset():
    """
    Smoke test: run on random data with random labels.
    Ensures nothing crashes end-to-end.
    """
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = np.random.randint(0, 3, size=50)

    X_train, X_test, y_train, y_test = simple_train_test_split(X, y)

    model = RegressionTree(max_depth=4)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert len(preds) == len(y_test)
    assert all(p in model.classes_ for p in preds)
