import numpy as np
import pytest
from rice2025.supervised_learning.decision_tree import DecisionTree


def test_decision_tree_basic_fit_and_predict():
    X = np.array([
        [0], [1], [2], [8], [9]
    ], dtype=float)
    y = np.array([0, 0, 0, 1, 1])

    model = DecisionTree().fit(X, y)
    preds = model.predict(X)

    assert np.array_equal(preds, y), "Tree should perfectly classify separable data"


def test_decision_tree_predict_before_fit_raises():
    model = DecisionTree()

    with pytest.raises(AttributeError):
        model.predict(np.array([[1]]))


def test_decision_tree_empty_X_or_y():
    model = DecisionTree()

    with pytest.raises(ValueError):
        model.fit(np.array([]), np.array([]))

    with pytest.raises(ValueError):
        model.fit(np.array([[1]]), np.array([]))


def test_decision_tree_mismatched_X_y_lengths():
    model = DecisionTree()

    X = np.array([[1], [2]])
    y = np.array([0])  # wrong length

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_decision_tree_max_depth_effect():
    """
    Depth 0 tree should simply predict the majority class.
    """
    X = np.array([[0], [1], [2], [10]])
    y = np.array([0, 0, 0, 1])  # majority = 0

    model = DecisionTree(max_depth=0).fit(X, y)
    preds = model.predict(X)

    assert np.all(preds == 0), "With max_depth=0, tree should predict majority class"


def test_decision_tree_single_feature_split():
    """
    Ensure the tree splits correctly on a feature threshold.
    """
    X = np.array([[0], [1], [5], [6]], dtype=float)
    y = np.array([0, 0, 1, 1])

    model = DecisionTree().fit(X, y)
    preds = model.predict(X)

    assert np.array_equal(preds, y), "Tree must separate two clean clusters"


def test_decision_tree_boolean_consistency():
    """
    Ensure repeatability: model.predict after fit is deterministic.
    """
    X = np.array([[0], [1], [5], [6]], dtype=float)
    y = np.array([0, 0, 1, 1])

    model = DecisionTree().fit(X, y)
    preds1 = model.predict(X)
    preds2 = model.predict(X)

    assert np.array_equal(preds1, preds2), "Predictions must be deterministic"
