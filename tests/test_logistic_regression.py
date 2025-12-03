import numpy as np
import pytest
from rice2025.supervised_learning.logistic_regression import LogisticRegression


def separable_data():
    """
    Simple linearly separable binary dataset.
    y = 1 if x > 0.5 else 0
    """
    X = np.array([[0.1], [0.2], [0.3], [0.7], [0.8], [0.9]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


# -------------------------------------------------------
# Basic tests
# -------------------------------------------------------

def test_fit_separable_data():
    X, y = separable_data()
    model = LogisticRegression(learning_rate=1.0, max_iter=5000)
    model.fit(X, y)

    preds = model.predict(X)
    assert np.all(preds == y)  # perfect linear separation


def test_predict_proba_validity():
    X, y = separable_data()
    model = LogisticRegression()
    model.fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)

    # probabilities must sum to 1
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_predict_thresholding():
    X = np.array([[0.0], [1.0]])
    y = np.array([0, 1])

    model = LogisticRegression(max_iter=3000, learning_rate=1.0)
    model.fit(X, y)

    preds = model.predict(X, threshold=0.5)
    assert np.array_equal(preds, y)


def test_score_accuracy():
    X, y = separable_data()
    model = LogisticRegression()
    model.fit(X, y)
    assert model.score(X, y) == 1.0


# -------------------------------------------------------
# Regularization tests
# -------------------------------------------------------

def test_regularization_shrinks_coefficients():
    X, y = separable_data()

    no_reg = LogisticRegression(C=1e9)  # nearly no penalty
    strong_reg = LogisticRegression(C=0.1)  # strong penalty

    no_reg.fit(X, y)
    strong_reg.fit(X, y)

    # strong regularization should shrink weights
    assert abs(strong_reg.coef_[0]) < abs(no_reg.coef_[0])


# -------------------------------------------------------
# Intercept tests
# -------------------------------------------------------

def test_intercept_learned():
    X, y = separable_data()
    model = LogisticRegression(fit_intercept=True)
    model.fit(X, y)
    assert model.intercept_ is not None


def test_no_intercept_behavior():
    X, y = separable_data()
    model = LogisticRegression(fit_intercept=False)
    model.fit(X, y)

    # intercept should be fixed at 0
    assert model.intercept_ == 0.0


# -------------------------------------------------------
# Decision function tests
# -------------------------------------------------------

def test_decision_function_relationship():
    X, y = separable_data()
    model = LogisticRegression()
    model.fit(X, y)

    z = model.decision_function(X)
    proba = model.predict_proba(X)[:, 1]

    # sigmoid relation
    assert np.allclose(proba, 1 / (1 + np.exp(-z)), atol=1e-6)


# -------------------------------------------------------
# Error handling tests
# -------------------------------------------------------

def test_invalid_labels_raises_error():
    X = np.array([[0.1], [0.2]])
    y = np.array([0, 2])  # invalid label

    model = LogisticRegression()
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_mismatched_shapes_error():
    X = np.array([[1.0], [2.0]])
    y = np.array([0])  # invalid shape

    model = LogisticRegression()
    with pytest.raises(ValueError):
        model.fit(X, y)


# -------------------------------------------------------
# Convergence / stability
# -------------------------------------------------------

def test_convergence_behavior():
    X, y = separable_data()
    model = LogisticRegression(learning_rate=0.5, max_iter=10000)
    model.fit(X, y)

    # After training on separable ydata:
    # predictions should all be correct
    preds = model.predict(X)
    assert np.all(preds == y)
