import numpy as np
from rice2025.supervised_learning.perceptron import Perceptron


def test_perceptron_fit_and_predict():
    # Simple AND dataset
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0, 0, 0, 1])

    model = Perceptron(learning_rate=0.1, n_iterations=20).fit(X, y)
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert model.weights is not None
    assert model.bias is not None


def test_perceptron_accuracy_improves():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0, 0, 0, 1])

    model = Perceptron(learning_rate=0.1, n_iterations=20).fit(X, y)

    acc = model.accuracy(X, y)
    assert acc >= 0.75  # perceptron should learn AND logic


def test_perceptron_loss_decreases():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0, 0, 0, 1])

    model = Perceptron(learning_rate=0.1, n_iterations=20).fit(X, y)

    assert len(model.loss_) >= 2
    assert model.loss_[0] >= model.loss_[-1]   # training should reduce loss
