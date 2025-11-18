"""
Perceptron 

A simple binary linear classifier using the classic perceptron update rule.
Supports:
- fit(X, y)
- predict(X)
- score(X, y)
- accuracy(X, y)

"""

import numpy as np


class Perceptron:
    """
    Perceptron classifier (binary).

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for weight updates.
    n_iterations : int, default=1000
        Number of passes through the training data.

    Notes
    -----
    - Labels must be 0/1. If your dataset uses -1/+1, convert before fitting.
    - Uses mean squared misclassification cost internally (for monitoring).
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = float(learning_rate)
        self.n_iterations = int(n_iterations)
        self.weights: np.ndarray | None = None
        self.bias: float | None = None
        self.loss_: list[float] = []

    # ------------------------- Helper -------------------------

    def _loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean-squared error between predicted (0/1) and true labels."""
        return float(np.mean((y_true - y_pred) ** 2))

    # ------------------------- API ----------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Train the perceptron using the perceptron update rule.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,), must be 0 or 1
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            for xi, yi in zip(X, y):
                linear = float(np.dot(xi, self.weights) + self.bias)
                y_pred = 1 if linear >= 0 else 0
                update = self.learning_rate * (yi - y_pred)
                self.weights += update * xi
                self.bias += update

            # Compute full-batch loss each epoch
            y_full_pred = self.predict(X)
            self.loss_.append(self._loss(y, y_full_pred))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions (0 or 1)."""
        X = np.asarray(X, dtype=float)
        linear = X @ self.weights + self.bias  # type: ignore[operator]
        return (linear >= 0).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy on (X, y)."""
        y = np.asarray(y, dtype=int)
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return accuracy of predictions (alias for score)."""
        return self.score(X, y)
