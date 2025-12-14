"""
k-Nearest Neighbors (kNN) Models (from scratch).

This module implements k-Nearest Neighbors (kNN) for both classification and
regression. Predictions are made by identifying the k closest training samples
using Euclidean distance and aggregating their labels via majority vote
(classification) or averaging (regression).

Implemented classes:
- KNNClassifier: kNN for classification using majority voting
- KNNRegressor: kNN for regression using mean aggregation

Key characteristics:
- Distance-based, non-parametric learning
- Supports both `k=` and `n_neighbors=` initialization styles
- Uses Euclidean distance as the similarity metric
- Requires no explicit training phase beyond storing data
- Designed for clarity and instructional use rather than performance

This implementation avoids external machine learning libraries and relies only
on NumPy and simple utility functions, making it suitable for educational use
in the INDE 577 course context.
"""

import numpy as np
from rice2025.supervised_learning.distances import euclidean_distance
from rice2025.utils.postprocess import majority_label, average_label


class _KNNBase:
    def __init__(self, n_neighbors=3):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        return self

    def _get_neighbors(self, x):
        distances = np.array([
            euclidean_distance(x, x_train) for x_train in self.X_train
        ])
        idx = np.argsort(distances)[: self.n_neighbors]
        return self.y_train[idx]


class KNNClassifier(_KNNBase):
    """
    KNN Classifier supporting both k= and n_neighbors=
    """

    def __init__(self, k=3, n_neighbors=None):
        # Allow both styles: k= or n_neighbors=
        if n_neighbors is not None:
            super().__init__(n_neighbors=n_neighbors)
        else:
            super().__init__(n_neighbors=k)

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([
            majority_label(self._get_neighbors(x)) for x in X
        ])


class KNNRegressor(_KNNBase):
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([
            average_label(self._get_neighbors(x)) for x in X
        ])
