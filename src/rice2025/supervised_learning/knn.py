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
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([majority_label(self._get_neighbors(x)) for x in X])


class KNNRegressor(_KNNBase):
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([average_label(self._get_neighbors(x)) for x in X])
