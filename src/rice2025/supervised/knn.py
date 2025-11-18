import numpy as np


class _KNNBase:
    """
    Base class for K-Nearest Neighbors models.
    Provides shared fit(), distance, and kneighbors() methods.

    Parameters
    ----------
    k : int, default=3
        Number of nearest neighbors.
    metric : {"euclidean", "manhattan"}, default="euclidean"
        Distance metric to use.
    """

    def __init__(self, k=3, metric="euclidean"):
        if k < 1:
            raise ValueError("k must be >= 1.")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'.")

        self.k = k
        self.metric = metric
        self.X_train = None
        self.y_train = None

    # -------------------------
    # Distance function
    # -------------------------
    def _distance(self, a, b):
        if self.metric == "euclidean":
            return np.sqrt(np.sum((a - b) ** 2))
        else:  # manhattan
            return np.sum(np.abs(a - b))

    # -------------------------
    # Fit
    # -------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        if y.ndim != 1:
            raise ValueError("y must be 1D.")
        if len(X) != len(y):
            raise ValueError("X and y must have the same length.")

        self.X_train = X
        self.y_train = y
        return self

    # -------------------------
    # Nearest neighbors
    # -------------------------
    def kneighbors(self, X):
        """
        Return distances + indices for the k nearest neighbors.

        Parameters
        ----------
        X : array-like, shape (n_query, n_features)

        Returns
        -------
        distances : ndarray, shape (n_query, k)
        indices   : ndarray, shape (n_query, k)
        """
        X = np.asarray(X, dtype=float)
        all_distances = []
        all_indices = []

        for x in X:
            distances = np.array([self._distance(x, x_tr) for x_tr in self.X_train])
            idx = np.argsort(distances)[:self.k]

            all_distances.append(distances[idx])
            all_indices.append(idx)

        return np.array(all_distances), np.array(all_indices)

class KNNClassifier(_KNNBase):
    """
    K-Nearest Neighbors classifier using majority vote.
    """

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        _, idx = self.kneighbors(X)
        preds = []

        for neighbors in idx:
            labels = self.y_train[neighbors]
            values, counts = np.unique(labels, return_counts=True)
            preds.append(values[np.argmax(counts)])

        return np.array(preds)

    def score(self, X, y):
        y_pred = self.predict(X)
        return float(np.mean(y_pred == np.asarray(y)))
    


class KNNRegressor(_KNNBase):
    """
    K-Nearest Neighbors regressor using mean of neighbor values.
    """

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        _, idx = self.kneighbors(X)
        preds = []

        for neighbors in idx:
            y_vals = self.y_train[neighbors].astype(float)
            preds.append(np.mean(y_vals))

        return np.array(preds)

    def score(self, X, y):
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1 - ss_res / ss_tot)

