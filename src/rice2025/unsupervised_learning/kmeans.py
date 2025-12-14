"""
K-Means Clustering Algorithm (Lloyd's Algorithm).

This module implements the classic K-Means clustering algorithm from scratch
using NumPy. K-Means is an unsupervised learning method that partitions data
into a fixed number of clusters by iteratively assigning points to the nearest
centroid and updating centroids as the mean of assigned points.

The algorithm alternates between:
1. Assignment step: assigning each sample to the closest centroid
2. Update step: recomputing centroids as the mean of their assigned samples

This implementation:
- Uses Euclidean distance for cluster assignment
- Supports random centroid initialization with reproducibility
- Detects convergence based on centroid movement tolerance
- Handles empty clusters by reinitializing their centroids
- Computes inertia (within-cluster sum of squares) after fitting

The focus of this implementation is clarity, correctness, and educational value,
rather than large-scale performance optimization.
"""


import numpy as np


class KMeans:
    """
    Simple NumPy implementation of K-Means clustering.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    max_iter : int, default=100
        Maximum iterations for Lloyd's algorithm.
    tol : float, default=1e-4
        Convergence tolerance on centroid movement.
    random_state : int or None
        Seed for reproducibility.

    Attributes
    ----------
    centroids_ : ndarray of shape (n_clusters, n_features)
        Final centroid positions.
    labels_ : ndarray of shape (n_samples,)
        Cluster assignment of each sample.
    inertia_ : float
        Sum of squared distances of samples to their closest centroid.
    """

    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=None):
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None

    # ---------------------------------------------------------
    def _init_centroids(self, X):
        """Random initialization of centroids."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        indices = np.random.choice(len(X), size=self.n_clusters, replace=False)
        return X[indices]

    def _compute_labels(self, X, centroids):
        """Assign each sample to the nearest centroid."""
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, labels):
        """Compute new centroids as mean of assigned points."""
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) == 0:
                # reinitialize empty cluster
                new_centroids[k] = X[np.random.choice(len(X))]
            else:
                new_centroids[k] = cluster_points.mean(axis=0)

        return new_centroids

    # ---------------------------------------------------------
    def fit(self, X):
        X = np.asarray(X, float)

        centroids = self._init_centroids(X)

        for _ in range(self.max_iter):
            labels = self._compute_labels(X, centroids)
            new_centroids = self._compute_centroids(X, labels)

            # Check convergence
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids

            if shift < self.tol:
                break

        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = float(
            np.sum((X - centroids[labels]) ** 2)
        )

        return self

    # ---------------------------------------------------------
    def predict(self, X):
        """Assign labels using trained centroids."""
        if self.centroids_ is None:
            raise AttributeError("Model not yet fitted.")

        X = np.asarray(X, float)
        return self._compute_labels(X, self.centroids_)

    # ---------------------------------------------------------
    def fit_predict(self, X):
        """Fit on X and return cluster labels."""
        self.fit(X)
        return self.labels_
