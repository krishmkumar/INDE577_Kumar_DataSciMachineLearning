"""
DBSCAN clustering algorithm implemented from scratch using NumPy.

This implementation follows the standard DBSCAN algorithm:

- `eps`: radius of the neighborhood
- `min_samples`: minimum number of points (including the point itself)
  required to form a dense region (i.e., a core point)
- Points are labeled with integers 0, 1, ..., n_clusters-1
- Noise points are labeled -1

Example
-------
>>> from rice2025.unsupervised.dbscan import DBSCAN
>>> import numpy as np
>>> X = np.array([[0, 0], [0, 1], [1, 0], [8, 8]])
>>> model = DBSCAN(eps=1.5, min_samples=2)
>>> labels = model.fit_predict(X)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for them to be considered
        as neighbors.

    min_samples : int, default=5
        The number of samples (or total weight) in a neighborhood for a point
        to be considered as a core point. This includes the point itself.

    Attributes
    ----------
    eps_ : float
        Effective neighborhood radius used.

    min_samples_ : int
        Effective minimum samples used.

    labels_ : np.ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset. Noisy samples are given
        the label -1.

    core_sample_indices_ : np.ndarray of shape (n_core_samples,)
        Indices of core samples.

    n_clusters_ : int
        The number of clusters found (excluding noise).
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None:
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if min_samples <= 0:
            raise ValueError("min_samples must be positive.")

        self.eps_ = float(eps)
        self.min_samples_ = int(min_samples)

        # Attributes set after fitting
        self.labels_: Optional[np.ndarray] = None
        self.core_sample_indices_: Optional[np.ndarray] = None
        self.n_clusters_: Optional[int] = None

    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------
    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute the full pairwise Euclidean distance matrix.

        This is O(n^2) in both time and memory, which is acceptable for
        small to medium datasets and for teaching purposes.
        """
        # Using (x - y)^2 = x^2 + y^2 - 2 xy^T
        # X: (n_samples, n_features)
        sq_norms = np.sum(X ** 2, axis=1, keepdims=True)  # (n, 1)
        # broadcasting plus matrix multiplication
        dist_sq = sq_norms + sq_norms.T - 2 * X @ X.T
        # numerical noise may give slight negatives
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.sqrt(dist_sq)

    def _region_query(self, dist_matrix: np.ndarray, idx: int) -> np.ndarray:
        """Return indices of all points within eps of point idx."""
        return np.where(dist_matrix[idx] <= self.eps_)[0]

    def _expand_cluster(
        self,
        X: np.ndarray,
        dist_matrix: np.ndarray,
        labels: np.ndarray,
        point_idx: int,
        cluster_id: int,
        visited: np.ndarray,
        is_core: np.ndarray,
    ) -> None:
        """Expand cluster starting from a core point."""
        # Points to process: start from this point's neighbors
        seeds = list(self._region_query(dist_matrix, point_idx))

        labels[point_idx] = cluster_id

        # Process seeds list like a queue
        i = 0
        while i < len(seeds):
            current_idx = seeds[i]

            if not visited[current_idx]:
                visited[current_idx] = True
                neighbors = self._region_query(dist_matrix, current_idx)

                # If current point is core, add its neighbors
                if neighbors.size >= self.min_samples_:
                    is_core[current_idx] = True
                    # Add new neighbors that are not already in seeds
                    for n_idx in neighbors:
                        if n_idx not in seeds:
                            seeds.append(n_idx)

            # If current point not yet assigned to a cluster, assign it
            if labels[current_idx] == -1:
                labels[current_idx] = cluster_id

            i += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "DBSCAN":
        """Perform DBSCAN clustering.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        self : DBSCAN
            Fitted instance.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")

        n_samples = X.shape[0]
        if n_samples == 0:
            # Handle empty input gracefully
            self.labels_ = np.array([], dtype=int)
            self.core_sample_indices_ = np.array([], dtype=int)
            self.n_clusters_ = 0
            return self

        # Compute distance matrix once
        dist_matrix = self._pairwise_distances(X)

        labels = np.full(n_samples, -1, dtype=int)  # -1 means noise/unassigned
        visited = np.zeros(n_samples, dtype=bool)
        is_core = np.zeros(n_samples, dtype=bool)

        cluster_id = 0

        for point_idx in range(n_samples):
            if visited[point_idx]:
                continue

            visited[point_idx] = True
            neighbors = self._region_query(dist_matrix, point_idx)

            if neighbors.size < self.min_samples_:
                # Mark as noise (already -1)
                continue

            # Otherwise, we have a new cluster
            is_core[point_idx] = True
            self._expand_cluster(
                X=X,
                dist_matrix=dist_matrix,
                labels=labels,
                point_idx=point_idx,
                cluster_id=cluster_id,
                visited=visited,
                is_core=is_core,
            )
            cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.where(is_core)[0]
        self.n_clusters_ = cluster_id

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit DBSCAN on X and return cluster labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster labels. Noise points are labeled -1.
        """
        self.fit(X)
        return self.labels_
