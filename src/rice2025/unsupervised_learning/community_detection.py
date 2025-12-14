"""
Label Propagation Algorithm (LPA) for Community Detection.

This module implements an unsupervised community detection algorithm based on
Label Propagation, using a pure NumPy formulation. The algorithm operates on an
undirected graph represented by a dense adjacency matrix and discovers
communities by iteratively propagating labels between neighboring nodes.

At initialization, each node is assigned a unique label. During each iteration,
nodes update their labels by adopting the most frequent label among their
neighbors. Over time, labels stabilize and form coherent communities without
requiring a predefined number of clusters.

This implementation:
- Requires no external graph libraries (e.g., NetworkX or SciPy)
- Supports randomized update order for convergence stability
- Allows optional damping to reduce oscillations
- Terminates early when label changes fall below a tolerance threshold

The algorithm is suitable for exploratory community detection in moderate-sized
graphs and emphasizes clarity and correctness over large-scale optimization.
"""

import numpy as np

class LabelPropagation:
    """
    Unsupervised community detection using Label Propagation (LPA).
    Pure NumPy implementation — no SciPy required.
    """

    def __init__(self, max_iter=1000, seed=42, damping=0.0, tol=0):
        self.max_iter = max_iter
        self.seed = seed
        self.damping = damping
        self.tol = tol
        self.labels_ = None

    def _get_neighbors(self, A, i):
        """Return neighbors of node i for dense adjacency matrix."""
        return np.where(A[i] > 0)[0]

    def fit(self, A):
        rng = np.random.default_rng(self.seed)
        A = np.array(A)

        # safety checks
        if A.shape[0] != A.shape[1]:
            raise ValueError("Adjacency matrix must be square.")
        if not np.allclose(A, A.T):
            raise ValueError("Adjacency matrix must be symmetric.")

        n = A.shape[0]
        labels = np.arange(n)

        for _ in range(self.max_iter):
            old_labels = labels.copy()

            # random update order
            update_order = rng.permutation(n)

            for i in update_order:
                neighbors = self._get_neighbors(A, i)
                if len(neighbors) == 0:
                    continue

                neighbor_labels = labels[neighbors]
                values, counts = np.unique(neighbor_labels, return_counts=True)
                max_count = counts.max()
                candidates = values[counts == max_count]

                chosen = rng.choice(candidates)

                if rng.random() >= self.damping:
                    labels[i] = chosen

            changes = np.sum(labels != old_labels)
            if changes < self.tol:
                break

        self.labels_ = labels
        return self

    def predict(self):
        if self.labels_ is None:
            raise ValueError("Model has not been fitted yet.")
        return self.labels_.copy()
