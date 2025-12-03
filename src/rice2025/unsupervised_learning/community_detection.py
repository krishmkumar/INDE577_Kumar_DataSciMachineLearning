import numpy as np

class LabelPropagation:
    """
    Unsupervised community detection using Label Propagation.
    Works on an undirected graph with adjacency matrix A.

    Nodes iteratively update their labels to the most frequent label
    among their neighbors until convergence.
    """

    def __init__(self, max_iter=1000, seed=42):
        self.max_iter = max_iter
        self.seed = seed
        self.labels_ = None

    def fit(self, A):
        """
        A : adjacency matrix (numpy array), shape (n, n)
        """
        rng = np.random.default_rng(self.seed)
        A = np.array(A)
        n = A.shape[0]

        # initialize labels as unique IDs
        labels = np.arange(n)

        for _ in range(self.max_iter):
            old_labels = labels.copy()

            # iterate deterministically from 0..n-1 for reproducibility
            for i in range(n):
                neighbors = np.where(A[i] > 0)[0]
                if len(neighbors) == 0:
                    continue

                # get labels of neighbors
                neighbor_labels = labels[neighbors]

                # pick the most frequent label
                values, counts = np.unique(neighbor_labels, return_counts=True)
                labels[i] = values[np.argmax(counts)]

            # convergence check
            if np.array_equal(labels, old_labels):
                break

        self.labels_ = labels
        return self

    def predict(self):
        """
        Return community labels for all nodes.
        """
        return self.labels_.copy()
