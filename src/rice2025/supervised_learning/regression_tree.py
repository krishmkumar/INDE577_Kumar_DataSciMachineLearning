"""
CART-Style Tree Model (from scratch).

This module implements a simple CART-style decision tree using Gini impurity
as the splitting criterion. The tree is constructed recursively by selecting
feature–threshold splits that minimize weighted Gini impurity, and predictions
are made via tree traversal.

Key features:
- Binary or multiclass classification
- Gini impurity–based splitting
- Recursive tree construction with depth and sample-size stopping rules
- Deterministic class probability estimates at leaf nodes
- Dictionary-based tree representation for transparency and inspection

Design notes:
- Although named `RegressionTree`, this implementation performs
  classification using class counts and Gini impurity
- No pruning or feature subsampling is performed
- Intended for clarity and instructional use rather than efficiency

This implementation relies only on NumPy and is designed for educational
demonstration in the INDE 577 course context, emphasizing interpretability
over production-level optimization.
"""


import numpy as np

class RegressionTree:
    """
    Simple CART-style regression tree for classification.
    Splits based on Gini impurity.
    """

    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None   # dictionary-based recursive tree

    # ---------------------------------------------------------
    # Public API (mirrors your other models)
    # ---------------------------------------------------------
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X):
        X = np.array(X)
        preds = [self._predict_row(row, self.tree) for row in X]
        return np.array(preds)

    def predict_proba(self, X):
        X = np.array(X)
        probs = []
        for row in X:
            node = self.tree
            while not node["is_leaf"]:
                if row[node["feature"]] < node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]
            # Normalize leaf probability distribution
            counts = node["class_counts"]
            p = counts / counts.sum()
            probs.append(p)
        return np.vstack(probs)

    # ---------------------------------------------------------
    # Tree-building internals
    # ---------------------------------------------------------
    def _build_tree(self, X, y, depth):
        num_samples = len(y)
        num_labels = len(np.unique(y))

        # stopping conditions
        if (depth >= self.max_depth or 
            num_labels == 1 or 
            num_samples < self.min_samples_split):
            return self._leaf(y)

        # find best split
        feat, thresh = self._best_split(X, y)
        if feat is None:
            return self._leaf(y)

        # partition
        left_idx = X[:, feat] < thresh
        right_idx = ~left_idx

        return {
            "is_leaf": False,
            "feature": feat,
            "threshold": thresh,
            "left": self._build_tree(X[left_idx], y[left_idx], depth + 1),
            "right": self._build_tree(X[right_idx], y[right_idx], depth + 1)
        }

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _leaf(self, y):
        classes, counts = np.unique(y, return_counts=True)
        leaf_counts = np.zeros(len(self.classes_))

        # fill class counts in deterministic order
        for i, c in enumerate(self.classes_):
            mask = (classes == c)
            leaf_counts[i] = counts[mask][0] if mask.any() else 0

        return {
            "is_leaf": True,
            "class_counts": leaf_counts
        }

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p ** 2)

    def _best_split(self, X, y):
        best_feat, best_thresh = None, None
        best_gini = np.inf
        n_samples, n_features = X.shape

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left = y[X[:, feat] < thresh]
                right = y[X[:, feat] >= thresh]

                if len(left) == 0 or len(right) == 0:
                    continue

                g = (len(left) * self._gini(left) +
                     len(right) * self._gini(right)) / n_samples

                if g < best_gini:
                    best_gini = g
                    best_feat = feat
                    best_thresh = thresh

        return best_feat, best_thresh

    def _predict_row(self, row, node):
        while not node["is_leaf"]:
            if row[node["feature"]] < node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        counts = node["class_counts"]
        return self.classes_[np.argmax(counts)]
