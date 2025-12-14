"""
Decision Tree Classifier (from scratch).

This module implements a simple binary decision tree classifier using
entropy and information gain as the splitting criterion. The tree is built
recursively by selecting the feature and threshold that maximize information
gain at each node, with optional depth control to prevent overfitting.

Key characteristics:
- Supports binary and multiclass classification
- Uses entropy-based information gain for splits
- Handles continuous features via thresholding
- Implements prediction via recursive tree traversal
- Designed for clarity and educational purposes rather than performance

This implementation avoids external machine learning libraries and relies
only on NumPy, making it suitable for instructional use and algorithmic
demonstration in the INDE 577 course context.
"""
import numpy as np

# ==========================
# Utility: Entropy
# ==========================
def entropy(y):
    """Compute entropy."""
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs))


# ==========================
# Node structure
# ==========================
class Node:
    """Simple node structure for the Decision Tree."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature       # feature index
        self.threshold = threshold   # threshold
        self.left = left             # Node
        self.right = right           # Node
        self.value = value           # leaf label


# ==========================
# Decision Tree Classifier
# ==========================
class DecisionTree:
    """
    Simple Decision Tree classifier using entropy + information gain.
    """

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    # ----------------------------------------------------
    # Find best (feature, threshold) split
    # ----------------------------------------------------
    def _best_split(self, X, y):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        base_entropy = entropy(y)

        n_samples, n_features = X.shape

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = ~left_mask

                # Skip invalid splits
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                p_left = len(y_left) / len(y)
                p_right = 1 - p_left

                gain = base_entropy - (
                    p_left * entropy(y_left) + p_right * entropy(y_right)
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = t

        return best_feature, best_threshold

    # ----------------------------------------------------
    # Recursive Tree Builder
    # ----------------------------------------------------
    def _build(self, X, y, depth):
        unique, counts = np.unique(y, return_counts=True)

        # Pure leaf
        if len(unique) == 1:
            return Node(value=unique[0])

        # Depth limit reached
        if self.max_depth is not None and depth >= self.max_depth:
            return Node(value=unique[np.argmax(counts)])

        feature, threshold = self._best_split(X, y)

        # No split found → leaf
        if feature is None:
            return Node(value=unique[np.argmax(counts)])

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[right_mask], y[right_mask], depth + 1)

        return Node(feature, threshold, left, right)

    # ----------------------------------------------------
    # Fit
    # ----------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # -------- Validation --------
        if X.ndim != 2:
            raise ValueError("X must be 2D (n_samples, n_features).")
        if y.ndim != 1:
            raise ValueError("y must be 1D (n_samples,).")
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("Empty X or y provided.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must match.")

        # Build tree
        self.root = self._build(X, y, depth=0)
        return self

    # ----------------------------------------------------
    # Prediction for one sample
    # ----------------------------------------------------
    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    # ----------------------------------------------------
    # Predict many samples
    # ----------------------------------------------------
    def predict(self, X):
        X = np.asarray(X, float)
        return np.array([self._predict_one(x, self.root) for x in X])

    # ----------------------------------------------------
    # Score (Accuracy)
    # ----------------------------------------------------
    def score(self, X, y):
        y = np.asarray(y, int)
        pred = self.predict(X)
        return np.mean(pred == y)
