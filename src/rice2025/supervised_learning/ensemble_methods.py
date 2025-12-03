import numpy as np
from collections import Counter
from .decision_tree import DecisionTree
from .knn import KNNClassifier


def _validate_inputs(X, y=None):
    X = np.asarray(X)
    if y is not None:
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        return X, y
    return X


def _majority_vote(preds_col):
    counts = Counter(preds_col)
    return sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]


class BaggingClassifier:
    def __init__(
        self,
        base_learner=DecisionTree,
        n_estimators=10,
        max_samples=1.0,
        random_state=None,
    ):
        self.base_learner = base_learner
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.models = []

    def fit(self, X, y):
        X, y = _validate_inputs(X, y)
        n = X.shape[0]
        sample_size = int(self.max_samples * n)

        self.models = []
        for _ in range(self.n_estimators):
            idx = self.rng.choice(n, size=sample_size, replace=True)
            X_boot, y_boot = X[idx], y[idx]

            model = self.base_learner()
            model.fit(X_boot, y_boot)
            self.models.append(model)

        return self

    def predict(self, X):
        X = _validate_inputs(X)
        preds = np.array([np.asarray(m.predict(X)) for m in self.models])
        return np.array([_majority_vote(preds[:, j]) for j in range(X.shape[0])])


class VotingClassifier:
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        X, y = _validate_inputs(X, y)
        for m in self.models:
            if hasattr(m, "fit"):
                m.fit(X, y)
        return self

    def predict(self, X):
        X = _validate_inputs(X)
        preds = np.array([np.asarray(m.predict(X)) for m in self.models])
        return np.array([_majority_vote(preds[:, j]) for j in range(X.shape[0])])


class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_samples=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.trees = []

    def fit(self, X, y):
        X, y = _validate_inputs(X, y)
        n = X.shape[0]
        sample_size = int(self.max_samples * n)

        self.trees = []
        for _ in range(self.n_estimators):
            idx = self.rng.choice(n, size=sample_size, replace=True)
            X_boot, y_boot = X[idx], y[idx]

            tree = DecisionTree()  # FIX: remove unsupported argument
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        return self

    def predict(self, X):
        X = _validate_inputs(X)
        preds = np.array([np.asarray(t.predict(X)) for t in self.trees])
        return np.array([_majority_vote(preds[:, j]) for j in range(X.shape[0])])
