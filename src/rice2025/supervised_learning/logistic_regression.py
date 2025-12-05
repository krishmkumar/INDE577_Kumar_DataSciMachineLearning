import numpy as np


def _validate_binary_y(y):
    """Ensure y contains only 0/1 labels."""
    unique_vals = np.unique(y)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError("LogisticRegression only supports binary labels 0/1.")


def _prepare_X_y(X, y=None):
    """Convert X and y to numpy arrays and validate shapes."""
    X = np.asarray(X, dtype=float)
    if y is None:
        return X

    y = np.asarray(y, dtype=float)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    _validate_binary_y(y)
    return X, y


class LogisticRegression:
    """
    A robust, sklearn-like implementation of binary Logistic Regression.
    """

    def __init__(
        self,
        penalty="l2",
        C=1.0,
        fit_intercept=True,
        max_iter=10000,
        tol=1e-4,
        learning_rate=1.0,
        random_state=None,
        solver="gd",
    ):
        if penalty not in ["l2", "none"]:
            raise ValueError("penalty must be 'l2' or 'none'")
        if C <= 0:
            raise ValueError("C must be positive (inverse regularization strength).")
        if solver not in ["gd"]:
            raise ValueError("Only solver='gd' is supported.")

        self.penalty = penalty
        self.C = C
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.solver = solver

        self.rng = np.random.default_rng(random_state)

        self.coef_ = None
        self.intercept_ = None

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _compute_gradient(self, X, y, w):
        y_pred = self._sigmoid(X @ w)
        error = y_pred - y
        grad = (X.T @ error) / X.shape[0]

        # L2 penalty
        if self.penalty == "l2":
            reg_grad = (1.0 / self.C) * w
            if self.fit_intercept:
                reg_grad[0] = 0  # do not regularize intercept
            grad += reg_grad

        # Prevent overflow
        grad = np.clip(grad, -1e6, 1e6)
        return grad

    def fit(self, X, y):
        X, y = _prepare_X_y(X, y)
        X_aug = self._add_intercept(X)

        n_features = X_aug.shape[1]
        w = np.zeros(n_features)

        # adaptive LR to avoid exploding with strong regularization
        eff_lr = self.learning_rate
        if self.penalty == "l2":
            eff_lr = eff_lr * min(1.0, self.C)

        for _ in range(self.max_iter):
            grad = self._compute_gradient(X_aug, y, w)
            w_new = w - eff_lr * grad

            if np.linalg.norm(w_new - w) < self.tol:
                w = w_new
                break

            w = w_new

        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w

        return self

    def decision_function(self, X):
        X = _prepare_X_y(X)
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X):
        X = _prepare_X_y(X)
        z = self.decision_function(X)
        p1 = self._sigmoid(z)
        p0 = 1 - p1
        return np.vstack([p0, p1]).T
   
    def roc_curve(self, X, y, num_thresholds=200):
        """
        Compute ROC curve (FPR, TPR) and AUC manually without sklearn.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            True binary labels.
        num_thresholds : int
            Number of probability thresholds to evaluate.

        Returns
        -------
        fprs : np.ndarray
            False positive rates.
        tprs : np.ndarray
            True positive rates.
        auc : float
            Area under the ROC curve.
        """
        X, y = _prepare_X_y(X, y)
        probs = self.predict_proba(X)[:, 1]

        thresholds = np.linspace(0, 1, num_thresholds)
        tprs = []
        fprs = []

        for t in thresholds:
            preds = (probs >= t).astype(int)

            TP = np.sum((preds == 1) & (y == 1))
            FP = np.sum((preds == 1) & (y == 0))
            FN = np.sum((preds == 0) & (y == 1))
            TN = np.sum((preds == 0) & (y == 0))

            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

            tprs.append(TPR)
            fprs.append(FPR)

        auc = np.trapz(tprs, fprs)
        return np.array(fprs), np.array(tprs), auc


    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def score(self, X, y):
        X, y = _prepare_X_y(X, y)
        preds = self.predict(X)
        return np.mean(preds == y)
