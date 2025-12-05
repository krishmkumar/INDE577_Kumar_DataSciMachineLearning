import numpy as np


def _validate_inputs(X, y=None):
    """
    Convert inputs to numpy arrays and validate shape consistency.
    """
    X = np.asarray(X, dtype=float)

    if y is not None:
        y = np.asarray(y, dtype=float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        return X, y
    return X


class LinearRegression:
    """
    Linear Regression (Ordinary Least Squares)

    Features:
    ----------
    • Normal Equation solution
    • Optional L2 regularization (Ridge)
    • Optional gradient descent optimizer
    • R² score and prediction
    • Handles bias/intercept internally

    Parameters
    ----------
    fit_intercept : bool
        Whether to include an intercept term.
    regularization : float
        L2 penalty λ. If λ=0 → plain OLS.
    use_gradient_descent : bool
        If True, fit using gradient descent instead of closed form.
    learning_rate : float
        Gradient descent learning rate.
    max_iter : int
        Maximum number of gradient descent iterations.
    tol : float
        Stop GD early if updates are small.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Fitted model coefficients.
    intercept_ : float
        Bias term.
    """

    def __init__(
        self,
        fit_intercept=True,
        regularization=0.0,
        use_gradient_descent=False,
        learning_rate=0.01,
        max_iter=1000,
        tol=1e-6,
    ):
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.use_gradient_descent = use_gradient_descent

        # GD parameters
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

        # Learned parameters
        self.coef_ = None
        self.intercept_ = None

    def _add_intercept(self, X):
        if not self.fit_intercept:
            return X
        intercept = np.ones((X.shape[0], 1))
        return np.hstack([intercept, X])

    def _closed_form_fit(self, X, y):
        """
        Solve using the normal equation:
            w = (XᵀX + λI)⁻¹ Xᵀ y
        """
        n_features = X.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0 if self.fit_intercept else 1  # don't regularize intercept

        A = X.T @ X + self.regularization * I
        b = X.T @ y

        w = np.linalg.solve(A, b)
        return w

    def _gradient_descent_fit(self, X, y):
        """
        Fit using gradient descent:
            w := w - α ∇Loss
        """
        n_samples, n_features = X.shape
        w = np.zeros(n_features)

        for _ in range(self.max_iter):
            y_pred = X @ w
            gradient = (2 / n_samples) * (X.T @ (y_pred - y))

            # L2 regularization
            if self.regularization > 0:
                reg_term = 2 * self.regularization * w
                if self.fit_intercept:
                    reg_term[0] = 0
                gradient += reg_term

            w_new = w - self.learning_rate * gradient

            if np.linalg.norm(w_new - w) < self.tol:
                break
            w = w_new

        return w

    def fit(self, X, y):
        """
        Fit linear regression using either:
        - Closed-form normal equation
        - Gradient descent (optional)
        """
        X, y = _validate_inputs(X, y)
        X_aug = self._add_intercept(X)

        if self.use_gradient_descent:
            w = self._gradient_descent_fit(X_aug, y)
        else:
            w = self._closed_form_fit(X_aug, y)

        if self.fit_intercept:
            self.intercept_ = w[0]
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w

        return self

    def predict(self, X):
        X = _validate_inputs(X)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """
        Return R²:
            R² = 1 - SS_res / SS_tot
        """
        X, y = _validate_inputs(X, y)
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot
    
    def residuals(self, X, y):
        """
        Return residuals: (y - y_pred)
        """
        X, y = _validate_inputs(X, y)
        return y - self.predict(X)

    def mse(self, X, y):
        """
        Mean Squared Error
        """
        res = self.residuals(X, y)
        return np.mean(res ** 2)

    def rmse(self, X, y):
        """
        Root Mean Squared Error
        """
        return np.sqrt(self.mse(X, y))

    def mae(self, X, y):
        """
        Mean Absolute Error
        """
        res = self.residuals(X, y)
        return np.mean(np.abs(res))

    def plot_residuals(self, X, y):
        """
        Residuals vs Predicted Plot
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        X, y = _validate_inputs(X, y)
        y_pred = self.predict(X)
        res = y - y_pred

        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=y_pred, y=res)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predicted Values")
        plt.show()

    def summary(self, X, y):
        """
        Print a summary of key regression metrics.
        """
        print("Model Summary")
        print("-" * 40)
        print(f"Intercept: {self.intercept_}")
        print(f"Coefficients: {self.coef_}")
        print(f"R² Score: {self.score(X, y):.4f}")
        print(f"MSE: {self.mse(X, y):.4f}")
        print(f"RMSE: {self.rmse(X, y):.4f}")
        print(f"MAE: {self.mae(X, y):.4f}")
        print("-" * 40)

