"""
Principal Component Analysis (PCA) implemented from scratch using NumPy.

This implementation is intentionally simple and educational:
- Uses SVD on the mean-centered data matrix (numerically stable)
- Supports an optional `n_components` parameter
- Exposes attributes similar to scikit-learn:
    - components_
    - explained_variance_
    - explained_variance_ratio_
    - mean_
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class PCA:
    """Principal Component Analysis (PCA).

    Parameters
    ----------
    n_components : int or None, default=None
        Number of principal components to keep.
        If None, all components are kept.

    center : bool, default=True
        Whether to subtract the mean of each feature before performing PCA.

    Attributes
    ----------
    n_components_ : int
        The number of components actually used.

    components_ : ndarray of shape (n_components_, n_features)
        Principal axes in feature space, representing the directions of
        maximum variance.

    explained_variance_ : ndarray of shape (n_components_,)
        The amount of variance explained by each of the selected components.

    explained_variance_ratio_ : ndarray of shape (n_components_,)
        Percentage of variance explained by each of the selected components.

    mean_ : ndarray of shape (n_features,)
        Per-feature empirical mean, estimated from the training set.
    """

    def __init__(self, n_components: Optional[int] = None, center: bool = True) -> None:
        if n_components is not None and n_components <= 0:
            raise ValueError("n_components must be positive or None.")

        self.n_components = n_components
        self.center = center

        # Set during fit
        self.n_components_: Optional[int] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _check_X(self, X: np.ndarray) -> np.ndarray:
        """Ensure X is a 2D numpy array of floats."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
        return X

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "PCA":
        """Fit the PCA model to X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : PCA
            Fitted estimator.
        """
        X = self._check_X(X)
        n_samples, n_features = X.shape

        if self.center:
            self.mean_ = np.mean(X, axis=0)
            X_centered = X - self.mean_
        else:
            # If not centering, treat mean as zeros for transform()
            self.mean_ = np.zeros(n_features)
            X_centered = X

        # Compute SVD on centered data
        # X_centered = U S V^T  -> rows of V^T are principal axes
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Eigenvalues of covariance matrix = S^2 / (n_samples - 1)
        eigenvalues = (S**2) / max(n_samples - 1, 1)

        # Decide how many components to keep
        if self.n_components is None:
            k = min(n_samples, n_features)
        else:
            k = min(self.n_components, n_samples, n_features)

        self.n_components_ = k
        self.components_ = Vt[:k]  # shape (k, n_features)
        self.explained_variance_ = eigenvalues[:k]

        total_var = eigenvalues.sum()
        if total_var > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_var
        else:
            self.explained_variance_ratio_ = np.zeros_like(self.explained_variance_)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X onto the principal components.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_components_)
        """
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA instance is not fitted yet. Call 'fit' first.")

        X = self._check_X(X)
        X_centered = X - self.mean_
        # Projection onto k components: X * components^T
        return X_centered @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA on X and return the transformed data."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Transform data back to the original feature space.

        Parameters
        ----------
        X_transformed : ndarray of shape (n_samples, n_components_)

        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features)
        """
        if self.components_ is None or self.mean_ is None:
            raise RuntimeError("PCA instance is not fitted yet. Call 'fit' first.")

        X_transformed = np.asarray(X_transformed, dtype=float)
        if X_transformed.ndim != 2:
            raise ValueError(
                "X_transformed must be 2D, got shape {}".format(X_transformed.shape)
            )

        # Reconstruct: X_hat = scores * components + mean
        return X_transformed @ self.components_ + self.mean_
