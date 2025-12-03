import numpy as np

from rice2025.unsupervised_learning.pca import PCA


def test_pca_full_reconstruction():
    """
    With n_components >= min(n_samples, n_features),
    PCA should reconstruct the original data (up to tiny FP error).
    """
    X = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 0.0, 1.0],
        ]
    )

    pca = PCA(n_components=None)
    Z = pca.fit_transform(X)
    X_rec = pca.inverse_transform(Z)

    assert X_rec.shape == X.shape
    # Perfect reconstruction up to numerical precision
    assert np.allclose(X, X_rec, atol=1e-10)

    # n_components_ should equal min(n_samples, n_features)
    assert pca.n_components_ == min(X.shape)


def test_pca_shapes_and_mean():
    """Check shapes and that mean_ matches the column-wise mean."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 5))

    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)

    # Transformed shape
    assert Z.shape == (20, 2)

    # mean_ should be the empirical column mean of X
    assert pca.mean_.shape == (5,)
    assert np.allclose(pca.mean_, X.mean(axis=0))


def test_pca_variance_ordering_and_ratio():
    """
    explained_variance_ should be non-increasing, and
    explained_variance_ratio_ should sum to 1 (or very close).
    """
    # Construct data with one dominant direction of variance
    x = np.linspace(-5, 5, 50)
    y = 0.1 * x  # much smaller variation along this direction
    X = np.column_stack([x, y])

    pca = PCA(n_components=2)
    pca.fit(X)

    # Variances non-increasing
    ev = pca.explained_variance_
    assert ev.shape == (2,)
    assert ev[0] >= ev[1]

    # Ratios sum to ~1
    ratio_sum = pca.explained_variance_ratio_.sum()
    assert np.isclose(ratio_sum, 1.0, atol=1e-8)


def test_pca_n_components_capped():
    """If n_components > min(n_samples, n_features), it should be capped."""
    X = np.random.rand(5, 3)  # min(5, 3) = 3

    pca = PCA(n_components=10)
    pca.fit(X)

    assert pca.n_components_ == 3
    assert pca.components_.shape == (3, 3)
