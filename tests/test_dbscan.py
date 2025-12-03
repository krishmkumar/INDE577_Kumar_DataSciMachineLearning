import numpy as np

from rice2025.unsupervised_learning.dbscan import DBSCAN


def test_dbscan_two_well_separated_clusters_with_noise():
    """
    Simple deterministic test for DBSCAN.

    We build:
    - 3 points near (0, 0)
    - 3 points near (10, 10)
    - 1 noise point far away

    With eps large enough to connect points within each blob and
    min_samples=2, DBSCAN should find exactly 2 clusters and mark
    the far-away point as noise (-1).
    """
    # Cluster 1
    c1 = np.array([[0.0, 0.0],
                   [0.1, -0.1],
                   [-0.2, 0.2]])

    # Cluster 2
    c2 = np.array([[10.0, 10.0],
                   [10.1, 9.9],
                   [9.8, 10.2]])

    # Noise point
    noise = np.array([[100.0, 100.0]])

    X = np.vstack([c1, c2, noise])

    model = DBSCAN(eps=0.5, min_samples=2)
    labels = model.fit_predict(X)

    # We expect two clusters and one noise point
    assert model.n_clusters_ == 2
    assert labels.shape == (7,)

    # Last point should be labeled as noise
    assert labels[-1] == -1

    # First three points should share the same non-negative label
    first_cluster_label = labels[0]
    assert first_cluster_label >= 0
    assert np.all(labels[1:3] == first_cluster_label)

    # Next three points should share another non-negative label,
    # different from the first one
    second_cluster_label = labels[3]
    assert second_cluster_label >= 0
    assert second_cluster_label != first_cluster_label
    assert np.all(labels[4:6] == second_cluster_label)


def test_dbscan_handles_empty_input():
    """DBSCAN should handle an empty array without crashing."""
    X_empty = np.empty((0, 2))

    model = DBSCAN(eps=0.5, min_samples=2)
    labels = model.fit_predict(X_empty)

    assert labels.size == 0
    assert model.n_clusters_ == 0
