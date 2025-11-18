import numpy as np
from rice2025.unsupervised_learning.kmeans import KMeans


def test_kmeans_basic_clustering():
    X = np.array([
        [1, 1],
        [1.1, 1.0],
        [0.9, 1.2],
        [5, 5],
        [5.2, 4.9],
        [4.8, 5.1],
    ])

    model = KMeans(n_clusters=2, random_state=0).fit(X)

    # labels should form exactly 2 clusters
    assert len(np.unique(model.labels_)) == 2

    # inertia must be non-negative
    assert model.inertia_ >= 0


def test_kmeans_predict_after_fit():
    X = np.array([[0, 0], [10, 10]])
    model = KMeans(n_clusters=2, random_state=0).fit(X)

    labels = model.predict(X)
    assert len(labels) == 2
    assert set(labels) == {0, 1}


def test_fit_predict_equivalence():
    X = np.array([[1, 1], [2, 2], [8, 8], [9, 9]])
    model = KMeans(n_clusters=2, random_state=0)

    labels1 = model.fit_predict(X)
    labels2 = model.predict(X)

    # After fitting, predict returns same partition
    assert np.array_equal(labels1, labels2)


def test_error_if_predict_called_before_fit():
    model = KMeans(n_clusters=2)
    X = np.array([[0, 0]])

    try:
        model.predict(X)
        assert False, "Expected AttributeError"
    except AttributeError:
        pass
