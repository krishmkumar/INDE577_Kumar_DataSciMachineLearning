import numpy as np
from rice2025.unsupervised_learning.community_detection import LabelPropagation


def test_basic_fit_predict():
    """Ensure fit() and predict() return labels of correct length."""
    A = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ])

    model = LabelPropagation(max_iter=100, seed=0)
    model.fit(A)
    labels = model.predict()

    assert len(labels) == 3
    assert labels.dtype.kind in {'i', 'u'}  # integer labels


def test_reproducibility():
    """Label propagation should be deterministic with the same seed."""
    A = np.array([
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
    ])

    m1 = LabelPropagation(seed=42)
    m2 = LabelPropagation(seed=42)

    m1.fit(A)
    m2.fit(A)

    assert np.array_equal(m1.predict(), m2.predict())


def test_two_communities():
    """
    A simple block graph with two clear communities should be separated.
    
    Community 1: nodes 0,1,2
    Community 2: nodes 3,4,5
    """
    A = np.array([
        [0,1,1,0,0,0],
        [1,0,1,0,0,0],
        [1,1,0,0,0,0],
        [0,0,0,0,1,1],
        [0,0,0,1,0,1],
        [0,0,0,1,1,0]
    ])

    model = LabelPropagation(max_iter=500, seed=123)
    labels = model.fit(A).predict()

    # first 3 nodes should share a label, last 3 should share a label
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_isolated_node():
    """An isolated node should keep its own label."""
    A = np.array([
        [0,1,0],
        [1,0,0],
        [0,0,0]  # isolated
    ])

    model = LabelPropagation(seed=1)
    labels = model.fit(A).predict()

    # node 2 is isolated → label stays unique
    assert labels[2] == 2


def test_convergence():
    """Model should converge quickly on a small graph."""
    A = np.array([
        [0,1,0,0],
        [1,0,1,0],
        [0,1,0,1],
        [0,0,1,0]
    ])

    model = LabelPropagation(max_iter=50, seed=7)
    labels = model.fit(A).predict()

    # Check labels are stable (no default all-different initialization)
    assert len(set(labels)) < 4  # at least some propagation occurred
