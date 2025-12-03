import numpy as np
from rice2025.unsupervised_learning.community_detection import LabelPropagation

def test_lp_runs():
    # Simple 4-node connected chain graph
    A = np.array([
        [0,1,0,0],
        [1,0,1,0],
        [0,1,0,1],
        [0,0,1,0]
    ])

    lp = LabelPropagation()
    lp.fit(A)

    labels = lp.predict()
    assert labels.shape == (4,)
    assert labels.dtype == int

def test_singleton_nodes():
    # graph with isolated nodes
    A = np.array([
        [0,1,0],
        [1,0,0],
        [0,0,0]  # isolated node
    ])

    lp = LabelPropagation()
    lp.fit(A)
    labels = lp.predict()

    assert labels[2] == 2  # isolated node keeps its own label

def test_two_communities():
    # Two disconnected cliques
    A = np.array([
        [0,1,1,0,0,0],
        [1,0,1,0,0,0],
        [1,1,0,0,0,0],

        [0,0,0,0,1,1],
        [0,0,0,1,0,1],
        [0,0,0,1,1,0]
    ])

    lp = LabelPropagation()
    lp.fit(A)
    labels = lp.predict()

    # group 1: nodes 0,1,2
    # group 2: nodes 3,4,5
    comm1 = labels[:3]
    comm2 = labels[3:]

    assert len(set(comm1)) == 1, "First 3 nodes should be same community"
    assert len(set(comm2)) == 1, "Last 3 nodes should be same community"
    assert comm1[0] != comm2[0], "Communities should differ"
