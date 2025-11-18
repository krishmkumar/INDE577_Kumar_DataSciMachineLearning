import numpy as np
from rice2025.supervised_learning.distances import euclidean_distance, manhattan_distance

def test_euclidean_distance():
    assert np.isclose(euclidean_distance(np.array([0, 0]), np.array([3, 4])), 5.0)

def test_manhattan_distance():
    assert manhattan_distance(np.array([0, 0]), np.array([3, 4])) == 7
