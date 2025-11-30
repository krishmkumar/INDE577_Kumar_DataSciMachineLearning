import numpy as np
from rice2025.utils.postprocess import majority_label, average_label

def test_majority_label():
    y = np.array([1, 2, 2, 3])
    assert majority_label(y) == 2

def test_average_label():
    y = np.array([2, 4, 6])
    assert average_label(y) == 4
