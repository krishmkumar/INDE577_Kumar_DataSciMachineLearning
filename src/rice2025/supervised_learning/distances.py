import numpy as np

def euclidean_distance(a, b):
    """Compute Euclidean distance between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan_distance(a, b):
    """Compute Manhattan distance between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return np.sum(np.abs(a - b))
