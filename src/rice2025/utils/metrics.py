import numpy as np

def euclidean_distance(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan_distance(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return np.sum(np.abs(a - b))
