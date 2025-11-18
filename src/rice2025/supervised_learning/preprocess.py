import numpy as np

def normalize(X):
    X = np.asarray(X, dtype=float)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1
    return (X - mean) / std


def train_test_split(X, y, test_size=0.2):
    X = np.asarray(X)
    y = np.asarray(y)

    n = len(X)
    split_idx = int(n * (1 - test_size))

    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]
