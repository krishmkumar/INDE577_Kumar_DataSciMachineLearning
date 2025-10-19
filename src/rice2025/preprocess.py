import numpy as np

def normalize(X):
    """Normalize X by subtracting mean and dividing by std."""
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def train_test_split(X, y, test_size=0.2, random_state=None):
    """Split data into training and testing sets."""
    if random_state:
        np.random.seed(random_state)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    test_count = int(len(X) * test_size)
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
