import numpy as np

def majority_label(labels):
    """Return the most common label."""
    values, counts = np.unique(labels, return_counts=True)
    return values[np.argmax(counts)]

def average_label(labels):
    """Return the mean of labels (for regression)."""
    return np.mean(labels)
