import numpy as np

def majority_label(labels):
    labels = np.asarray(labels)
    vals, counts = np.unique(labels, return_counts=True)
    return vals[np.argmax(counts)]

def average_label(labels):
    labels = np.asarray(labels, dtype=float)
    return float(np.mean(labels))
