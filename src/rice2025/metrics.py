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


import numpy as np
from collections import Counter

def accuracy(y_true, y_pred):
    """
    Compute accuracy between true and predicted labels.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)

def confusion_matrix_custom(y_true, y_pred):
    """
    Compute a simple confusion matrix for classification.
    """
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    class_to_index = {cls: idx for idx, cls in enumerate(classes)}
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[class_to_index[true_label], class_to_index[pred_label]] += 1

    return matrix
