import numpy as np
from rice2025.metrics import euclidean_distance
from rice2025.postprocess import majority_label, average_label

class KNN:
    def __init__(self, k=3, classify=True):
        self.k = k
        self.classify = classify

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
            k_idx = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_idx]
            if self.classify:
                predictions.append(majority_label(k_labels))
            else:
                predictions.append(average_label(k_labels))
        return np.array(predictions)
