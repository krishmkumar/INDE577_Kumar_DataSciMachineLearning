class KNNRegressor(_KNNBase):
    """
    K-Nearest Neighbors regressor using mean of neighbor values.
    """

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        _, idx = self.kneighbors(X)
        preds = []

        for neighbors in idx:
            y_vals = self.y_train[neighbors].astype(float)
            preds.append(np.mean(y_vals))

        return np.array(preds)

    def score(self, X, y):
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return float(1 - ss_res / ss_tot)
