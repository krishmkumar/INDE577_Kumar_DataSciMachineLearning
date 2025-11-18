import numpy as np
from rice2025.supervised_learning.knn import KNNRegressor, KNNClassifier

def test_knn_regressor_basic():
    X = np.array([[0],[1],[2],[3]], dtype=float)
    y = np.array([0.0, 1.0, 1.5, 3.0])
    
    model = KNNRegressor(n_neighbors=2)
    model.fit(X, y)
    
    pred = model.predict([[1.5]])
    assert np.isclose(pred[0], 1.25)

def test_knn_classifier_basic():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = np.array([0,0,1,1])

    model = KNNClassifier(n_neighbors=3)
    model.fit(X, y)

    pred = model.predict([[0.1, 0.1]])
    assert pred[0] == 0
