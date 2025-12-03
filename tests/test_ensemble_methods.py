import numpy as np

from rice2025.supervised_learning.ensemble_methods import (
    BaggingClassifier,
    VotingClassifier,
    RandomForestClassifier,
)
from rice2025.supervised_learning.knn import KNNClassifier
from rice2025.supervised_learning.decision_tree import DecisionTree


def small_dataset():
    # Simple binary classification
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y = np.array([0, 0, 1, 1])
    return X, y


def test_bagging_classifier_runs():
    X, y = small_dataset()
    model = BaggingClassifier(
        base_learner=DecisionTree, n_estimators=5, random_state=42
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_voting_classifier_runs():
    X, y = small_dataset()
    m1 = KNNClassifier(k=1)
    m2 = DecisionTree()
    voter = VotingClassifier(models=[m1, m2])
    voter.fit(X, y)
    preds = voter.predict(X)
    assert preds.shape == y.shape


def test_random_forest_runs():
    X, y = small_dataset()
    rf = RandomForestClassifier(n_estimators=5, random_state=123)
    rf.fit(X, y)
    preds = rf.predict(X)
    assert preds.shape == y.shape


def test_ensemble_stability_with_random_state():
    X, y = small_dataset()
    rf1 = RandomForestClassifier(n_estimators=5, random_state=123)
    rf2 = RandomForestClassifier(n_estimators=5, random_state=123)
    rf1.fit(X, y)
    rf2.fit(X, y)
    assert np.all(rf1.predict(X) == rf2.predict(X))
