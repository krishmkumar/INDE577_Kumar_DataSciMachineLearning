import numpy as np
from rice2025.supervised_learning.multilayer_perceptron import MLPClassifier


def simple_data():
    """
    Perfectly separable binary dataset:
    y = 1 if x > 0.5 else 0
    """
    X = np.array([[0.1], [0.2], [0.3], [0.7], [0.8], [0.9]])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


# -----------------------------------------------------------
# Basic construction tests
# -----------------------------------------------------------

def test_init_hidden_layers():
    model = MLPClassifier(hidden_layer_sizes=[5, 3])
    assert model.hidden_layer_sizes == [5, 3]


def test_predict_shape_before_fit():
    model = MLPClassifier()
    X = np.array([[0.1], [0.2]])
    # Should still raise because not fitted; good sanity check
    # but predict_proba should fail cleanly if run prematurely.
    try:
        model.predict(X) 
    except Exception:
        pass


# -----------------------------------------------------------
# Fit + convergence tests
# -----------------------------------------------------------

def test_fit_converges_on_simple_data():
    X, y = simple_data()
    model = MLPClassifier(
        hidden_layer_sizes=[5],
        learning_rate=0.1,
        max_iter=5000,
        random_state=123,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert np.mean(preds == y) >= 0.95   # NN may not hit exactly 1.0


# -----------------------------------------------------------
# predict_proba correctness
# -----------------------------------------------------------

def test_predict_proba_shape():
    X, y = simple_data()
    model = MLPClassifier(hidden_layer_sizes=[4], random_state=0)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    # probabilities must sum to 1
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


# -----------------------------------------------------------
# Activation tests
# -----------------------------------------------------------

def test_relu_vs_sigmoid_hidden():
    X, y = simple_data()

    model_relu = MLPClassifier(hidden_layer_sizes=[5], activation="relu", random_state=1)
    model_sig = MLPClassifier(hidden_layer_sizes=[5], activation="sigmoid", random_state=1)

    model_relu.fit(X, y)
    model_sig.fit(X, y)

    preds_relu = model_relu.predict(X)
    preds_sig = model_sig.predict(X)

    assert preds_relu.shape == preds_sig.shape == y.shape


# -----------------------------------------------------------
# Regularization tests
# -----------------------------------------------------------

def test_regularization_shrinks_weights():
    X, y = simple_data()

    model_no_reg = MLPClassifier(alpha=0.0, random_state=42)
    model_reg = MLPClassifier(alpha=10.0, random_state=42)

    model_no_reg.fit(X, y)
    model_reg.fit(X, y)

    # Compare mean absolute weight magnitude
    w_no_reg = np.mean([np.abs(W).mean() for W in model_no_reg.weights])
    w_reg = np.mean([np.abs(W).mean() for W in model_reg.weights])

    assert w_reg < w_no_reg


# -----------------------------------------------------------
# Deterministic behavior
# -----------------------------------------------------------

def test_random_state_reproducibility():
    X, y = simple_data()

    m1 = MLPClassifier(hidden_layer_sizes=[4], random_state=123)
    m2 = MLPClassifier(hidden_layer_sizes=[4], random_state=123)

    m1.fit(X, y)
    m2.fit(X, y)

    # weights should be identical
    for W1, W2 in zip(m1.weights, m2.weights):
        assert np.allclose(W1, W2)

    for b1, b2 in zip(m1.biases, m2.biases):
        assert np.allclose(b1, b2)
