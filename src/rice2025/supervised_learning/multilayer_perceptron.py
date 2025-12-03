import numpy as np


def _validate_binary_y(y):
    """Ensure y contains only 0/1 labels."""
    unique_vals = np.unique(y)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError("MLPClassifier only supports binary labels 0/1.")


def _prepare_X_y(X, y=None):
    X = np.asarray(X, dtype=float)
    if y is None:
        return X
    y = np.asarray(y, dtype=float)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    _validate_binary_y(y)
    return X, y


class MLPClassifier:
    """
    A simple Multilayer Perceptron (feedforward neural network) for binary classification.

    Features:
    ----------
    • Multiple hidden layers
    • ReLU or Sigmoid activations
    • Sigmoid output layer
    • Binary cross entropy loss
    • L2 regularization via 'alpha'
    • Full-batch gradient descent
    • sklearn-like API

    Parameters
    ----------
    hidden_layer_sizes : list
        Example: [10, 5] for 2 hidden layers.
    activation : {"relu", "sigmoid"}
    learning_rate : float
    max_iter : int
    tol : float
    alpha : float
        L2 penalty (λ).
    random_state : int or None
    """

    def __init__(
        self,
        hidden_layer_sizes=[10],
        activation="relu",
        learning_rate=0.01,
        max_iter=2000,
        tol=1e-4,
        alpha=0.0,
        random_state=None,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.random_state = random_state

        # Internal
        self.rng = np.random.default_rng(random_state)
        self.weights = []  # list of W matrices
        self.biases = []   # list of b vectors

    # --------------------------
    # Activation functions
    # --------------------------

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_grad(z):
        return (z > 0).astype(float)

    @staticmethod
    def _sigmoid_grad(a):
        return a * (1 - a)

    def _activate(self, z):
        if self.activation == "relu":
            return self._relu(z)
        elif self.activation == "sigmoid":
            return self._sigmoid(z)
        else:
            raise ValueError("activation must be 'relu' or 'sigmoid'")

    def _activate_grad(self, a, z):
        if self.activation == "relu":
            return self._relu_grad(z)
        else:
            return self._sigmoid_grad(a)

    # --------------------------
    # Weight initialization
    # --------------------------
    def _init_weights(self, input_dim):
        layer_dims = [input_dim] + self.hidden_layer_sizes + [1]

        self.weights = []
        self.biases = []

        for i in range(len(layer_dims) - 1):
            fan_in = layer_dims[i]

            # Xavier init for sigmoid, He init for ReLU
            if self.activation == "relu":
                limit = np.sqrt(2.0 / fan_in)
            else:
                limit = np.sqrt(1.0 / fan_in)

            W = self.rng.normal(0, limit, size=(fan_in, layer_dims[i + 1]))
            b = np.zeros(layer_dims[i + 1])

            self.weights.append(W)
            self.biases.append(b)

    # --------------------------
    # Forward pass
    # --------------------------
    def _forward(self, X):
        activations = [X]
        zs = []

        a = X
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            a = self._activate(z)
            zs.append(z)
            activations.append(a)

        # Output layer: sigmoid
        z = activations[-1] @ self.weights[-1] + self.biases[-1]
        a = self._sigmoid(z)

        zs.append(z)
        activations.append(a)

        return activations, zs

    # --------------------------
    # Backpropagation
    # --------------------------
    def _backward(self, activations, zs, y):
        m = y.shape[0]

        # Lists for gradients
        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)

        # Convert y shape to match predictions
        y = y.reshape(-1, 1)
        a_out = activations[-1]

        # Output layer delta
        delta = (a_out - y)

        # Gradient for output layer
        dW[-1] = (activations[-2].T @ delta) / m + (self.alpha * self.weights[-1])
        db[-1] = delta.mean(axis=0)

        # Hidden layers backward
        for i in reversed(range(len(self.hidden_layer_sizes))):
            z = zs[i]
            a = activations[i + 1]

            delta = (delta @ self.weights[i + 1].T) * self._activate_grad(a, z)
            dW[i] = (activations[i].T @ delta) / m + (self.alpha * self.weights[i])
            db[i] = delta.mean(axis=0)

        return dW, db

    # --------------------------
    # Fit
    # --------------------------
    def fit(self, X, y):
        X, y = _prepare_X_y(X, y)
        y = y.reshape(-1, 1)

        n_samples, n_features = X.shape
        self._init_weights(n_features)

        for _ in range(self.max_iter):
            activations, zs = self._forward(X)
            dW, db = self._backward(activations, zs, y)

            # Gradient descent update
            max_change = 0
            for i in range(len(self.weights)):
                update_W = self.learning_rate * dW[i]
                update_b = self.learning_rate * db[i]

                self.weights[i] -= update_W
                self.biases[i] -= update_b

                max_change = max(max_change, np.max(np.abs(update_W)))

            if max_change < self.tol:
                break

        return self

    # --------------------------
    # Prediction API
    # --------------------------

    def predict_proba(self, X):
        X = _prepare_X_y(X)
        out = self._forward(X)[0][-1].reshape(-1)
        return np.vstack([1 - out, out]).T

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def score(self, X, y):
        X, y = _prepare_X_y(X, y)
        preds = self.predict(X)
        return np.mean(preds == y)
