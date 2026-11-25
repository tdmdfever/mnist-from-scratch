import numpy as np

from src.softmax import softmax


def init_params(n_features=784, n_hidden=128, n_classes=10, seed=0):
    """Initialize weights for a single hidden layer MLP classifier.

    Hidden layer weights W1 (n_features, n_hidden) and biases b1 (1, n_hidden);
    output layer weights W2 (n_hidden, n_classes) and b2 (1, n_classes).
    """
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, np.sqrt(2 / n_features), size=(n_features, n_hidden))
    W2 = rng.normal(0.0, np.sqrt(2 / n_hidden), size=(n_hidden, n_classes))

    params = {"W1": W1, "b1": np.zeros((1, n_hidden)), "W2": W2, "b2": np.zeros((1, n_classes))}
    return params


def affine(X, W, b):
    """Compute the linear scores Z = X @ W + b."""
    Z = X @ W + b
    return Z


def relu(Z):
    """Compute ReLU values from linear scores: (m, h) -> (m, h)."""
    A = np.maximum(0, Z)
    return A


def forward(X, params):
    """Compute the probabilities for input X given parameters.

    Return P along with a cache dict of values for backpropagation:
    X (m, 784), Z1 (m, 128), A1 (m, 128), and P (m, 10).
    """
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    Z1 = affine(X, W1, b1)
    A1 = relu(Z1)
    Z2 = affine(A1, W2, b2)
    P = softmax(Z2)

    cache = {"X": X, "Z1": Z1, "A1": A1, "P": P}
    return P, cache


def backward(params, cache, Y):
    """Compute gradients of the loss with respect to W1, b1, W2, and b2.

    From the derivation: G2 = dL/dZ2 = (P - Y)/m, so dW2 = A1^T @ G2 and db2 = column sums of G2
                         G1 = (G2 @ W2.T) * (Z1 > 0), so dW1 = X^T @ G1 and db1 = column sums of G1
    """
    W2 = params["W2"]

    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    P = cache["P"]
    m = len(X)

    G2 = (P - Y) / m
    dW2 = A1.T @ G2
    db2 = G2.sum(axis=0, keepdims=True)
    G1 = (G2 @ W2.T) * (Z1 > 0)
    dW1 = X.T @ G1
    db1 = G1.sum(axis=0, keepdims=True)

    grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    return grads


def predict(X, params):
    """Predict class labels for X given parameters.

    Return the index of the max probability for each sample.
    """
    P, _ = forward(X, params)
    pred = P.argmax(axis=1)
    return pred
