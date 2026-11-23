import numpy as np

def init_params(n_features=784, n_classes=10):
    """Initialize weights W (n_features, n_classes) and biases b (1, n_classes) for a softmax classifier."""
    W = np.zeros((n_features, n_classes))
    b = np.zeros((1, n_classes))
    params = {'W': W, 'b': b}
    return params

def score(X, params):
    """Compute the linear scores Z = X @ W + b."""
    W, b = params['W'], params['b']
    Z = X @ W + b
    return Z

def softmax(Z):
    """Row-wise softmax: (m, k) scores -> (m, k) probabilities.

    Subtracts each row's max first: exp overflows past ~709, and shifting doesn't change the result.
    """
    row_max = Z.max(axis=1, keepdims=True)
    Z_shifted = Z - row_max
    E = np.exp(Z_shifted)
    row_total = E.sum(axis=1, keepdims=True)
    P = E / row_total
    return P

def forward(X, params):
    """Compute the softmax probabilities for input X given parameters."""
    Z = score(X, params)
    P = softmax(Z)
    return P

def cross_entropy(P, Y):
    """Compute mean cross-entropy over the batch. P, Y: (m, 10).

    Adds 1e-12 before taking log to avoid log(0).
    """
    loss = -(Y * np.log(P + 1e-12)).sum(axis=1).mean()
    return loss

def gradients(X, P, Y):
    """Compute gradients of the loss with respect to W and b.

    From the derivation: G = dL/dZ = (P - Y)/m, so dW = X^T @ G and db = column sums of G
    """
    m = len(X)
    G = (P - Y)/m
    dW = X.T @ G
    db = G.sum(axis=0, keepdims=True)
    grads = {'W': dW, 'b': db}
    return grads

def update(params, grads, lr=0.1):
    """Update parameters in place using gradient descent."""
    for key in params:
        params[key] -= lr * grads[key]

def predict(X, params):
    """Predict class labels for input X given parameters. Returns the index of the max probability for each sample."""
    P = forward(X, params)
    pred = P.argmax(axis=1)
    return pred

def accuracy(pred, y):
    """Compute the accuracy of predictions pred against true labels y."""
    acc = (pred == y).mean()
    return acc
