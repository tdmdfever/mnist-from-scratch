import numpy as np
import pytest

from src.data import load_train, one_hot
from src.mlp import init_params, forward, backward
from src.softmax import cross_entropy


@pytest.fixture(scope="module")
def small_batch():
    X, y = load_train()
    X_small_batch = X[:10]
    Y_small_batch = one_hot(y[:10])
    return X_small_batch, Y_small_batch


def test_backward_matches_finite_difference(small_batch):
    # Gradient check: for 5 random entries of each parameter, nudge the entry by +h and -h,
    # and measure the loss's slope directly: (L(w + h) - L(w - h)) / (2h).
    # Each slope must match backward()'s gradient for that entry (relative error < 1e-6).
    X, Y = small_batch
    params = init_params()
    _, cache = forward(X, params)
    grads = backward(params, cache, Y)

    rng = np.random.default_rng(0)
    h = 1e-5
    errors = []
    for key in params:
        for _ in range(5):
            idx = tuple(rng.integers(0, s) for s in params[key].shape)
            old_param = params[key][idx]
            grad = grads[key][idx]

            params[key][idx] = old_param + h
            P_plus, _ = forward(X, params)
            loss_plus = cross_entropy(P_plus, Y)
            params[key][idx] = old_param - h
            P_minus, _ = forward(X, params)
            loss_minus = cross_entropy(P_minus, Y)
            params[key][idx] = old_param  # restore before next entry

            slope = (loss_plus - loss_minus) / (2 * h)

            # Floor avoids 0/0 where both are exactly 0 (blank pixels or dead hidden units).
            denom = max(abs(grad) + abs(slope), 1e-12)
            rel_error = abs(grad - slope) / denom
            errors.append(rel_error)

    # Use np.max(), not max(); Python's max() skips nan, so a nan gradient would pass.
    assert np.max(errors) < 1e-6


def test_gradient_shapes_match_params(small_batch):
    X, Y = small_batch
    params = init_params()
    _, cache = forward(X, params)
    grads = backward(params, cache, Y)

    for key in params:
        assert grads[key].shape == params[key].shape


def test_he_initialization():
    params = init_params()
    W1 = params["W1"]
    assert np.isclose(W1.std(), np.sqrt(2 / 784), rtol=0.01)
