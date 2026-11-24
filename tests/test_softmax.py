import pytest
import numpy as np
from src.data import load_train, one_hot
from src.softmax import init_params, softmax, forward, cross_entropy, backward, update, predict, accuracy

@pytest.fixture(scope="module")
def small_batch():
    X, y = load_train()
    X_small_batch = X[:256]
    Y_small_batch = one_hot(y[:256])
    return X_small_batch, Y_small_batch

def test_softmax_rows_sum_to_one():
    Z1 = np.array([[1., 2., 3.],
                   [0., 0., 0.]])
    Z2 = np.array([[1000., 1000., 1001.]])
    P1 = softmax(Z1)
    P2 = softmax(Z2)
    assert np.allclose(P1.sum(axis=1), 1) and np.allclose(P2.sum(axis=1), 1), "softmax should sum to 1"

def test_softmax_non_negative():
    Z = np.array([[1., 2., 3.],
                  [0., 0., 0.]])
    P = softmax(Z)
    assert (P >= 0).all(), "softmax should be non-negative"

def test_softmax_overflow_guard():
    Z = np.array([[1000., 1000., 1001.]])
    P = softmax(Z)
    assert np.isfinite(P).all(), "softmax should have only finite values (no overflow/nan)"

def test_softmax_invariant_to_shifting_rows():
    Z = np.array([[1., 2., 3.],
                  [0., 0., 0.]])
    P1 = softmax(Z)
    P2 = softmax(Z + np.array([[-.5], [-.3]]))
    assert np.allclose(P1, P2), "softmax should not change after shifting rows"

def test_equal_scores_uniform():
    Z = np.array([[1., 1., 1.],
                  [1., 1., 1.]])
    P = softmax(Z)
    assert np.allclose(P, 1/3), "equal scores should give uniform softmax"

def test_uniform_prediction_loss_ln_ten():
    # Every class gets 0.1, so each row's loss is -log(0.1) = ln 10, whatever the label.
    # Same value as the first training loss at zero init.
    P = np.full((5, 10), 0.1)
    Y = one_hot(np.array([5, 7, 9, 1, 3]))
    assert np.isclose(cross_entropy(P, Y), np.log(10))

def test_gradient_shapes_match_params(small_batch):
    X, Y = small_batch
    params = init_params()
    P, cache = forward(X, params)
    grads = backward(params, cache, Y)

    for key in params:
        assert grads[key].shape == params[key].shape, "gradient shapes should match parameter shapes"

def test_loss_decreases(small_batch):
    X, Y = small_batch
    params = init_params()
    P, cache = forward(X, params)
    loss1 = cross_entropy(P, Y)
    grads = backward(params, cache, Y)
    update(params, grads)
    new_P, _ = forward(X, params)
    loss2 = cross_entropy(new_P, Y)
    assert loss2 < loss1, "loss should decrease after one update"

def test_predict():
    # By hand: X @ W = [30, 36, 42]; + b = [33, 45, 44] -> class 1 wins.
    X = np.array([[1., 2., 3.]])
    W = np.array([[1., 2., 3.],
                  [4., 5., 6.],
                  [7, 8., 9.]])
    b = np.array([[3., 9., 2.]])
    params = {'W': W, 'b': b}
    assert np.array_equal(predict(X, params), [1])

def test_accuracy():
    assert np.isclose(accuracy(np.array([1, 0]), np.array([1, 2])), 0.5)
