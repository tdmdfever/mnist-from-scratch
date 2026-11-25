import numpy as np
import pytest

from src.data import load_train, load_test, split_data, one_hot


@pytest.fixture(scope="module")
def train_data():
    return load_train()


@pytest.fixture(scope="module")
def kaggle_data():
    return load_test()


def test_scaling(train_data, kaggle_data):
    # Exact ==, not isclose: 255/255 is exactly 1.0. Catches scaling twice (max would be ~0.004).
    X, _ = train_data
    X_test = kaggle_data
    assert X.max() == 1.0 and X.min() == 0.0, "X should be scaled to [0, 1]"
    assert X_test.max() == 1.0 and X_test.min() == 0.0, "test data should be scaled to [0, 1]"


def test_shape(train_data, kaggle_data):
    X, y = train_data
    X_test = kaggle_data
    assert X.shape == (42000, 784)
    assert y.shape == (42000,)
    assert X_test.shape == (28000, 784)


def test_split_sizes(train_data):
    X, y = train_data
    _, _, X_train, X_val, y_train, y_val = split_data(X, y)

    assert len(X_train) == len(y_train) == 37000, "X_train and y_train should have 37000 samples"
    assert len(X_val) == len(y_val) == 5000, "X_val and y_val should have 5000 samples"


def test_split_span_with_no_overlap(train_data):
    X, y = train_data
    train_idx, val_idx, _, _, _, _ = split_data(X, y)
    assert np.intersect1d(train_idx, val_idx).size == 0
    assert np.array_equal(np.union1d(train_idx, val_idx), np.arange(len(X)))


def test_every_digit_in_both(train_data):
    X, y = train_data
    _, _, _, _, y_train, y_val = split_data(X, y)
    assert np.unique(y_train).size == 10, "Training set should contain all digits 0-9"
    assert np.unique(y_val).size == 10, "Validation set should contain all digits 0-9"


def test_split_reproducibility(train_data):
    X, y = train_data
    names = ["train_idx", "val_idx", "X_train", "X_val", "y_train", "y_val"]
    first = split_data(X, y, seed=42)
    again = split_data(X, y, seed=42)
    other = split_data(X, y, seed=0)

    for name, a, b, c in zip(names, first, again, other):
        assert np.array_equal(a, b), f"{name} should be the same for the same seed"
        assert not np.array_equal(a, c), f"{name} should be different for different seeds"


def test_one_hot_encoding(train_data):
    X, y = train_data
    _, _, _, _, y_train, _ = split_data(X, y)
    Y_train = one_hot(y_train)

    assert np.all(np.sum(Y_train, axis=1) == 1), (
        "Each one-hot encoded label should have exactly one '1'"
    )
    assert np.array_equal(Y_train.argmax(axis=1), y_train), (
        "The 1 should be at each label's index"
    )
    assert Y_train.shape == (len(y_train), 10), (
        "One-hot encoded labels should have the same number of rows as y_train and 10 columns"
    )
