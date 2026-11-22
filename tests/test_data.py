import pytest
import numpy as np

from src.data import load_train, split_data, one_hot

@pytest.fixture(scope="module")
def train_data():
    return load_train()

def test_scaling(train_data):
    X, y = train_data
    assert X.max() <= 1.0 and X.min() >= 0.0, "X should be scaled to [0, 1]"

def test_split_sizes(train_data):
    X, y = train_data
    train_idx, val_idx, X_train, X_val, y_train, y_val = split_data(X, y)

    assert len(X_train) == len(y_train) == 37000, "X_train and y_train should both have 37000 samples"
    assert len(X_val) == len(y_val) == 5000, "X_val and y_val should both have 5000 samples"

def test_split_no_overlap(train_data):
    X, y = train_data
    train_idx, val_idx, X_train, X_val, y_train, y_val = split_data(X, y)
    assert np.intersect1d(train_idx, val_idx).size == 0, "Training and validation sets should not overlap"
    assert np.array_equal(np.union1d(train_idx, val_idx), np.arange(len(X))), "Training and validation indices should together span all data"

def test_every_digit_in_both(train_data):
    X, y = train_data
    train_idx, val_idx, X_train, X_val, y_train, y_val = split_data(X, y)
    assert np.unique(y_train).size == 10, "Training set should contain all digits 0-9"
    assert np.unique(y_val).size == 10, "Validation set should contain all digits 0-9"

def test_split_reproducibility(train_data):
    X, y = train_data
    train_idx1, val_idx1, X_train1, X_val1, y_train1, y_val1 = split_data(X, y, seed=42)
    train_idx2, val_idx2, X_train2, X_val2, y_train2, y_val2 = split_data(X, y, seed=42)

    assert np.array_equal(X_train1, X_train2), "X_train should be the same for the same seed"
    assert np.array_equal(y_train1, y_train2), "y_train should be the same for the same seed"
    assert np.array_equal(X_val1, X_val2), "X_val should be the same for the same seed"
    assert np.array_equal(y_val1, y_val2), "y_val should be the same for the same seed"

def test_one_hot_encoding(train_data):
    X, y = train_data
    _, _, _, _, y_train, _ = split_data(X, y)
    Y_train = one_hot(y_train)

    assert Y_train.shape[1] == 10, "One-hot encoded labels should have 10 columns"
    assert np.all(np.sum(Y_train, axis=1) == 1), "Each one-hot encoded label should have exactly one '1'"
    assert np.array_equal(Y_train.argmax(axis=1), y_train), "The 1 should be at each label's index"
    assert Y_train.shape == (len(y_train), 10), "One-hot encoded labels should have the same number of rows as y_train"