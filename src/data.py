from pathlib import Path

import numpy as np
import pandas as pd

# Absolute path to data/ so loading works from any folder
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_train():
    """Load training data from CSV and return X (42000, 784) scaled to [0, 1] and y (42000,)."""
    train_data = pd.read_csv(DATA_DIR / "train.csv")
    train_arr = train_data.to_numpy()
    y = train_arr[:, 0]
    X = train_arr[:, 1:] / 255
    return X, y


def load_test():
    """Load test data from CSV and return X_test (28000, 784) scaled to [0, 1] in file order."""
    test_data = pd.read_csv(DATA_DIR / "test.csv")
    X_test = test_data.to_numpy() / 255
    return X_test


def split_data(X, y, n_val=5000, seed=0):
    """Shuffle once with `seed`, then split.

    Return train_idx, val_idx, X_train, X_val, y_train, y_val.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X_shuffled = X[perm]
    y_shuffled = y[perm]

    train_size = len(X) - n_val
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]
    X_train = X_shuffled[:train_size]
    X_val = X_shuffled[train_size:]
    y_train = y_shuffled[:train_size]
    y_val = y_shuffled[train_size:]

    return train_idx, val_idx, X_train, X_val, y_train, y_val


def one_hot(y):
    """Convert a 1D array of labels (0-9) to one-hot encoded 2D array (m, 10)."""
    identity = np.eye(10)
    return identity[y.astype(int)]
