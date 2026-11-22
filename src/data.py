import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# __file__            -> .../nn-from-scratch/src/data.py
# .parent             -> .../nn-from-scratch/src
# .parent             -> .../nn-from-scratch
# / "data"            -> .../nn-from-scratch/data

# load training data, normalize, and split labels and features
def load_train():
    train_data = pd.read_csv(DATA_DIR / "train.csv")
    train_arr = train_data.to_numpy()
    y = train_arr[:, 0]
    X = train_arr[:, 1:]/255
    return X, y

# load test data and normalize
def load_test():
    test_data = pd.read_csv(DATA_DIR / "test.csv")
    test_arr = test_data.to_numpy()/255
    return test_arr

# split training data into training and validation sets
def split_data(X, y, n_val=5000, seed=0):
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

# convert labels to one-hot encoding
def one_hot(y):
    I = np.eye(10)
    return I[y.astype(int)]

if __name__ == "__main__":
    X, y = load_train()
    test_arr = load_test()
    train_idx, val_idx, X_train, X_val, y_train, y_val = split_data(X, y)
    Y_train = one_hot(y_train)
    Y_val = one_hot(y_val)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_val shape:", X_val.shape)
    print("y_val shape:", y_val.shape)
    print("Y_val shape:", Y_val.shape)
    print("test_arr shape:", test_arr.shape)