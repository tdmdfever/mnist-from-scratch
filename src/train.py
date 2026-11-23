import numpy as np
import pandas as pd
from pathlib import Path
from src.data import load_train, load_test, split_data, one_hot
from src.softmax import init_params, forward, cross_entropy, gradients, update, predict, accuracy

SUB_DIR = Path(__file__).resolve().parent.parent / "submissions"

def train(params, X_train, Y_train, X_val, y_val, lr=0.1, batch_size=128, n_epochs=10, seed=0):
    """Train the softmax classifier using mini-batch gradient descent.

    Return a history dict with train_loss, train_acc, val_acc for each epoch, while params is trained in place.
    """
    rng = np.random.default_rng(seed)
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(n_epochs):
        order = rng.permutation(len(X_train))

        for start in range(0, len(X_train), batch_size):
            batch_idx = order[start:start + batch_size]
            X_batch = X_train[batch_idx]
            Y_batch = Y_train[batch_idx]
            P_batch = forward(X_batch, params)
            grads = gradients(X_batch, P_batch, Y_batch)
            update(params, grads, lr)

        P = forward(X_train, params)
        train_loss = cross_entropy(P, Y_train)
        train_pred = P.argmax(axis=1)
        train_acc = accuracy(train_pred, Y_train.argmax(axis=1))
        val_pred = predict(X_val, params)
        val_acc = accuracy(val_pred, y_val)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch + 1:2d} | loss {train_loss:.4f} | train {train_acc:.4f} | val {val_acc:.4f}")

    return history

def write_submission(pred, path):
    """Write the prediction to a CSV file, stored at 'path'"""
    submission = pd.DataFrame({"ImageId": np.arange(1, len(pred) + 1), "Label": pred})
    submission.to_csv(path, index=False)

def main():
    X, y = load_train()
    X_test = load_test()
    _, _, X_train, X_val, y_train, y_val = split_data(X, y)
    Y_train = one_hot(y_train)
    params = init_params()
    P = forward(X_train, params)
    starting_loss = cross_entropy(P, Y_train)
    print(f"initial loss {starting_loss:.4f}")

    history = train(params, X_train, Y_train, X_val, y_val)

    pred = predict(X_test, params)

    SUB_DIR.mkdir(exist_ok=True)
    write_submission(pred, SUB_DIR / "softmax.csv")

if __name__ == "__main__":
    main()
