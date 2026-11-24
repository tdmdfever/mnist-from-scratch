import argparse
from pathlib import Path
from datetime import datetime
import json
import subprocess
import numpy as np
import pandas as pd
from src.data import load_train, load_test, split_data, one_hot
from src import softmax, mlp
from src.softmax import cross_entropy, update, accuracy

MODELS = {"softmax": softmax, "mlp": mlp}
ROOT_DIR = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT_DIR / "runs"

def train(model, params, X_train, Y_train, X_val, y_val, lr=0.1, batch_size=128, n_epochs=10, seed=0):
    """Train the given model using mini-batch gradient descent.

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
            _, cache_batch = model.forward(X_batch, params)
            grads = model.backward(params, cache_batch, Y_batch)
            update(params, grads, lr)

        P, _ = model.forward(X_train, params)
        train_loss = cross_entropy(P, Y_train)
        train_pred = P.argmax(axis=1)
        train_acc = accuracy(train_pred, Y_train.argmax(axis=1))
        val_pred = model.predict(X_val, params)
        val_acc = accuracy(val_pred, y_val)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(f"epoch {epoch + 1:2d} | loss {train_loss:.4f} | train {train_acc:.4f} | val {val_acc:.4f}")

    return history

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), default="mlp")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.1)
    args = parser.parse_args()
    return args

def write_submission(pred, path):
    """Write the prediction to a CSV file, stored at 'path'"""
    submission = pd.DataFrame({"ImageId": np.arange(1, len(pred) + 1), "Label": pred})
    submission.to_csv(path, index=False)

def main():
    commit = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT_DIR, capture_output=True, text=True).stdout.strip()
    started = datetime.now()
    stamp = started.strftime("%Y-%m-%d_%H%M%S")

    args = parse_args()
    model_name = args.model
    model = MODELS[model_name]

    n_epochs = args.epochs
    lr = args.lr
    batch_size = 128

    data_shuffle_seed = 0
    init_params_seed = 0
    if model_name == "softmax":
        init_params_seed = None
    epochs_shuffle_seed = 0
    seeds = {"data_shuffle": data_shuffle_seed, "init_params": init_params_seed, "epoch_shuffle": epochs_shuffle_seed}

    X, y = load_train()
    X_test = load_test()
    _, _, X_train, X_val, y_train, y_val = split_data(X, y, seed=data_shuffle_seed)
    Y_train = one_hot(y_train)

    params = model.init_params(seed=init_params_seed)
    P, _ = model.forward(X_train, params)
    starting_loss = cross_entropy(P, Y_train)
    print(f"initial loss {starting_loss:.4f}")

    history = train(model, params, X_train, Y_train, X_val, y_val, lr=lr, batch_size=batch_size, n_epochs=n_epochs, seed=epochs_shuffle_seed)
    pred = model.predict(X_test, params)

    seconds = (datetime.now() - started).total_seconds()

    folder_path = RUN_DIR / f"{stamp}_{model_name}"
    folder_path.mkdir(parents=True, exist_ok=True)

    with open(folder_path / "config.json", "w") as cf:
        json.dump({"model": model_name, "epochs": n_epochs, "lr": lr, "batch_size": batch_size, "seeds": seeds, "commit": commit, "runtime_seconds": seconds}, cf, indent=2)
    with open(folder_path / "history.json", "w") as his:
        json.dump(history, his, indent=2)
    write_submission(pred, folder_path / "submission.csv")

if __name__ == "__main__":
    main()
