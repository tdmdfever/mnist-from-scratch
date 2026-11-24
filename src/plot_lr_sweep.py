import argparse
from pathlib import Path
import json
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "runs"
FIG_DIR = ROOT / "figures"

def load_runs(commit):
    run_paths = list(RUN_DIR.glob("*_mlp"))
    runs = {}

    for path in run_paths:
        config_path = path / "config.json"
        with open(config_path, "r") as cf:
            config = json.load(cf)
        if config["commit"] == commit:
            if config["lr"] in runs:
                dupe_lr = config["lr"]
                raise ValueError(f"multiple runs of learning rate {dupe_lr} found for commit {commit}")
            history_path = path / "history.json"
            with open(history_path, "r") as his:
                history = json.load(his)
            runs[config["lr"]] = {"val": history["val_acc"], "loss": history["train_loss"]}

    if not runs:
        raise ValueError(f"no runs found for commit {commit}")

    runs = dict(sorted(runs.items()))
    return runs

def plot_accuracy(commit, runs, path, title):
    n = len(runs)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    acc_fig, acc_axes = plt.subplots(rows, cols, figsize=(11, 3 * rows), squeeze=False, sharex=True, sharey=True)

    for acc_ax, lr in zip(acc_axes.flat, runs):
        epochs = np.arange(1, len(runs[lr]["val"]) + 1)
        for other in runs:
            acc_ax.plot(epochs, runs[other]["val"], color="lightgrey", linewidth=1)
        acc_ax.plot(epochs, runs[lr]["val"], color="tab:blue", linewidth=2)
        acc_ax.set_title(f"lr: {lr}")
        acc_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        acc_ax.grid(alpha=0.3)
        acc_ax.spines[["top", "right"]].set_visible(False)

    for empty_ax in acc_axes.flat[n:]:
        empty_ax.set_visible(False)

    acc_fig.supxlabel("Epoch")
    acc_fig.supylabel("Validation accuracy")
    acc_fig.suptitle(title)
    acc_fig.text(0.99, 0.01, f"runs @ {commit}", ha="right", fontsize=8, color="gray")

    acc_fig.tight_layout()
    acc_fig.savefig(path, dpi=150)
    plt.close(acc_fig)

def plot_loss(commit, runs, path, title):
    colors = plt.cm.Blues(np.linspace(0.45, 1.0, len(runs)))
    loss_fig, loss_ax = plt.subplots()
    for lr, color in zip(runs, colors):
        epochs = np.arange(1, len(runs[lr]["loss"]) + 1)
        loss_ax.plot(epochs, runs[lr]["loss"], color=color, linewidth=2, label=f"lr = {lr}")
        loss_ax.annotate(f"lr {lr}", xy=(epochs[-1], runs[lr]["loss"][-1]),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)

    loss_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    loss_ax.grid(alpha=0.3)
    loss_ax.spines[["top", "right"]].set_visible(False)
    loss_ax.legend(frameon=False)

    loss_ax.set(xlabel="Epoch", ylabel="Training loss (log scale)", yscale="log")
    loss_fig.suptitle(title)
    loss_fig.text(0.99, 0.01, f"runs @ {commit}", ha="right", fontsize=8, color="gray")

    loss_fig.tight_layout()
    loss_fig.savefig(path, dpi=150)
    plt.close(loss_fig)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    commit = args.commit
    runs = load_runs(commit)
    plot_accuracy(commit, runs, FIG_DIR / "lr_sweep_accuracy.png", "")
    plot_loss(commit, runs, FIG_DIR / "lr_sweep_loss.png", "")

if __name__ == "__main__":
    main()
