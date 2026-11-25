import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
SWEEP_DIR = RESULTS_DIR / "lr_sweep"
FIG_DIR = ROOT / "figures"
SOFTMAX_VAL_ACC = 0.9068  # softmax baseline, README.md results table


def load_runs(commit, runs_dir):
    """Load all runs for the given commit from `runs_dir`.

    Return a dictionary of learning rates to their validation accuracy
    and training loss histories for the given commit.
    The returned dictionary is of the form `{lr: {"val": [], "loss": []}}`.
    """
    run_paths = sorted(runs_dir.glob("*_mlp"))
    runs = {}

    for path in run_paths:
        config_path = path / "config.json"
        with open(config_path) as f:
            config = json.load(f)
        if config["commit"] == commit:
            if config["lr"] in runs:
                dupe_lr = config["lr"]
                raise ValueError(f"multiple runs with lr={dupe_lr} found for commit {commit}")
            history_path = path / "history.json"
            with open(history_path) as f:
                history = json.load(f)
            runs[config["lr"]] = {"val": history["val_acc"], "loss": history["train_loss"]}

    if not runs:
        raise ValueError(f"no runs found for commit {commit}")

    runs = dict(sorted(runs.items()))
    return runs


def plot_accuracy(commit, runs, path, title):
    """Plot small multiples of validation accuracy for each learning rate and save to `path`."""
    n = len(runs)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    acc_fig, acc_axes = plt.subplots(rows, cols, figsize=(11, 3 * rows),
                                     squeeze=False, sharex=True, sharey=True)

    for acc_ax, lr in zip(acc_axes.flat, runs):
        epochs = np.arange(1, len(runs[lr]["val"]) + 1)
        for other in runs:
            acc_ax.plot(epochs, runs[other]["val"], color="lightgrey", linewidth=1)
        acc_ax.plot(epochs, runs[lr]["val"], color="tab:blue", linewidth=2)
        acc_ax.set_title(f"lr: {lr}")
        acc_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        acc_ax.grid(alpha=0.3)
        acc_ax.spines[["top", "right"]].set_visible(False)
        acc_ax.axhline(SOFTMAX_VAL_ACC, color="gray", linestyle="--", linewidth=1)
        acc_ax.text(7.5, SOFTMAX_VAL_ACC - 0.01, f"softmax accuracy ({SOFTMAX_VAL_ACC:.4f})",
                    va="top", fontsize=8, color="gray")

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
    """Plot training loss for each learning rate and save to `path`."""
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

    loss_ax.axhline(np.log(10), color="gray", linestyle="--", linewidth=1)
    loss_ax.text(1, np.log(10), " random guessing (ln 10)", va="bottom", fontsize=8, color="gray")

    loss_fig.tight_layout()
    loss_fig.savefig(path, dpi=150)
    plt.close(loss_fig)


def parse_args():
    """Parse command line arguments for the learning rate sweep plotter."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--runs-dir", type=Path, default=SWEEP_DIR)
    return parser.parse_args()


def main():
    """Plot the results of the learning rate sweep for the given commit and save to figures/."""
    args = parse_args()
    commit = args.commit
    runs_dir = args.runs_dir
    runs = load_runs(commit, runs_dir)
    acc_title = ("Above lr=0.3, training collapses mid-run and recovers; "
                 "final accuracies near-tie")
    loss_title = ("Higher learning rates lower loss faster;\n"
                  "at 0.5 and 1.0 it spikes back to about random guessing")
    plot_accuracy(commit, runs, FIG_DIR / "lr_sweep_accuracy.png", acc_title)
    plot_loss(commit, runs, FIG_DIR / "lr_sweep_loss.png", loss_title)


if __name__ == "__main__":
    main()
