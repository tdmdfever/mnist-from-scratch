import argparse
import json
from pathlib import Path

import numpy as np


def export_weights(run_dir):
    """Export model weights to a JSON file."""
    with np.load(run_dir / "params.npz") as params:
        params_dict = {key: np.round(params[key], 5).tolist() for key in params}

    with open(run_dir / "params.json", "w") as f:
        json.dump(params_dict, f, separators=(",", ":"))


def parse_args():
    """Parse command line arguments for run path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    """Export model weights to a JSON file for the given run directory."""
    args = parse_args()
    run_dir = args.run
    export_weights(run_dir)


if __name__ == "__main__":
    main()
