import json

import numpy as np

from src.export_weights import export_weights
from src.mlp import init_params


def test_export_weights(tmp_path):
    # Check that after rounding to five decimal places and exporting,
    # every number moves by at most 5e-6
    params = init_params()
    np.savez(tmp_path / "params.npz", **params)
    export_weights(tmp_path)
    with open(tmp_path / "params.json") as f:
        params_rounded = json.load(f)

    for key in params:
        assert np.allclose(params[key], params_rounded[key], atol=5e-6, rtol=0)
