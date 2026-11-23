import numpy as np
import pandas as pd
from src.data import DATA_DIR
from src.train import write_submission

def test_write_submission_format(tmp_path):
    # The sample submission labels all 28000 ImageIds as 0, so 28000 zero predictions should reproduce it exactly.
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    sub_path = tmp_path / "submission.csv"
    write_submission(np.zeros(28000, dtype=int), sub_path)
    assert pd.read_csv(sub_path).equals(sample), "submission should match sample submission formatting"

def test_write_submission_labels(tmp_path):
    sub_path = tmp_path / "submission.csv"
    pred = np.array([8, 2, 5, 7, 3])
    write_submission(pred, sub_path)
    df = pd.read_csv(sub_path)
    assert np.array_equal(df["ImageId"].to_numpy(), np.arange(1, 6)), "ImageId column should be 1-indexed and match the number of predictions"
    assert np.array_equal(df["Label"].to_numpy(), pred), "Label column should match passed predictions in same order"
