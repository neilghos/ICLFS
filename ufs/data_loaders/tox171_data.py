from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from src.data import register_dataset


PATH = Path(__file__).resolve().parents[2] / "dataset" / "TOX_171.mat"


def load_tox171_raw(**_) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading tox171 from {PATH}...")
    mat = loadmat(PATH)
    x = np.asarray(mat["X"], dtype=np.float32)
    y = np.asarray(mat["Y"]).reshape(-1)
    return x, y


register_dataset("tox171", load_tox171_raw)
# kbs
