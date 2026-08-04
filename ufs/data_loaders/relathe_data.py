from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from src.data import register_dataset


PATH = Path(__file__).resolve().parents[2] / "dataset" / "RELATHE.mat"


def load_relathe_raw(**_) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading RELATHE from {PATH}...")
    mat = loadmat(PATH)
    x = np.asarray(mat["X"], dtype=np.float32)
    y = np.asarray(mat["Y"]).reshape(-1)
    return x, y


register_dataset("relathe", load_relathe_raw)
# kbs
