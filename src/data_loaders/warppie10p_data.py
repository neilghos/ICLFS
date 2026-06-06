from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from data import register_dataset


PATH = Path(__file__).resolve().parents[2] / "data" / "warpPIE10P.mat"


def load_warppie10p_raw(**_) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading warpPIE10P from {PATH}...")
    mat = loadmat(PATH)
    x = np.asarray(mat["X"], dtype=np.float32)
    y = np.asarray(mat["Y"]).reshape(-1)
    return x, y


register_dataset("warppie10p", load_warppie10p_raw)
