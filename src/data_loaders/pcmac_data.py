from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from data import register_dataset


PATH = Path(__file__).resolve().parents[2] / "dataset" / "PCMAC.mat"


def load_pcmac_raw(**_) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading PCMAC from {PATH}...")
    mat = loadmat(PATH)
    x = np.asarray(mat["X"], dtype=np.float32)
    y = np.asarray(mat["Y"]).reshape(-1)
    return x, y


register_dataset("pcmac", load_pcmac_raw)
# kbs
