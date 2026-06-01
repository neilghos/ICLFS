from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from data import register_dataset


PATH = Path("/home/utsab/Desktop/ICLFE/ICLFE/data/arcene.mat")


def load_arcene_raw(**_) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading arcene from {PATH}...")
    mat = loadmat(PATH)
    x = np.asarray(mat["X"], dtype=np.float32)
    y = np.asarray(mat["Y"]).reshape(-1)
    return x, y


register_dataset("arcene", load_arcene_raw)
