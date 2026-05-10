from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from data import get_dataset_bundle, register_dataset


YALE_PATH = Path("/home/utsab/Desktop/ICLFE/ICLFE/data/Yale.mat")


def load_yale_raw(**_) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading Yale from {YALE_PATH}...")
    mat = loadmat(YALE_PATH)
    x = np.asarray(mat["X"], dtype=np.float32)
    y = np.asarray(mat["Y"]).reshape(-1)
    return x, y


register_dataset("yale", load_yale_raw)


def get_yale_loaders(**kwargs):
    bundle = get_dataset_bundle("yale", **kwargs)
    return bundle.train_loader, bundle.labels
