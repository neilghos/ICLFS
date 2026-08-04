from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from src.data import get_dataset_bundle, register_dataset


COIL20_PATH = Path(__file__).resolve().parents[2] / "dataset" / "COIL20.mat"


def load_coil20_raw(**_) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading COIL20 from {COIL20_PATH}...")
    mat = loadmat(COIL20_PATH)
    x = np.asarray(mat["X"], dtype=np.float32)
    y = np.asarray(mat["Y"]).reshape(-1)
    return x, y


register_dataset("coil20", load_coil20_raw)


def get_coil20_loaders(**kwargs):
    bundle = get_dataset_bundle("coil20", **kwargs)
    return bundle.train_loader, bundle.labels
# kbs
