from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from data import get_dataset_bundle, register_dataset

PROSTATE_PATH = Path("/home/utsab/Desktop/ICLFE/ICLFE/data/Prostate-GE.mat")

def load_prostate_raw(**_) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading Prostate-GE from {PROSTATE_PATH}...")
    mat = loadmat(PROSTATE_PATH)
    x = np.asarray(mat["X"], dtype=np.float32)
    y = np.asarray(mat["Y"]).reshape(-1)
    return x, y

register_dataset("prostate", load_prostate_raw)
