from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from data import get_dataset_bundle, register_dataset

TOX171_PATH = Path("/home/utsab/Desktop/ICLFE/ICLFE/data/TOX-171.mat")

def load_tox171_raw(**_) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading TOX-171 from {TOX171_PATH}...")
    mat = loadmat(TOX171_PATH)
    x = np.asarray(mat["X"], dtype=np.float32)
    y = np.asarray(mat["Y"]).reshape(-1)
    return x, y

register_dataset("tox171", load_tox171_raw)
