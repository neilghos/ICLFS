from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from data import register_dataset

PATH = Path("/home/utsab/Desktop/ICLFE/ICLFE/data/pixraw10P.mat")

def load_pixraw10p_raw(**_) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading PIXRAW10P from {PATH}...")
    mat = loadmat(PATH)
    x = np.asarray(mat["X"], dtype=np.float32)
    y = np.asarray(mat["Y"]).reshape(-1)
    return x, y

register_dataset("pixraw10p", load_pixraw10p_raw)
