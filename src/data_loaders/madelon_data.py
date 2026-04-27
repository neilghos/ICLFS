from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

from data import get_dataset_bundle, register_dataset


DEFAULT_MADELON_PATH = Path("/home/utsab/Desktop/ICL/phpfLuQE4.arff")


def load_madelon_raw(data_path: str | Path = DEFAULT_MADELON_PATH, **_) -> tuple[np.ndarray, np.ndarray]:
    path = Path(data_path)
    print(f"Loading Madelon from {path}...")
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    x = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values
    return x, y


register_dataset("madelon", load_madelon_raw)


def get_madelon_loaders(**kwargs):
    bundle = get_dataset_bundle("madelon", **kwargs)
    return bundle.train_loader, bundle.val_loader, bundle.test_loader, bundle.labels


if __name__ == "__main__":
    bundle = get_dataset_bundle("madelon")
    v1, v2 = next(iter(bundle.train_loader))
    print(f"Madelon Inverted Feature Batch: {v1.shape}")

    x_val = next(iter(bundle.val_loader))
    print(f"Madelon Standard Patient Batch: {x_val.shape}")
