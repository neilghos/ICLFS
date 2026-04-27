from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer

from data import get_dataset_bundle, register_dataset


def load_wisconsin_raw(**_) -> tuple[np.ndarray, np.ndarray]:
    data = load_breast_cancer()
    return data.data, data.target


register_dataset("wisconsin", load_wisconsin_raw)


def get_wisconsin_loaders(**kwargs):
    bundle = get_dataset_bundle("wisconsin", **kwargs)
    return bundle.train_loader, bundle.val_loader, bundle.test_loader, bundle.labels


def get_dataloaders(**kwargs):
    return get_wisconsin_loaders(**kwargs)


if __name__ == "__main__":
    bundle = get_dataset_bundle("wisconsin", batch_size=30, mask_prob=0.1)
    v1, v2 = next(iter(bundle.train_loader))
    print("--- INVERTED TRAINING (ICL EXPERT) ---")
    print(f"Feature-Batch shape: {v1.shape}")

    x_val = next(iter(bundle.val_loader))
    print("\n--- STANDARD EVALUATION ---")
    print(f"Patient-Batch shape: {x_val.shape}")
