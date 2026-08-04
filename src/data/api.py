from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.augmentor import build_augmentor
from src.runtime_config import get_augmentor_config


DatasetLoader = Callable[..., tuple[np.ndarray, np.ndarray]]
DATASET_REGISTRY: Dict[str, DatasetLoader] = {}


class InvertedFeatureDataset(Dataset):

    def __init__(
        self,
        x: np.ndarray,
        augmentor_config: dict | None = None,
    ):
        # Convert sample-major input (n_samples, n_features) into feature-major
        # tensors so each row is a feature profile over all samples.
        self.x = torch.from_numpy(x.T).float()
        self.n_patients = self.x.shape[1]
        self.augmentor = build_augmentor(
            n_patients=self.n_patients,
            strategy="four_view_mask",
            config=augmentor_config,
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.augmentor(self.x[idx])


@dataclass
class DatasetBundle:
    name: str
    train_loader: DataLoader
    labels: np.ndarray
    x: np.ndarray
    y: np.ndarray
    input_dim: int
    num_features: int
    num_samples: int


def register_dataset(name: str, loader_fn: DatasetLoader) -> None:
    DATASET_REGISTRY[name] = loader_fn


def _normalize_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    if y.size == 0:
        return y.astype(int)
    if y.dtype.kind in {"U", "S", "O"}:
        y = np.asarray(y).astype(str)
        _, y = np.unique(y, return_inverse=True)
        return y.astype(int)
    y = y.astype(int)
    if np.min(y) == 1:
        y = y - 1
    return y


def build_dataset_bundle(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    config_path: str | None = None,
) -> DatasetBundle:
    # Standardize the full dataset before constructing feature-wise instances.
    x = StandardScaler().fit_transform(np.asarray(x, dtype=np.float32)).astype(np.float32)
    y = _normalize_labels(y)

    augmentor_config = get_augmentor_config(config_path)

    train_ds = InvertedFeatureDataset(
        x,
        augmentor_config=augmentor_config,
    )
    # The full inverted feature set is processed jointly as one batch.
    train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)

    return DatasetBundle(
        name=name,
        train_loader=train_loader,
        labels=y,
        x=x,
        y=y,
        input_dim=x.shape[0],
        num_features=x.shape[1],
        num_samples=x.shape[0],
    )


def get_dataset_bundle(name: str, **kwargs) -> DatasetBundle:
    if name not in DATASET_REGISTRY:
        import ufs.data_loaders

    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")

    x, y = DATASET_REGISTRY[name](**kwargs)
    bundle_kwargs = {
        key: kwargs[key]
        for key in ("config_path",)
        if key in kwargs
    }
    return build_dataset_bundle(name, x, y, **bundle_kwargs)


def get_dataloaders(name: str, **kwargs):
    bundle = get_dataset_bundle(name, **kwargs)
    return bundle.train_loader, bundle.labels
