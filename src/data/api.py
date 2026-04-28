from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from augmentor import build_augmentor
from runtime_config import get_augmentor_config


DatasetLoader = Callable[..., tuple[np.ndarray, np.ndarray]]
DATASET_REGISTRY: Dict[str, DatasetLoader] = {}


class InvertedFeatureDataset(Dataset):
    """
    Training dataset where each item is a feature profile across patients.
    """

    def __init__(
        self,
        x: np.ndarray,
        mask_prob: float = 0.5,
        augmentor_strategy: str = "random_mask",
        augmentor_config: dict | None = None,
    ):
        self.x = torch.from_numpy(x.T).float()
        self.n_patients = self.x.shape[1]
        self.mask_prob = mask_prob
        self.augmentor = build_augmentor(
            n_patients=self.n_patients,
            mask_prob=mask_prob,
            strategy=augmentor_strategy,
            config=augmentor_config,
        )

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        feat_vector = self.x[idx]
        return self.augmentor(feat_vector)


class PatientDataset(Dataset):
    """
    Evaluation dataset where each item is a standard patient/sample vector.
    """

    def __init__(self, x: np.ndarray):
        self.x = torch.from_numpy(x).float()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx]


@dataclass
class DatasetBundle:
    name: str
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    labels: tuple[np.ndarray, np.ndarray, np.ndarray]
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    input_dim: int
    num_features: int
    num_train_samples: int


def register_dataset(name: str, loader_fn: DatasetLoader) -> None:
    DATASET_REGISTRY[name] = loader_fn


def _normalize_labels(y: np.ndarray) -> np.ndarray:
    if len(y) == 0:
        return y
    y = np.asarray(y)
    if isinstance(y[0], (bytes, str, np.bytes_, np.str_)):
        y = y.astype(str).astype(int)
    if np.min(y) < 0:
        y = np.where(y <= 0, 0, 1)
    if np.min(y) == 1:
        y = y - 1
    return y.astype(int)


def _to_dense_float32(x):
    if sparse.issparse(x):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def build_dataset_bundle(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 128,
    random_state: int = 42,
    val_size: float = 0.1,
    test_size: float = 0.1,
    mask_prob: float = 0.15,
    augmentor_strategy: str | None = None,
    config_path: str | None = None,
) -> DatasetBundle:
    """
    Shared split + scale + dataloader pipeline for all datasets.
    """

    y = _normalize_labels(y)

    held_out_size = val_size + test_size
    if held_out_size <= 0 or held_out_size >= 1:
        raise ValueError("val_size + test_size must be between 0 and 1.")

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=held_out_size,
        random_state=random_state,
        stratify=y,
    )

    test_fraction = test_size / held_out_size
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=test_fraction,
        random_state=random_state,
        stratify=y_temp,
    )

    scaler = StandardScaler(with_mean=not sparse.issparse(x_train))
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    x_train = _to_dense_float32(x_train)
    x_val = _to_dense_float32(x_val)
    x_test = _to_dense_float32(x_test)

    augmentor_config = get_augmentor_config(config_path)
    if augmentor_strategy is None:
        augmentor_strategy = augmentor_config.get("strategy", "four_view_mask")

    train_ds = InvertedFeatureDataset(
        x_train,
        mask_prob=mask_prob,
        augmentor_strategy=augmentor_strategy,
        augmentor_config=augmentor_config,
    )
    val_ds = PatientDataset(x_val)
    test_ds = PatientDataset(x_test)

    train_loader = DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return DatasetBundle(
        name=name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        labels=(y_train, y_val, y_test),
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        input_dim=x_train.shape[0],
        num_features=x_train.shape[1],
        num_train_samples=x_train.shape[0],
    )


def get_dataset_bundle(name: str, **kwargs) -> DatasetBundle:
    if name not in DATASET_REGISTRY:
        try:
            import data_loaders  # noqa: F401
        except ImportError:
            pass

    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY)) or "<none>"
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {available}")

    x, y = DATASET_REGISTRY[name](**kwargs)
    bundle_kwargs = {
        key: kwargs[key]
        for key in (
            "batch_size",
            "random_state",
            "val_size",
            "test_size",
            "mask_prob",
            "augmentor_strategy",
            "config_path",
        )
        if key in kwargs
    }
    return build_dataset_bundle(name, x, y, **bundle_kwargs)


def get_dataloaders(name: str = "madelon", **kwargs):
    bundle = get_dataset_bundle(name, **kwargs)
    return bundle.train_loader, bundle.val_loader, bundle.test_loader, bundle.labels


def get_split_arrays(name: str, **kwargs) -> DatasetBundle:
    return get_dataset_bundle(name, **kwargs)
