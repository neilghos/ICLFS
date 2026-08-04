from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.api import InvertedFeatureDataset
from src.runtime_config import get_augmentor_config


def _load_gradenfs_data_module():
    module_name = "sfs_gradenfs_data_loading_util"
    if module_name in sys.modules:
        return sys.modules[module_name]

    loader_path = Path(__file__).resolve().parent / "data_loading_util.py"
    spec = importlib.util.spec_from_file_location(module_name, loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load GradEnFS data loader from {loader_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _to_class_indices(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim == 2:
        return np.argmax(y, axis=1).astype(np.int64)
    return y.reshape(-1).astype(np.int64)


class SampleDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


@dataclass
class SFSSplitBundle:
    dataset_name: str
    x_train: np.ndarray
    y_train_onehot: np.ndarray
    y_train_index: np.ndarray
    x_valid: np.ndarray
    y_valid_onehot: np.ndarray
    y_valid_index: np.ndarray
    x_test: np.ndarray
    y_test_onehot: np.ndarray
    y_test_index: np.ndarray

    @property
    def input_dim(self) -> int:
        return int(self.x_train.shape[1])

    @property
    def output_dim(self) -> int:
        return int(self.y_train_onehot.shape[1] if self.y_train_onehot.ndim == 2 else np.unique(self.y_train_index).shape[0])

    def build_inverted_train_loader(
        self,
        *,
        config_path: str | None = None,
    ) -> DataLoader:
        train_ds = InvertedFeatureDataset(
            self.x_train,
            augmentor_config=get_augmentor_config(config_path),
        )
        return DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)

    def build_sample_loaders(
        self,
        *,
        training_batch_size: int = 100,
        evaluating_batch_size: int = 10_000,
        use_onehot_labels: bool = True,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        if use_onehot_labels:
            y_train = self.y_train_onehot
            y_valid = self.y_valid_onehot
            y_test = self.y_test_onehot
        else:
            y_train = self.y_train_index
            y_valid = self.y_valid_index
            y_test = self.y_test_index
        train_loader = DataLoader(SampleDataset(self.x_train, y_train), batch_size=training_batch_size, shuffle=False)
        valid_loader = DataLoader(SampleDataset(self.x_valid, y_valid), batch_size=evaluating_batch_size, shuffle=False)
        test_loader = DataLoader(SampleDataset(self.x_test, y_test), batch_size=evaluating_batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader


def load_gradenfs_splits(dataset_name: str, *, args=None, seed: int | None = None) -> SFSSplitBundle:
    module = _load_gradenfs_data_module()
    if args is None:
        class _Args:
            pass
        args = _Args()
    args.dataset = dataset_name.lower()
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    x_train, y_train, x_valid, y_valid, x_test, y_test = module.get_dataset(args)
    x_train = np.asarray(x_train, dtype=np.float32)
    x_valid = np.asarray(x_valid, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_train = np.asarray(y_train)
    y_valid = np.asarray(y_valid)
    y_test = np.asarray(y_test)

    return SFSSplitBundle(
        dataset_name=args.dataset,
        x_train=x_train,
        y_train_onehot=y_train,
        y_train_index=_to_class_indices(y_train),
        x_valid=x_valid,
        y_valid_onehot=y_valid,
        y_valid_index=_to_class_indices(y_valid),
        x_test=x_test,
        y_test_onehot=y_test,
        y_test_index=_to_class_indices(y_test),
    )
