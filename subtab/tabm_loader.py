from __future__ import annotations

import enum
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn.preprocessing

PartKey = str


class NumPolicy(enum.Enum):
    STANDARD = 'standard'
    NOISY_QUANTILE = 'noisy-quantile'


class CatPolicy(enum.Enum):
    ORDINAL = 'ordinal'
    ONE_HOT = 'one-hot'


@dataclass(frozen=True, kw_only=True)
class RegressionLabelStats:
    mean: float
    std: float


@dataclass
class TabMPreparedDataset:
    x_train: np.ndarray
    x_valid: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    metadata: dict[str, Any]


def _load_part_arrays(path: Path, key: str) -> dict[PartKey, np.ndarray] | None:
    train_path = path / f'{key}_train.npy'
    if not train_path.exists():
        return None
    return {
        part: np.load(path / f'{key}_{part}.npy', allow_pickle=True)
        for part in ['train', 'val', 'test']
    }


def transform_num(
    x_num: dict[PartKey, np.ndarray],
    policy: None | str,
    seed: int,
) -> dict[PartKey, np.ndarray]:
    if policy is not None:
        policy = NumPolicy(policy)
        x_num_train = x_num['train']
        if policy == NumPolicy.STANDARD:
            normalizer = sklearn.preprocessing.StandardScaler()
        elif policy == NumPolicy.NOISY_QUANTILE:
            normalizer = sklearn.preprocessing.QuantileTransformer(
                n_quantiles=max(min(x_num_train.shape[0] // 30, 1000), 10),
                output_distribution='normal',
                subsample=1_000_000_000,
                random_state=seed,
            )
            x_num_train = x_num_train + np.random.RandomState(seed).normal(
                0.0, 1e-5, x_num_train.shape
            ).astype(x_num_train.dtype)
        else:
            raise ValueError(f'Unknown numeric policy: {policy}')
        normalizer.fit(x_num_train)
        x_num = {k: normalizer.transform(v) for k, v in x_num.items()}

    x_num = {k: np.nan_to_num(v) for k, v in x_num.items()}
    mask = np.array([len(np.unique(col)) > 1 for col in x_num['train'].T])
    x_num = {k: v[:, mask] for k, v in x_num.items()}
    return {k: v.astype(np.float32) for k, v in x_num.items()}


def transform_cat(
    x_cat: dict[PartKey, np.ndarray],
    policy: None | str,
) -> dict[PartKey, np.ndarray]:
    if policy is None:
        return x_cat

    policy = CatPolicy(policy)
    unknown_value = np.iinfo('int64').max - 3
    encoder = sklearn.preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=unknown_value,
        dtype='int64',
    ).fit(x_cat['train'])
    x_cat = {k: encoder.transform(v) for k, v in x_cat.items()}
    max_values = x_cat['train'].max(axis=0)
    for part in ['val', 'test']:
        for column_idx in range(x_cat[part].shape[1]):
            mask = x_cat[part][:, column_idx] == unknown_value
            x_cat[part][mask, column_idx] = max_values[column_idx] + 1

    if policy == CatPolicy.ORDINAL:
        return x_cat
    if policy == CatPolicy.ONE_HOT:
        encoder = sklearn.preprocessing.OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False,
            dtype=np.float32,
        )
        encoder.fit(x_cat['train'])
        return {k: encoder.transform(v) for k, v in x_cat.items()}
    raise ValueError(f'Unknown categorical policy: {policy}')


def standardize_labels(
    y: dict[PartKey, np.ndarray],
) -> tuple[dict[PartKey, np.ndarray], RegressionLabelStats]:
    y_train = y['train'].astype(np.float32, copy=False)
    mean = float(y_train.mean())
    std = float(y_train.std())
    if std == 0.0:
        std = 1.0
    y_scaled = {k: ((v.astype(np.float32, copy=False) - mean) / std).astype(np.float32) for k, v in y.items()}
    return y_scaled, RegressionLabelStats(mean=mean, std=std)


def _concat_available_blocks(blocks: list[dict[PartKey, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.concatenate([b['train'] for b in blocks], axis=1)
    valid = np.concatenate([b['val'] for b in blocks], axis=1)
    test = np.concatenate([b['test'] for b in blocks], axis=1)
    return train.astype(np.float32), valid.astype(np.float32), test.astype(np.float32)


def load_tabm_prepared_dataset(
    dataset_dir: str | Path,
    *,
    task: str,
    seed: int,
    num_policy: str | None = 'standard',
    cat_policy: str | None = 'one-hot',
    standardize_regression_labels_flag: bool = True,
) -> TabMPreparedDataset:
    dataset_dir = Path(dataset_dir).resolve()
    x_num = _load_part_arrays(dataset_dir, 'X_num')
    x_bin = _load_part_arrays(dataset_dir, 'X_bin')
    x_cat = _load_part_arrays(dataset_dir, 'X_cat')
    y = _load_part_arrays(dataset_dir, 'Y')
    if y is None:
        raise FileNotFoundError(f'Missing labels in {dataset_dir}')

    blocks: list[dict[PartKey, np.ndarray]] = []
    output_feature_names: list[str] = []

    if x_num is not None:
        x_num = transform_num(x_num, num_policy, seed)
        blocks.append(x_num)
        output_feature_names.extend([f'x_num_{i}' for i in range(x_num['train'].shape[1])])

    if x_bin is not None:
        x_bin = {k: np.asarray(v, dtype=np.float32) for k, v in x_bin.items()}
        mask = np.array([len(np.unique(col)) > 1 for col in x_bin['train'].T])
        x_bin = {k: v[:, mask] for k, v in x_bin.items()}
        blocks.append(x_bin)
        output_feature_names.extend([f'x_bin_{i}' for i in range(x_bin['train'].shape[1])])

    if x_cat is not None:
        x_cat = transform_cat(x_cat, cat_policy)
        x_cat = {k: np.asarray(v, dtype=np.float32) for k, v in x_cat.items()}
        blocks.append(x_cat)
        output_feature_names.extend([f'x_cat_{i}' for i in range(x_cat['train'].shape[1])])

    if not blocks:
        raise ValueError(f'No usable feature blocks found in {dataset_dir}')

    x_train, x_valid, x_test = _concat_available_blocks(blocks)

    y = {k: np.asarray(v).reshape(-1) for k, v in y.items()}
    label_stats = None
    if task == 'classification':
        y_train = y['train'].astype(np.int64)
        y_valid = y['val'].astype(np.int64)
        y_test = y['test'].astype(np.int64)
    else:
        y = {k: v.astype(np.float32) for k, v in y.items()}
        if standardize_regression_labels_flag:
            y, label_stats = standardize_labels(y)
        y_train = y['train']
        y_valid = y['val']
        y_test = y['test']

    info_path = dataset_dir / 'info.json'
    info = json.loads(info_path.read_text(encoding='utf-8')) if info_path.exists() else {}
    metadata: dict[str, Any] = {
        'tabm_dataset_dir': str(dataset_dir),
        'tabm_info': info,
        'num_train': int(x_train.shape[0]),
        'num_valid': int(x_valid.shape[0]),
        'num_test': int(x_test.shape[0]),
        'input_dim': int(x_train.shape[1]),
        'output_feature_names': output_feature_names,
        'num_policy': num_policy,
        'cat_policy': cat_policy,
    }
    if label_stats is not None:
        metadata['label_standardization'] = {
            'mean': label_stats.mean,
            'std': label_stats.std,
        }

    return TabMPreparedDataset(
        x_train=x_train,
        x_valid=x_valid,
        x_test=x_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
        metadata=metadata,
    )
