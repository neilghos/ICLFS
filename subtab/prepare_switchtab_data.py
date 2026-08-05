from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.datasets import load_svmlight_file
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


SUBTAB_ROOT = Path(__file__).resolve().parent
RAW_DATA_ROOT = SUBTAB_ROOT / "subtab_datasets"
PROCESSED_ROOT = SUBTAB_ROOT / "processed"

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


@dataclass
class DatasetSpec:
    name: str
    task: str
    loader: str
    target_column: str | None = None
    path: str | None = None
    extra_path: str | None = None
    image_like: bool = False
    predefined_split: bool = False
    split_root: str | None = None
    feature_dim: int | None = None
    max_rows: int | None = None


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "adult": DatasetSpec(
        name="adult",
        task="classification",
        loader="adult",
        path="adult/adult.data",
        extra_path="adult/adult.test",
        target_column="income",
    ),
    "california": DatasetSpec(
        name="california",
        task="regression",
        loader="csv",
        path="california.csv",
    ),
    "covertype": DatasetSpec(
        name="covertype",
        task="classification",
        loader="csv_no_header",
        path="covertype/covtype.data",
    ),
    "yearpredictionmsd": DatasetSpec(
        name="yearpredictionmsd",
        task="regression",
        loader="yearpredictionmsd",
        path="yearpredictionmsd/YearPredictionMSD.txt",
    ),
    "aloi": DatasetSpec(
        name="aloi",
        task="classification",
        loader="arff",
        path="aloi.arff",
    ),
    "helena": DatasetSpec(
        name="helena",
        task="classification",
        loader="arff",
        path="helena.arff",
    ),
    "jannis": DatasetSpec(
        name="jannis",
        task="classification",
        loader="arff",
        path="jannis.arff",
    ),
    "higgs": DatasetSpec(
        name="higgs",
        task="classification",
        loader="csv_no_header_label_first",
        path="HIGGS.csv",
        max_rows=98_050,
    ),
    "epsilon": DatasetSpec(
        name="epsilon",
        task="classification",
        loader="svmlight_pair",
        predefined_split=True,
        split_root="epsilon",
        feature_dim=2000,
    ),
    "microsoft": DatasetSpec(
        name="microsoft",
        task="ranking",
        loader="letor",
        predefined_split=True,
        split_root="MS/Fold1",
        feature_dim=136,
    ),
    "yahoo": DatasetSpec(
        name="yahoo",
        task="ranking",
        loader="letor",
        predefined_split=True,
        split_root="ltrc_yahoo",
        feature_dim=699,
    ),
}

COMMON_TARGET_CANDIDATES = (
    "target",
    "label",
    "class",
    "income",
    "cover_type",
    "medhouseval",
    "median_house_value",
    "y",
)


def _is_categorical_series(series: pd.Series) -> bool:
    return pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare SFE datasets with SwitchTab-style preprocessing and train/val/test splits."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_REGISTRY),
        help="Dataset key under subtab/subtab_datasets.",
    )
    parser.add_argument("--raw-root", type=Path, default=RAW_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _decode_if_bytes(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _clean_object_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in result.columns:
        if _is_categorical_series(result[col]):
            result[col] = result[col].map(_decode_if_bytes)
            result[col] = result[col].astype("string").str.strip()
            result[col] = result[col].replace({"?": pd.NA, "": pd.NA, "nan": pd.NA, "None": pd.NA})
    return result


def _prepare_categorical_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = df[columns].copy()
    for col in columns:
        frame[col] = frame[col].astype(object)
        frame[col] = frame[col].where(~frame[col].isna(), np.nan)
    return frame


def _load_adult(raw_root: Path, spec: DatasetSpec) -> tuple[pd.DataFrame, pd.Series]:
    assert spec.path is not None and spec.extra_path is not None and spec.target_column is not None
    train_path = raw_root / spec.path
    test_path = raw_root / spec.extra_path

    train_df = pd.read_csv(
        train_path,
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
        na_values=["?"],
    )
    test_df = pd.read_csv(
        test_path,
        header=None,
        names=ADULT_COLUMNS,
        skiprows=1,
        comment="|",
        skipinitialspace=True,
        na_values=["?"],
    )
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    full_df = _clean_object_frame(full_df)
    full_df[spec.target_column] = full_df[spec.target_column].str.rstrip(".")
    y = full_df.pop(spec.target_column)
    return full_df, y


def _load_csv(raw_root: Path, spec: DatasetSpec) -> tuple[pd.DataFrame, pd.Series]:
    assert spec.path is not None
    df = pd.read_csv(raw_root / spec.path)
    df = _clean_object_frame(df)
    target_col = resolve_target_column(df, spec)
    y = df.pop(target_col)
    return df, y


def _load_csv_no_header(raw_root: Path, spec: DatasetSpec) -> tuple[pd.DataFrame, pd.Series]:
    assert spec.path is not None
    df = pd.read_csv(raw_root / spec.path, header=None, nrows=spec.max_rows)
    feature_count = df.shape[1] - 1
    df.columns = [f"feature_{idx}" for idx in range(feature_count)] + ["target"]
    y = df.pop("target")
    return df, y


def _load_csv_no_header_label_first(raw_root: Path, spec: DatasetSpec) -> tuple[pd.DataFrame, pd.Series]:
    assert spec.path is not None
    df = pd.read_csv(raw_root / spec.path, header=None, nrows=spec.max_rows)
    y = df.iloc[:, 0].copy()
    x = df.iloc[:, 1:].copy()
    x.columns = [f"feature_{idx}" for idx in range(x.shape[1])]
    return x, y


def _load_yearpredictionmsd(raw_root: Path, spec: DatasetSpec) -> tuple[pd.DataFrame, pd.Series]:
    assert spec.path is not None
    df = pd.read_csv(raw_root / spec.path, header=None)
    y = df.iloc[:, 0].copy()
    x = df.iloc[:, 1:].copy()
    x.columns = [f"feature_{idx}" for idx in range(x.shape[1])]
    return x, y


def _load_arff(raw_root: Path, spec: DatasetSpec) -> tuple[pd.DataFrame, pd.Series]:
    assert spec.path is not None
    raw, _ = arff.loadarff(raw_root / spec.path)
    df = pd.DataFrame(raw)
    df = _clean_object_frame(df)
    target_col = resolve_target_column(df, spec)
    y = df.pop(target_col)
    return df, y


def _load_svmlight_file(path: Path, *, feature_dim: int | None) -> tuple[pd.DataFrame, pd.Series]:
    x_sparse, y = load_svmlight_file(path, n_features=feature_dim)
    x = pd.DataFrame.sparse.from_spmatrix(
        x_sparse,
        columns=[f"feature_{idx}" for idx in range(x_sparse.shape[1])],
    )
    return x, pd.Series(y, name="target")


def load_predefined_svmlight_splits(
    raw_root: Path,
    spec: DatasetSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    if spec.split_root is None:
        raise ValueError(f"Dataset '{spec.name}' is missing split_root.")

    split_root = raw_root / spec.split_root
    train_path = _resolve_existing_path([
        split_root / 'epsilon_normalized',
        split_root / 'train.libsvm',
        split_root / 'train.svm',
    ])
    test_path = _resolve_existing_path([
        split_root / 'epsilon_normalized.t',
        split_root / 'test.libsvm',
        split_root / 'test.svm',
    ])

    x_train_full, y_train_full = _load_svmlight_file(train_path, feature_dim=spec.feature_dim)
    x_test, y_test = _load_svmlight_file(test_path, feature_dim=spec.feature_dim)

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full,
    )

    metadata = {
        'predefined_split': True,
        'split_root': str(split_root),
        'source_train_rows': int(x_train_full.shape[0]),
        'source_test_rows': int(x_test.shape[0]),
        'valid_fraction_of_source_train': 0.2,
    }
    return x_train, x_valid, x_test, y_train, y_valid, y_test, metadata


def _parse_letor_line(
    line: str,
    *,
    feature_dim: int | None,
) -> tuple[float, int, list[float]]:
    tokens = line.strip().split()
    if len(tokens) < 2:
        raise ValueError("LETOR line must contain at least label and qid.")
    label = float(tokens[0])
    qid_token = tokens[1]
    if not qid_token.startswith("qid:"):
        raise ValueError(f"Expected qid token, got '{qid_token}'.")
    qid = int(qid_token.split(":", 1)[1])

    feature_map: dict[int, float] = {}
    max_idx = 0
    for token in tokens[2:]:
        if token.startswith("#"):
            break
        idx_str, value_str = token.split(":", 1)
        idx = int(idx_str)
        if value_str.upper() == "NULL":
            value = np.nan
        else:
            value = float(value_str)
        feature_map[idx] = value
        max_idx = max(max_idx, idx)

    resolved_dim = feature_dim or max_idx
    features = [0.0] * resolved_dim
    for idx, value in feature_map.items():
        if 1 <= idx <= resolved_dim:
            features[idx - 1] = value
    return label, qid, features


def _read_letor_split(
    path: Path,
    *,
    feature_dim: int | None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    labels: list[float] = []
    qids: list[int] = []
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            label, qid, features = _parse_letor_line(stripped, feature_dim=feature_dim)
            labels.append(label)
            qids.append(qid)
            rows.append(features)

    x = pd.DataFrame(rows, columns=[f"feature_{idx}" for idx in range(len(rows[0]))])
    y = pd.Series(labels, name="target")
    query_ids = pd.Series(qids, name="qid")
    return x, y, query_ids


def _resolve_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("None of the expected paths exist:\n" + "\n".join(str(p) for p in candidates))


def load_predefined_letor_splits(
    raw_root: Path,
    spec: DatasetSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, dict[str, Any]]:
    if spec.split_root is None:
        raise ValueError(f"Dataset '{spec.name}' is missing split_root.")

    split_root = raw_root / spec.split_root
    if spec.name == "microsoft":
        train_path = _resolve_existing_path([split_root / "train.txt"])
        valid_path = _resolve_existing_path([split_root / "vali.txt", split_root / "valid.txt"])
        test_path = _resolve_existing_path([split_root / "test.txt"])
    elif spec.name == "yahoo":
        train_path = _resolve_existing_path(
            [
                split_root / "train.txt",
                split_root / "set1.train.txt",
                split_root / "train.tsv",
            ]
        )
        valid_path = _resolve_existing_path(
            [
                split_root / "vali.txt",
                split_root / "valid.txt",
                split_root / "set1.valid.txt",
                split_root / "set1.vali.txt",
                split_root / "validation.txt",
            ]
        )
        test_path = _resolve_existing_path(
            [
                split_root / "test.txt",
                split_root / "set1.test.txt",
            ]
        )
    else:
        raise ValueError(f"Unknown LETOR predefined-split dataset '{spec.name}'.")

    x_train, y_train, qid_train = _read_letor_split(train_path, feature_dim=spec.feature_dim)
    x_valid, y_valid, qid_valid = _read_letor_split(valid_path, feature_dim=spec.feature_dim)
    x_test, y_test, qid_test = _read_letor_split(test_path, feature_dim=spec.feature_dim)

    metadata = {
        "predefined_split": True,
        "split_root": str(split_root),
        "train_query_count": int(qid_train.nunique()),
        "valid_query_count": int(qid_valid.nunique()),
        "test_query_count": int(qid_test.nunique()),
    }
    return x_train, x_valid, x_test, y_train, y_valid, y_test, metadata


def resolve_target_column(df: pd.DataFrame, spec: DatasetSpec) -> str:
    if spec.target_column is not None and spec.target_column in df.columns:
        return spec.target_column

    lowered = {str(col).strip().lower(): col for col in df.columns}
    for candidate in COMMON_TARGET_CANDIDATES:
        if candidate in lowered:
            return str(lowered[candidate])

    return str(df.columns[-1])


def load_dataset(raw_root: Path, dataset_name: str) -> tuple[pd.DataFrame, pd.Series, DatasetSpec]:
    spec = DATASET_REGISTRY[dataset_name]
    if spec.loader == "adult":
        x, y = _load_adult(raw_root, spec)
    elif spec.loader == "csv":
        x, y = _load_csv(raw_root, spec)
    elif spec.loader == "csv_no_header":
        x, y = _load_csv_no_header(raw_root, spec)
    elif spec.loader == "csv_no_header_label_first":
        x, y = _load_csv_no_header_label_first(raw_root, spec)
    elif spec.loader == "yearpredictionmsd":
        x, y = _load_yearpredictionmsd(raw_root, spec)
    elif spec.loader == "arff":
        x, y = _load_arff(raw_root, spec)
    else:
        raise ValueError(f"Unknown loader '{spec.loader}' for dataset '{dataset_name}'.")

    x = maybe_flatten_image_like(x, image_like=spec.image_like)
    return x, y, spec


def load_predefined_dataset(
    raw_root: Path,
    dataset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, DatasetSpec, dict[str, Any]]:
    spec = DATASET_REGISTRY[dataset_name]
    if spec.loader == "letor":
        x_train, x_valid, x_test, y_train, y_valid, y_test, metadata = load_predefined_letor_splits(raw_root, spec)
    elif spec.loader == "svmlight_pair":
        x_train, x_valid, x_test, y_train, y_valid, y_test, metadata = load_predefined_svmlight_splits(raw_root, spec)
    else:
        raise ValueError(f"Dataset '{dataset_name}' does not support predefined split loading.")

    x_train = maybe_flatten_image_like(x_train, image_like=spec.image_like)
    x_valid = maybe_flatten_image_like(x_valid, image_like=spec.image_like)
    x_test = maybe_flatten_image_like(x_test, image_like=spec.image_like)
    return x_train, x_valid, x_test, y_train, y_valid, y_test, spec, metadata


def maybe_flatten_image_like(x: pd.DataFrame, *, image_like: bool) -> pd.DataFrame:
    if not image_like:
        return x
    array = np.asarray(x)
    if array.ndim <= 2:
        return pd.DataFrame(array, columns=x.columns if hasattr(x, "columns") else None)
    flat = array.reshape(array.shape[0], -1)
    return pd.DataFrame(flat, columns=[f"pixel_{idx}" for idx in range(flat.shape[1])])


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train/val/test ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio} = {total}."
        )


def split_dataset(
    x: pd.DataFrame,
    y: pd.Series,
    *,
    task: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    validate_split_ratios(train_ratio, val_ratio, test_ratio)

    stratify = y if task == "classification" else None
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=test_ratio,
        random_state=seed,
        stratify=stratify,
    )

    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    stratify_train_val = y_train_val if task == "classification" else None
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_ratio_adjusted,
        random_state=seed,
        stratify=stratify_train_val,
    )
    return x_train, x_valid, x_test, y_train, y_valid, y_test


def preprocess_splits(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        from category_encoders.backward_difference import BackwardDifferenceEncoder
    except ImportError as exc:
        raise ImportError(
            "category_encoders is required for backward difference encoding. "
            "Install it in your torch environment before running SFE preprocessing."
        ) from exc

    train_df = x_train.copy()
    valid_df = x_valid.copy()
    test_df = x_test.copy()

    keep_columns = [col for col in train_df.columns if not train_df[col].isna().all()]
    train_df = train_df.loc[:, keep_columns]
    valid_df = valid_df.loc[:, keep_columns]
    test_df = test_df.loc[:, keep_columns]

    categorical_cols = [
        col
        for col in train_df.columns
        if pd.api.types.is_object_dtype(train_df[col])
        or pd.api.types.is_string_dtype(train_df[col])
        or isinstance(train_df[col].dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(train_df[col])
    ]
    numerical_cols = [col for col in train_df.columns if col not in categorical_cols]

    numeric_blocks: list[np.ndarray] = []
    categorical_blocks: list[np.ndarray] = []
    output_feature_names: list[str] = []

    if numerical_cols:
        num_imputer = SimpleImputer(strategy="mean")
        train_num = num_imputer.fit_transform(train_df[numerical_cols])
        valid_num = num_imputer.transform(valid_df[numerical_cols])
        test_num = num_imputer.transform(test_df[numerical_cols])
        numeric_blocks = [train_num, valid_num, test_num]
        output_feature_names.extend([str(col) for col in numerical_cols])

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        train_cat_input = _prepare_categorical_frame(train_df, categorical_cols)
        valid_cat_input = _prepare_categorical_frame(valid_df, categorical_cols)
        test_cat_input = _prepare_categorical_frame(test_df, categorical_cols)
        train_cat = pd.DataFrame(
            cat_imputer.fit_transform(train_cat_input),
            columns=categorical_cols,
            index=train_df.index,
        )
        valid_cat = pd.DataFrame(
            cat_imputer.transform(valid_cat_input),
            columns=categorical_cols,
            index=valid_df.index,
        )
        test_cat = pd.DataFrame(
            cat_imputer.transform(test_cat_input),
            columns=categorical_cols,
            index=test_df.index,
        )

        encoder = BackwardDifferenceEncoder(cols=categorical_cols, return_df=True)
        train_cat_encoded = encoder.fit_transform(train_cat)
        valid_cat_encoded = encoder.transform(valid_cat)
        test_cat_encoded = encoder.transform(test_cat)

        categorical_blocks = [
            train_cat_encoded.to_numpy(dtype=np.float32),
            valid_cat_encoded.to_numpy(dtype=np.float32),
            test_cat_encoded.to_numpy(dtype=np.float32),
        ]
        output_feature_names.extend([str(col) for col in train_cat_encoded.columns])

    if not numeric_blocks and not categorical_blocks:
        raise ValueError("No usable features remain after dropping all-missing columns.")

    if numeric_blocks and categorical_blocks:
        train_array = np.concatenate([numeric_blocks[0], categorical_blocks[0]], axis=1)
        valid_array = np.concatenate([numeric_blocks[1], categorical_blocks[1]], axis=1)
        test_array = np.concatenate([numeric_blocks[2], categorical_blocks[2]], axis=1)
    elif numeric_blocks:
        train_array, valid_array, test_array = numeric_blocks
    else:
        train_array, valid_array, test_array = categorical_blocks

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_array).astype(np.float32)
    valid_scaled = scaler.transform(valid_array).astype(np.float32)
    test_scaled = scaler.transform(test_array).astype(np.float32)

    metadata = {
        "input_dim": int(train_scaled.shape[1]),
        "kept_raw_columns": [str(col) for col in keep_columns],
        "categorical_columns": [str(col) for col in categorical_cols],
        "numerical_columns": [str(col) for col in numerical_cols],
        "output_feature_names": output_feature_names,
    }
    return train_scaled, valid_scaled, test_scaled, metadata


def encode_targets(
    y_train: pd.Series,
    y_valid: pd.Series,
    y_test: pd.Series,
    *,
    task: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if task == "classification":
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train.astype(str))
        y_valid_encoded = label_encoder.transform(y_valid.astype(str))
        y_test_encoded = label_encoder.transform(y_test.astype(str))
        metadata = {
            "task": task,
            "class_names": label_encoder.classes_.tolist(),
            "output_dim": int(len(label_encoder.classes_)),
        }
        return (
            y_train_encoded.astype(np.int64),
            y_valid_encoded.astype(np.int64),
            y_test_encoded.astype(np.int64),
            metadata,
        )

    if task == "ranking":
        y_train_array = y_train.to_numpy(dtype=np.float32)
        y_valid_array = y_valid.to_numpy(dtype=np.float32)
        y_test_array = y_test.to_numpy(dtype=np.float32)
        metadata = {"task": task, "output_dim": 1}
        return y_train_array, y_valid_array, y_test_array, metadata

    y_train_array = y_train.to_numpy(dtype=np.float32)
    y_valid_array = y_valid.to_numpy(dtype=np.float32)
    y_test_array = y_test.to_numpy(dtype=np.float32)
    metadata = {"task": task, "output_dim": 1}
    return y_train_array, y_valid_array, y_test_array, metadata


def save_processed_dataset(
    out_root: Path,
    dataset_name: str,
    *,
    x_train: np.ndarray,
    x_valid: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_test: np.ndarray,
    metadata: dict[str, Any],
) -> Path:
    dataset_dir = out_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    split_path = dataset_dir / "splits.npz"
    np.savez_compressed(
        split_path,
        x_train=x_train,
        x_valid=x_valid,
        x_test=x_test,
        y_train=y_train,
        y_valid=y_valid,
        y_test=y_test,
    )
    metadata_path = dataset_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return split_path


def main() -> None:
    args = parse_args()
    spec = DATASET_REGISTRY[args.dataset]
    split_metadata: dict[str, Any] = {}
    if spec.predefined_split:
        x_train, x_valid, x_test, y_train, y_valid, y_test, spec, split_metadata = load_predefined_dataset(
            args.raw_root,
            args.dataset,
        )
    else:
        x, y, spec = load_dataset(args.raw_root, args.dataset)
        x_train, x_valid, x_test, y_train, y_valid, y_test = split_dataset(
            x,
            y,
            task=spec.task,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    x_train_processed, x_valid_processed, x_test_processed, feature_metadata = preprocess_splits(
        x_train,
        x_valid,
        x_test,
    )
    y_train_encoded, y_valid_encoded, y_test_encoded, target_metadata = encode_targets(
        y_train,
        y_valid,
        y_test,
        task=spec.task,
    )

    metadata = {
        "dataset": args.dataset,
        "task": spec.task,
        "seed": args.seed,
        "split_ratios": None
        if spec.predefined_split
        else {
            "train": args.train_ratio,
            "valid": args.val_ratio,
            "test": args.test_ratio,
        },
        "num_train": int(x_train_processed.shape[0]),
        "num_valid": int(x_valid_processed.shape[0]),
        "num_test": int(x_test_processed.shape[0]),
        **split_metadata,
        **feature_metadata,
        **target_metadata,
    }
    split_path = save_processed_dataset(
        args.out_root,
        args.dataset,
        x_train=x_train_processed,
        x_valid=x_valid_processed,
        x_test=x_test_processed,
        y_train=y_train_encoded,
        y_valid=y_valid_encoded,
        y_test=y_test_encoded,
        metadata=metadata,
    )

    print(
        f"Prepared {args.dataset}: train={x_train_processed.shape[0]} "
        f"valid={x_valid_processed.shape[0]} test={x_test_processed.shape[0]} "
        f"features={x_train_processed.shape[1]}"
    )
    print(f"Saved processed splits to {split_path}")


if __name__ == "__main__":
    main()
