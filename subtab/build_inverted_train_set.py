from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.augmentor import build_augmentor
from src.runtime_config import get_augmentor_config


SUBTAB_ROOT = Path(__file__).resolve().parent
PROCESSED_ROOT = SUBTAB_ROOT / "processed"
INVERTED_ROOT = SUBTAB_ROOT / "inverted"


class InvertedFeatureDataset(Dataset):
    """Feature-major dataset where each item is one feature profile over samples."""

    def __init__(
        self,
        x_train: np.ndarray,
        augmentor_config: dict | None = None,
    ):
        x_train = np.asarray(x_train, dtype=np.float32)
        if x_train.ndim != 2:
            raise ValueError(f"x_train must be 2D, got shape {x_train.shape}.")
        self.x = torch.from_numpy(x_train.T).float()
        self.n_patients = self.x.shape[1]
        self.augmentor = build_augmentor(
            n_patients=self.n_patients,
            strategy="four_view_mask",
            config=augmentor_config,
        )

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int):
        return self.augmentor(self.x[idx])


@dataclass
class InvertedTrainBundle:
    x_train: np.ndarray
    x_train_inverted: np.ndarray
    y_train: np.ndarray
    metadata: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the inverted feature-major train set for SFE training."
    )
    parser.add_argument("--dataset", required=True, help="Dataset key under subtab/processed.")
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--out-root", type=Path, default=INVERTED_ROOT)
    parser.add_argument("--config-path", type=Path, default=None)
    return parser.parse_args()


def load_processed_dataset(processed_root: Path, dataset_name: str) -> InvertedTrainBundle:
    dataset_dir = processed_root / dataset_name
    split_path = dataset_dir / "splits.npz"
    metadata_path = dataset_dir / "metadata.json"
    if not split_path.exists():
        raise FileNotFoundError(f"Processed split file not found: {split_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Processed metadata file not found: {metadata_path}")

    bundle = np.load(split_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    x_train = np.asarray(bundle["x_train"], dtype=np.float32)
    y_train = np.asarray(bundle["y_train"])
    x_train_inverted = np.asarray(x_train.T, dtype=np.float32)
    return InvertedTrainBundle(
        x_train=x_train,
        x_train_inverted=x_train_inverted,
        y_train=y_train,
        metadata=metadata,
    )


def build_inverted_loader(
    x_train: np.ndarray,
    *,
    config_path: str | Path | None = None,
) -> DataLoader:
    dataset = InvertedFeatureDataset(
        x_train,
        augmentor_config=get_augmentor_config(config_path),
    )
    return DataLoader(dataset, batch_size=len(dataset), shuffle=False)


def save_inverted_dataset(out_root: Path, dataset_name: str, bundle: InvertedTrainBundle) -> Path:
    dataset_dir = out_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    tensor_path = dataset_dir / "train_inverted.pt"
    npy_path = dataset_dir / "train_inverted.npy"
    metadata_path = dataset_dir / "metadata.json"

    torch.save(torch.from_numpy(bundle.x_train_inverted), tensor_path)
    np.save(npy_path, bundle.x_train_inverted)

    metadata = dict(bundle.metadata)
    metadata.update(
        {
            "num_train_samples": int(bundle.x_train.shape[0]),
            "num_train_features": int(bundle.x_train.shape[1]),
            "inverted_shape": list(bundle.x_train_inverted.shape),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return tensor_path


def main() -> None:
    args = parse_args()
    bundle = load_processed_dataset(args.processed_root, args.dataset)
    output_path = save_inverted_dataset(args.out_root, args.dataset, bundle)

    preview_loader = build_inverted_loader(bundle.x_train, config_path=args.config_path)
    preview_batch = next(iter(preview_loader))

    print(
        f"Built inverted train set for {args.dataset}: "
        f"train={bundle.x_train.shape[0]} features={bundle.x_train.shape[1]} "
        f"inverted_shape={bundle.x_train_inverted.shape}"
    )
    print(f"Saved inverted tensor to {output_path}")
    print(f"Preview batch views: {len(preview_batch)} tensors")


if __name__ == "__main__":
    main()
