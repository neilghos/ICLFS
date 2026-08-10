from __future__ import annotations

import argparse
import gc
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.api import InvertedFeatureDataset
from src.extractor import get_feature_scores, get_topk_feature_indices
from src.loss import contrastive_loss, decorrelation_loss
from src.models import InvertedFeatureExpert
from src.runtime_config import get_augmentor_config
from subtab.metrics import calculate_metrics, summarize_metrics
from subtab.tabm_loader import load_tabm_prepared_dataset


SUBTAB_ROOT = Path(__file__).resolve().parent
TABM_DATA_ROOT = SUBTAB_ROOT / "tabm-data"
RESULTS_DIR = SUBTAB_ROOT / "results"
TABM_DATASET_REGISTRY: dict[str, tuple[str, str]] = {
    "adult": ("adult", "classification"),
    "california": ("california", "regression"),
    "covertype": ("covtype2", "classification"),
    "jannis": ("classif-num-large-0-jannis", "classification"),
    "higgs": ("higgs-small", "classification"),
    "otto": ("otto", "classification"),
    "churn": ("churn", "classification"),
    "house": ("house", "regression"),
    "diamond": ("diamond", "regression"),
    "yearpredictionmsd": ("regression-num-large-0-year", "regression"),
    "microsoft": ("microsoft", "ranking"),
    # Ultra-fast iteration suite (N <= 15,000)
    "cpu_act": ("regression-num-medium-0-cpu_act", "regression"),
    "brazilian_houses": ("regression-cat-medium-0-Brazilian_houses", "regression"),
    "ailerons": ("regression-num-medium-0-Ailerons", "regression"),
    "miami_housing": ("regression-num-medium-0-MiamiHousing2016", "regression"),
    "pol": ("regression-num-medium-0-pol", "regression"),
    "elevators": ("regression-num-medium-0-elevators", "regression"),
    "bank_marketing": ("classif-num-medium-0-bank-marketing", "classification"),
    "magic_telescope": ("classif-num-medium-0-MagicTelescope", "classification"),
    "credit": ("classif-num-medium-0-credit", "classification"),
}
DEV_DATASETS = (
    "adult",
    "jannis",
    "higgs",
    "otto",
    "churn",
    "california",
    "house",
    "diamond",
    "yearpredictionmsd",
    "cpu_act",
    "brazilian_houses",
    "ailerons",
    "miami_housing",
    "pol",
    "elevators",
    "bank_marketing",
    "magic_telescope",
    "credit",
)
DEFAULT_TOP_K = 256
DEFAULT_SEEDS = list(range(5))

REGRESSION_HYPERPARAMS = {
    "lr": 2e-3,
    "weight_decay": 3e-4,
    "epochs": 200,
    "warmup_ratio": 0.10,
    "cycle_len": 90,
    "restart_decay": 0.8,
}


@dataclass
class DatasetBundle:
    dataset_name: str
    task: str
    x_train: np.ndarray
    x_valid: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    metadata: dict[str, Any]


@dataclass
class BackboneConfig:
    encoder_hidden_dim: int
    latent_dim: int
    projector_hidden_dim: int
    projector_output_dim: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unsupervised subtab evaluation with ICL feature training, top-k selection, and sample projection."
    )
    parser.add_argument(
        "--dataset",
        default="california",
        choices=sorted((*TABM_DATASET_REGISTRY.keys(), "dev", "classification", "regression")),
        help="Dataset key under subtab/tabm-data, or 'dev' for the 18-dataset suite, or 'classification'/'regression' groups.",
    )
    parser.add_argument("--tabm-root", type=Path, default=TABM_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional hard override for selected feature count. If omitted, selection-ratio is used.",
    )
    parser.add_argument(
        "--selection-ratio",
        type=float,
        default=0.9,
        help="Fraction of original processed features kept after saliency ranking when --top-k is not set.",
    )
    parser.add_argument("--encoder-hidden-dim", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--projector-hidden-dim", type=int, default=256)
    parser.add_argument("--projector-output-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--decorrelation-weight", type=float, default=0.40)
    parser.add_argument("--probe-hidden-dim", type=int, default=256)
    parser.add_argument("--probe-max-iter", type=int, default=200)
    parser.add_argument("--config-path", default=None)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Training device for the unsupervised ICL backbone. 'auto' tries CUDA first and falls back to CPU on OOM.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_bundle(tabm_root: Path, dataset_name: str, seed: int) -> DatasetBundle:
    if dataset_name not in TABM_DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{dataset_name}'.")
    tabm_name, task_name = TABM_DATASET_REGISTRY[dataset_name]
    prepared = load_tabm_prepared_dataset(
        tabm_root / tabm_name,
        task=task_name,
        seed=seed,
        num_policy='standard',
        cat_policy='one-hot',
        standardize_regression_labels_flag=False,
    )
    metadata = {
        'dataset': dataset_name,
        'task': task_name,
        'seed': seed,
        **prepared.metadata,
    }
    if task_name == 'classification':
        metadata['class_names'] = [str(x) for x in sorted(np.unique(prepared.y_train).tolist())]
        metadata['output_dim'] = int(len(np.unique(prepared.y_train)))
    else:
        metadata['output_dim'] = 1

    x_train = np.asarray(prepared.x_train, dtype=np.float32)
    y_train = np.asarray(prepared.y_train)
    if x_train.shape[0] > 50000:
        x_train = x_train[:50000]
        y_train = y_train[:50000]

    x_valid = np.asarray(prepared.x_valid, dtype=np.float32)
    y_valid = np.asarray(prepared.y_valid)
    if x_valid.shape[0] > 15000:
        x_valid = x_valid[:15000]
        y_valid = y_valid[:15000]

    return DatasetBundle(
        dataset_name=dataset_name,
        task=task_name,
        x_train=x_train,
        x_valid=x_valid,
        x_test=np.asarray(prepared.x_test, dtype=np.float32),
        y_train=y_train,
        y_valid=y_valid,
        y_test=np.asarray(prepared.y_test),
        metadata=metadata,
    )


def resolve_backbone_config(bundle: DatasetBundle, args: argparse.Namespace) -> BackboneConfig:
    return BackboneConfig(
        encoder_hidden_dim=args.encoder_hidden_dim,
        latent_dim=args.latent_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        projector_output_dim=args.projector_output_dim,
    )


from torch.utils.data import Dataset, DataLoader
from src.models import SupervisedICLModel


class SupervisedTabularDataset(Dataset):
    """
    Dataset for End-to-End Supervised ICL (SupICL).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, task: str = "classification"):
        super().__init__()
        self.x = torch.from_numpy(np.asarray(x, dtype=np.float32))
        self.task = task
        if task == "classification":
            classes = np.unique(y)
            class_map = {c: idx for idx, c in enumerate(classes)}
            y_mapped = np.array([class_map[val] for val in y], dtype=np.int64)
            self.y = torch.from_numpy(y_mapped)
        else:
            self.y = torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(-1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def train_supicl_model(
    bundle,
    *,
    epochs: int = 100,
    seed: int = 42,
    d_token: int = 128,
    n_heads: int = 8,
    n_layers: int = 3,
    lr: float = 2e-3,
    weight_decay: float = 3e-4,
    batch_size: int = 256,
    device_name: str = "auto",
) -> tuple[dict[str, Any], dict[str, Any]]:
    set_seed(seed)
    if device_name == "cuda":
        candidate_devices = ["cuda"]
    elif device_name == "cpu":
        candidate_devices = ["cpu"]
    else:
        candidate_devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]

    device = torch.device(candidate_devices[0])
    num_samples, num_features = bundle.x_train.shape

    # Target Label Standardization for Regression
    if bundle.task == "regression":
        y_tr_raw = np.asarray(bundle.y_train, dtype=np.float32).reshape(-1)
        y_mean = float(y_tr_raw.mean())
        y_std = float(y_tr_raw.std())
        if y_std == 0.0:
            y_std = 1.0
        y_tr_norm = (y_tr_raw - y_mean) / y_std
    else:
        y_mean, y_std = 0.0, 1.0
        y_tr_norm = bundle.y_train

    if bundle.task == "classification":
        from sklearn.preprocessing import QuantileTransformer
        # Fit QuantileTransformer to map empirical distribution to normal
        qt = QuantileTransformer(output_distribution='normal', random_state=seed, n_quantiles=min(1000, num_samples))
        x_train_qt = qt.fit_transform(bundle.x_train)
        x_valid_qt = qt.transform(bundle.x_valid)
        x_test_qt = qt.transform(bundle.x_test)
        
        # Stack quantile and standard scaling to shape [N, D, 2]
        x_train_input = np.stack([x_train_qt, bundle.x_train], axis=-1)
        x_valid_input = np.stack([x_valid_qt, bundle.x_valid], axis=-1)
        x_test_input = np.stack([x_test_qt, bundle.x_test], axis=-1)
        in_channels = 2
    else:
        from sklearn.preprocessing import QuantileTransformer
        # Fit normal and uniform QuantileTransformers on regression features
        qt_norm = QuantileTransformer(output_distribution='normal', random_state=seed, n_quantiles=min(1000, num_samples))
        x_train_qt_norm = qt_norm.fit_transform(bundle.x_train)
        x_valid_qt_norm = qt_norm.transform(bundle.x_valid)
        x_test_qt_norm = qt_norm.transform(bundle.x_test)

        qt_uni = QuantileTransformer(output_distribution='uniform', random_state=seed, n_quantiles=min(1000, num_samples))
        x_train_qt_uni = qt_uni.fit_transform(bundle.x_train)
        x_valid_qt_uni = qt_uni.transform(bundle.x_valid)
        x_test_qt_uni = qt_uni.transform(bundle.x_test)
        
        # Stack quantile normal, uniform, and standard scaling to shape [N, D, 3]
        x_train_input = np.stack([x_train_qt_norm, x_train_qt_uni, bundle.x_train], axis=-1)
        x_valid_input = np.stack([x_valid_qt_norm, x_valid_qt_uni, bundle.x_valid], axis=-1)
        x_test_input = np.stack([x_test_qt_norm, x_test_qt_uni, bundle.x_test], axis=-1)
        in_channels = 3

    train_dataset = SupervisedTabularDataset(x_train_input, y_tr_norm, task=bundle.task)
    valid_dataset = SupervisedTabularDataset(x_valid_input, bundle.y_valid, task=bundle.task)
    test_dataset = SupervisedTabularDataset(x_test_input, bundle.y_test, task=bundle.task)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_outputs = len(np.unique(bundle.y_train)) if bundle.task == "classification" else 1
    model = SupervisedICLModel(
        num_features=num_features,
        num_outputs=num_outputs,
        in_channels=in_channels,
        d_token=d_token,
        n_heads=n_heads,
        n_layers=n_layers,
    ).to(device)

    if bundle.task == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    import math
    if bundle.task == "regression":
        # Warm Restarts scheduler for regression
        def lr_lambda(epoch):
            warmup_epochs = max(1, int(REGRESSION_HYPERPARAMS["warmup_ratio"] * epochs))
            if epoch < warmup_epochs:
                return float(epoch) / float(warmup_epochs)
            else:
                elapsed = epoch - warmup_epochs
                cycle_len = REGRESSION_HYPERPARAMS["cycle_len"]
                decay = REGRESSION_HYPERPARAMS["restart_decay"]
                
                cycle_idx = elapsed // cycle_len
                epoch_in_cycle = elapsed % cycle_len
                
                alpha_k = decay ** cycle_idx
                progress = float(epoch_in_cycle) / float(cycle_len)
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                
                return alpha_k * cosine_decay
    else:
        # Standard warmup cosine scheduler for classification
        def lr_lambda(epoch):
            warmup_epochs = max(1, int(0.10 * epochs))
            if epoch < warmup_epochs:
                return float(epoch) / float(warmup_epochs)
            else:
                progress = float(epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
                return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Multi-horizon sweep setup
    horizons = [100, 200, 300, 400, 500] if bundle.task == "regression" else [200]
    horizons = [h for h in horizons if h < epochs] + [epochs]
    horizons = sorted(list(set(horizons)))

    best_valid_scores = {h: -float("inf") for h in horizons}
    best_weights_dict = {h: None for h in horizons}
    best_valid_metrics_dict = {h: {} for h in horizons}

    epoch_bar = tqdm(range(epochs), desc=f"train_supicl[{seed}]", leave=False)
    for epoch in epoch_bar:
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            # 20% random feature mask for self-supervised subtab reconstruction (shape [B, D])
            mask = (torch.rand(batch_x.shape[0], batch_x.shape[1], device=device) < 0.20).float()
            logits, recon_x = model(batch_x, mask=mask)

            task_loss = criterion(logits, batch_y)
            # Compare reconstruction with standard-scaled features (the last channel)
            recon_loss = ((recon_x - batch_x[:, :, -1]) ** 2 * mask).sum() / (mask.sum() + 1e-6)
            loss = task_loss + 0.25 * recon_loss

            loss.backward()
            optimizer.step()

        scheduler.step()

        # Fast Validation Evaluation
        model.eval()
        with torch.no_grad():
            valid_preds = []
            for batch_x, _ in valid_loader:
                batch_x = batch_x.to(device)
                out, _ = model(batch_x)
                if bundle.task == "classification":
                    preds = out.argmax(dim=-1).cpu().numpy()
                else:
                    preds = (out.cpu().numpy().reshape(-1) * y_std) + y_mean
                valid_preds.extend(preds)

            valid_preds = np.array(valid_preds)
            valid_metrics = calculate_metrics(bundle.task, bundle.y_valid, valid_preds)

        # Check and update checkpoints for active horizons
        for h in horizons:
            if epoch < h:
                if valid_metrics["score"] > best_valid_scores[h] or best_weights_dict[h] is None:
                    best_valid_scores[h] = valid_metrics["score"]
                    best_valid_metrics_dict[h] = valid_metrics
                    best_weights_dict[h] = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        epoch_bar.set_postfix(valid=f"{valid_metrics['score']:.4f}")

    # Evaluate Test Metrics for all epoch horizons using stored best weights
    test_metrics_dict = {}
    for h in horizons:
        if best_weights_dict[h] is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_weights_dict[h].items()})

        model.eval()
        with torch.no_grad():
            test_preds = []
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(device)
                out, _ = model(batch_x)
                if bundle.task == "classification":
                    preds = out.argmax(dim=-1).cpu().numpy()
                else:
                    preds = (out.cpu().numpy().reshape(-1) * y_std) + y_mean
                test_preds.extend(preds)

            test_preds = np.array(test_preds)
            test_metrics = calculate_metrics(bundle.task, bundle.y_test, test_preds)
            test_metrics_dict[h] = test_metrics

    del model, best_weights_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_valid_metrics_dict, test_metrics_dict


def _run_single_dataset(dataset_name: str, args: argparse.Namespace) -> dict[str, Any]:
    bundle = load_dataset_bundle(args.tabm_root, dataset_name, seed=args.seeds[0])

    if args.epochs is None:
        epochs = REGRESSION_HYPERPARAMS["epochs"] if bundle.task == "regression" else 200
    else:
        epochs = args.epochs

    print(
        f"Loaded {bundle.dataset_name}: train={bundle.x_train.shape[0]} "
        f"valid={bundle.x_valid.shape[0]} test={bundle.x_test.shape[0]} "
        f"features={bundle.x_train.shape[1]} task={bundle.task} epochs={epochs}"
    )

    run_rows: list[dict[str, Any]] = []
    seed_bar = tqdm(list(enumerate(args.seeds)), desc="seeds")
    for run_idx, seed in seed_bar:
        seed_bar.set_postfix(seed=seed)
        print(f"\nRun {run_idx + 1}/{len(args.seeds)} | seed={seed}")

        valid_metrics_dict, test_metrics_dict = train_supicl_model(
            bundle,
            epochs=epochs,
            seed=seed,
            device_name=args.device,
        )

        for h in sorted(valid_metrics_dict.keys()):
            h_valid = valid_metrics_dict[h]
            h_test = test_metrics_dict[h]
            print(f"[valid-{h}] {summarize_metrics(bundle.task, h_valid)}")
            print(f"[test -{h}] {summarize_metrics(bundle.task, h_test)}")

            run_rows.append(
                {
                    "Dataset": bundle.dataset_name,
                    "Task": bundle.task,
                    "Method": f"End2End-SupICL-{h}",
                    "Run": run_idx,
                    "Seed": seed,
                    "Horizon": h,
                    "ValidationScore": float(h_valid["score"]),
                    "TestScore": float(h_test["score"]),
                    "ValidationMetric": float(
                        h_valid["accuracy"] if bundle.task == "classification" else h_valid["rmse"]
                    ),
                    "TestMetric": float(
                        h_test["accuracy"] if bundle.task == "classification" else h_test["rmse"]
                    ),
                }
            )

    runs_df = pd.DataFrame(run_rows)
    summary_rows = []
    
    # Calculate statistics grouped by each epoch horizon
    for method_grp, grp_df in runs_df.groupby("Method"):
        summary_df = grp_df.agg(
            {
                "ValidationScore": ["mean", "std"],
                "TestScore": ["mean", "std"],
                "ValidationMetric": ["mean", "std"],
                "TestMetric": ["mean", "std"],
            }
        )

        summary_row = {
            "Dataset": bundle.dataset_name,
            "Task": bundle.task,
            "Method": method_grp,
            "ValidationScoreMean": float(summary_df.loc["mean", "ValidationScore"]),
            "ValidationScoreStd": float(summary_df.loc["std", "ValidationScore"]),
            "TestScoreMean": float(summary_df.loc["mean", "TestScore"]),
            "TestScoreStd": float(summary_df.loc["std", "TestScore"]),
            "ValidationMetricMean": float(summary_df.loc["mean", "ValidationMetric"]),
            "ValidationMetricStd": float(summary_df.loc["std", "ValidationMetric"]),
            "TestMetricMean": float(summary_df.loc["mean", "TestMetric"]),
            "TestMetricStd": float(summary_df.loc["std", "TestMetric"]),
        }
        summary_rows.append(summary_row)

    summary_path = args.out_dir / f"{bundle.dataset_name}_supicl_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print("\nSummary")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"Saved summaries to {summary_path}")
    
    # Return the summary row for the maximum epoch horizon to keep compatibility with Dev summaries
    max_horizon = max(runs_df["Horizon"])
    max_summary = next(row for row in summary_rows if row["Method"] == f"End2End-SupICL-{max_horizon}")
    return max_summary


def main() -> None:
    args = parse_args()
    if not args.seeds:
        raise ValueError("Provide at least one seed.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    CLASSIFICATION_DATASETS = [
        "adult", "bank_marketing", "churn", "credit", "higgs", "jannis", "magic_telescope", "otto"
    ]
    REGRESSION_DATASETS = [
        "cpu_act", "brazilian_houses", "ailerons", "miami_housing", "pol", "elevators", "california", "house", "diamond"
    ]

    if args.dataset == "classification":
        dataset_names = CLASSIFICATION_DATASETS
    elif args.dataset == "regression":
        dataset_names = REGRESSION_DATASETS
    elif args.dataset == "dev":
        dataset_names = REGRESSION_DATASETS + CLASSIFICATION_DATASETS
    else:
        dataset_names = [args.dataset]

    all_summaries: list[dict[str, Any]] = []
    for dataset_idx, dataset_name in enumerate(dataset_names):
        if len(dataset_names) > 1:
            print(f"\n{'=' * 80}")
            print(f"Dataset {dataset_idx + 1}/{len(dataset_names)} | {dataset_name}")
            print(f"{'=' * 80}")
        summary_row = _run_single_dataset(dataset_name, args)
        all_summaries.append(summary_row)

    if args.dataset in ("dev", "classification", "regression"):
        combined_path = args.out_dir / f"{args.dataset}_subtab_summary.csv"
        combined_df = pd.DataFrame(all_summaries)
        combined_df.to_csv(combined_path, index=False)
        print(f"\n{args.dataset.capitalize()} Summary")
        print(combined_df.to_string(index=False))
        print(f"Saved {args.dataset} summary to {combined_path}")


if __name__ == "__main__":
    main()
