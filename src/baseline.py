from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.cluster as sklearn_cluster
import torch
from lscae import Lscae
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from skfeature.function.similarity_based import lap_score
from skfeature.function.sparse_learning_based import MCFS, NDFS
from skfeature.utility.construct_W import construct_W
from torch.utils.data import DataLoader, TensorDataset

import data_loaders  # noqa: F401
from data.api import DATASET_REGISTRY


RESULTS_DIR = Path("/home/utsab/Desktop/ICLFE/ICLFE/src/results")
PAPER_KS = (50, 100, 150, 200, 250, 300)
PAPER_NUM_KMEANS_RUNS = 20
PAPER_NUM_EPOCHS = 300


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone unsupervised feature-selection baseline runner."
    )
    parser.add_argument("--dataset", required=True, help="Registered dataset name.")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=["LS", "MCFS", "NDFS", "CAE", "LSCAE"],
        choices=["LS", "MCFS", "NDFS", "CAE", "LSCAE"],
        help="Baseline methods to run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmeans-runs", type=int, default=PAPER_NUM_KMEANS_RUNS)
    parser.add_argument("--epochs", type=int, default=PAPER_NUM_EPOCHS)
    parser.add_argument(
        "--out",
        default=None,
        help="Optional CSV output path. Defaults to results/baseline_<dataset>.csv",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_full_dataset(dataset_name: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'")
    x, y = DATASET_REGISTRY[dataset_name](random_state=seed)
    x = StandardScaler().fit_transform(np.asarray(x, dtype=np.float32))
    y = pd.factorize(np.asarray(y))[0].astype(int)
    return x.astype(np.float32), y


def valid_ks(num_features: int) -> list[int]:
    return [k for k in PAPER_KS if k <= num_features]


def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    d = max(y_true.max(), y_pred.max()) + 1
    contingency = np.zeros((d, d), dtype=np.int64)
    for pred, true in zip(y_pred, y_true):
        contingency[pred, true] += 1
    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    return contingency[row_ind, col_ind].sum() / y_true.shape[0]


def evaluate_selected_features(
    x: np.ndarray,
    y: np.ndarray,
    selected_idx: np.ndarray,
    *,
    num_clusters: int,
    kmeans_runs: int,
    seed: int,
) -> dict[str, float]:
    x_sel = x[:, selected_idx]
    accs = []
    nmis = []
    for run_idx in range(kmeans_runs):
        pred = KMeans(
            n_clusters=num_clusters,
            n_init=1,
            random_state=seed + run_idx,
        ).fit_predict(x_sel)
        accs.append(clustering_accuracy(y, pred))
        nmis.append(normalized_mutual_info_score(y, pred))
    return {
        "AccuracyMean": float(np.mean(accs)),
        "AccuracyStd": float(np.std(accs)),
        "NMIMean": float(np.mean(nmis)),
        "NMIStd": float(np.std(nmis)),
    }


def build_affinity_graph(x: np.ndarray):
    # Use the package's default graph construction to stay aligned with the
    # published scikit-feature baselines while avoiding their missing-W bug.
    return construct_W(x)


def laplacian_score_select(x: np.ndarray, k_selected: int, W) -> np.ndarray:
    ranking = lap_score.lap_score(x, mode="rank", W=W)
    ranking = np.asarray(ranking).astype(int).reshape(-1)
    return ranking[:k_selected]


def mcfs_select(
    x: np.ndarray,
    k_selected: int,
    num_clusters: int,
    seed: int,
    W,
) -> np.ndarray:
    ranking = MCFS.mcfs(
        x,
        n_selected_features=k_selected,
        mode="rank",
        n_clusters=num_clusters,
        W=W,
    )
    ranking = np.asarray(ranking).astype(int).reshape(-1)
    return ranking[:k_selected]


def ndfs_select(
    x: np.ndarray,
    k_selected: int,
    num_clusters: int,
    seed: int,
    W,
) -> np.ndarray:
    original_kmeans = sklearn_cluster.KMeans

    def compat_kmeans(*args, **kwargs):
        kwargs.pop("precompute_distances", None)
        kwargs.pop("n_jobs", None)
        kwargs.setdefault("random_state", seed)
        return original_kmeans(*args, **kwargs)

    sklearn_cluster.KMeans = compat_kmeans
    try:
        ranking = NDFS.ndfs(
            x,
            mode="rank",
            W=W,
            n_clusters=num_clusters,
            alpha=1,
            beta=1,
            gamma=1,
        )
    finally:
        sklearn_cluster.KMeans = original_kmeans
    ranking = np.asarray(ranking).astype(int).reshape(-1)
    return ranking[:k_selected]


def lscae_select(
    x: np.ndarray,
    k_selected: int,
    *,
    model_name: str,
    epochs: int,
    seed: int,
) -> np.ndarray:
    set_seed(seed)
    tensor_x = torch.as_tensor(x, dtype=torch.float32)
    dataset = TensorDataset(tensor_x)
    batch_size = min(64, x.shape[0])
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=x.shape[0] > batch_size,
    )

    selector = Lscae(
        input_dim=x.shape[1],
        kwargs={
            "input_dim": x.shape[1],
            "k_selected": k_selected,
            "decoder_lr": 0.01,
            "selector_lr": 1.0,
            "batch_size": batch_size,
            "num_epochs": epochs,
            "model": model_name,
            "laplacian_k": min(50, batch_size),
            "scale_k": min(2, min(50, batch_size)),
            "verbose": False,
        },
    )
    selected = sorted(int(i) for i in selector.select_features(dataloader))
    return np.asarray(selected, dtype=int)


def select_features(
    method: str,
    x: np.ndarray,
    *,
    k_selected: int,
    num_clusters: int,
    epochs: int,
    seed: int,
    W=None,
) -> np.ndarray:
    if method == "LS":
        return laplacian_score_select(x, k_selected, W)
    if method == "MCFS":
        return mcfs_select(x, k_selected, num_clusters, seed, W)
    if method == "NDFS":
        return ndfs_select(x, k_selected, num_clusters, seed, W)
    if method == "CAE":
        return lscae_select(x, k_selected, model_name="cae", epochs=epochs, seed=seed)
    if method == "LSCAE":
        return lscae_select(x, k_selected, model_name="lscae", epochs=epochs, seed=seed)
    raise ValueError(f"Unknown method '{method}'")


def run_method(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int,
    kmeans_runs: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    n_clusters = np.unique(y).shape[0]
    W = build_affinity_graph(x) if method in {"LS", "MCFS", "NDFS"} else None
    rows = []
    for k_selected in valid_ks(x.shape[1]):
        selected_idx = select_features(
            method,
            x,
            k_selected=k_selected,
            num_clusters=n_clusters,
            epochs=epochs,
            seed=seed,
            W=W,
        )
        metrics = evaluate_selected_features(
            x,
            y,
            selected_idx,
            num_clusters=n_clusters,
            kmeans_runs=kmeans_runs,
            seed=seed,
        )
        row = {
            "Method": method,
            "NumSelected": k_selected,
            **metrics,
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        ["AccuracyMean", "NMIMean", "NumSelected"],
        ascending=[False, False, True],
    )
    best = df.iloc[0].to_dict()
    return df, best


def main():
    args = parse_args()
    set_seed(args.seed)

    x, y = load_full_dataset(args.dataset, args.seed)
    print(
        f"Loaded {args.dataset}: {x.shape[0]} samples, {x.shape[1]} features, "
        f"{np.unique(y).shape[0]} clusters/classes"
    )

    all_rows = []
    best_rows = []
    for method in args.methods:
        print(f"\nRunning {method}...")
        method_df, best = run_method(
            method,
            x,
            y,
            epochs=args.epochs,
            kmeans_runs=args.kmeans_runs,
            seed=args.seed,
        )
        all_rows.append(method_df)
        best_rows.append(best)
        print(
            f"Best {method}: ACC={best['AccuracyMean']:.4f} "
            f"(k={int(best['NumSelected'])}), NMI={best['NMIMean']:.4f}"
        )

    raw_df = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.DataFrame(best_rows).sort_values(
        ["AccuracyMean", "NMIMean"],
        ascending=[False, False],
    )
    summary_df.insert(0, "Dataset", args.dataset)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out or RESULTS_DIR / f"baseline_{args.dataset}.csv")
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    raw_df.to_csv(out_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nSummary")
    print(summary_df.to_string(index=False))
    print(f"\nSaved raw baseline sweep to {out_path}")
    print(f"Saved summary baseline table to {summary_path}")


if __name__ == "__main__":
    main()
