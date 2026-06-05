from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import data_loaders  # noqa: F401
from data.api import DATASET_REGISTRY, InvertedFeatureDataset
from extractor import get_feature_scores, get_topk_feature_indices
from loss import contrastive_loss, diversity_loss
from models import InvertedFeatureExpert
from runtime_config import get_augmentor_config


RESULTS_DIR = Path("/home/utsab/Desktop/ICLFE/ICLFE/src/results")
PAPER_KS = (50, 100, 150, 200, 250, 300)
PAPER_NUM_KMEANS_RUNS = 20
PAPER_DATASETS = [
    "coil20",
    "allaml",
    "arcene",
    "basehock",
    "lung",
    "nci9",
    "pcmac",
    "prostate",
    "relathe",
    "warppie10p",
]
SHORT_DATASET_ARCH_DATASETS = {"allaml", "arcene", "lung", "nci9", "prostate", "warppie10p","orl"}
SHORT_DATASET_ARCH_PRESET = {
    "encoder_hidden_dim": 16,
    "latent_dim": 512,
    "projector_hidden_dim": 128,
    "projector_output_dim": 16,
    "diversity_weight": 0.20,
}
LARGE_DATASET_ARCH_DATASETS = {"basehock", "coil20", "pcmac", "relathe"}
LARGE_DATASET_ARCH_PRESET = {
    "encoder_hidden_dim": 64,
    "latent_dim": 1440,
    "projector_hidden_dim": 2048,
    "projector_output_dim": 32,
    "diversity_weight": 0.40,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone ICL evaluator using the paper-style unsupervised clustering protocol."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Registered dataset name. If omitted, runs the full hardcoded paper dataset list.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmeans-runs", type=int, default=PAPER_NUM_KMEANS_RUNS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--encoder-hidden-dim", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--projector-hidden-dim", type=int, default=256)
    parser.add_argument("--projector-output-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--diversity-weight", type=float, default=0.40,
                        help="Weight for the redundancy pruning (de-correlation) loss.")
    parser.add_argument(
        "--config-path",
        default=None,
        help="Optional path to config.yaml. Defaults to the repo config.yaml.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional summary CSV output path. Defaults to results/icl_<dataset>_summary.csv",
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


def architecture_for_dataset(
    dataset_name: str,
    *,
    encoder_hidden_dim: int,
    latent_dim: int,
    projector_hidden_dim: int,
    projector_output_dim: int,
) -> dict[str, int | float]:
    if dataset_name in SHORT_DATASET_ARCH_DATASETS:
        return dict(SHORT_DATASET_ARCH_PRESET)
    if dataset_name in LARGE_DATASET_ARCH_DATASETS:
        return dict(LARGE_DATASET_ARCH_PRESET)
    return {
        "encoder_hidden_dim": encoder_hidden_dim,
        "latent_dim": latent_dim,
        "projector_hidden_dim": projector_hidden_dim,
        "projector_output_dim": projector_output_dim,
        "diversity_weight": None,
    }


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


def build_inverted_loader(x: np.ndarray, config_path: str | None = None) -> DataLoader:
    train_ds = InvertedFeatureDataset(
        x,
        augmentor_config=get_augmentor_config(config_path),
    )
    return DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)


def train_and_rank_features(
    x: np.ndarray,
    *,
    epochs: int,
    seed: int,
    encoder_hidden_dim: int,
    latent_dim: int,
    n_heads: int,
    projector_hidden_dim: int,
    projector_output_dim: int,
    temperature: float,
    diversity_weight: float = 0.005,
    config_path: str | None = None,
) -> np.ndarray:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_inverted_loader(x, config_path=config_path)
    model = InvertedFeatureExpert(
        n_patients=x.shape[0],
        latent_dim=latent_dim,
        n_heads=n_heads,
        encoder_hidden_dim=encoder_hidden_dim,
        projector_hidden_dim=projector_hidden_dim,
        projector_out_dim=projector_output_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in loader:
            # batch is [Anchor, Pos1, Pos2, Pos3, Pos4, Neg]
            views = [view.to(device) for view in batch]
            anchor = views[0]
            pos_views = views[1:5]
            neg_view = views[5]
            
            optimizer.zero_grad()
            
            # Memory Optimization: Process views one-by-one and accumulate gradients
            num_pos = len(pos_views)
            for v in pos_views:
                _, z_anchor = model(anchor)
                _, z_neg = model(neg_view)
                _, z_v = model(v)
                
                # Task 1: Sovereignty (Contrastive)
                l_con = contrastive_loss(z_anchor, z_v, temperature=temperature, z_neg=z_neg)
                
                # Task 2: Diversity (Redundancy Pruning)
                l_div = diversity_loss(z_anchor)
                
                total_loss = (l_con / num_pos) + (diversity_weight * l_div)
                total_loss.backward()
                epoch_loss += total_loss.item()
            
            optimizer.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"ICL Epoch {epoch:03d} | Loss: {epoch_loss / len(loader):.4f}")

    feature_scores = get_feature_scores(model, loader)
    max_k = max(valid_ks(x.shape[1]))
    return get_topk_feature_indices(feature_scores, max_k)


def main():
    args = parse_args()
    set_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    datasets = [args.dataset] if args.dataset else list(PAPER_DATASETS)
    unknown = sorted(set(datasets) - set(DATASET_REGISTRY))
    if unknown:
        raise ValueError(f"Unknown datasets requested: {', '.join(unknown)}")

    summary_rows = []
    for dataset_name in datasets:
        arch = architecture_for_dataset(
            dataset_name,
            encoder_hidden_dim=args.encoder_hidden_dim,
            latent_dim=args.latent_dim,
            projector_hidden_dim=args.projector_hidden_dim,
            projector_output_dim=args.projector_output_dim,
        )
        diversity_weight = (
            args.diversity_weight
            if arch["diversity_weight"] is None
            else float(arch["diversity_weight"])
        )

        x, y = load_full_dataset(dataset_name, args.seed)
        n_clusters = np.unique(y).shape[0]
        ks = valid_ks(x.shape[1])
        if not ks:
            raise ValueError(
                f"No valid paper-style feature counts for dataset with {x.shape[1]} features."
            )

        print(
            f"Loaded {dataset_name}: {x.shape[0]} samples, {x.shape[1]} features, "
            f"{n_clusters} clusters/classes"
        )
        print(
            "Architecture preset | "
            f"encoder_hidden_dim={arch['encoder_hidden_dim']} "
            f"latent_dim={arch['latent_dim']} "
            f"projector_hidden_dim={arch['projector_hidden_dim']} "
            f"projector_output_dim={arch['projector_output_dim']} "
            f"diversity_weight={diversity_weight}"
        )

        ranking = train_and_rank_features(
            x,
            epochs=args.epochs,
            seed=args.seed,
            encoder_hidden_dim=arch["encoder_hidden_dim"],
            latent_dim=arch["latent_dim"],
            n_heads=args.n_heads,
            projector_hidden_dim=arch["projector_hidden_dim"],
            projector_output_dim=arch["projector_output_dim"],
            temperature=args.temperature,
            diversity_weight=diversity_weight,
            config_path=args.config_path,
        )

        rows = []
        for k_selected in ks:
            selected_idx = np.asarray(ranking[:k_selected], dtype=int)
            metrics = evaluate_selected_features(
                x,
                y,
                selected_idx,
                num_clusters=n_clusters,
                kmeans_runs=args.kmeans_runs,
                seed=args.seed,
            )
            rows.append(
                {
                    "Method": "ICL",
                    "NumSelected": k_selected,
                    **metrics,
                }
            )

        raw_df = pd.DataFrame(rows).sort_values(
            ["AccuracyMean", "NMIMean", "NumSelected"],
            ascending=[False, False, True],
        )
        best = raw_df.iloc[0].to_dict()
        summary_df = pd.DataFrame([{**best, "Dataset": dataset_name}])[
            ["Dataset", "Method", "NumSelected", "AccuracyMean", "AccuracyStd", "NMIMean", "NMIStd"]
        ]
        summary_rows.append(summary_df.iloc[0].to_dict())

        summary_path = RESULTS_DIR / f"icl_{dataset_name}_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        print("\nSummary")
        print(summary_df.to_string(index=False))
        print(f"Saved summary ICL table to {summary_path}")

    if len(summary_rows) > 1:
        combined_df = pd.DataFrame(summary_rows)[
            ["Dataset", "Method", "NumSelected", "AccuracyMean", "AccuracyStd", "NMIMean", "NMIStd"]
        ]
        combined_path = Path(args.out or RESULTS_DIR / "icl_all_summary.csv")
        combined_df.to_csv(combined_path, index=False)
        print("\nCombined Summary")
        print(combined_df.to_string(index=False))
        print(f"Saved combined ICL table to {combined_path}")


if __name__ == "__main__":
    main()
