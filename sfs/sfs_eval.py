from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sfs.data import load_gradenfs_splits
from src.data.api import InvertedFeatureDataset
from src.extractor import get_feature_scores, get_topk_feature_indices
from src.loss import contrastive_loss, decorrelation_loss
from src.models import InvertedFeatureExpert
from src.runtime_config import get_augmentor_config
from evaluation_model import ExtraTree_Model, KNN_Model, SVM_Model


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_KS = (20, 40, 60, 80, 100, 150, 200, 250, 300)
DEFAULT_DATASETS = (
    "basehock",
    "pcmac",
    "prostate_ge",
    "tox",
    "madelon",
    "isolet",
    "coil20",
    "usps",
)
SMALL_DATASET_ARCH_DATASETS = {"prostate_ge", "tox"}
SMALL_DATASET_ARCH_PRESET = {
    "encoder_hidden_dim": 16,
    "latent_dim": 512,
    "projector_hidden_dim": 128,
    "projector_output_dim": 16,
    "decorrelation_weight": 0.20,
}
LARGE_DATASET_ARCH_DATASETS = {"basehock", "pcmac", "madelon", "isolet", "usps"}
LARGE_DATASET_ARCH_PRESET = {
    "encoder_hidden_dim": 64,
    "latent_dim": 1440,
    "projector_hidden_dim": 2048,
    "projector_output_dim": 32,
    "decorrelation_weight": 0.40,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="SFS evaluator using GradEnFS preprocessing with the shared ICL saliency backbone."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="GradEnFS dataset name. Use 'all' or omit it to run the default SFS dataset list.",
    )
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--encoder-hidden-dim", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--projector-hidden-dim", type=int, default=256)
    parser.add_argument("--projector-output-dim", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument(
        "--decorrelation-weight",
        type=float,
        default=0.40,
        help="Weight for the decorrelation loss.",
    )
    parser.add_argument(
        "--evaluation-model",
        default="svm",
        choices=("svm", "knn", "extratree"),
    )
    parser.add_argument("--k-list", type=int, nargs="*", default=list(DEFAULT_KS))
    parser.add_argument(
        "--config-path",
        default=None,
        help="Optional path to config.yaml. Defaults to the repo config.yaml.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional combined summary CSV output path. Defaults to sfs/results/SFS_all_summary.csv",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def architecture_for_dataset(
    dataset_name: str,
    *,
    encoder_hidden_dim: int,
    latent_dim: int,
    projector_hidden_dim: int,
    projector_output_dim: int,
) -> dict[str, int | float]:
    if dataset_name in SMALL_DATASET_ARCH_DATASETS:
        return dict(SMALL_DATASET_ARCH_PRESET)
    if dataset_name in LARGE_DATASET_ARCH_DATASETS:
        return dict(LARGE_DATASET_ARCH_PRESET)
    return {
        "encoder_hidden_dim": encoder_hidden_dim,
        "latent_dim": latent_dim,
        "projector_hidden_dim": projector_hidden_dim,
        "projector_output_dim": projector_output_dim,
        "decorrelation_weight": None,
    }


def valid_ks(num_features: int, requested_ks: list[int]) -> list[int]:
    return [k for k in sorted(set(requested_ks)) if k <= num_features]


def build_inverted_loader(x: np.ndarray, config_path: str | None = None) -> DataLoader:
    train_ds = InvertedFeatureDataset(
        x,
        augmentor_config=get_augmentor_config(config_path),
    )
    return DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)


def build_evaluation_model(
    model_name: str,
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
):
    if model_name == "svm":
        return SVM_Model(x_train, y_train, x_test, y_test)
    if model_name == "knn":
        return KNN_Model(x_train, y_train, x_test, y_test)
    if model_name == "extratree":
        return ExtraTree_Model(x_train, y_train, x_test, y_test)
    raise ValueError(f"Unknown evaluation model '{model_name}'")


def train_and_score_features(
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
    decorrelation_weight: float = 0.005,
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
            views = [view.to(device) for view in batch]
            anchor = views[0]
            pos_views = views[1:5]
            neg_view = views[5]

            optimizer.zero_grad()
            num_pos = len(pos_views)
            for v in pos_views:
                _, z_anchor = model(anchor)
                _, z_neg = model(neg_view)
                _, z_v = model(v)

                l_con = contrastive_loss(z_anchor, z_v, temperature=temperature, z_neg=z_neg)
                l_div = decorrelation_loss(z_anchor)
                total_loss = (l_con / num_pos) + (decorrelation_weight * l_div)
                total_loss.backward()
                epoch_loss += total_loss.item()
            optimizer.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"SFS Epoch {epoch:03d} | Loss: {epoch_loss / len(loader):.4f}")
    return get_feature_scores(model, loader)


def train_and_rank_features(
    x: np.ndarray,
    **kwargs,
) -> np.ndarray:
    feature_scores = train_and_score_features(x, **kwargs)
    return get_topk_feature_indices(feature_scores, feature_scores.shape[0])


def evaluate_selected_features(
    evaluation_model,
    ranking: np.ndarray,
    *,
    k_list: list[int],
) -> list[dict[str, float | int | str]]:
    rows = []
    for k_selected in k_list:
        selected_idx = np.asarray(ranking[:k_selected], dtype=int)
        accuracy = float(evaluation_model.train_and_test(selected_idx))
        rows.append(
            {
                "Method": "ICLFS-sfs",
                "NumSelected": k_selected,
                "AccuracyMean": accuracy,
                "AccuracyStd": 0.0,
            }
        )
    return rows


def main():
    args = parse_args()
    repeat = len(args.seeds)
    if repeat == 0:
        raise ValueError("Provide at least one seed via --seeds.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.dataset and args.dataset.lower() != "all":
        datasets = [args.dataset.lower()]
    else:
        datasets = list(DEFAULT_DATASETS)
    summary_rows = []

    for dataset_name in datasets:
        dataset_seed = args.seeds[0]
        set_seed(dataset_seed)
        bundle = load_gradenfs_splits(dataset_name, seed=dataset_seed)
        ks = valid_ks(bundle.input_dim, args.k_list)
        if not ks:
            raise ValueError(
                f"No valid selected-feature counts for dataset with {bundle.input_dim} features."
            )

        arch = architecture_for_dataset(
            dataset_name,
            encoder_hidden_dim=args.encoder_hidden_dim,
            latent_dim=args.latent_dim,
            projector_hidden_dim=args.projector_hidden_dim,
            projector_output_dim=args.projector_output_dim,
        )
        decorrelation_weight = (
            args.decorrelation_weight
            if arch["decorrelation_weight"] is None
            else float(arch["decorrelation_weight"])
        )

        print(
            f"Loaded {dataset_name}: train={bundle.x_train.shape[0]} "
            f"valid={bundle.x_valid.shape[0]} test={bundle.x_test.shape[0]} "
            f"features={bundle.input_dim} classes={bundle.output_dim}"
        )
        print(
            "Architecture preset | "
            f"encoder_hidden_dim={arch['encoder_hidden_dim']} "
            f"latent_dim={arch['latent_dim']} "
            f"projector_hidden_dim={arch['projector_hidden_dim']} "
            f"projector_output_dim={arch['projector_output_dim']} "
            f"decorrelation_weight={decorrelation_weight}"
        )

        per_run_frames = []
        for run_idx, seed in enumerate(args.seeds):
            print(f"\nRun {run_idx + 1}/{repeat} | seed={seed}")
            ranking = train_and_rank_features(
                bundle.x_train,
                epochs=args.epochs,
                seed=seed,
                encoder_hidden_dim=arch["encoder_hidden_dim"],
                latent_dim=arch["latent_dim"],
                n_heads=args.n_heads,
                projector_hidden_dim=arch["projector_hidden_dim"],
                projector_output_dim=arch["projector_output_dim"],
                temperature=args.temperature,
                decorrelation_weight=decorrelation_weight,
                config_path=args.config_path,
            )
            evaluation_model = build_evaluation_model(
                args.evaluation_model,
                x_train=bundle.x_train,
                y_train=bundle.y_train_index,
                x_test=bundle.x_test,
                y_test=bundle.y_test_index,
            )
            run_rows = evaluate_selected_features(
                evaluation_model,
                ranking,
                k_list=ks,
            )
            run_df = pd.DataFrame(run_rows)
            run_df["Run"] = run_idx
            per_run_frames.append(run_df)

        raw_df = pd.concat(per_run_frames, ignore_index=True)
        summary_df = (
            raw_df.groupby(["Method", "NumSelected"])["AccuracyMean"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "AccuracyMean", "std": "AccuracyStd"})
        )
        summary_df["AccuracyStd"] = summary_df["AccuracyStd"].fillna(0.0)
        summary_df = summary_df.sort_values(
            ["AccuracyMean", "NumSelected"],
            ascending=[False, True],
        )
        best = summary_df.iloc[0].to_dict()
        best_row = pd.DataFrame(
            [
                {
                    "Dataset": dataset_name,
                    "Method": best["Method"],
                    "NumSelected": int(best["NumSelected"]),
                    "AccuracyMean": float(best["AccuracyMean"]),
                    "AccuracyStd": float(best["AccuracyStd"]),
                }
            ]
        )
        summary_rows.append(best_row.iloc[0].to_dict())

        summary_path = RESULTS_DIR / f"SFS_{dataset_name}_summary.csv"
        best_row.to_csv(summary_path, index=False)
        print("\nSummary")
        print(best_row.to_string(index=False))
        print(f"Saved summary SFS table to {summary_path}")

    if len(summary_rows) > 1:
        combined_df = pd.DataFrame(summary_rows)[
            ["Dataset", "Method", "NumSelected", "AccuracyMean", "AccuracyStd"]
        ]
        combined_path = Path(args.out or RESULTS_DIR / "SFS_all_summary.csv")
        combined_df.to_csv(combined_path, index=False)
        print("\nCombined Summary")
        print(combined_df.to_string(index=False))
        print(f"Saved combined SFS table to {combined_path}")


if __name__ == "__main__":
    main()
