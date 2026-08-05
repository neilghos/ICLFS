from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
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


SUBTAB_ROOT = Path(__file__).resolve().parent
PROCESSED_ROOT = SUBTAB_ROOT / "processed"
RESULTS_DIR = SUBTAB_ROOT / "results"
DEFAULT_TOP_K = 256
DEFAULT_SEEDS = list(range(10))


@dataclass
class ProcessedSplitBundle:
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
    parser.add_argument("--dataset", default="california", help="Dataset key under subtab/processed.")
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_ROOT)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=100)
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


def load_processed_splits(processed_root: Path, dataset_name: str) -> ProcessedSplitBundle:
    dataset_dir = processed_root / dataset_name
    split_path = dataset_dir / "splits.npz"
    metadata_path = dataset_dir / "metadata.json"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Processed split file not found: {split_path}. Run prepare_switchtab_data.py first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Processed metadata file not found: {metadata_path}. Run prepare_switchtab_data.py first."
        )

    bundle = np.load(split_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return ProcessedSplitBundle(
        dataset_name=dataset_name,
        task=metadata["task"],
        x_train=np.asarray(bundle["x_train"], dtype=np.float32),
        x_valid=np.asarray(bundle["x_valid"], dtype=np.float32),
        x_test=np.asarray(bundle["x_test"], dtype=np.float32),
        y_train=np.asarray(bundle["y_train"]),
        y_valid=np.asarray(bundle["y_valid"]),
        y_test=np.asarray(bundle["y_test"]),
        metadata=metadata,
    )


def resolve_backbone_config(bundle: ProcessedSplitBundle, args: argparse.Namespace) -> BackboneConfig:
    return BackboneConfig(
        encoder_hidden_dim=args.encoder_hidden_dim,
        latent_dim=args.latent_dim,
        projector_hidden_dim=args.projector_hidden_dim,
        projector_output_dim=args.projector_output_dim,
    )


def build_inverted_loader(x: np.ndarray, config_path: str | None = None) -> DataLoader:
    train_ds = InvertedFeatureDataset(
        x,
        augmentor_config=get_augmentor_config(config_path),
    )
    return DataLoader(train_ds, batch_size=len(train_ds), shuffle=False)


def train_unsupervised_feature_model(
    x_train: np.ndarray,
    *,
    epochs: int,
    seed: int,
    encoder_hidden_dim: int,
    latent_dim: int,
    n_heads: int,
    projector_hidden_dim: int,
    projector_output_dim: int,
    temperature: float,
    decorrelation_weight: float,
    config_path: str | None = None,
    device_name: str = "auto",
) -> tuple[InvertedFeatureExpert, np.ndarray, np.ndarray]:
    set_seed(seed)
    if device_name == "cuda":
        candidate_devices = ["cuda"]
    elif device_name == "cpu":
        candidate_devices = ["cpu"]
    else:
        candidate_devices = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]

    last_error: RuntimeError | None = None
    for candidate in candidate_devices:
        device = torch.device(candidate)
        try:
            loader = build_inverted_loader(x_train, config_path=config_path)
            model = InvertedFeatureExpert(
                n_patients=x_train.shape[0],
                latent_dim=latent_dim,
                encoder_hidden_dim=encoder_hidden_dim,
                projector_hidden_dim=projector_hidden_dim,
                projector_out_dim=projector_output_dim,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

            model.train()
            epoch_bar = tqdm(
                range(epochs),
                desc=f"train[{candidate}]",
                leave=False,
            )
            for epoch in epoch_bar:
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
                        epoch_loss += float(total_loss.item())
                    optimizer.step()

                epoch_bar.set_postfix(loss=f"{epoch_loss / len(loader):.4f}")
                if epoch % 20 == 0 or epoch == epochs - 1:
                    print(
                        f"ICL Epoch {epoch:03d} | Loss: {epoch_loss / len(loader):.4f} | device={candidate}"
                    )

            feature_scores = get_feature_scores(model, loader)
            with torch.no_grad():
                model.eval()
                batch = next(iter(loader))
                anchor = batch[0].to(device)
                _, projector_embeddings = model(anchor)
                feature_embeddings = projector_embeddings.detach().cpu().numpy()
            del batch, anchor, projector_embeddings, model, optimizer, loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return None, feature_scores, feature_embeddings
        except RuntimeError as err:
            last_error = err
            is_cuda_oom = candidate == "cuda" and "out of memory" in str(err).lower()
            locals_to_clear = ["model", "optimizer", "loader", "batch", "views", "anchor", "pos_views", "neg_view"]
            for name in locals_to_clear:
                if name in locals():
                    del locals()[name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not is_cuda_oom or candidate == candidate_devices[-1]:
                raise
            print("CUDA OOM in unsupervised backbone training. Falling back to CPU for this run.")

    assert last_error is not None
    raise last_error


def build_sample_representations(
    x: np.ndarray,
    feature_embeddings: np.ndarray,
    selected_idx: np.ndarray,
) -> np.ndarray:
    selected_idx = np.asarray(selected_idx, dtype=np.int64)
    selected_x = np.asarray(x[:, selected_idx], dtype=np.float32)
    selected_feature_embeddings = np.asarray(feature_embeddings[selected_idx], dtype=np.float32)
    scale = np.sqrt(max(1, selected_idx.shape[0]))
    return (selected_x @ selected_feature_embeddings) / scale


def resolve_selected_feature_count(
    num_features: int,
    *,
    top_k: int | None,
    selection_ratio: float,
) -> tuple[int, str]:
    if top_k is not None:
        return min(top_k, num_features), f"top_k={top_k}"
    if not 0.0 < selection_ratio <= 1.0:
        raise ValueError(f"selection_ratio must be in (0, 1], got {selection_ratio}.")
    selected = max(1, int(np.ceil(selection_ratio * num_features)))
    return min(selected, num_features), f"selection_ratio={selection_ratio}"


def fit_supervised_probe(
    task: str,
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_valid: np.ndarray,
    y_valid: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int,
    run_label: str,
    probe_hidden_dim: int,
    probe_max_iter: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scaler = StandardScaler()
    z_train_scaled = scaler.fit_transform(z_train)
    z_valid_scaled = scaler.transform(z_valid)
    z_test_scaled = scaler.transform(z_test)

    if task == "classification":
        grid = [1e-5, 1e-4, 1e-3]
        best = None
        probe_bar = tqdm(grid, desc=f"eval[{run_label}]", leave=False)
        for alpha in probe_bar:
            clf = MLPClassifier(
                hidden_layer_sizes=(probe_hidden_dim,),
                activation="relu",
                solver="adam",
                alpha=alpha,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=probe_max_iter,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=seed,
            )
            clf.fit(z_train_scaled, y_train)
            valid_pred = clf.predict(z_valid_scaled)
            valid_metrics = calculate_metrics(task, y_valid, valid_pred)
            probe_bar.set_postfix(alpha=alpha, valid=f"{valid_metrics['score']:.4f}")
            if best is None or valid_metrics["score"] > best["valid_metrics"]["score"]:
                test_pred = clf.predict(z_test_scaled)
                best = {
                    "hyperparam": alpha,
                    "valid_metrics": valid_metrics,
                    "test_metrics": calculate_metrics(task, y_test, test_pred),
                }
        assert best is not None
        return best["valid_metrics"], best["test_metrics"], {
            "probe": "mlp",
            "alpha": best["hyperparam"],
            "hidden_dim": probe_hidden_dim,
        }

    grid = [1e-5, 1e-4, 1e-3]
    best = None
    probe_bar = tqdm(grid, desc=f"eval[{run_label}]", leave=False)
    for alpha in probe_bar:
        reg = MLPRegressor(
            hidden_layer_sizes=(probe_hidden_dim,),
            activation="relu",
            solver="adam",
            alpha=alpha,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=probe_max_iter,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=seed,
        )
        reg.fit(z_train_scaled, y_train)
        valid_pred = reg.predict(z_valid_scaled)
        valid_metrics = calculate_metrics(task, y_valid, valid_pred)
        probe_bar.set_postfix(alpha=alpha, valid=f"{valid_metrics['score']:.4f}")
        if best is None or valid_metrics["score"] > best["valid_metrics"]["score"]:
            test_pred = reg.predict(z_test_scaled)
            best = {
                "hyperparam": alpha,
                "valid_metrics": valid_metrics,
                "test_metrics": calculate_metrics(task, y_test, test_pred),
            }
    assert best is not None
    return best["valid_metrics"], best["test_metrics"], {
        "probe": "mlp",
        "alpha": best["hyperparam"],
        "hidden_dim": probe_hidden_dim,
    }


def main() -> None:
    args = parse_args()
    if not args.seeds:
        raise ValueError("Provide at least one seed.")

    bundle = load_processed_splits(args.processed_root, args.dataset)
    backbone = resolve_backbone_config(bundle, args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loaded {bundle.dataset_name}: train={bundle.x_train.shape[0]} "
        f"valid={bundle.x_valid.shape[0]} test={bundle.x_test.shape[0]} "
        f"features={bundle.x_train.shape[1]} task={bundle.task}"
    )
    print(
        "Backbone config | "
        f"encoder_hidden_dim={backbone.encoder_hidden_dim} "
        f"latent_dim={backbone.latent_dim} "
        f"projector_hidden_dim={backbone.projector_hidden_dim} "
        f"projector_output_dim={backbone.projector_output_dim}"
    )

    run_rows: list[dict[str, Any]] = []
    seed_bar = tqdm(list(enumerate(args.seeds)), desc="seeds")
    for run_idx, seed in seed_bar:
        seed_bar.set_postfix(seed=seed)
        print(f"\nRun {run_idx + 1}/{len(args.seeds)} | seed={seed}")
        _, feature_scores, feature_embeddings = train_unsupervised_feature_model(
            bundle.x_train,
            epochs=args.epochs,
            seed=seed,
            encoder_hidden_dim=backbone.encoder_hidden_dim,
            latent_dim=backbone.latent_dim,
            n_heads=args.n_heads,
            projector_hidden_dim=backbone.projector_hidden_dim,
            projector_output_dim=backbone.projector_output_dim,
            temperature=args.temperature,
            decorrelation_weight=args.decorrelation_weight,
            config_path=args.config_path,
            device_name=args.device,
        )

        k_selected, selection_mode = resolve_selected_feature_count(
            feature_scores.shape[0],
            top_k=args.top_k,
            selection_ratio=args.selection_ratio,
        )
        selected_idx = get_topk_feature_indices(feature_scores, k_selected)

        z_train = build_sample_representations(bundle.x_train, feature_embeddings, selected_idx)
        z_valid = build_sample_representations(bundle.x_valid, feature_embeddings, selected_idx)
        z_test = build_sample_representations(bundle.x_test, feature_embeddings, selected_idx)

        valid_metrics, test_metrics, probe_info = fit_supervised_probe(
            bundle.task,
            z_train,
            bundle.y_train,
            z_valid,
            bundle.y_valid,
            z_test,
            bundle.y_test,
            seed=seed,
            run_label=f"seed{seed}",
            probe_hidden_dim=args.probe_hidden_dim,
            probe_max_iter=args.probe_max_iter,
        )

        print(f"Selected features = {k_selected} via {selection_mode} | probe={probe_info}")
        print(f"[valid] {summarize_metrics(bundle.task, valid_metrics)}")
        print(f"[test ] {summarize_metrics(bundle.task, test_metrics)}")

        run_rows.append(
            {
                "Dataset": bundle.dataset_name,
                "Task": bundle.task,
                "Method": f"ICL-SubTab-{backbone.projector_output_dim}",
                "Run": run_idx,
                "Seed": seed,
                "NumSelected": k_selected,
                "SelectionMode": selection_mode,
                "ValidationScore": float(valid_metrics["score"]),
                "TestScore": float(test_metrics["score"]),
                "ValidationMetric": float(
                    valid_metrics["accuracy"] if bundle.task == "classification" else valid_metrics["rmse"]
                ),
                "TestMetric": float(
                    test_metrics["accuracy"] if bundle.task == "classification" else test_metrics["rmse"]
                ),
            }
        )

        del feature_scores, feature_embeddings, selected_idx, z_train, z_valid, z_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    runs_df = pd.DataFrame(run_rows)
    summary_df = runs_df.agg(
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
        "Method": f"ICL-SubTab-{backbone.projector_output_dim}",
        "NumSelected": int(run_rows[0]["NumSelected"]),
        "SelectionMode": run_rows[0]["SelectionMode"],
        "ValidationScoreMean": float(summary_df.loc["mean", "ValidationScore"]),
        "ValidationScoreStd": float(summary_df.loc["std", "ValidationScore"]),
        "TestScoreMean": float(summary_df.loc["mean", "TestScore"]),
        "TestScoreStd": float(summary_df.loc["std", "TestScore"]),
        "ValidationMetricMean": float(summary_df.loc["mean", "ValidationMetric"]),
        "ValidationMetricStd": float(summary_df.loc["std", "ValidationMetric"]),
        "TestMetricMean": float(summary_df.loc["mean", "TestMetric"]),
        "TestMetricStd": float(summary_df.loc["std", "TestMetric"]),
    }

    runs_path = args.out_dir / f"{bundle.dataset_name}_subtab_runs.csv"
    summary_path = args.out_dir / f"{bundle.dataset_name}_subtab_summary.csv"
    runs_df.to_csv(runs_path, index=False)
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)

    print("\nSummary")
    print(pd.DataFrame([summary_row]).to_string(index=False))
    print(f"Saved per-run results to {runs_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
