import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from extractor import extract_topk_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from xgboost import XGBClassifier


def get_probe_grid():
    """
    Fixed downstream classifier suite for representation benchmarking.
    """
    return {
        "LogReg": [
            {"C": 0.1, "max_iter": 2000},
            {"C": 1.0, "max_iter": 2000},
            {"C": 10.0, "max_iter": 2000},
        ],
        "LinearSVM": [
            {"C": 0.1, "max_iter": 5000},
            {"C": 1.0, "max_iter": 5000},
            {"C": 10.0, "max_iter": 5000},
        ],
        "SVM-RBF": [
            {"C": 0.1, "gamma": "scale"},
            {"C": 1.0, "gamma": "scale"},
            {"C": 10.0, "gamma": "scale"},
            {"C": 1.0, "gamma": "auto"},
        ],
        "kNN": [
            {"n_neighbors": 3, "weights": "uniform"},
            {"n_neighbors": 5, "weights": "uniform"},
            {"n_neighbors": 11, "weights": "distance"},
        ],
        "RandomForest": [
            {"n_estimators": 200, "max_depth": None},
            {"n_estimators": 300, "max_depth": 8},
            {"n_estimators": 500, "max_depth": None},
        ],
        "XGBoost": [
            {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05},
            {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05},
            {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
            {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
        ],
    }


def build_probe(probe_name, params, seed=42):
    if probe_name == "LogReg":
        return LogisticRegression(
            solver="lbfgs",
            random_state=seed,
            **params,
        )
    if probe_name == "LinearSVM":
        return LinearSVC(
            random_state=seed,
            **params,
        )
    if probe_name == "SVM-RBF":
        return SVC(
            kernel="rbf",
            random_state=seed,
            **params,
        )
    if probe_name == "kNN":
        return KNeighborsClassifier(**params)
    if probe_name == "RandomForest":
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            **params,
        )
    if probe_name == "XGBoost":
        return XGBClassifier(
            random_state=seed,
            eval_metric="logloss",
            **params,
        )
    raise ValueError(f"Unknown probe '{probe_name}'")


def tune_probe(probe_name, x_train, y_train, x_val, y_val, seed=42):
    best_params = None
    best_val_acc = -1.0

    for params in get_probe_grid()[probe_name]:
        clf = build_probe(probe_name, params, seed=seed)
        clf.fit(x_train, y_train)
        val_acc = accuracy_score(y_val, clf.predict(x_val))
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params

    return best_params, best_val_acc


def fit_and_score_probe(
    probe_name,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    seed=42,
):
    best_params, best_val_acc = tune_probe(
        probe_name,
        x_train,
        y_train,
        x_val,
        y_val,
        seed=seed,
    )
    x_train_full = np.concatenate([x_train, x_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)

    clf = build_probe(probe_name, best_params, seed=seed)
    clf.fit(x_train_full, y_train_full)
    y_pred = clf.predict(x_test)

    return {
        "Probe": probe_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1Score": f1_score(y_test, y_pred, average="binary"),
        "BestParams": str(best_params),
    }


def run_probe_suite(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    *,
    probe_names=None,
    seed=42,
):
    if probe_names is None:
        probe_names = list(get_probe_grid())

    results = []
    for probe_name in probe_names:
        results.append(
            fit_and_score_probe(
                probe_name,
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                seed=seed,
            )
        )

    return pd.DataFrame(results).sort_values(["Accuracy", "F1Score"], ascending=[False, False])


def build_baseline_representations(bundle, dataset_name, n, seed=42):
    from baselines import (
        encode_autoencoder,
        encode_vae,
        make_classical_reducers,
        make_umap_reducers,
        train_deep_rivals,
    )

    representations = []
    latent_dims = [n]

    for name, reducer in make_classical_reducers(dataset_name=dataset_name, latent_dims=latent_dims):
        x_train = reducer.fit_transform(bundle.x_train)
        x_val = reducer.transform(bundle.x_val)
        x_test = reducer.transform(bundle.x_test)
        representations.append((name, x_train, x_val, x_test))

    for name, reducer in make_umap_reducers(n_components=n):
        x_train = reducer.fit_transform(bundle.x_train)
        x_val = reducer.transform(bundle.x_val)
        x_test = reducer.transform(bundle.x_test)
        representations.append((name, x_train, x_val, x_test))

    best_ae, best_vae = train_deep_rivals(
        bundle.x_train,
        bundle.x_val,
        latent_dims=(n,),
        seed=seed,
    )
    representations.append(
        (
            f"Deep AE [d={n}]",
            encode_autoencoder(best_ae["model"], bundle.x_train),
            encode_autoencoder(best_ae["model"], bundle.x_val),
            encode_autoencoder(best_ae["model"], bundle.x_test),
        )
    )
    representations.append(
        (
            f"Deep VAE [d={n}]",
            encode_vae(best_vae["model"], bundle.x_train),
            encode_vae(best_vae["model"], bundle.x_val),
            encode_vae(best_vae["model"], bundle.x_test),
        )
    )

    return representations


@torch.no_grad()
def load_icl_model(bundle, checkpoint_path, latent_dim=64, n_heads=5):
    from models import InvertedFeatureExpert

    model = InvertedFeatureExpert(
        n_patients=bundle.num_train_samples,
        latent_dim=latent_dim,
        n_heads=n_heads,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def build_icl_topk_features(bundle, checkpoint_path, top_k, latent_dim=64, n_heads=5):
    model = load_icl_model(bundle, checkpoint_path, latent_dim=latent_dim, n_heads=n_heads)
    x_train, x_val, x_test, _, _ = extract_topk_features(
        model,
        bundle.train_loader,
        bundle.x_train,
        bundle.x_val,
        bundle.x_test,
        top_k,
    )
    return x_train, x_val, x_test


def parse_args():
    parser = argparse.ArgumentParser(description="Run a fixed downstream probe suite.")
    parser.add_argument("--dataset", default="madelon", help="Registered dataset name.")
    parser.add_argument(
        "--mode",
        choices=["baseline", "icl"],
        default="baseline",
        help="Probe baseline reducer representations or ICL top-k feature selection.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Target representation dimension, e.g. 20 for Madelon.",
    )
    parser.add_argument(
        "--probes",
        nargs="*",
        default=None,
        help="Optional subset of probes to run.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional trained ICL checkpoint. If omitted in icl mode, the model is trained first.",
    )
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=5)
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k original features to keep for ICL feature-selection mode. Defaults to --n.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        default=None,
        help="Optional CSV output path. Defaults to results/linear_probe_<dataset>.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    from data import get_dataset_bundle

    args = parse_args()
    bundle = get_dataset_bundle(args.dataset)

    if args.mode == "icl":
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            from train_madelon import train_icl
            from models import InvertedFeatureExpert

            if args.dataset != "madelon":
                raise ValueError("Auto-training from linear_probe.py is currently only supported for the madelon dataset.")

            checkpoint_path = f"checkpoints/{args.dataset}_last.pt"
            feature_list_path = f"checkpoints/{args.dataset}_topk_features.csv"
            input_dim = bundle.num_train_samples
            model = InvertedFeatureExpert(
                n_patients=input_dim,
                latent_dim=args.latent_dim,
                n_heads=args.n_heads,
            )
            train_icl(
                model,
                bundle.train_loader,
                bundle.val_loader,
                bundle.test_loader,
                bundle.labels,
                checkpoint_path=checkpoint_path,
                top_k=args.top_k or args.n,
                feature_list_path=feature_list_path,
            )

        checkpoint_path = str(Path(checkpoint_path))
        top_k = args.top_k or args.n

        selection_train, selection_val, selection_test = build_icl_topk_features(
            bundle,
            checkpoint_path,
            top_k=top_k,
            latent_dim=args.latent_dim,
            n_heads=args.n_heads,
        )
        df = run_probe_suite(
            selection_train,
            bundle.y_train,
            selection_val,
            bundle.y_val,
            selection_test,
            bundle.y_test,
            probe_names=args.probes,
            seed=args.seed,
        )
        df.insert(0, "View", f"Feature Selection [k={top_k}]")
        df = df.sort_values(["Accuracy", "F1Score"], ascending=[False, False])
    else:
        frames = []
        representations = build_baseline_representations(
            bundle,
            args.dataset,
            args.n,
            seed=args.seed,
        )
        for method_name, x_train, x_val, x_test in representations:
            probe_df = run_probe_suite(
                x_train,
                bundle.y_train,
                x_val,
                bundle.y_val,
                x_test,
                bundle.y_test,
                probe_names=args.probes,
                seed=args.seed,
            )
            probe_df.insert(0, "Method", method_name)
            frames.append(probe_df)
        df = pd.concat(frames, ignore_index=True).sort_values(
            ["Method", "Accuracy", "F1Score"],
            ascending=[True, False, False],
        )

    out_path = args.out or f"results/linear_probe_{args.dataset}_{args.mode}_d{args.n}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved probe results to {out_path}")
