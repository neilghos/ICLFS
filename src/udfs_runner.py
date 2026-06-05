from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from skfeature.function.sparse_learning_based import UDFS

from baseline import (
    RESULTS_DIR,
    evaluate_selected_features,
    load_full_dataset,
    set_seed,
    valid_ks,
)


PAPER_NUM_KMEANS_RUNS = 20
DATASETS = [
    "prostate",
    "nci9",
    "arcene",
    "relathe",
    "pcmac",
    "basehock",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone UDFS runner under the paper-style clustering protocol."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Optional dataset name. Defaults to the full hardcoded benchmark list.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmeans-runs", type=int, default=PAPER_NUM_KMEANS_RUNS)
    parser.add_argument(
        "--out",
        default=None,
        help="Optional single-dataset CSV output path. Ignored when running the full benchmark list.",
    )
    return parser.parse_args()


def udfs_ranking(x: np.ndarray, *, num_clusters: int) -> np.ndarray:
    ranking = UDFS.udfs(
        x,
        mode="rank",
        n_clusters=num_clusters,
    )
    return np.asarray(ranking).astype(int).reshape(-1)


def run_dataset(
    dataset_name: str,
    *,
    seed: int,
    kmeans_runs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, y = load_full_dataset(dataset_name, seed)
    n_clusters = len(set(y.tolist()))
    ks = valid_ks(x.shape[1])

    print(
        f"Loaded {dataset_name}: {x.shape[0]} samples, {x.shape[1]} features, "
        f"{n_clusters} clusters/classes"
    )

    print("Computing UDFS ranking once for the full dataset...")
    ranking = udfs_ranking(x, num_clusters=n_clusters)

    rows = []
    for k_selected in ks:
        selected_idx = ranking[:k_selected]
        metrics = evaluate_selected_features(
            x,
            y,
            selected_idx,
            num_clusters=n_clusters,
            kmeans_runs=kmeans_runs,
            seed=seed,
        )
        rows.append(
            {
                "Method": "UDFS",
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
    return raw_df, summary_df


def main():
    args = parse_args()
    set_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [args.dataset] if args.dataset is not None else DATASETS
    all_summaries = []

    for dataset_name in datasets:
        raw_df, summary_df = run_dataset(
            dataset_name,
            seed=args.seed,
            kmeans_runs=args.kmeans_runs,
        )
        all_summaries.append(summary_df)

        out_path = Path(args.out or RESULTS_DIR / f"udfs_{dataset_name}.csv")
        if len(datasets) > 1:
            out_path = RESULTS_DIR / f"udfs_{dataset_name}.csv"
        summary_path = out_path.with_name(out_path.stem + "_summary.csv")
        raw_df.to_csv(out_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        print("\nSummary")
        print(summary_df.to_string(index=False))
        print(f"\nSaved raw UDFS sweep to {out_path}")
        print(f"Saved summary UDFS table to {summary_path}")

    if len(all_summaries) > 1:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_path = RESULTS_DIR / "udfs_all_summary.csv"
        combined_summary.to_csv(combined_path, index=False)
        print(f"\nSaved combined UDFS summary to {combined_path}")


if __name__ == "__main__":
    main()
