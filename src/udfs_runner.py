from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baseline import (
    RESULTS_DIR,
    evaluate_selected_features,
    load_full_dataset,
    set_seed,
    udfs_select,
    valid_ks,
)


PAPER_NUM_KMEANS_RUNS = 20


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone UDFS runner under the paper-style clustering protocol."
    )
    parser.add_argument("--dataset", required=True, help="Registered dataset name.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kmeans-runs", type=int, default=PAPER_NUM_KMEANS_RUNS)
    parser.add_argument(
        "--out",
        default=None,
        help="Optional CSV output path. Defaults to results/udfs_<dataset>.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    x, y = load_full_dataset(args.dataset, args.seed)
    n_clusters = len(set(y.tolist()))
    ks = valid_ks(x.shape[1])

    print(
        f"Loaded {args.dataset}: {x.shape[0]} samples, {x.shape[1]} features, "
        f"{n_clusters} clusters/classes"
    )

    rows = []
    for k_selected in ks:
        selected_idx = udfs_select(
            x,
            k_selected=k_selected,
            num_clusters=n_clusters,
        )
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
    summary_df = pd.DataFrame([{**best, "Dataset": args.dataset}])[
        ["Dataset", "Method", "NumSelected", "AccuracyMean", "AccuracyStd", "NMIMean", "NMIStd"]
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out or RESULTS_DIR / f"udfs_{args.dataset}.csv")
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    raw_df.to_csv(out_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nSummary")
    print(summary_df.to_string(index=False))
    print(f"\nSaved raw UDFS sweep to {out_path}")
    print(f"Saved summary UDFS table to {summary_path}")


if __name__ == "__main__":
    main()
