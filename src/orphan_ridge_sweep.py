from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from icl_eval import RESULTS_DIR, set_seed, valid_ks
from orphan_ridge_analysis import analyze_cached_dataset, prepare_dataset_cache, write_csv
from redundancy import LAPLACIAN_PRUNER_LAP_PERCENTILE, LAPLACIAN_PRUNER_NEIGHBORS


SWEEP_DATASETS = [
    "coil20",
    "allaml",
    "arcene",
    "basehock",
    "lung",
    "nci9",
    "pcmac",
    "prostate",
    "relathe",
    "tox171",
    "warppie10p",
]

SWEEP_GRID = {
    "rejection_pool_multiplier": [1.0],
    "ridge_alpha": [1.0],
    "max_swaps": [10],
    "redundancy_lap_percentile": [LAPLACIAN_PRUNER_LAP_PERCENTILE],
    "redundancy_neighbors": [LAPLACIAN_PRUNER_NEIGHBORS],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep orphan ridge grafting hyperparameters over the hardcoded dataset suite."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--kmeans-runs", type=int, default=20)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument(
        "--config-path",
        default=None,
        help="Optional path to config.yaml. Defaults to the repo config.yaml.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory. Defaults to src/results.",
    )
    return parser.parse_args()


def build_configs() -> list[dict[str, int | float]]:
    configs = []
    for rejection_pool_multiplier, ridge_alpha, max_swaps, lap_percentile, neighbors in itertools.product(
        SWEEP_GRID["rejection_pool_multiplier"],
        SWEEP_GRID["ridge_alpha"],
        SWEEP_GRID["max_swaps"],
        SWEEP_GRID["redundancy_lap_percentile"],
        SWEEP_GRID["redundancy_neighbors"],
    ):
        configs.append(
            {
                "RejectionPoolMultiplier": float(rejection_pool_multiplier),
                "RidgeAlpha": float(ridge_alpha),
                "MaxSwaps": int(max_swaps),
                "LapPercentile": float(lap_percentile),
                "Neighbors": int(neighbors),
            }
        )
    return configs


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir or RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    all_path = out_dir / "orphan_ridge_sweep_all.csv"
    grouped_path = out_dir / "orphan_ridge_sweep_grouped.csv"

    configs = build_configs()
    dataset_cache = {}
    for dataset_name in SWEEP_DATASETS:
        print(f"\nPreparing orphan ridge cache for {dataset_name}...")
        dataset_cache[dataset_name] = prepare_dataset_cache(
            dataset=dataset_name,
            seed=args.seed,
            epochs=args.epochs,
            config_path=args.config_path,
        )

    for cfg in configs:
        print(
            f"\n=== Sweeping orphan ridge: rejection_pool_multiplier={cfg['RejectionPoolMultiplier']}, "
            f"ridge_alpha={cfg['RidgeAlpha']}, max_swaps={cfg['MaxSwaps']} ==="
        )
        for dataset_name in SWEEP_DATASETS:
            cached = dataset_cache[dataset_name]
            ks = valid_ks(cached["x"].shape[1])
            dataset_rows = []
            for target_k in ks:
                rejection_pool_size = max(
                    1,
                    int(round(float(cfg["RejectionPoolMultiplier"]) * target_k)),
                )
                _, graft_rows = analyze_cached_dataset(
                    dataset_cache=cached,
                    target_k=target_k,
                    rejection_pool_size=rejection_pool_size,
                    ridge_alpha=float(cfg["RidgeAlpha"]),
                    max_swaps=int(cfg["MaxSwaps"]),
                    kmeans_runs=args.kmeans_runs,
                    test_ratio=args.test_ratio,
                    redundancy_lap_percentile=float(cfg["LapPercentile"]),
                    redundancy_neighbors=int(cfg["Neighbors"]),
                )
                best_for_k = max(
                    graft_rows,
                    key=lambda row: (row["AccuracyMean"], -row["NumSwaps"]),
                )
                dataset_rows.append(
                    {
                        "Dataset": dataset_name,
                        "TargetK": target_k,
                        "RejectionPoolMultiplier": float(cfg["RejectionPoolMultiplier"]),
                        "EffectiveRejectionPoolSize": rejection_pool_size,
                        "RidgeAlpha": float(cfg["RidgeAlpha"]),
                        "MaxSwaps": int(cfg["MaxSwaps"]),
                        "LapPercentile": float(cfg["LapPercentile"]),
                        "Neighbors": int(cfg["Neighbors"]),
                        "Protocol": best_for_k["Protocol"],
                        "NumSwaps": int(best_for_k["NumSwaps"]),
                        "AccuracyMean": float(best_for_k["AccuracyMean"]),
                        "AccuracyStd": float(best_for_k["AccuracyStd"]),
                    }
                )
            best = max(
                dataset_rows,
                key=lambda row: (row["AccuracyMean"], -row["TargetK"], -row["NumSwaps"]),
            )
            summary_row = {
                **best,
            }
            all_rows.append(summary_row)
            write_csv(all_rows, all_path)
            print(pd.DataFrame([summary_row]).to_string(index=False))

    all_df = pd.DataFrame(all_rows).sort_values(
        [
            "Dataset",
            "RejectionPoolMultiplier",
            "RidgeAlpha",
            "MaxSwaps",
            "Protocol",
        ]
    )
    grouped_df = (
        all_df.groupby(
            [
                "RejectionPoolMultiplier",
                "RidgeAlpha",
                "MaxSwaps",
                "LapPercentile",
                "Neighbors",
                "Protocol",
            ],
            as_index=False,
        )
        .agg(
            MeanAccuracy=("AccuracyMean", "mean"),
            StdAcrossDatasets=("AccuracyMean", "std"),
        )
        .sort_values(["MeanAccuracy"], ascending=[False])
    )
    all_df.to_csv(all_path, index=False)
    grouped_df.to_csv(grouped_path, index=False)

    print("\nGrouped orphan ridge summary")
    print(grouped_df.to_string(index=False))
    print(f"Saved full sweep to {all_path}")
    print(f"Saved grouped summary to {grouped_path}")


if __name__ == "__main__":
    main()
