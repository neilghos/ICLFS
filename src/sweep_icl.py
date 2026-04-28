from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from icl_eval import (
    RESULTS_DIR,
    evaluate_selected_features,
    load_full_dataset,
    set_seed,
    train_and_rank_features,
    valid_ks,
)


DEFAULT_TEMPERATURES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep ICL temperature under the paper-style clustering protocol."
    )
    parser.add_argument("--dataset", required=True, help="Registered dataset name.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--kmeans-runs", type=int, default=20)
    parser.add_argument(
        "--temperatures",
        nargs="*",
        type=float,
        default=list(DEFAULT_TEMPERATURES),
        help="Temperatures to sweep.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional CSV output path. Defaults to results/icl_<dataset>_temperature_sweep.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    x, y = load_full_dataset(args.dataset, args.seed)
    ks = valid_ks(x.shape[1])
    if not ks:
        raise ValueError(
            f"No valid paper-style feature counts for dataset with {x.shape[1]} features."
        )

    n_clusters = len(set(y.tolist()))
    max_k = max(ks)
    rows = []

    print(
        f"Loaded {args.dataset}: {x.shape[0]} samples, {x.shape[1]} features, "
        f"{n_clusters} clusters/classes"
    )

    for temperature in args.temperatures:
        effective_temperature = max(float(temperature), 1e-6)
        print(
            f"\nSweeping temperature={temperature}"
            + (f" (effective {effective_temperature})" if effective_temperature != temperature else "")
        )
        ranking = train_and_rank_features(
            x,
            epochs=args.epochs,
            seed=args.seed,
            latent_dim=args.latent_dim,
            n_heads=args.n_heads,
            temperature=effective_temperature,
        )
        ranking = ranking[:max_k]

        best_row = None
        for k_selected in ks:
            selected_idx = ranking[:k_selected]
            metrics = evaluate_selected_features(
                x,
                y,
                selected_idx,
                num_clusters=n_clusters,
                kmeans_runs=args.kmeans_runs,
                seed=args.seed,
            )
            row = {
                "Dataset": args.dataset,
                "Temperature": temperature,
                "EffectiveTemperature": effective_temperature,
                "LatentDim": args.latent_dim,
                "Epochs": args.epochs,
                "Heads": args.n_heads,
                "NumSelected": k_selected,
                **metrics,
            }
            rows.append(row)
            if best_row is None or (
                row["AccuracyMean"],
                row["NMIMean"],
                -row["NumSelected"],
            ) > (
                best_row["AccuracyMean"],
                best_row["NMIMean"],
                -best_row["NumSelected"],
            ):
                best_row = row

        assert best_row is not None
        print(
            f"Best temperature={temperature}: ACC={best_row['AccuracyMean']:.4f} "
            f"(k={best_row['NumSelected']}), NMI={best_row['NMIMean']:.4f}"
        )

    raw_df = pd.DataFrame(rows).sort_values(
        ["AccuracyMean", "NMIMean", "Temperature", "NumSelected"],
        ascending=[False, False, True, True],
    )
    summary_df = (
        raw_df.sort_values(
            ["Temperature", "AccuracyMean", "NMIMean", "NumSelected"],
            ascending=[True, False, False, True],
        )
        .groupby("Temperature", as_index=False)
        .first()
        .sort_values(["AccuracyMean", "NMIMean"], ascending=[False, False])
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out or RESULTS_DIR / f"icl_{args.dataset}_temperature_sweep.csv")
    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    raw_df.to_csv(out_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nSummary")
    print(summary_df.to_string(index=False))
    print(f"\nSaved raw temperature sweep to {out_path}")
    print(f"Saved summary temperature sweep to {summary_path}")


if __name__ == "__main__":
    main()
