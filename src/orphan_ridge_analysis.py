from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from icl_eval import (
    RESULTS_DIR,
    architecture_for_dataset,
    evaluate_selected_features,
    load_full_dataset,
    set_seed,
    train_and_score_features,
)
from redundancy import (
    LAPLACIAN_PRUNER_LAP_PERCENTILE,
    LAPLACIAN_PRUNER_NEIGHBORS,
    adaptive_laplacian_pool_prune,
    laplacian_score,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze post-Laplacian orphan candidates with ridge reconstruction."
    )
    parser.add_argument("--dataset", required=True, help="Registered dataset name.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--target-k", type=int, default=50)
    parser.add_argument(
        "--rejection-pool-size",
        type=int,
        default=50,
        help="Number of post-Laplacian rejected candidates to analyze alongside the selected set.",
    )
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--max-swaps",
        type=int,
        default=10,
        help="Maximum number of selected/rejected swaps to evaluate.",
    )
    parser.add_argument("--kmeans-runs", type=int, default=20)
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.20,
        help="Holdout ratio over samples for ridge reconstruction error.",
    )
    parser.add_argument(
        "--redundancy-lap-percentile",
        type=float,
        default=LAPLACIAN_PRUNER_LAP_PERCENTILE,
    )
    parser.add_argument(
        "--redundancy-neighbors",
        type=int,
        default=LAPLACIAN_PRUNER_NEIGHBORS,
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Optional path to config.yaml. Defaults to the repo config.yaml.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional CSV output path. Defaults to results/orphan_ridge_<dataset>_k<k>_r<r>.csv",
    )
    return parser.parse_args()


def minmax_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(values)
    if not finite_mask.any():
        return np.zeros_like(values, dtype=np.float64)
    finite = values[finite_mask]
    v_min = float(finite.min())
    v_max = float(finite.max())
    if abs(v_max - v_min) <= 1e-12:
        scaled = np.zeros_like(values, dtype=np.float64)
        scaled[finite_mask] = 0.5
        return scaled
    scaled = np.zeros_like(values, dtype=np.float64)
    scaled[finite_mask] = (finite - v_min) / (v_max - v_min)
    return scaled


def ridge_reconstruction_error(
    x_pool: np.ndarray,
    target_pos: int,
    *,
    alpha: float,
    test_ratio: float,
    seed: int,
) -> float:
    num_samples = x_pool.shape[0]
    permutation = np.random.default_rng(seed).permutation(num_samples)
    test_size = max(1, int(round(test_ratio * num_samples)))
    test_idx = permutation[:test_size]
    train_idx = permutation[test_size:]
    if train_idx.size == 0:
        train_idx = test_idx

    predictor_mask = np.ones(x_pool.shape[1], dtype=bool)
    predictor_mask[target_pos] = False
    x_train = x_pool[train_idx][:, predictor_mask]
    y_train = x_pool[train_idx, target_pos]
    x_test = x_pool[test_idx][:, predictor_mask]
    y_test = x_pool[test_idx, target_pos]

    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    return float(np.mean((pred - y_test) ** 2))


def write_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prepare_dataset_cache(
    *,
    dataset: str,
    seed: int,
    epochs: int,
    config_path: str | None,
) -> dict[str, object]:
    set_seed(seed)

    x, y = load_full_dataset(dataset, seed)
    arch = architecture_for_dataset(
        dataset,
        encoder_hidden_dim=1024,
        latent_dim=512,
        projector_hidden_dim=256,
        projector_output_dim=128,
    )
    decorrelation_weight = 0.40 if arch["decorrelation_weight"] is None else float(
        arch["decorrelation_weight"]
    )

    feature_scores = train_and_score_features(
        x,
        epochs=epochs,
        seed=seed,
        encoder_hidden_dim=int(arch["encoder_hidden_dim"]),
        latent_dim=int(arch["latent_dim"]),
        n_heads=1,
        projector_hidden_dim=int(arch["projector_hidden_dim"]),
        projector_output_dim=int(arch["projector_output_dim"]),
        temperature=0.05,
        decorrelation_weight=decorrelation_weight,
        config_path=config_path,
    )
    return {
        "dataset": dataset,
        "seed": seed,
        "x": x,
        "y": y,
        "feature_scores": feature_scores,
        "ranking": np.argsort(feature_scores)[::-1],
    }


def analyze_cached_dataset(
    *,
    dataset_cache: dict[str, object],
    target_k: int,
    rejection_pool_size: int,
    ridge_alpha: float,
    max_swaps: int,
    kmeans_runs: int,
    test_ratio: float,
    redundancy_lap_percentile: float,
    redundancy_neighbors: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    dataset = str(dataset_cache["dataset"])
    seed = int(dataset_cache["seed"])
    x = np.asarray(dataset_cache["x"])
    y = np.asarray(dataset_cache["y"])
    feature_scores = np.asarray(dataset_cache["feature_scores"])
    ranking = np.asarray(dataset_cache["ranking"], dtype=int)

    analysis_pool_size = min(x.shape[1], target_k + rejection_pool_size)
    _, _, repaired_pool = adaptive_laplacian_pool_prune(
        x,
        ranking,
        pool_size=analysis_pool_size,
        final_k=target_k,
        n_neighbors=redundancy_neighbors,
        lap_percentile=redundancy_lap_percentile,
    )

    pool_idx = np.asarray(repaired_pool[:analysis_pool_size], dtype=int)
    x_pool = x[:, pool_idx]
    pool_lap_scores = laplacian_score(
        x_pool,
        n_neighbors=redundancy_neighbors,
    )
    pool_norm_scores = feature_scores[pool_idx]
    scaled_norm = minmax_scale(pool_norm_scores)
    scaled_lap = minmax_scale(pool_lap_scores)
    quality = scaled_norm + (1.0 - scaled_lap)

    feature_rows: list[dict[str, object]] = []
    for pos, feature_idx in enumerate(pool_idx):
        recon_mse = ridge_reconstruction_error(
            x_pool,
            pos,
            alpha=ridge_alpha,
            test_ratio=test_ratio,
            seed=seed,
        )
        feature_rows.append(
            {
                "Dataset": dataset,
                "TargetK": target_k,
                "RejectionPoolSize": rejection_pool_size,
                "AnalysisPoolSize": analysis_pool_size,
                "PoolRank": pos + 1,
                "FeatureIndex": int(feature_idx),
                "Membership": "selected" if pos < target_k else "rejected",
                "ProjectorNorm": float(pool_norm_scores[pos]),
                "LaplacianScore": float(pool_lap_scores[pos]),
                "ScaledNorm": float(scaled_norm[pos]),
                "ScaledLap": float(scaled_lap[pos]),
                "QualityScore": float(quality[pos]),
                "RidgeTestMSE": recon_mse,
            }
        )

    selected_rows = [row for row in feature_rows if row["Membership"] == "selected"]
    rejected_rows = [row for row in feature_rows if row["Membership"] == "rejected"]
    graft_rows = graft_and_score(
        x,
        y,
        selected_rows=selected_rows,
        rejected_rows=rejected_rows,
        ranking_key="RidgeTestMSE",
        max_swaps=max_swaps,
        kmeans_runs=kmeans_runs,
        seed=seed,
    )
    return feature_rows, graft_rows


def analyze_dataset(
    *,
    dataset: str,
    seed: int,
    epochs: int,
    target_k: int,
    rejection_pool_size: int,
    ridge_alpha: float,
    max_swaps: int,
    kmeans_runs: int,
    test_ratio: float,
    redundancy_lap_percentile: float,
    redundancy_neighbors: int,
    config_path: str | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    dataset_cache = prepare_dataset_cache(
        dataset=dataset,
        seed=seed,
        epochs=epochs,
        config_path=config_path,
    )
    return analyze_cached_dataset(
        dataset_cache=dataset_cache,
        target_k=target_k,
        rejection_pool_size=rejection_pool_size,
        ridge_alpha=ridge_alpha,
        max_swaps=max_swaps,
        kmeans_runs=kmeans_runs,
        test_ratio=test_ratio,
        redundancy_lap_percentile=redundancy_lap_percentile,
        redundancy_neighbors=redundancy_neighbors,
    )


def graft_and_score(
    x: np.ndarray,
    y: np.ndarray,
    *,
    selected_rows: list[dict[str, object]],
    rejected_rows: list[dict[str, object]],
    ranking_key: str,
    max_swaps: int,
    kmeans_runs: int,
    seed: int,
) -> list[dict[str, object]]:
    num_clusters = np.unique(y).shape[0]
    selected_sorted = sorted(selected_rows, key=lambda row: row["QualityScore"])
    rejected_sorted = sorted(rejected_rows, key=lambda row: row[ranking_key], reverse=True)
    limit = min(max_swaps, len(selected_sorted), len(rejected_sorted))

    baseline_idx = np.asarray([row["FeatureIndex"] for row in selected_rows], dtype=int)
    results = []
    baseline_metrics = evaluate_selected_features(
        x,
        y,
        baseline_idx,
        num_clusters=num_clusters,
        kmeans_runs=kmeans_runs,
        seed=seed,
    )
    results.append(
        {
            "Protocol": ranking_key,
            "NumSwaps": 0,
            "AccuracyMean": baseline_metrics["AccuracyMean"],
            "AccuracyStd": baseline_metrics["AccuracyStd"],
            "Sacrificed": "",
            "Admitted": "",
        }
    )

    for num_swaps in range(1, limit + 1):
        kept = selected_sorted[num_swaps:]
        admitted = rejected_sorted[:num_swaps]
        swap_idx = np.asarray(
            [row["FeatureIndex"] for row in kept + admitted],
            dtype=int,
        )
        metrics = evaluate_selected_features(
            x,
            y,
            swap_idx,
            num_clusters=num_clusters,
            kmeans_runs=kmeans_runs,
            seed=seed,
        )
        results.append(
            {
                "Protocol": ranking_key,
                "NumSwaps": num_swaps,
                "AccuracyMean": metrics["AccuracyMean"],
                "AccuracyStd": metrics["AccuracyStd"],
                "Sacrificed": ",".join(
                    str(row["FeatureIndex"]) for row in selected_sorted[:num_swaps]
                ),
                "Admitted": ",".join(
                    str(row["FeatureIndex"]) for row in admitted
                ),
            }
        )
    return results


def main():
    args = parse_args()
    rows, graft_rows = analyze_dataset(
        dataset=args.dataset,
        seed=args.seed,
        epochs=args.epochs,
        target_k=args.target_k,
        rejection_pool_size=args.rejection_pool_size,
        ridge_alpha=args.ridge_alpha,
        max_swaps=args.max_swaps,
        kmeans_runs=args.kmeans_runs,
        test_ratio=args.test_ratio,
        redundancy_lap_percentile=args.redundancy_lap_percentile,
        redundancy_neighbors=args.redundancy_neighbors,
        config_path=args.config_path,
    )
    out_path = Path(
        args.out
        or RESULTS_DIR
        / f"orphan_ridge_{args.dataset}_k{args.target_k}_r{args.rejection_pool_size}.csv"
    )
    write_csv(rows, out_path)

    selected_rows = [row for row in rows if row["Membership"] == "selected"]
    rejected_rows = [row for row in rows if row["Membership"] == "rejected"]
    weakest_selected = sorted(selected_rows, key=lambda row: row["QualityScore"])[:5]
    strongest_orphans = sorted(
        rejected_rows,
        key=lambda row: (row["RidgeTestMSE"], row["QualityScore"]),
        reverse=True,
    )[:5]
    graft_out_path = out_path.with_name(out_path.stem + "_grafts.csv")
    write_csv(graft_rows, graft_out_path)

    print(f"Saved orphan ridge analysis to {out_path}")
    print(f"Saved grafting analysis to {graft_out_path}")
    print("\nLowest-quality selected features")
    for row in weakest_selected:
        print(
            f"feature={row['FeatureIndex']} rank={row['PoolRank']} "
            f"quality={row['QualityScore']:.4f} mse={row['RidgeTestMSE']:.6f}"
        )
    print("\nHighest-MSE rejected features")
    for row in strongest_orphans:
        print(
            f"feature={row['FeatureIndex']} rank={row['PoolRank']} "
            f"quality={row['QualityScore']:.4f} mse={row['RidgeTestMSE']:.6f}"
        )
    best = max(graft_rows, key=lambda row: (row["AccuracyMean"], -row["NumSwaps"]))
    print("\nBest graft result")
    print(
        f"RidgeTestMSE: swaps={best['NumSwaps']} "
        f"acc={best['AccuracyMean']:.6f} std={best['AccuracyStd']:.6f}"
    )


if __name__ == "__main__":
    main()
