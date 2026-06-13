from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


def _load_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    vmin = values[finite].min()
    vmax = values[finite].max()
    if np.isclose(vmin, vmax):
        out = np.zeros_like(values)
        out[finite] = 1.0
        return out
    out = np.zeros_like(values)
    out[finite] = (values[finite] - vmin) / (vmax - vmin)
    return out


def _forward_prefixes(max_k: int) -> list[int]:
    if max_k <= 0:
        return []
    prefixes = list(range(1, min(max_k, 10) + 1))
    step = 5 if max_k <= 50 else 10
    prefixes.extend(range(15, max_k + 1, step))
    prefixes.append(max_k)
    return sorted(set(prefixes))


def _safe_cv_splits(y: np.ndarray) -> int:
    _, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min()) if counts.size else 2
    return max(2, min(5, min_count))


def _forward_selection_curve(
    x: np.ndarray,
    y: np.ndarray,
    ranking: np.ndarray,
    *,
    seed: int,
    max_forward_k: int,
) -> pd.DataFrame:
    max_forward_k = min(max_forward_k, len(ranking), x.shape[1])
    prefixes = _forward_prefixes(max_forward_k)
    if not prefixes:
        return pd.DataFrame(columns=["NumSelected", "ForwardAccuracy"])

    cv = StratifiedKFold(n_splits=_safe_cv_splits(y), shuffle=True, random_state=seed)
    rows = []
    for k in prefixes:
        idx = np.asarray(ranking[:k], dtype=int)
        x_sel = x[:, idx]
        clf = LogisticRegression(max_iter=2000, solver="liblinear")
        scores = cross_val_score(clf, x_sel, y, cv=cv, scoring="accuracy")
        rows.append({"NumSelected": k, "ForwardAccuracy": float(np.mean(scores))})
    return pd.DataFrame(rows)


def _pairwise_mi_matrix(x: np.ndarray, *, seed: int) -> np.ndarray:
    n_features = x.shape[1]
    mi = np.zeros((n_features, n_features), dtype=float)
    for j in range(n_features):
        if n_features == 1:
            break
        others = [idx for idx in range(n_features) if idx != j]
        vals = mutual_info_regression(x[:, others], x[:, j], random_state=seed)
        for off, other_idx in enumerate(others):
            mi[j, other_idx] = vals[off]
    return 0.5 * (mi + mi.T)


def _prefix_redundancy_curves(
    x: np.ndarray,
    ranked_idx: np.ndarray,
    *,
    seed: int,
    redundancy_k: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    top_k = min(redundancy_k, len(ranked_idx), x.shape[1])
    idx = np.asarray(ranked_idx[:top_k], dtype=int)
    x_top = x[:, idx]
    corr = np.abs(np.corrcoef(x_top, rowvar=False))
    corr = np.nan_to_num(corr, nan=0.0)
    mi = _pairwise_mi_matrix(x_top, seed=seed)

    rows = []
    for k in range(2, top_k + 1):
        corr_block = corr[:k, :k]
        mi_block = mi[:k, :k]
        tri = np.triu_indices(k, k=1)
        rows.append(
            {
                "NumSelected": k,
                "AvgAbsCorrelation": float(corr_block[tri].mean()) if tri[0].size else 0.0,
                "AvgPairwiseMI": float(mi_block[tri].mean()) if tri[0].size else 0.0,
            }
        )
    return pd.DataFrame(rows), mi


def _mrmr_curve(
    mi_label_scores: np.ndarray,
    ranked_idx: np.ndarray,
    mi_feature_matrix: np.ndarray,
) -> pd.DataFrame:
    top_k = min(len(ranked_idx), mi_feature_matrix.shape[0])
    if top_k == 0:
        return pd.DataFrame(columns=["NumSelected", "AvgLabelMI", "MRMRLike"])

    ranked_relevance = mi_label_scores[np.asarray(ranked_idx[:top_k], dtype=int)]
    rows = []
    for k in range(1, top_k + 1):
        avg_rel = float(np.mean(ranked_relevance[:k]))
        if k == 1:
            avg_red = 0.0
        else:
            tri = np.triu_indices(k, k=1)
            avg_red = float(mi_feature_matrix[:k, :k][tri].mean()) if tri[0].size else 0.0
        rows.append(
            {
                "NumSelected": k,
                "AvgLabelMI": avg_rel,
                "MRMRLike": avg_rel - avg_red,
            }
        )
    return pd.DataFrame(rows)


def generate_analysis_bundle(
    *,
    dataset_name: str,
    x: np.ndarray,
    y: np.ndarray,
    ranking: np.ndarray,
    eval_df: pd.DataFrame,
    out_dir: Path,
    seed: int,
    analysis_top_k: int = 100,
    redundancy_top_k: int = 30,
    forward_top_k: int = 100,
) -> Path:
    plt = _load_matplotlib()
    analysis_dir = Path(out_dir) / "analysis" / dataset_name
    analysis_dir.mkdir(parents=True, exist_ok=True)

    ranked_idx = np.asarray(ranking[: min(analysis_top_k, len(ranking))], dtype=int)
    ranks = np.arange(1, len(ranked_idx) + 1)

    mi_label = mutual_info_classif(x, y, random_state=seed)
    anova_scores, _ = f_classif(x, y)
    anova_scores = np.nan_to_num(anova_scores, nan=0.0, posinf=0.0, neginf=0.0)

    forward_df = _forward_selection_curve(
        x,
        y,
        ranking,
        seed=seed,
        max_forward_k=forward_top_k,
    )
    redundancy_df, pairwise_mi = _prefix_redundancy_curves(
        x,
        ranked_idx,
        seed=seed,
        redundancy_k=redundancy_top_k,
    )
    mrmr_df = _mrmr_curve(mi_label, ranked_idx[: pairwise_mi.shape[0]], pairwise_mi)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(ranks, _normalized(mi_label[ranked_idx]), label="Label MI", linewidth=2)
    axes[0, 0].plot(ranks, _normalized(anova_scores[ranked_idx]), label="ANOVA F", linewidth=2)
    axes[0, 0].set_title("Relevance By Rank")
    axes[0, 0].set_xlabel("Feature rank")
    axes[0, 0].set_ylabel("Normalized score")
    axes[0, 0].legend()

    if not forward_df.empty:
        axes[0, 1].plot(
            forward_df["NumSelected"],
            forward_df["ForwardAccuracy"],
            marker="o",
            linewidth=2,
            label="Forward selection acc.",
        )
    if not eval_df.empty:
        axes[0, 1].plot(
            eval_df["NumSelected"],
            eval_df["AccuracyMean"],
            marker="s",
            linewidth=2,
            label="KMeans acc.",
        )
    axes[0, 1].set_title("Prefix Performance")
    axes[0, 1].set_xlabel("Number of selected features")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()

    if not redundancy_df.empty:
        axes[1, 0].plot(
            redundancy_df["NumSelected"],
            redundancy_df["AvgAbsCorrelation"],
            label="Avg |corr|",
            linewidth=2,
        )
        axes[1, 0].plot(
            redundancy_df["NumSelected"],
            _normalized(redundancy_df["AvgPairwiseMI"].to_numpy()),
            label="Norm. pairwise MI",
            linewidth=2,
        )
    axes[1, 0].set_title("Redundancy By Prefix")
    axes[1, 0].set_xlabel("Number of selected features")
    axes[1, 0].set_ylabel("Redundancy")
    axes[1, 0].legend()

    if not mrmr_df.empty:
        axes[1, 1].plot(
            mrmr_df["NumSelected"],
            mrmr_df["AvgLabelMI"],
            label="Avg label MI",
            linewidth=2,
        )
        axes[1, 1].plot(
            mrmr_df["NumSelected"],
            mrmr_df["MRMRLike"],
            label="mRMR-like",
            linewidth=2,
        )
    axes[1, 1].set_title("Label-Aware Redundancy")
    axes[1, 1].set_xlabel("Number of selected features")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].legend()

    fig.suptitle(f"ICLFS analysis: {dataset_name}")
    fig.tight_layout()
    plot_path = analysis_dir / "analysis_overview.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics_path = analysis_dir / "analysis_metrics.csv"
    combined = (
        forward_df.merge(redundancy_df, on="NumSelected", how="outer")
        .merge(mrmr_df, on="NumSelected", how="outer")
        .sort_values("NumSelected")
    )
    combined.to_csv(metrics_path, index=False)
    return plot_path
