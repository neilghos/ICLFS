from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph

LAPLACIAN_PRUNER_POOL_MULTIPLIER = 1.5
LAPLACIAN_PRUNER_LAP_PERCENTILE = 0.75
LAPLACIAN_PRUNER_NEIGHBORS = 3


def _resolve_bandwidth(distances: np.ndarray, connectivity: np.ndarray) -> float:
    nonzero = distances[connectivity > 0]
    if nonzero.size == 0:
        return 1.0
    value = float(np.mean(nonzero))
    return max(value, 1e-12)


def _build_affinity(x: np.ndarray, *, n_neighbors: int = 5) -> np.ndarray:
    n_samples = x.shape[0]
    n_neighbors = max(1, min(n_neighbors, n_samples - 1))
    connectivity = kneighbors_graph(
        x,
        n_neighbors=n_neighbors,
        mode="connectivity",
        metric="cosine",
        include_self=False,
    ).toarray()

    distances = pairwise_distances(x, metric="cosine")
    resolved_bandwidth = _resolve_bandwidth(distances, connectivity)
    affinity = np.exp(-distances / resolved_bandwidth) * connectivity
    affinity = np.maximum(affinity, affinity.T)
    np.fill_diagonal(affinity, 0.0)
    return affinity


def laplacian_score(x: np.ndarray, *, n_neighbors: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    affinity = _build_affinity(x, n_neighbors=n_neighbors)
    degree = np.sum(affinity, axis=1)
    laplacian = np.diag(degree) - affinity

    ones = np.ones(x.shape[0], dtype=np.float64)
    degree_sum = float(degree.sum())
    if degree_sum <= 0:
        return np.zeros(x.shape[1], dtype=np.float64)

    scores = np.zeros(x.shape[1], dtype=np.float64)
    for j in range(x.shape[1]):
        feature = x[:, j]
        weighted_mean = float(feature @ degree) / degree_sum
        centered = feature - weighted_mean * ones
        denom = float(centered @ (degree * centered))
        if denom <= 1e-12:
            scores[j] = np.inf
            continue
        numer = float(centered @ (laplacian @ centered))
        scores[j] = numer / denom
    return scores


def adaptive_laplacian_pool_prune(
    x: np.ndarray,
    ranking: np.ndarray,
    *,
    pool_size: int,
    final_k: int = 50,
    n_neighbors: int = LAPLACIAN_PRUNER_NEIGHBORS,
    lap_percentile: float = LAPLACIAN_PRUNER_LAP_PERCENTILE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranking = np.asarray(ranking, dtype=int)
    if ranking.ndim != 1:
        raise ValueError("ranking must be a 1D array of feature indices.")

    total_features = min(ranking.shape[0], x.shape[1])
    pool_size = min(pool_size, total_features)
    final_k = min(final_k, pool_size)

    global_scores = laplacian_score(
        np.asarray(x, dtype=np.float64),
        n_neighbors=n_neighbors,
    )
    finite_scores = global_scores[np.isfinite(global_scores)]
    if finite_scores.size == 0:
        threshold = np.inf
    else:
        threshold = float(np.quantile(finite_scores, lap_percentile))

    pool = list(ranking[:pool_size].tolist())
    next_ptr = pool_size

    while next_ptr < total_features:
        pool_scores = global_scores[np.asarray(pool, dtype=int)]
        worst_pos = int(np.argmax(pool_scores))
        worst_score = float(pool_scores[worst_pos])
        if worst_score <= threshold:
            break

        removed = pool.pop(worst_pos)
        replacement = int(ranking[next_ptr])
        next_ptr += 1
        if replacement == removed:
            continue
        pool.append(replacement)

    pool_array = np.asarray(pool, dtype=int)
    pool_norm_order = np.argsort([np.where(ranking == idx)[0][0] for idx in pool_array])
    pool_sorted_by_norm = pool_array[pool_norm_order]
    selected_idx = pool_sorted_by_norm[:final_k]
    selected_scores = global_scores[selected_idx]
    return selected_idx.astype(int), selected_scores.astype(np.float64), pool_sorted_by_norm.astype(int)
