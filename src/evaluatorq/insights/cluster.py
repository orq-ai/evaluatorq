"""Two-level agglomerative clustering over L2-normalised embedding vectors.

Base clusters come from `scipy.cluster.hierarchy.linkage`/`fcluster` (ward) with the
cut chosen by `gap_optimal_k` (ported from `trace_intelligence/cluster/agglomerative.py`'s
`_gap_optimal_k`); top clusters are the same cut applied to the base centroids, capped so
no top group holds more than `max_subclusters` base clusters. Outlier detection (optional,
`outlier_zscore`) ports `cluster_runner.py`'s per-cluster z-score step. Pure numpy/scipy —
no LLM or network calls, so there is no retry layer here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from scipy.cluster.hierarchy import fcluster, linkage


@dataclass
class ClusterTree:
    """Two-level cluster assignment for one discovered dimension's vectors.

    `base_labels` is one entry per input vector (`-1` marks a noise/outlier point).
    `top_of_base` maps every non-noise base cluster id to its top cluster id.
    """

    base_labels: np.ndarray
    top_of_base: dict[int, int] = field(default_factory=dict)
    n_base: int = 0
    n_top: int = 0


def centroids(vectors: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """Mean vector per non-noise label. `labels` must align 1:1 with `vectors` rows."""
    result: dict[int, np.ndarray] = {}
    for lbl in np.unique(labels):
        if lbl == -1:
            continue
        result[int(lbl)] = vectors[labels == lbl].mean(axis=0)
    return result


def gap_optimal_k(z: np.ndarray, n: int, *, k_min: int = 2, k_max: int) -> int:
    """Port of `_gap_optimal_k`: the k whose merge-distance gap to the next merge is largest.

    `z` is a scipy linkage matrix over `n` observations. Falls back to `k_min` when there
    are too few merges to compare, or when every merge distance is ~0 (identical inputs).
    """
    k_max = max(k_min, min(k_max, n - 1)) if n > 1 else k_min
    distances = sorted(float(d) for d in z[:, 2])
    if len(distances) < 2 or max(distances) < 1e-10:
        return k_min

    best_k = k_min
    best_gap = -1.0
    for i in range(len(distances) - 1):
        if distances[i] <= 0:
            continue
        gap = (distances[i + 1] - distances[i]) / distances[i]
        k = n - i - 1
        if k_min <= k <= k_max and gap > best_gap:
            best_gap = gap
            best_k = k
    return best_k


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _flag_outliers(vectors: np.ndarray, base_labels: np.ndarray, *, zscore: float) -> np.ndarray:
    """Port of `cluster_runner.py`'s z-score outlier step, applied to base clusters only.

    Per base cluster with >= 3 members, a point whose distance to the (unit) centroid
    exceeds `mean + zscore * std(ddof=1)` is relabelled `-1` (noise).
    """
    labels = base_labels.copy()
    for lbl in np.unique(base_labels):
        if lbl == -1:
            continue
        mask = base_labels == lbl
        if mask.sum() < 3:
            continue
        cluster_vecs = vectors[mask]
        centroid = cluster_vecs.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm
        dists = 1.0 - cluster_vecs @ centroid
        threshold = dists.mean() + zscore * dists.std(ddof=1)
        cluster_indices = np.where(mask)[0]
        outliers = cluster_indices[dists > threshold]
        labels[outliers] = -1
    n_flagged = int((labels == -1).sum() - (base_labels == -1).sum())
    if n_flagged:
        logger.warning('cluster: flagged {} point(s) as outliers (zscore={})', n_flagged, zscore)
    return labels


def _relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    """Remap non-noise labels to a contiguous `0..k-1` range, preserving `-1`."""
    unique = sorted(int(v) for v in np.unique(labels) if v != -1)
    remap = {old: new for new, old in enumerate(unique)}
    out = labels.copy()
    for old, new in remap.items():
        out[labels == old] = new
    return out


def _merge_base_into_nearest_sibling(
    base_labels: np.ndarray,
    vectors: np.ndarray,
    top_labels: dict[int, int],
    group_top: int,
    max_subclusters: int,
) -> dict[int, int]:
    """Reduce the number of distinct base clusters mapped to `group_top` to `max_subclusters`.

    Repeatedly merges the smallest base cluster in the offending group into its nearest
    (by centroid cosine similarity) sibling base cluster within the same group, relabelling
    every point of the victim to the target's base id.
    """
    while True:
        members = [b for b, t in top_labels.items() if t == group_top]
        if len(members) <= max_subclusters:
            return top_labels
        cents = centroids(vectors, base_labels)
        sizes = {b: int((base_labels == b).sum()) for b in members}
        victim = min(members, key=lambda b: sizes[b])
        siblings = [b for b in members if b != victim]
        target = max(siblings, key=lambda b: _cosine_sim(cents[victim], cents[b]))
        base_labels[base_labels == victim] = target
        del top_labels[victim]


def cluster_two_level(
    vectors: np.ndarray,
    *,
    max_clusters: int = 15,
    max_subclusters: int = 15,
    min_cluster_size: int = 5,
    outlier_zscore: float | None = None,
) -> ClusterTree:
    """Cluster `vectors` (one row per trace) into a two-level hierarchy.

    L2-normalises first. Below `2 * min_cluster_size` points there is too little signal
    to split: everything is one base cluster under one top cluster.
    """
    n = vectors.shape[0]
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1.0, norms)
    normed = vectors / safe_norms

    if n < 2 * min_cluster_size:
        base_labels = np.zeros(n, dtype=int)
        return ClusterTree(base_labels=base_labels, top_of_base={0: 0}, n_base=1, n_top=1)

    z_base = linkage(normed, method='ward')
    k_max_base = min(max_clusters * max_subclusters, n // min_cluster_size)
    k_base = gap_optimal_k(z_base, n, k_min=2, k_max=k_max_base)
    base_labels = fcluster(z_base, k_base, criterion='maxclust') - 1

    base_cents = centroids(normed, base_labels)
    base_ids = sorted(base_cents)
    n_base = len(base_ids)

    if n_base < 2:
        base_labels = _relabel_contiguous(base_labels)
        top_of_base = {new: 0 for new in range(n_base)}
        result_labels = base_labels
        if outlier_zscore is not None:
            result_labels = _flag_outliers(normed, result_labels, zscore=outlier_zscore)
        return ClusterTree(base_labels=result_labels, top_of_base=top_of_base, n_base=n_base, n_top=1)

    centroid_matrix = np.array([base_cents[b] for b in base_ids])
    z_top = linkage(centroid_matrix, method='ward')
    k_top = gap_optimal_k(z_top, n_base, k_min=1, k_max=max_clusters)
    top_labels_arr = fcluster(z_top, k_top, criterion='maxclust') - 1
    top_of_base = {base_ids[i]: int(top_labels_arr[i]) for i in range(n_base)}

    def _max_group_size(mapping: dict[int, int]) -> int:
        counts: dict[int, int] = {}
        for t in mapping.values():
            counts[t] = counts.get(t, 0) + 1
        return max(counts.values()) if counts else 0

    while _max_group_size(top_of_base) > max_subclusters and k_top < max_clusters:
        k_top += 1
        top_labels_arr = fcluster(z_top, k_top, criterion='maxclust') - 1
        top_of_base = {base_ids[i]: int(top_labels_arr[i]) for i in range(n_base)}

    while _max_group_size(top_of_base) > max_subclusters:
        counts: dict[int, int] = {}
        for t in top_of_base.values():
            counts[t] = counts.get(t, 0) + 1
        offending_top = max(counts, key=lambda t: counts[t])
        logger.warning(
            'cluster: top group {} holds {} base clusters (cap {}); merging smallest into nearest sibling',
            offending_top,
            counts[offending_top],
            max_subclusters,
        )
        top_of_base = _merge_base_into_nearest_sibling(base_labels, normed, top_of_base, offending_top, max_subclusters)

    base_labels = _relabel_contiguous(base_labels)
    old_to_new = {old: new for new, old in enumerate(sorted(top_of_base))}
    top_of_base = {old_to_new[old]: top for old, top in top_of_base.items()}
    top_ids_used = sorted({t for t in top_of_base.values()})
    top_remap = {old: new for new, old in enumerate(top_ids_used)}
    top_of_base = {b: top_remap[t] for b, t in top_of_base.items()}

    result_labels = base_labels
    if outlier_zscore is not None:
        result_labels = _flag_outliers(normed, result_labels, zscore=outlier_zscore)

    return ClusterTree(
        base_labels=result_labels,
        top_of_base=top_of_base,
        n_base=len(top_of_base),
        n_top=len(top_ids_used),
    )
