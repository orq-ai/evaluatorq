"""Unit tests for `evaluatorq.insights.cluster` — two-level agglomerative clustering."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from evaluatorq.insights.cluster import (
    MAX_LINKAGE_OBSERVATIONS,
    ClusterTree,
    cluster_two_level,
    nearest_neighbours,
)


def _blobs(
    n_blobs: int, n_per_blob: int, dim: int = 32, spread: float = 0.3, seed: int = 0
) -> tuple[NDArray[Any], NDArray[Any]]:
    """`n_blobs` well-separated gaussian blobs in `dim`-d space, plus their true labels."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=8.0, size=(n_blobs, dim))
    vectors = np.concatenate([centers[i] + rng.normal(scale=spread, size=(n_per_blob, dim)) for i in range(n_blobs)])
    true_labels = np.repeat(np.arange(n_blobs), n_per_blob)
    return vectors, true_labels


def test_four_blobs_recover_four_base_clusters():
    vectors, true_labels = _blobs(n_blobs=4, n_per_blob=30)
    tree = cluster_two_level(vectors, min_cluster_size=5)

    assert tree.n_base == 4
    # Every true blob maps to exactly one base label (base clusters may not be numbered
    # the same way, but within a blob the assigned base label must be constant).
    for blob in range(4):
        labels_in_blob = tree.base_labels[true_labels == blob]
        assert len(set(labels_in_blob.tolist())) == 1


def test_max_clusters_caps_n_top():
    vectors, _ = _blobs(n_blobs=4, n_per_blob=30)
    tree = cluster_two_level(vectors, min_cluster_size=5, max_clusters=2)
    assert tree.n_top <= 2


def test_max_subclusters_caps_group_size():
    vectors, _ = _blobs(n_blobs=6, n_per_blob=15)
    tree = cluster_two_level(vectors, min_cluster_size=5, max_subclusters=2)

    counts: dict[int, int] = {}
    for top in tree.top_of_base.values():
        counts[top] = counts.get(top, 0) + 1
    assert all(count <= 2 for count in counts.values())


def _blobs_with_within_cluster_outlier(n_blobs: int = 3, n_per_blob: int = 20, seed: int = 0) -> NDArray[Any]:
    """Blobs where point 0 is pushed far from its own blob's centroid, but not far enough
    to form its own base cluster — a within-cluster distance outlier, not a structural one."""
    vectors, _ = _blobs(n_blobs=n_blobs, n_per_blob=n_per_blob, seed=seed)
    vectors = vectors.copy()
    vectors[0] = vectors[0] + 3.0
    return vectors


def test_far_point_flagged_as_outlier_with_zscore():
    vectors = _blobs_with_within_cluster_outlier()
    tree = cluster_two_level(vectors, min_cluster_size=5, outlier_zscore=2.0)
    assert tree.base_labels[0] == -1
    assert {label for label in tree.base_labels if label != -1} == set(tree.top_of_base)
    assert tree.n_base == len(tree.top_of_base)
    assert tree.n_top == len(set(tree.top_of_base.values()))


def test_no_outlier_zscore_never_flags_noise():
    vectors = _blobs_with_within_cluster_outlier()
    tree = cluster_two_level(vectors, min_cluster_size=5, outlier_zscore=None)
    assert -1 not in tree.base_labels.tolist()


def test_few_points_single_cluster_no_exception():
    rng = np.random.default_rng(1)
    vectors = rng.normal(size=(3, 32))
    tree = cluster_two_level(vectors, min_cluster_size=5)

    assert isinstance(tree, ClusterTree)
    assert tree.n_base == 1
    assert tree.n_top == 1
    assert set(tree.base_labels.tolist()) == {0}


def test_all_identical_vectors_single_cluster():
    vectors = np.ones((12, 32))
    tree = cluster_two_level(vectors, min_cluster_size=5)

    assert tree.n_base == 1
    assert tree.n_top == 1
    assert set(tree.base_labels.tolist()) == {0}


def test_below_two_times_min_cluster_size_is_one_cluster():
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(9, 32))  # < 2 * min_cluster_size(5)
    tree = cluster_two_level(vectors, min_cluster_size=5)

    assert tree.n_base == 1
    assert tree.n_top == 1
    assert tree.top_of_base == {0: 0}


@pytest.mark.parametrize('path', ['single_base', 'multi_base'])
def test_all_outliers_leave_empty_hierarchy_metadata(monkeypatch, path):
    if path == 'single_base':
        vectors = np.ones((12, 8))
    else:
        vectors, _ = _blobs(n_blobs=3, n_per_blob=10)

    def flag_everything(_vectors, labels, *, zscore):
        del zscore
        return np.full_like(labels, -1)

    monkeypatch.setattr('evaluatorq.insights.cluster._flag_outliers', flag_everything)
    tree = cluster_two_level(vectors, min_cluster_size=5, outlier_zscore=2.0)

    assert set(tree.base_labels.tolist()) == {-1}
    assert tree.top_of_base == {}
    assert tree.n_base == 0
    assert tree.n_top == 0


def test_low_sample_fallback_applies_outlier_metadata_consistently(monkeypatch):
    def flag_everything(_vectors, labels, *, zscore):
        del zscore
        return np.full_like(labels, -1)

    monkeypatch.setattr('evaluatorq.insights.cluster._flag_outliers', flag_everything)
    tree = cluster_two_level(np.ones((9, 8)), min_cluster_size=5, outlier_zscore=2.0)

    assert set(tree.base_labels.tolist()) == {-1}
    assert tree.top_of_base == {}
    assert tree.n_base == 0
    assert tree.n_top == 0


def test_removed_base_clusters_keep_their_original_top_assignments(monkeypatch):
    vectors, _ = _blobs(n_blobs=4, n_per_blob=20)
    baseline = cluster_two_level(vectors, min_cluster_size=5)
    removed_id = min(baseline.top_of_base)
    surviving_old_ids = sorted(set(baseline.top_of_base) - {removed_id})
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(surviving_old_ids)}
    surviving_top_ids = sorted({baseline.top_of_base[old_id] for old_id in surviving_old_ids})
    top_to_new = {old_id: new_id for new_id, old_id in enumerate(surviving_top_ids)}

    def remove_first_base(_vectors, labels, *, zscore):
        del zscore
        out = labels.copy()
        out[labels == removed_id] = -1
        return out

    monkeypatch.setattr('evaluatorq.insights.cluster._flag_outliers', remove_first_base)
    tree = cluster_two_level(vectors, min_cluster_size=5, outlier_zscore=2.0)

    assert tree.top_of_base == {
        old_to_new[old_id]: top_to_new[baseline.top_of_base[old_id]] for old_id in surviving_old_ids
    }
    assert {label for label in tree.base_labels if label != -1} == set(tree.top_of_base)


def test_linkage_population_bound_rejects_without_dropping_traces():
    vectors = np.ones((MAX_LINKAGE_OBSERVATIONS + 1, 2))

    with pytest.raises(ValueError, match=f'at most {MAX_LINKAGE_OBSERVATIONS} traces'):
        cluster_two_level(vectors, min_cluster_size=1)


@pytest.mark.parametrize('outlier_zscore', [-1.0, float('nan'), float('inf')])
def test_invalid_direct_outlier_threshold_is_rejected(outlier_zscore):
    with pytest.raises(ValueError, match='outlier_zscore must be non-negative and finite'):
        cluster_two_level(np.ones((10, 2)), outlier_zscore=outlier_zscore)


@pytest.mark.parametrize('n_blobs,n_per_blob', [(3, 8), (5, 6)])
def test_top_of_base_keys_match_present_base_ids(n_blobs, n_per_blob):
    vectors, _ = _blobs(n_blobs=n_blobs, n_per_blob=n_per_blob, seed=3)
    tree = cluster_two_level(vectors, min_cluster_size=5)

    present_base_ids = {b for b in tree.base_labels.tolist() if b != -1}
    assert present_base_ids == set(tree.top_of_base.keys())


def test_nearest_neighbours_returns_k_closest_by_cosine_similarity():
    # 0 and 1 point the same direction (closest to each other); 2 is orthogonal to both.
    cents = {
        0: np.array([1.0, 0.0]),
        1: np.array([0.9, 0.1]),
        2: np.array([0.0, 1.0]),
    }
    result = nearest_neighbours(cents, k=1)

    assert result[0] == [1]
    assert result[1] == [0]
    # 2 is closer to 1 (small positive x-component) than to 0 (pure x-axis).
    assert result[2] == [1]


def test_nearest_neighbours_caps_at_k_and_excludes_self():
    cents = {i: np.array([float(i), 1.0]) for i in range(6)}
    result = nearest_neighbours(cents, k=3)

    assert len(result) == 6
    for cid, neighbours in result.items():
        assert len(neighbours) <= 3
        assert cid not in neighbours


def test_nearest_neighbours_rejects_negative_k():
    with pytest.raises(ValueError, match='k must be non-negative'):
        nearest_neighbours({0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}, k=-1)


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_non_finite_vectors_are_rejected_before_linkage(bad):
    vectors = np.ones((20, 3))
    vectors[4, 1] = bad

    with pytest.raises(ValueError, match='only finite values'):
        cluster_two_level(vectors)


def test_nearest_neighbours_single_cluster_has_no_neighbours():
    result = nearest_neighbours({0: np.array([1.0, 0.0])}, k=3)
    assert result == {0: []}


def test_empty_input_returns_empty_tree():
    tree = cluster_two_level(np.empty((0, 32)))

    assert tree.n_base == 0
    assert tree.n_top == 0
    assert tree.top_of_base == {}
    assert tree.base_labels.size == 0


@pytest.mark.parametrize(
    'limits',
    [
        {'max_clusters': 0},
        {'max_subclusters': 0},
        {'min_cluster_size': 0},
        {'max_clusters': -1},
    ],
)
def test_non_positive_limits_are_rejected(limits):
    with pytest.raises(ValueError, match='must be positive'):
        cluster_two_level(np.ones((10, 3)), **limits)


def test_structureless_data_is_cut_finer_for_the_merge_step_to_join() -> None:
    rng = np.random.default_rng(0)
    tree = cluster_two_level(rng.normal(size=(100, 32)), min_cluster_size=5)

    assert tree.n_base >= 10  # one per 2 * min_cluster_size, not the 2-3 the largest-gap rule picks


def test_every_top_group_large_enough_gets_two_base_clusters() -> None:
    rng = np.random.default_rng(3)
    far = np.zeros(32)
    far[0] = 50.0
    vectors = np.vstack([rng.normal(size=(40, 32)), far + rng.normal(scale=0.5, size=(12, 32))])

    tree = cluster_two_level(vectors, min_cluster_size=5)

    children: dict[int, int] = {}
    for top in tree.top_of_base.values():
        children[top] = children.get(top, 0) + 1
    assert all(count >= 2 for count in children.values())
