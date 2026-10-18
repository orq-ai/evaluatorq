"""Unit tests for `evaluatorq.insights.reduce` — 3D UMAP reduction."""

from __future__ import annotations

import numpy as np
import pytest

from evaluatorq.insights import reduce as reduce_module
from evaluatorq.insights.reduce import reduce_3d


@pytest.mark.filterwarnings('error:n_jobs value.*overridden to 1:UserWarning')
def test_reduce_3d_shape_and_determinism():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(40, 16))

    coords_1 = reduce_3d(vectors, random_state=42)
    coords_2 = reduce_3d(vectors, random_state=42)

    assert coords_1 is not None
    assert coords_1.shape == (40, 3)
    np.testing.assert_array_equal(coords_1, coords_2)


@pytest.mark.filterwarnings('error:n_jobs value.*overridden to 1:UserWarning')
def test_project_2d_produces_repeatable_layout_from_saved_3d_coordinates():
    coords = np.random.default_rng(0).normal(size=(12, 3))
    original = coords.copy()

    first = reduce_module.project_2d(coordinates=coords)
    second = reduce_module.project_2d(coordinates=coords)

    assert first is not None
    assert first.shape == (12, 2)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(coords, original)


@pytest.mark.parametrize('coordinates', [np.zeros((4, 3)), np.zeros((6, 2)), np.full((6, 3), np.nan)])
def test_project_2d_rejects_unusable_coordinates(coordinates):
    assert reduce_module.project_2d(coordinates=coordinates) is None


def test_project_2d_degrades_when_umap_fails(monkeypatch):
    class BrokenReducer:
        def __init__(self, **kwargs):
            raise RuntimeError('projection failed')

    monkeypatch.setitem(__import__('sys').modules, 'umap', type('FakeUmap', (), {'UMAP': BrokenReducer}))
    assert reduce_module.project_2d(coordinates=np.arange(18).reshape(6, 3)) is None


def test_analysis_serializes_umap_fits_across_both_layouts(monkeypatch):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    guard = Lock()
    active = maximum = 0

    class ProbeReducer:
        def __init__(self, *, n_components, **kwargs):
            self.dimensions = n_components

        def fit_transform(self, values):
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            try:
                time.sleep(0.03)
                return values[:, :self.dimensions]
            finally:
                with guard:
                    active -= 1

    monkeypatch.setitem(__import__('sys').modules, 'umap', type('FakeUmap', (), {'UMAP': ProbeReducer}))
    values = np.arange(18, dtype=float).reshape(6, 3) + 1
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(reduce_3d, values)
        second = pool.submit(reduce_module.project_2d, values)
        assert first.result() is not None
        assert second.result() is not None
    assert maximum == 1


def test_reduce_3d_none_below_five_points():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(4, 16))

    assert reduce_3d(vectors) is None


@pytest.mark.parametrize(
    'vectors',
    [
        np.full((5, 3), np.nan),
        np.zeros((5, 3)),
        np.ones((5,)),
    ],
)
def test_reduce_3d_degrades_on_invalid_or_degenerate_vectors(vectors):
    assert reduce_3d(vectors) is None


def test_reduce_3d_degrades_when_umap_fails(monkeypatch):
    class BrokenReducer:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, vectors):
            raise ValueError('broken UMAP input')

    monkeypatch.setitem(__import__('sys').modules, 'umap', type('FakeUmap', (), {'UMAP': BrokenReducer}))
    vectors = np.arange(30, dtype=float).reshape(10, 3) + 1

    assert reduce_module.reduce_3d(vectors) is None
