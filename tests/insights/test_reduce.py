"""Unit tests for `evaluatorq.insights.reduce` — 3D UMAP reduction."""

from __future__ import annotations

import numpy as np
import pytest

from evaluatorq.insights import reduce as reduce_module
from evaluatorq.insights.reduce import reduce_3d


def test_reduce_3d_shape_and_determinism():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(40, 16))

    coords_1 = reduce_3d(vectors, random_state=42)
    coords_2 = reduce_3d(vectors, random_state=42)

    assert coords_1 is not None
    assert coords_1.shape == (40, 3)
    np.testing.assert_array_equal(coords_1, coords_2)


def test_reduce_3d_none_below_five_points(caplog):
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
