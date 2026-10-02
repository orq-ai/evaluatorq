"""Deterministic 2D projections for the Insights review map."""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from threading import RLock
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from evaluatorq.insights.models import InsightsRun

_CACHE_LIMIT = 32
_PROJECTIONS: OrderedDict[str, dict[str, tuple[float, float]]] = OrderedDict()
_CACHE_LOCK = RLock()


def _coordinates(run: InsightsRun, dimension: str, *, warn: bool = True) -> tuple[list[str], list[list[float]]]:
    ids: list[str] = []
    rows: list[list[float]] = []
    for trace in run.traces:
        coords = trace.coords.get(dimension)
        if coords is None:
            continue
        try:
            row = [float(value) for value in coords]
        except (TypeError, ValueError):
            if warn:
                logger.warning(
                    'Insights review map skipped invalid coordinates for {} in dimension {}', trace.trace_id, dimension
                )
            continue
        if len(row) != 3 or not all(math.isfinite(value) for value in row):
            if warn:
                logger.warning(
                    'Insights review map skipped invalid coordinates for {} in dimension {}', trace.trace_id, dimension
                )
            continue
        ids.append(f'{trace.trace_id}:{trace.span_id}')
        rows.append(row)
    return ids, rows


def _cache_key(run: InsightsRun, dimension: str, ids: list[str], rows: list[list[float]]) -> str:
    content = json.dumps([ids, rows], separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(f'{run.run_id}\0{dimension}\0{content}'.encode()).hexdigest()


def review_xy(run: InsightsRun, dimension: str) -> dict[str, tuple[float, float]]:
    """Project saved finite 3D coordinates to deterministic display XY points.

    Fewer than five valid points and UMAP failures return an empty map. UMAP is
    imported only when this review projection is requested.
    """
    ids, rows = _coordinates(run, dimension)
    if len(rows) < 5:
        logger.warning(
            'Insights review map unavailable for dimension {}: only {} valid points (need >= 5)', dimension, len(rows)
        )
        return {}
    key = _cache_key(run, dimension, ids, rows)
    # UMAP's default Numba workqueue layer is not safe for concurrent calls from
    # multiple request threads. Keep the lock through cache misses so all UMAP
    # fitting in this process is serialized, including misses for different runs.
    with _CACHE_LOCK:
        cached = _PROJECTIONS.get(key)
        if cached is not None:
            _PROJECTIONS.move_to_end(key)
            return cached.copy()
        try:
            import numpy as np
            import umap

            projected = np.asarray(
                umap.UMAP(
                    n_components=2,
                    n_neighbors=min(15, len(rows) - 1),
                    metric='euclidean',
                    random_state=7,
                ).fit_transform(np.asarray(rows, dtype=float))
            )
            if projected.shape != (len(rows), 2) or not np.isfinite(projected).all():
                raise ValueError(f'UMAP returned invalid coordinates with shape {projected.shape}')
            result = {
                trace_id: (float(point[0]), float(point[1])) for trace_id, point in zip(ids, projected, strict=True)
            }
        except Exception as exc:  # noqa: BLE001 - display projection degrades to an explicit empty map
            logger.warning(
                'Insights review map unavailable for dimension {}: UMAP projection failed: {}', dimension, exc
            )
            return {}
        _PROJECTIONS[key] = result.copy()
        _PROJECTIONS.move_to_end(key)
        while len(_PROJECTIONS) > _CACHE_LIMIT:
            _PROJECTIONS.popitem(last=False)
        return result


def warm_review_projection() -> None:
    """Pay UMAP's import and first-fit compile (~8s cold) before the first review request does."""
    with _CACHE_LOCK:
        try:
            import numpy as np
            import umap

            umap.UMAP(n_components=2, n_neighbors=4, random_state=7).fit_transform(
                np.random.default_rng(7).random((8, 3))
            )
        except Exception as exc:  # noqa: BLE001 - warm-up is best effort; review_xy reports real failures
            logger.debug('Insights review map warm-up skipped: {}', exc)


def review_xy_state(
    run: InsightsRun,
    dimension: str,
    xy: dict[str, tuple[float, float]] | None = None,
) -> dict[str, object]:
    """Return explicit availability metadata for a review-map projection."""
    if xy is None:
        xy = review_xy(run, dimension)
    if xy:
        return {'available': True, 'reason': None, 'n_points': len(xy)}
    valid_ids, _ = _coordinates(run, dimension, warn=False)
    if len(valid_ids) < 5:
        reason = f'Only {len(valid_ids)} valid points are available; at least 5 are needed.'
    else:
        reason = 'The 2D projection could not be produced.'
    return {'available': False, 'reason': reason, 'n_points': len(valid_ids)}


def _clear_review_xy_cache() -> None:
    """Clear the process cache for isolated tests."""
    with _CACHE_LOCK:
        _PROJECTIONS.clear()
