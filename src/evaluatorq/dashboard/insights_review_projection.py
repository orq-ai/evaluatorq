"""Read persisted 2D coordinates for the Insights review map."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from evaluatorq.insights.models import InsightsRun, TraceInsight


def review_trace_key(trace: TraceInsight) -> str:
    """Return the shared map and review-payload key for one trace span."""
    return f'{trace.trace_id}:{trace.span_id or ""}'


def _finite_pair(values: object) -> tuple[float, float] | None:
    if not isinstance(values, (list, tuple)):
        return None
    try:
        pair = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if len(pair) != 2 or not all(math.isfinite(value) for value in pair):
        return None
    return pair[0], pair[1]


def _legacy_xy(run: InsightsRun, dimension: str) -> dict[str, tuple[float, float]]:
    points: dict[str, tuple[float, float]] = {}
    for trace in run.traces:
        coords = trace.coords.get(dimension)
        if coords is None:
            continue
        try:
            values = tuple(float(value) for value in coords)
        except (TypeError, ValueError):
            values = ()
        if len(values) == 3 and all(math.isfinite(value) for value in values):
            points[review_trace_key(trace)] = (values[0], values[1])
        else:
            logger.warning(
                'Insights review map skipped invalid coordinates for {} in dimension {}', trace.trace_id, dimension
            )
    return points


def review_xy(run: InsightsRun, dimension: str) -> dict[str, tuple[float, float]]:
    """Read saved finite 2D coordinates, with an announced 3D fallback for legacy reports."""
    if all(trace.coords_2d is None for trace in run.traces):
        logger.warning('Insights review map is using the legacy report 3D XY fallback for dimension {}', dimension)
        points = _legacy_xy(run, dimension)
    else:
        points = {}
        for trace in run.traces:
            coords_2d = trace.coords_2d or {}
            if dimension not in coords_2d:
                continue
            pair = _finite_pair(coords_2d[dimension])
            if pair is None:
                logger.warning(
                    'Insights review map skipped invalid coordinates for {} in dimension {}', trace.trace_id, dimension
                )
                continue
            points[review_trace_key(trace)] = pair
    if len(points) < 5:
        logger.warning(
            'Insights review map unavailable for dimension {}: only {} valid points (need >= 5)', dimension, len(points)
        )
        return {}
    return points


def review_xy_state(
    run: InsightsRun,
    dimension: str,
    xy: dict[str, tuple[float, float]] | None = None,
) -> dict[str, object]:
    """Return explicit availability metadata for saved review-map coordinates."""
    if xy is None:
        xy = review_xy(run, dimension)
    if xy:
        return {'available': True, 'reason': None, 'n_points': len(xy)}
    if all(trace.coords_2d is None for trace in run.traces):
        valid_points = _legacy_xy(run, dimension)
    else:
        valid_points = {
            review_trace_key(trace)
            for trace in run.traces
            if dimension in (trace.coords_2d or {}) and _finite_pair((trace.coords_2d or {})[dimension]) is not None
        }
    has_saved_dimension = any(dimension in (trace.coords_2d or {}) for trace in run.traces)
    if not has_saved_dimension and not all(trace.coords_2d is None for trace in run.traces):
        reason = 'Saved 2D coordinates are unavailable for this dimension.'
    elif len(valid_points) < 5:
        reason = f'Only {len(valid_points)} valid points are available; at least 5 are needed.'
    else:
        reason = 'Saved 2D coordinates are unavailable for this dimension.'
    return {'available': False, 'reason': reason, 'n_points': len(valid_points)}
