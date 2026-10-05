"""Projection behavior for the Insights review map."""

# Assertions are the test framework's reporting contract.
# ruff: noqa: S101

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from types import ModuleType
from typing import TYPE_CHECKING, cast

import numpy as np

from evaluatorq.dashboard import insights_review_projection as projection
from evaluatorq.dashboard.insights_review_data import build_review_payload
from evaluatorq.insights.models import DimensionResult, InsightsConfig, InsightsRun, TraceInsight

if TYPE_CHECKING:
    import pytest


def _run(*, count: int = 7) -> InsightsRun:
    traces = [
        TraceInsight(
            trace_id='same' if index < 2 else f'trace-{index}',
            span_id=f'span-{index}',
            timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
            coords={'intent': (float(index), float(index + 1), float(index + 2))},
        )
        for index in range(count)
    ]
    return InsightsRun(
        run_id='projection-test',
        run_name='Projection test',
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        status='completed',
        stage_failures=[],
        population={},
        config=InsightsConfig(labels=[], dimensions=['intent']),
        traces=traces,
        dimensions={'intent': DimensionResult(name='intent', source_field='request', clusters=[])},
        labels={},
        priority=None,
        priority_reason=None,
        counts={'n_traces': count},
        warnings=[],
    )


def _fake_umap(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    module = ModuleType('umap')

    class Reducer:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

        def fit_transform(self, values: np.ndarray) -> np.ndarray:
            return np.asarray([[row[0], row[1]] for row in values])

    module.UMAP = Reducer
    monkeypatch.setitem(sys.modules, 'umap', module)
    return calls


def test_review_xy_maps_saved_points_by_full_trace_key_and_uses_fixed_umap(monkeypatch: pytest.MonkeyPatch):
    projection._clear_review_xy_cache()  # noqa: SLF001
    calls = _fake_umap(monkeypatch)

    points = projection.review_xy(_run(), 'intent')

    assert points['same:span-0'] == (0.0, 1.0)
    assert points['same:span-1'] == (1.0, 2.0)
    assert points['trace-6:span-6'] == (6.0, 7.0)
    assert calls == [{'n_components': 2, 'n_neighbors': 6, 'metric': 'euclidean', 'random_state': 7}]


def test_review_xy_returns_explicit_no_map_state_below_five_points(monkeypatch: pytest.MonkeyPatch):
    projection._clear_review_xy_cache()  # noqa: SLF001
    calls = _fake_umap(monkeypatch)
    run = _run(count=4)

    assert projection.review_xy(run, 'intent') == {}
    state = projection.review_xy_state(run, 'intent')
    assert state == {
        'available': False,
        'reason': 'Only 4 valid points are available; at least 5 are needed.',
        'n_points': 4,
    }
    assert calls == []


def test_review_xy_skips_invalid_coordinates_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    projection._clear_review_xy_cache()  # noqa: SLF001
    _fake_umap(monkeypatch)
    run = _run(count=7)
    run.traces[0].coords['intent'] = (float('nan'), 1.0, 2.0)
    run.traces[1].coords['intent'] = cast('tuple[float, float, float]', (1.0, 2.0))

    points = projection.review_xy(run, 'intent')

    assert 'same:span-0' not in points
    assert 'same:span-1' not in points
    assert len(points) == 5
    assert 'skipped invalid coordinates' in caplog.text


def test_review_xy_cache_hits_and_coordinate_changes_miss(monkeypatch: pytest.MonkeyPatch):
    projection._clear_review_xy_cache()  # noqa: SLF001
    calls = _fake_umap(monkeypatch)
    run = _run()

    first = projection.review_xy(run, 'intent')
    second = projection.review_xy(run, 'intent')
    run.traces[0].coords['intent'] = (99.0, 100.0, 101.0)
    changed = projection.review_xy(run, 'intent')

    assert first == second
    assert len(calls) == 2
    assert changed['same:span-0'] == (99.0, 100.0)


def test_review_xy_serializes_concurrent_umap_fits(monkeypatch: pytest.MonkeyPatch):
    projection._clear_review_xy_cache()  # noqa: SLF001
    state_lock = Lock()
    active = 0
    maximum_active = 0
    module = ModuleType('umap')

    class ConcurrentProbeReducer:
        def __init__(self, **kwargs: object) -> None:
            pass

        def fit_transform(self, values: np.ndarray) -> np.ndarray:
            nonlocal active, maximum_active
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                time.sleep(0.03)
                return np.asarray([[row[0], row[1]] for row in values])
            finally:
                with state_lock:
                    active -= 1

    module.UMAP = ConcurrentProbeReducer
    monkeypatch.setitem(sys.modules, 'umap', module)
    runs = [_run().model_copy(update={'run_id': f'projection-test-{index}'}) for index in range(4)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda run: projection.review_xy(run, 'intent'), runs))

    assert all(len(result) == 7 for result in results)
    assert maximum_active == 1


def test_review_payload_exposes_xy_without_mutating_saved_xyz(monkeypatch: pytest.MonkeyPatch):
    projection._clear_review_xy_cache()  # noqa: SLF001
    _fake_umap(monkeypatch)
    run = _run()

    trace = cast(dict[str, object], cast(list[object], build_review_payload(run)['traces'])[0])

    assert cast(dict[str, object], trace['xy']) == {'intent': [0.0, 1.0]}
    assert 'xyz' not in trace
    assert run.traces[0].coords == {'intent': (0.0, 1.0, 2.0)}
    assert trace['id'] == 'same:span-0'
