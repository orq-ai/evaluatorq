"""Persisted coordinate behavior for the Insights review map."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, cast

from evaluatorq.dashboard import insights_review_projection as projection
from evaluatorq.insights.models import DimensionResult, InsightsConfig, InsightsRun, TraceInsight

if TYPE_CHECKING:
    import pytest


def _run(*, count: int = 7, saved: bool = True) -> InsightsRun:
    traces = [
        TraceInsight(
            trace_id='same' if index < 2 else f'trace-{index}',
            span_id=f'span-{index}',
            timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
            coords={'intent': (float(index), float(index + 1), float(index + 2))},
            coords_2d={'intent': (float(index + 10), float(index + 20))} if saved else None,
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


def test_review_xy_uses_saved_points_by_full_trace_key_without_mutating_run():
    run = _run()
    original = [trace.model_dump() for trace in run.traces]

    points = projection.review_xy(run, 'intent')

    assert points['same:span-0'] == (10.0, 20.0)
    assert points['same:span-1'] == (11.0, 21.0)
    assert points['trace-6:span-6'] == (16.0, 26.0)
    assert [trace.model_dump() for trace in run.traces] == original


def test_review_xy_uses_3d_xy_only_for_legacy_missing_field(caplog: pytest.LogCaptureFixture):
    run = _run(saved=False)

    points = projection.review_xy(run, 'intent')

    assert points['same:span-0'] == (0.0, 1.0)
    assert len(points) == 7
    assert 'legacy report' in caplog.text
    assert 'fallback' in caplog.text


def test_new_report_missing_dimension_does_not_fallback():
    run = _run()
    for trace in run.traces:
        trace.coords_2d = {}

    assert projection.review_xy(run, 'intent') == {}
    assert (
        projection.review_xy_state(run, 'intent')['reason']
        == 'Saved 2D coordinates are unavailable for this dimension.'
    )


def test_review_xy_skips_invalid_saved_coordinates(caplog: pytest.LogCaptureFixture):
    run = _run()
    run.traces[0].coords_2d = {'intent': (float('nan'), 2.0)}
    run.traces[1].coords_2d = {'intent': cast('tuple[float, float]', (1.0,))}

    points = projection.review_xy(run, 'intent')

    assert 'same:span-0' not in points
    assert 'same:span-1' not in points
    assert len(points) == 5
    assert 'skipped invalid coordinates' in caplog.text


def test_review_xy_returns_explicit_no_map_state_below_five_points():
    run = _run(count=4)

    assert projection.review_xy(run, 'intent') == {}
    assert projection.review_xy_state(run, 'intent') == {
        'available': False,
        'reason': 'Only 4 valid points are available; at least 5 are needed.',
        'n_points': 4,
    }


def test_dashboard_payload_and_startup_do_not_import_heavy_reducers():
    script = r"""import importlib.abc
import os
import sys
import tempfile
from datetime import datetime, timezone

forbidden = {'umap', 'numpy', 'sklearn', 'numba'}
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in forbidden:
            raise AssertionError(f'unexpected heavy import: {fullname}')
        return None
sys.meta_path.insert(0, BlockHeavy())

from evaluatorq.insights.models import DimensionResult, InsightsConfig, InsightsRun, TraceInsight
from evaluatorq.dashboard.insights_review_data import build_review_payload
traces = [TraceInsight(trace_id=f't-{i}', span_id=f's-{i}', coords_2d={'intent': (float(i), float(i + 1))}) for i in range(5)]
run = InsightsRun(run_id='fresh-process', run_name='Fresh', created_at=datetime.now(timezone.utc), status='completed', stage_failures=[], population={}, config=InsightsConfig(labels=[], dimensions=['intent']), traces=traces, dimensions={'intent': DimensionResult(name='intent', source_field='request', clusters=[])}, labels={}, priority=None, priority_reason=None, counts={'n_traces': 5}, warnings=[])
payload = build_review_payload(run)
assert payload['traces'][0]['xy'] == {'intent': [0.0, 1.0]}
with tempfile.TemporaryDirectory() as cwd:
    os.chdir(cwd)
    from evaluatorq.dashboard.launch import build_app_from_env
    build_app_from_env()
print('dashboard payload and startup stayed free of heavy reducers')
"""
    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[2],
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert 'dashboard payload and startup stayed free of heavy reducers' in result.stdout
