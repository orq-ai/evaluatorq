from __future__ import annotations

# Assertions are the test framework's reporting contract.
# ruff: noqa: S101

from datetime import datetime, timezone

from starlette.testclient import TestClient

from evaluatorq.common.trace_document import DatasetRef, Outcome
from evaluatorq.dashboard.app import build_app
from evaluatorq.insights.models import InsightsConfig, InsightsRun, TraceInsight
from evaluatorq.insights.store import get_insights_runs_dir, save_run
from evaluatorq.signals.models import Evidence, SignalReport, SignalResult


def _run() -> InsightsRun:
    trace = TraceInsight(
        trace_id='bench-row-7',
        span_id=None,
        timestamp=None,
        dataset=DatasetRef(name='org/bench', revision='abc123', split='test', row_id='row-7'),
        outcome=Outcome(
            passed=True,
            score=1.0,
            source='programmatic',
            definition='The task completed successfully.',
        ),
        signals=SignalReport(
            trajectory_id='bench-row-7',
            config_version='signals-v3',
            results={
                'tool_call_count': SignalResult(
                    name='tool_call_count',
                    group='A',
                    value=0,
                    evidence=[Evidence(step_id=1)],
                )
            },
        ),
    )
    return InsightsRun(
        run_id='dataset-run',
        run_name='Dataset run',
        created_at=datetime(2026, 10, 6, tzinfo=timezone.utc),
        status='completed',
        stage_failures=[],
        population={'mode': 'dataset'},
        config=InsightsConfig(labels=[], dimensions=[]),
        traces=[trace],
        dimensions={},
        labels={},
        priority=None,
        priority_reason=None,
        counts={'n_traces': 1},
        warnings=[],
    )


def test_dataset_trace_without_orq_metadata_renders_list_detail_and_signal_routes(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    save_run(_run(), get_insights_runs_dir())
    client = TestClient(build_app())

    review = client.get('/insights/dataset-run/review-data.json')
    row = review.json()['traces'][0]
    listing = client.get('/insights/dataset-run/traces')
    detail = client.get('/insights/dataset-run/trace?trace_id=bench-row-7&span_id=')
    signal_url = row['signal_detail_url']
    signals = client.get(signal_url)

    assert review.status_code == 200
    assert row['span_id'] is None
    assert row['ts'] is None
    assert row['trace_url'] == '/insights/dataset-run/trace?trace_id=bench-row-7&span_id='
    assert row['orq_url'] is None
    assert row['signal_detail_url'] == (
        '/insights/dataset-run/trace-signals.json?trace_id=bench-row-7&span_id='
    )
    assert listing.status_code == 200
    assert 'Time unavailable' in listing.text
    assert '↗ Orq' not in listing.text
    assert detail.status_code == 200
    assert 'Time unavailable' in detail.text
    assert ' · span None' not in detail.text
    assert 'Open full trace in Orq' not in detail.text
    assert signals.status_code == 200
    assert signals.json()['trace_id'] == 'bench-row-7'
    assert signals.json()['span_id'] is None
