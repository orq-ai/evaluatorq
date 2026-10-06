from __future__ import annotations

import json
from datetime import datetime, timezone
from html.parser import HTMLParser

from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.insights_review_views import review_page
from evaluatorq.insights.models import InsightsConfig, InsightsRun, TraceInsight
from evaluatorq.insights.store import get_insights_runs_dir, save_run
from evaluatorq.signals.models import Evidence, SignalReport, SignalResult


class _SavedSignalsMountParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.options: dict[str, object] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == 'div':
            value = dict(attrs).get('data-saved-signals')
            if value is not None:
                self.options = json.loads(value)


def _run(*, with_signals: bool = True) -> InsightsRun:
    report = SignalReport(
        trajectory_id='trace-1',
        config_version='signals-v3',
        results={
            'tool_call_count': SignalResult(
                name='tool_call_count',
                group='A',
                value=1,
                evidence=[Evidence(step_id=2, call_id='call-1')],
            )
        },
    )
    return InsightsRun(
        run_id='signals-route-run',
        run_name='Signal route',
        created_at=datetime(2026, 10, 5, tzinfo=timezone.utc),
        status='completed',
        stage_failures=[],
        population={'mode': 'snapshot'},
        config=InsightsConfig(labels=[], dimensions=[]),
        traces=[
            TraceInsight(
                trace_id='trace-1',
                span_id='span-1',
                timestamp=datetime(2026, 10, 5, tzinfo=timezone.utc),
                signals=report if with_signals else None,
                source_coverage={'source': 'snapshot', 'partial': True},
            )
        ],
        dimensions={},
        labels={},
        priority=None,
        priority_reason=None,
        counts={},
        warnings=[],
    )


def test_signal_detail_route_reads_saved_report_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    save_run(_run(), get_insights_runs_dir())

    def unexpected_source_request(*args, **kwargs):
        raise AssertionError('signal details must come from the saved run')

    monkeypatch.setattr('evaluatorq.dashboard.insights_routes.resolve_orq_client', unexpected_source_request)
    client = TestClient(build_app())

    response = client.get(
        '/insights/signals-route-run/trace-signals.json?trace_id=trace-1&span_id=span-1'
    )

    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    assert response.json() == {
        'trace_id': 'trace-1',
        'span_id': 'span-1',
        'has_signals': True,
        'signals': _run().traces[0].signals.model_dump(mode='json'),
        'source_coverage': {'source': 'snapshot', 'partial': True},
    }


def test_signal_detail_route_handles_missing_trace_run_and_old_report(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    save_run(_run(with_signals=False), get_insights_runs_dir())
    client = TestClient(build_app())

    old_report = client.get(
        '/insights/signals-route-run/trace-signals.json?trace_id=trace-1&span_id=span-1'
    )
    missing_trace = client.get(
        '/insights/signals-route-run/trace-signals.json?trace_id=wrong&span_id=span-1'
    )
    missing_run = client.get(
        '/insights/missing-run/trace-signals.json?trace_id=trace-1&span_id=span-1'
    )
    missing_key = client.get('/insights/signals-route-run/trace-signals.json?trace_id=trace-1')

    assert old_report.status_code == 200 and old_report.json()['has_signals'] is False
    assert old_report.json()['signals'] is None
    assert missing_trace.status_code == 404
    assert missing_run.status_code == 404
    assert missing_key.status_code == 400


def test_trace_detail_page_mounts_saved_signals(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    run = _run()
    save_run(run, get_insights_runs_dir())
    client = TestClient(build_app())

    response = client.get('/insights/signals-route-run/trace?trace_id=trace-1&span_id=span-1')
    parser = _SavedSignalsMountParser()
    parser.feed(response.text)

    assert response.status_code == 200
    assert '<div data-saved-signals=' in response.text
    assert parser.options == {
        'report': run.traces[0].signals.model_dump(mode='json'),
        'detail': {
            'signals': run.traces[0].signals.model_dump(mode='json'),
            'source_coverage': {'source': 'snapshot', 'partial': True},
        },
    }
    assert '/static/insights-signals.css' in response.text
    assert '/static/insights-signals.js' in response.text


def test_trace_detail_page_mounts_missing_signals_as_null(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    save_run(_run(with_signals=False), get_insights_runs_dir())
    client = TestClient(build_app())

    response = client.get('/insights/signals-route-run/trace?trace_id=trace-1&span_id=span-1')
    parser = _SavedSignalsMountParser()
    parser.feed(response.text)

    assert response.status_code == 200
    assert parser.options == {
        'report': None,
        'detail': {'signals': None, 'source_coverage': {'source': 'snapshot', 'partial': True}},
    }


def test_review_page_loads_shared_signal_widget_before_review_script() -> None:
    html = review_page(_run())

    assert '/static/insights-signals.css' in html
    assert html.index('/static/insights-signals.js') < html.index('/static/insights-review.js')
