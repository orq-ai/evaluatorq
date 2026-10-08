"""Trace tab of the Insights trace page: lazy conversation and span routes."""

from __future__ import annotations

import hashlib
import re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from loguru import logger
from starlette.testclient import TestClient

from evaluatorq.dashboard import insights_routes
from evaluatorq.dashboard.app import build_app
from evaluatorq.insights.models import InsightsRun
from evaluatorq.trace_finder.models import Snapshot, TraceRecord
from tests.dashboard.test_insights_page import _write_run, minimal_run  # noqa: F401 — fixture reused
from tests.trace_finder.test_orq_source import FakeOrq, FakeTraces, span

TRACE_PAGE = '/insights/run-1/trace?trace_id=trace-1&span_id=span-1'
CONVERSATION = '/insights/run-1/trace-conversation?trace_id=trace-1&span_id=span-1'


def _record(trace_id: str, span_id: str, content: str) -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id=trace_id,
        span_id=span_id,
        timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
        messages=({'role': 'user', 'content': content},),
        project='default',
        model='m',
        provider='p',
        status='ok',
        product='deployments',
        trace_type='trace',
    )


def _write_snapshot(path: Path, records: list[TraceRecord]) -> str:
    path.write_bytes(Snapshot(traces=tuple(records)).model_dump_json().encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_run(run: InsightsRun, path: Path, digest: str) -> InsightsRun:
    return run.model_copy(
        update={'population': {'mode': 'snapshot', 'snapshot_path': str(path), 'snapshot_sha256': digest}}
    )


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    monkeypatch.setenv('ORQ_WORKSPACE', 'example-workspace')
    return tmp_path


def test_trace_view_renders_a_lazy_conversation_placeholder(env: Path, minimal_run: InsightsRun) -> None:
    _write_run(env, minimal_run)

    page = TestClient(build_app()).get(f'{TRACE_PAGE}&view=trace')

    assert page.status_code == 200
    assert 'hx-get="/insights/run-1/trace-conversation?trace_id=trace-1&amp;span_id=span-1"' in page.text
    assert 'hx-trigger="load"' in page.text
    assert 'hx-swap="outerHTML"' in page.text
    assert 'Loading conversation…' in page.text
    assert '<h3>Summary</h3>' not in page.text
    assert re.search(r'class="(?:[^"]*\s)?insights-tabs(?:\s[^"]*)?"', page.text) is None
    assert (
        '<a class="insights-tab active" href="/insights/run-1/trace?trace_id=trace-1&amp;span_id=span-1&amp;view=trace" '
        'aria-current="page">Trace</a>'
    ) in page.text
    assert (
        '<a class="insights-tab" href="/insights/run-1/trace?trace_id=trace-1&amp;span_id=span-1">Analysis</a>'
    ) in page.text


def test_unknown_view_falls_back_to_the_analysis_page(env: Path, minimal_run: InsightsRun) -> None:
    _write_run(env, minimal_run)

    page = TestClient(build_app()).get(f'{TRACE_PAGE}&view=bogus')

    assert page.status_code == 200
    assert '<h3>Summary</h3>' in page.text
    assert 'trace-conversation' not in page.text


def test_snapshot_run_reads_the_conversation_from_its_snapshot(env: Path, minimal_run: InsightsRun) -> None:
    path = env / 'snap.json'
    digest = _write_snapshot(path, [_record('trace-1', 'span-1', 'hello from the snapshot')])
    _write_run(env, _snapshot_run(minimal_run, path, digest))

    response = TestClient(build_app()).get(CONVERSATION)

    assert response.status_code == 200
    assert 'hello from the snapshot' in response.text
    assert response.text.startswith('<div class="fd-traces insights-trace-view">')
    assert 'trace-spans' not in response.text


def test_snapshot_run_with_the_file_gone_says_so_without_the_path(env: Path, minimal_run: InsightsRun) -> None:
    path = env / 'snap.json'
    digest = _write_snapshot(path, [_record('trace-1', 'span-1', 'x')])
    _write_run(env, _snapshot_run(minimal_run, path, digest))
    path.unlink()

    response = TestClient(build_app()).get(CONVERSATION)

    assert response.status_code == 200
    assert 'Conversation unavailable' in response.text
    assert str(env) not in response.text


def _fake_source(monkeypatch: pytest.MonkeyPatch, record: TraceRecord | str) -> None:
    sentinel = object()

    @asynccontextmanager
    async def source(app: Any):
        yield sentinel

    async def load(run: InsightsRun, trace: Any, *, source: Any) -> TraceRecord | str:
        assert source is sentinel
        return record

    monkeypatch.setattr(insights_routes, '_trace_source', source)
    monkeypatch.setattr(insights_routes, 'load_trace_record', load)


def test_orq_run_shows_the_transcript_and_a_spans_url(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(env, minimal_run)
    _fake_source(monkeypatch, _record('trace-1', 'span-1', 'hello from orq'))

    response = TestClient(build_app()).get(CONVERSATION)

    assert response.status_code == 200
    assert 'hello from orq' in response.text
    assert 'hx-get="/insights/run-1/trace-spans?trace_id=trace-1"' in response.text
    assert 'now selects span' not in response.text


def test_orq_run_with_a_different_span_shows_a_mismatch_notice(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(env, minimal_run)
    _fake_source(monkeypatch, _record('trace-1', 'span-other', 'hello from orq'))
    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(str(message)), level='WARNING', format='{message}')

    try:
        response = TestClient(build_app()).get(CONVERSATION)
    finally:
        logger.remove(sink)

    assert 'Orq now selects span span-other' in response.text
    assert 'the analysis used span span-1' in response.text
    assert 'hello from orq' in response.text
    assert [message for message in warnings if 'span-other' in message] == [
        'Insights trace trace-1 was analysed on span span-1 but Orq now selects span span-other\n'
    ]


def test_orq_run_without_credentials_degrades_visibly(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(env, minimal_run)

    def no_auth(app: Any) -> Any:
        raise ValueError('no credentials')

    monkeypatch.setattr(insights_routes, 'selected_dashboard_auth', no_auth)
    client = TestClient(build_app())
    warnings: list[str] = []
    sink = logger.add(lambda message: warnings.append(str(message)), level='WARNING', format='{message}')

    try:
        conversation = client.get(CONVERSATION)
    finally:
        logger.remove(sink)
    spans = client.get('/insights/run-1/trace-spans?trace_id=trace-1')

    assert conversation.status_code == 200
    assert 'Connect an Orq account in Settings' in conversation.text
    assert warnings == ['Insights trace source unavailable: no credentials\n']
    assert spans.status_code == 200
    assert 'Could not load spans. Try again.' in spans.text


def _named_span(span_id: str, *, name: str, minute: int, parent_span_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(**vars(span(span_id, minute=minute, parent_span_id=parent_span_id)), name=name)


def test_spans_route_renders_the_span_tree_from_orq(
    env: Path, minimal_run: InsightsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_run(env, minimal_run)
    traces = FakeTraces(
        {},
        spans={
            'trace-1': [
                _named_span('span-root', name='answer refund question', minute=0),
                _named_span('span-child', name='look up refund policy', minute=1, parent_span_id='span-root'),
            ]
        },
    )
    monkeypatch.setattr(insights_routes, 'build_orq_client', lambda auth, **kwargs: FakeOrq(traces))

    response = TestClient(build_app()).get('/insights/run-1/trace-spans?trace_id=trace-1')

    assert response.status_code == 200
    assert 'answer refund question' in response.text
    assert 'look up refund policy' in response.text


def test_unknown_trace_and_snapshot_runs_have_no_span_or_conversation_routes(
    env: Path, minimal_run: InsightsRun
) -> None:
    _write_run(env, minimal_run)
    path = env / 'snap.json'
    digest = _write_snapshot(path, [_record('trace-1', 'span-1', 'x')])
    _write_run(env, _snapshot_run(minimal_run, path, digest).model_copy(update={'run_id': 'snapshot-run'}))
    client = TestClient(build_app())

    unknown_spans = client.get('/insights/run-1/trace-spans?trace_id=unknown')
    snapshot_spans = client.get('/insights/snapshot-run/trace-spans?trace_id=trace-1')

    assert unknown_spans.status_code == 404
    assert 'Span loading is unavailable.' in unknown_spans.text
    assert snapshot_spans.status_code == 404
    assert client.get('/insights/run-1/trace-conversation?trace_id=unknown&span_id=span-1').status_code == 404
    assert client.get('/insights/missing/trace-conversation?trace_id=trace-1&span_id=span-1').status_code == 404
