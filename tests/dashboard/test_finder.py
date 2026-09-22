from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard import finder_routes
from evaluatorq.dashboard.app import build_app
from evaluatorq.trace_finder import (
    CompiledQuery,
    FacetCatalogue,
    FacetSelection,
    JevProjection,
    NumericFilters,
    RunSnapshot,
    Snapshot,
    TraceClassification,
    TraceDetail,
    TraceRecord,
    ValueSelection,
)
from evaluatorq.common.judge import ClassifyQuestion


def _trace() -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id='trace-1',
        span_id='span-1',
        timestamp=datetime(2026, 9, 21, 12, 41, tzinfo=timezone.utc),
        messages=(
            {'role': 'user', 'content': 'This is the third time I am asking.'},
            {'role': 'assistant', 'content': 'I will check that for you.'},
        ),
        project='support-agent',
        model='gpt-5.6-luna',
        provider='openai',
        status='ok',
        product='production',
        trace_type='llm',
        agent_name='support',
        tool_names=('lookup_refund',),
        total_tokens=120,
        duration_ms=250,
    )


def _compiled() -> CompiledQuery:
    return CompiledQuery(
        task=ClassifyQuestion(
            kind='choice',
            instructions="Judge the customer's emotional state.",
            criteria={'frustrated': 'Annoyed or escalating.', 'neutral': 'Transactional.'},
            state={},
        ),
        selection=ValueSelection(kind='values', values=('frustrated',)),
    )


class FakeStore:
    def __init__(self) -> None:
        self.trace = _trace()
        self.projection = JevProjection(
            payload={'trace_status': 'ok', 'messages': list(self.trace.messages)},
            serialized='{"messages": []}',
            estimated_tokens=10,
            omitted_messages=0,
            omitted_bytes=0,
        )
        self.compiled = _compiled()
        self.snapshot_value = RunSnapshot()
        self.started = False
        self.compile_request: Any | None = None
        self.started_compiled: CompiledQuery | None = None

    async def compile(self, request: Any) -> RunSnapshot:
        self.compile_request = request
        state = 'awaiting_review' if request.mode == 'review' else 'classifying'
        self.snapshot_value = RunSnapshot(
            generation=1,
            state=state,
            request=request,
            compiled=self.compiled,
            generated_filters=FacetSelection(project=frozenset({'support-agent'})),
            generated_numeric=NumericFilters(),
            trace_ids=(self.trace.trace_id,),
            traces=(self.trace,),
            projections={self.trace.trace_id: self.projection} if state == 'classifying' else {},
            total=1,
            active=1 if state == 'classifying' else 0,
            queued=0,
        )
        return self.snapshot_value

    async def start(self, request: Any, compiled: CompiledQuery) -> RunSnapshot:
        self.started = True
        self.started_compiled = compiled
        result = TraceClassification(
            trace_id=self.trace.trace_id,
            span_id=self.trace.span_id,
            value='frustrated',
            confidence=0.91,
            probabilities={'frustrated': 0.91, 'neutral': 0.09},
            matched=True,
            summary='The customer repeats the request.',
            raw_result={'value': 'frustrated'},
        )
        self.snapshot_value = replace(
            self.snapshot_value,
            state='completed',
            compiled=compiled,
            projections={self.trace.trace_id: self.projection},
            results=MappingProxyType({self.trace.trace_id: result}),
            completed=1,
            matched=1,
            active=0,
            queued=0,
        )
        return self.snapshot_value

    async def snapshot(self) -> RunSnapshot:
        return self.snapshot_value

    async def cancel(self) -> RunSnapshot:
        self.snapshot_value = replace(self.snapshot_value, state='cancelled')
        return self.snapshot_value

    async def reset(self) -> RunSnapshot:
        self.snapshot_value = RunSnapshot(generation=self.snapshot_value.generation + 1)
        return self.snapshot_value

    async def trace_detail(self, trace_id: str) -> TraceDetail | None:
        if trace_id != self.trace.trace_id:
            return None
        return TraceDetail(
            trace=self.trace,
            projection=self.projection,
            classification=next(iter(self.snapshot_value.results.values()), None),
            compiled=self.snapshot_value.compiled,
        )

    def complete(self) -> None:
        result = TraceClassification(
            trace_id=self.trace.trace_id,
            span_id=self.trace.span_id,
            value='frustrated',
            confidence=0.91,
            probabilities={'frustrated': 0.91, 'neutral': 0.09},
            matched=True,
            summary='The customer repeats the request.',
            raw_result={'value': 'frustrated'},
        )
        self.snapshot_value = replace(
            self.snapshot_value,
            state='completed',
            results={self.trace.trace_id: result},
            completed=1,
            matched=1,
            active=0,
            queued=0,
        )


@pytest.fixture
def setup_finder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))
    store = FakeStore()
    monkeypatch.setattr(finder_routes, '_build_store', lambda app: store)
    return store, TestClient(build_app(roots=[tmp_path]), raise_server_exceptions=True)


def test_find_idle_page_and_nav(setup_finder) -> None:
    _store, client = setup_finder
    response = client.get('/find')
    assert response.status_code == 200
    assert 'Ask <em>JEV</em>' in response.text
    assert 'Every dot is a trace' in response.text
    assert 'Find' in response.text
    assert 'about 1,240 traces in window' not in response.text
    assert response.text.count('class="idle"') == 500


def test_find_without_api_key_renders_empty_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv('ORQ_API_KEY', raising=False)
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))
    client = TestClient(build_app(roots=[tmp_path]), raise_server_exceptions=True)
    response = client.get('/find')
    assert response.status_code == 200
    assert 'Set ORQ_API_KEY to load traces' in response.text
    assert '<textarea name="query"' in response.text
    assert '<textarea name="query"' in response.text and 'disabled' in response.text.split('<textarea name="query"', 1)[1].split('>', 1)[0]
    assert 'class="finder-hint"' in response.text
    assert '<span class="finder-key-hint"' not in response.text


def test_find_run_starts_polling_and_completed_poll_shows_matches(setup_finder) -> None:
    store, client = setup_finder
    response = client.post('/find/run', data={'query': 'frustrated customers', 'mode': 'immediate'})
    assert response.status_code == 200
    assert 'hx-trigger="every 1s"' in response.text

    store.complete()
    poll = client.get('/find/poll')
    assert poll.status_code == 200
    assert 'frustrated' in poll.text
    assert 'included' in poll.text
    assert 'trace-1' in poll.text


def test_find_run_passes_numeric_filter_and_renders_chip(setup_finder) -> None:
    store, client = setup_finder
    response = client.post('/find/run', data={'query': 'long traces', 'mode': 'immediate', 'tokens_min': '5000'})
    assert response.status_code == 200
    assert store.compile_request is not None
    assert store.compile_request.population.numeric.tokens_min == 5000
    assert '<span class="chip"><b>tokens</b>≥ 5000' in response.text
    assert 'name="tokens_min"' in response.text


def test_find_review_start_transitions_to_classification(setup_finder) -> None:
    store, client = setup_finder
    review = client.post('/find/run', data={'query': 'frustrated customers', 'mode': 'review'})
    assert review.status_code == 200
    assert 'Review the plan before spending JEV calls.' in review.text
    assert 'name="instructions"' in review.text

    started = client.post(
        '/find/start',
        data={
            'kind': 'choice',
            'instructions': 'Edited instructions',
            'criteria_label_0': 'frustrated',
            'criteria_description_0': 'Changed criterion',
            'criteria_label_1': 'neutral',
            'criteria_description_1': 'Transactional.',
            'selection_value': 'frustrated',
        },
    )
    assert started.status_code == 200
    assert store.started
    assert store.started_compiled is not None
    assert store.started_compiled.task.instructions == 'Edited instructions'
    assert store.started_compiled.task.criteria['frustrated'] == 'Changed criterion'


def test_find_failed_snapshot_renders_escaped_error(setup_finder) -> None:
    store, client = setup_finder
    store.snapshot_value = RunSnapshot(state='failed', error='<compiler failed>')
    response = client.get('/find/poll')
    assert response.status_code == 200
    assert '&lt;compiler failed&gt;' in response.text
    assert '<compiler failed>' not in response.text


def test_find_trace_drawer_renders_thread_and_jev_input(setup_finder, monkeypatch: pytest.MonkeyPatch) -> None:
    store, client = setup_finder
    client.post('/find/run', data={'query': 'frustrated customers', 'mode': 'immediate'})
    store.complete()
    monkeypatch.setenv('ORQ_WORKSPACE', 'workspace')
    drawer = client.get('/find/trace/trace-1')
    assert drawer.status_code == 200
    assert 'rt-drawer' in drawer.text
    assert 'Full thread' in drawer.text
    assert 'JEV input' in drawer.text
    assert 'Raw result' in drawer.text
    assert 'This is the third time' in drawer.text
    assert 'eqFinderTab(this' in drawer.text
    assert 'navigator.clipboard.writeText' in drawer.text


def test_find_export_is_404_until_completed_then_downloads_json(setup_finder) -> None:
    store, client = setup_finder
    assert client.get('/find/export.json').status_code == 404
    client.post('/find/run', data={'query': 'frustrated customers', 'mode': 'immediate'})
    store.complete()
    response = client.get('/find/export.json')
    assert response.status_code == 200
    assert response.headers['content-disposition'] == 'attachment; filename="trace-finder-1.json"'
    assert json.loads(response.text)['counts']['matched'] == 1


def test_find_facets_menu_lists_catalogue_values(setup_finder, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, client = setup_finder
    async def load_catalogue(app: Any) -> FacetCatalogue:
        return FacetCatalogue(project=('support-agent',), model=('gpt-5.6-luna',))

    monkeypatch.setattr(finder_routes, '_load_catalogue', load_catalogue)
    response = client.get('/find/facets')
    assert response.status_code == 200
    assert 'support-agent' in response.text
    assert 'gpt-5.6-luna' in response.text
