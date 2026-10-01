from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, Literal

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.trace_finder import routes as finder_routes, views as finder_views
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.security import CSRF_FIELD, _CSRF_TOKEN
from evaluatorq.dashboard.trace_links import trace_span_url
from evaluatorq.trace_finder.explorer import ExplorerStore
from evaluatorq.trace_finder.rows import TraceRow
from evaluatorq.trace_finder import (
    CompiledQuery,
    DashboardSettings,
    DimensionAnswer,
    FacetCatalogue,
    FacetSelection,
    TraceProjection,
    NumericFilters,
    RunSnapshot,
    ThresholdSelection,
    TraceClassification,
    TraceDetail,
    TraceRecord,
    ValueSelection,
)
from evaluatorq.common.judge import ClassifyAnswer, ClassifyQuestion, ClassifyResponse


def test_failed_facet_warmup_releases_background_task(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail_to_load(_app: Any, _window_days: int) -> FacetCatalogue | None:
        raise RuntimeError('catalogue failed')

    monkeypatch.setattr(finder_routes, '_load_catalogue', fail_to_load)

    async def check() -> None:
        app = SimpleNamespace(state=SimpleNamespace(finder_store_lock=asyncio.Lock(), finder_generation=1))
        await finder_routes._warm_catalogue(app, 7)
        task = app.state.finder_catalogue_warmup[2]
        await asyncio.wait({task})
        await asyncio.sleep(0)
        assert app.state.finder_catalogue_warmup is None

    asyncio.run(check())


def csrf_data(values: dict[str, str] | None = None) -> dict[str, str]:
    """Build a finder POST body with the token used by the dashboard security module."""

    return {CSRF_FIELD: _CSRF_TOKEN, **(values or {})}


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


def test_review_form_preserves_zero_thresholds() -> None:
    noul = CompiledQuery(
        task=ClassifyQuestion(kind='noul', instructions='Did it happen?', noul_threshold=0.5, state={}),
        selection=ValueSelection(kind='values', values=(True,)),
    )
    score = CompiledQuery(
        task=ClassifyQuestion(kind='score', instructions='Score it.', criteria=['Low', 'High'], state={}),
        selection=ThresholdSelection(kind='threshold', operator='gte', value=0.5),
    )

    assert finder_routes._compiled_from_form(noul, {'noul_threshold': '0'}, prefix='').task.noul_threshold == 0
    assert finder_routes._compiled_from_form(score, {'selection_threshold': '0'}, prefix='').selection.value == 0


class FakeStore:
    def __init__(self) -> None:
        self.created_at = datetime(2026, 9, 1, tzinfo=timezone.utc)
        self.finished_at = self.created_at + timedelta(minutes=1)
        self.trace = _trace()
        self.projection = TraceProjection(
            payload={'trace_status': 'ok', 'messages': list(self.trace.messages)},
            serialized='{"messages": []}',
            estimated_tokens=10,
            omitted_messages=0,
            omitted_bytes=0,
        )
        self.compiled = _compiled()
        self.snapshot_value = RunSnapshot()
        self.explorer: Any | None = None
        self.started = False
        self.compile_request: Any | None = None
        self.compile_wait: bool | None = None
        self.started_request: Any | None = None
        self.started_dimensions: tuple[CompiledQuery, ...] | None = None
        self.start_wait: bool | None = None

    async def compile(
        self, request: Any, *, wait: bool = True, table: Any = None, source_generation: Any = None
    ) -> RunSnapshot:
        self.compile_request = request
        self.compile_wait = wait
        state = 'awaiting_review' if request.mode == 'review' else 'classifying'
        self.snapshot_value = RunSnapshot(
            generation=1,
            state=state,
            request=request,
            dimensions=(self.compiled,),
            generated_filters=FacetSelection(project=frozenset({'support-agent'})),
            generated_numeric=NumericFilters(),
            explicit_filters=request.population.facets,
            explicit_numeric=request.population.numeric,
            trace_ids=(self.trace.trace_id,),
            traces=(self.trace,),
            projections={self.trace.trace_id: self.projection} if state == 'classifying' else {},
            total=1,
            active=1 if state == 'classifying' else 0,
            queued=0,
            created_at=self.created_at,
        )
        return self.snapshot_value

    async def start(self, request: Any, dimensions: tuple[CompiledQuery, ...], *, wait: bool = True) -> RunSnapshot:
        self.started = True
        self.start_wait = wait
        self.started_request = request
        self.started_dimensions = tuple(dimensions)
        result = TraceClassification(
            trace_id=self.trace.trace_id,
            span_id=self.trace.span_id,
            answers=(
                DimensionAnswer(
                    value='frustrated',
                    confidence=0.91,
                    probabilities={'frustrated': 0.91, 'neutral': 0.09},
                    matched=True,
                    summary='The customer repeats the request.',
                ),
            ),
            matched=True,
            raw_result={'value': 'frustrated'},
        )
        self.snapshot_value = replace(
            self.snapshot_value,
            state='completed',
            dimensions=self.started_dimensions,
            projections={self.trace.trace_id: self.projection},
            results=MappingProxyType({self.trace.trace_id: result}),
            completed=1,
            matched=1,
            active=0,
            queued=0,
            finished_at=self.finished_at,
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
            dimensions=self.snapshot_value.dimensions,
        )

    def complete(self) -> None:
        result = TraceClassification(
            trace_id=self.trace.trace_id,
            span_id=self.trace.span_id,
            answers=(
                DimensionAnswer(
                    value='frustrated',
                    confidence=0.91,
                    probabilities={'frustrated': 0.91, 'neutral': 0.09},
                    matched=True,
                    summary='The customer repeats the request.',
                ),
            ),
            matched=True,
            raw_result={'value': 'frustrated'},
        )
        self.snapshot_value = replace(
            self.snapshot_value,
            state='completed',
            dimensions=self.snapshot_value.dimensions or (self.compiled,),
            results={self.trace.trace_id: result},
            completed=1,
            matched=1,
            active=0,
            queued=0,
            finished_at=self.finished_at,
        )


@pytest.fixture
def setup_finder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))
    store = FakeStore()
    async def build_store(_app: Any) -> FakeStore:
        return store

    monkeypatch.setattr(finder_routes, '_build_store', build_store)
    return store, TestClient(build_app(roots=[tmp_path]), raise_server_exceptions=True)


def test_find_compiling_state_shows_an_indicator(setup_finder) -> None:
    store, client = setup_finder
    store.snapshot_value = replace(store.snapshot_value, state='compiling', phase='planning')
    response = client.get('/find')
    assert response.status_code == 200
    assert 'finder-compiling' in response.text
    assert 'Planning the search' in response.text
    assert 'compiling the question and selecting metadata filters' in response.text
    assert 'No traces loaded.' not in response.text
    assert '<b>0 / 0</b> judged' not in response.text


def test_async_controls_and_trace_drawer_have_request_feedback(setup_finder) -> None:
    store, client = setup_finder
    page = client.get('/find').text
    assert 'class="finder-go-working" role="status">Starting search…' in page
    assert 'id="finder-mode-working" role="status">Resetting review…' in page
    assert 'id="finder-drawer-loading" role="status">Loading trace…' in page

    client.post('/find/run', data=csrf_data({'surface': 'search', 'query': 'frustrated customers', 'mode': 'immediate'}))
    running = client.get('/find/poll?surface=search').text
    assert 'role="status">Cancelling…' in running
    assert 'hx-indicator="#finder-drawer-loading"' in running

    store.complete()
    completed = client.get('/find/poll?surface=search').text
    assert 'role="status">Clearing…' in completed


def test_classifier_picked_filters_do_not_carry_into_the_next_query(setup_finder) -> None:
    """Chips show the run's whole population, but only the user's own filters are re-submitted."""
    store, client = setup_finder
    client.post(
        '/find/run',
        data=csrf_data({'query': 'frustrated customers', 'mode': 'immediate', 'facet_model': 'gpt-5', 'tokens_min': '900'}),
    )
    request = store.snapshot_value.request
    assert request is not None
    population = request.population.model_copy(
        update={
            'facets': FacetSelection(project=frozenset({'support-agent'}), model=frozenset({'gpt-5'})),
            'numeric': NumericFilters(tokens_min=900, duration_ms_min=50),
        }
    )
    store.snapshot_value = replace(
        store.snapshot_value,
        request=request.model_copy(update={'population': population}),
        generated_numeric=NumericFilters(duration_ms_min=50),
    )
    store.complete()

    html = client.get('/find').text
    assert 'data-chip-name="facet_project" data-finder-value="support-agent"' in html
    assert 'data-chip-name="duration_ms_min"' in html
    assert 'name="facet_model" value="gpt-5"' in html
    assert '<input type="hidden" form="finder-query-form" name="facet_project"' not in html
    assert 'name="tokens_min" type="number" min="0" placeholder="min" value="900"' in html
    assert 'name="duration_ms_min" type="number" min="0" placeholder="min" value=""' in html

    client.post('/find/run', data=csrf_data({'query': 'frustrated customers', 'mode': 'review'}))
    review_request = store.snapshot_value.request
    assert review_request is not None
    store.snapshot_value = replace(
        store.snapshot_value,
        request=review_request.model_copy(update={'population': population}),
    )
    html = client.get('/find').text
    assert '<input type="hidden" form="finder-start-form" name="facet_project" value="support-agent">' in html


def test_explicit_filters_survive_matching_classifier_picks(setup_finder) -> None:
    store, client = setup_finder
    client.post('/find/run', data=csrf_data({
        'query': 'support agent traces', 'mode': 'immediate',
        'facet_project': 'support-agent', 'tokens_min': '900',
    }))
    request = store.snapshot_value.request
    assert request is not None
    store.snapshot_value = replace(
        store.snapshot_value,
        generated_numeric=NumericFilters(tokens_min=900),
    )
    store.complete()

    html = client.get('/find').text
    assert '<input type="hidden" form="finder-query-form" name="facet_project" value="support-agent">' in html
    assert 'name="tokens_min" type="number" min="0" placeholder="min" value="900"' in html


def test_facet_menu_loads_itself_after_the_page_renders(setup_finder, monkeypatch: pytest.MonkeyPatch) -> None:
    """The page never waits on the Orq facet call: the menu fetches its values with an htmx load trigger."""
    _store, client = setup_finder
    loads: list[int | None] = []

    async def load_catalogue(app: Any, window_days: int | None = None) -> FacetCatalogue:
        loads.append(window_days)
        catalogue = FacetCatalogue(project=('support-agent',))
        app.state.finder_catalogue_cache = (datetime.now(timezone.utc) + timedelta(minutes=5), 7, catalogue)
        return catalogue

    monkeypatch.setattr(finder_routes, '_load_catalogue', load_catalogue)
    page = client.get('/find').text
    assert loads == []
    assert 'hx-get="/find/facets?form_id=finder-query-form" hx-trigger="load, refreshFacets" hx-include="#finder-controls"' in page
    assert 'Loading facet values…' in page
    assert 'class="finder-facet-loading" role="status">Loading filters…' in page
    assert '<button class="add" type="button" aria-haspopup="true">+ Filter</button>' in page
    assert 'name="window_days"' in page and 'value="7"' in page

    menu = client.get('/find/facets?form_id=finder-query-form&window_days=7').text
    assert loads == [7]
    assert 'hx-trigger="load"' not in menu
    assert 'support-agent' in menu

    page = client.get('/find').text
    assert 'hx-trigger="load"' not in page
    assert 'support-agent' in page
    assert loads == [7]


def test_legacy_search_defaults_to_immediate_mode(setup_finder) -> None:
    _store, client = setup_finder
    html = client.get('/find').text
    assert 'value="immediate" form="finder-query-form" checked' in html


def test_find_idle_page_is_standalone_legacy_search_and_reuses_facets(setup_finder) -> None:
    _store, client = setup_finder
    response = client.get('/find')
    html = response.text
    assert response.status_code == 200
    assert 'Trace search' in html
    assert 'Find the signal.' in html
    assert 'id="finder-query-form"' in html
    assert 'id="finder-controls"' in html
    assert 'class="finder-facets' in html
    assert 'id="explorer-results-slot"' not in html
    assert 'Load traces to start' not in html
    assert "hx-vals='{\"surface\":\"search\"}'" in html
    assert 'href="/traces"' in html


def test_traces_page_is_separate_explorer_and_reuses_facet_controls(setup_finder) -> None:
    _store, client = setup_finder
    response = client.get('/traces')
    html = response.text
    assert response.status_code == 200
    assert 'id="explorer-results-slot"' in html
    assert 'Load traces' in html
    assert 'id="finder-controls"' in html
    assert 'class="finder-facets' in html
    assert 'href="/find"' in html
    assert 'Find the signal.' not in html


def test_search_reset_does_not_clear_traces_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))
    stores: list[FakeStore] = []

    async def build_store(_app: Any) -> FakeStore:
        store = FakeStore()
        stores.append(store)
        return store

    monkeypatch.setattr(finder_routes, '_build_store', build_store)

    async def load_facet_catalogue(
        _orq: Any,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> FacetCatalogue:
        return FacetCatalogue()

    monkeypatch.setattr(finder_routes, 'load_facet_catalogue', load_facet_catalogue)
    app = build_app(roots=[tmp_path])
    client = TestClient(app, raise_server_exceptions=True)
    assert client.get('/traces').status_code == 200
    assert client.get('/find').status_code == 200
    assert len(stores) == 2
    assert not hasattr(app.state, 'finder_store')
    assert client.cookies.get('evaluatorq_dashboard_session') is not None
    assert app.state.finder_search_store is stores[1]

    client.post('/find/run', data=csrf_data({'surface': 'search', 'query': 'frustrated customers'}))
    assert stores[1].snapshot_value.state == 'classifying'
    assert stores[0].snapshot_value.state == 'idle'
    stores[1].complete()
    assert 'href="/find/export.json?surface=search"' in client.get('/find').text
    assert client.get('/find/export.json?surface=search').status_code == 200
    assert client.get('/find/export.json').status_code == 404
    client.post('/find/reset', data=csrf_data({'surface': 'search'}))
    assert stores[1].snapshot_value.state == 'idle'
    assert stores[0].snapshot_value.state == 'idle'


def test_traces_state_is_isolated_between_browser_sessions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))
    stores: list[FakeStore] = []

    async def warm_catalogue(_app: Any, _window_days: int) -> None:
        return None

    monkeypatch.setattr(finder_routes, '_warm_catalogue', warm_catalogue)

    async def build_store(_app: Any) -> FakeStore:
        store = FakeStore()
        trace_id = f'trace-{len(stores) + 1}'
        store.trace = store.trace.model_copy(update={'trace_id': trace_id})
        row = TraceRow(trace_id=trace_id, status='ok', tokens_in=100 + len(stores), agent_name=trace_id)

        async def search_rows(
            start: Any,
            end: Any,
            limit: int,
            *,
            facets: Any,
            numeric: Any,
            on_page: Any = None,
        ) -> tuple[TraceRow, ...]:
            if on_page is not None:
                on_page((row,))
            return (row,)

        async def hydrate_rows(_rows: Any) -> dict[str, Any]:
            return {}

        store.explorer = ExplorerStore(search=search_rows, hydrate=hydrate_rows)
        store.snapshot_for_render = store.snapshot
        now = datetime.now(timezone.utc)
        await store.explorer.load(
            now - timedelta(days=1),
            now,
            1,
            facets=FacetSelection(),
            numeric=NumericFilters(),
            wait=True,
            warm_trajectories=False,
        )
        stores.append(store)
        return store

    monkeypatch.setattr(finder_routes, '_build_store', build_store)

    async def load_facet_catalogue(
        _orq: Any,
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> FacetCatalogue:
        return FacetCatalogue()

    monkeypatch.setattr(finder_routes, 'load_facet_catalogue', load_facet_catalogue)
    app = build_app(roots=[tmp_path])
    first = TestClient(app, raise_server_exceptions=True)
    second = TestClient(app, raise_server_exceptions=True)

    first_page = first.get('/traces')
    second_page = second.get('/traces')
    assert first_page.status_code == second_page.status_code == 200
    assert first_page.headers['set-cookie'].startswith('evaluatorq_dashboard_session=')
    assert second_page.headers['set-cookie'].startswith('evaluatorq_dashboard_session=')
    assert 'httponly' in first_page.headers['set-cookie'].lower()
    assert 'samesite=lax' in first_page.headers['set-cookie'].lower()
    assert 'max-age=' not in first_page.headers['set-cookie'].lower()
    assert first.cookies.get('evaluatorq_dashboard_session') != second.cookies.get('evaluatorq_dashboard_session')
    assert len(stores) == 2

    first.post('/find/run', data=csrf_data({'query': 'first browser query'}))
    second.post('/find/run', data=csrf_data({'query': 'second browser query'}))
    first_request = stores[0].compile_request
    second_request = stores[1].compile_request
    assert first_request is not None
    assert second_request is not None
    assert first_request.query == 'first browser query'
    assert second_request.query == 'second browser query'
    first_rows = first.get('/find/rows?sort=tokens_in&dir=asc')
    second_rows = second.get('/find/rows?sort=tokens_in&dir=desc')
    assert 'data-tv-row="trace-1"' in first_rows.text
    assert 'data-tv-row="trace-2"' not in first_rows.text
    assert 'data-tv-row="trace-2"' in second_rows.text
    assert 'data-tv-row="trace-1"' not in second_rows.text
    assert first.get('/find/poll').status_code == 200
    assert second.get('/find/poll').status_code == 200
    assert first.get('/find/trace/trace-1').status_code == 200
    assert second.get('/find/trace/trace-2').status_code == 200
    assert first.get('/find/trace/trace-2').status_code == 200
    assert 'not part of the current run' in first.get('/find/trace/trace-2').text
    assert first.get('/find/export.json').status_code == 404
    stores[0].complete()
    stores[1].complete()
    assert json.loads(first.get('/find/export.json').text)['counts']['matched'] == 1
    assert json.loads(second.get('/find/export.json').text)['counts']['matched'] == 1

    first_cookie = first.cookies.get('evaluatorq_dashboard_session')
    assert first_cookie is not None
    first.cookies.clear()
    first.cookies.set('evaluatorq_dashboard_session', f'{first_cookie}tampered')
    rotated = first.get('/find/rows')
    assert rotated.status_code == 200
    assert f'{first_cookie}tampered' not in rotated.headers['set-cookie']
    assert len(stores) == 3
    assert stores[2].snapshot_value.state == 'idle'
    assert 'data-tv-row="trace-3"' in rotated.text
    assert 'data-tv-row="trace-1"' not in rotated.text


def test_legacy_find_surface_is_shared_between_browser_sessions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))
    stores: list[FakeStore] = []

    async def build_store(_app: Any) -> FakeStore:
        store = FakeStore()
        stores.append(store)
        return store

    monkeypatch.setattr(finder_routes, '_build_store', build_store)
    app = build_app(roots=[tmp_path])
    first = TestClient(app, raise_server_exceptions=True)
    second = TestClient(app, raise_server_exceptions=True)
    first.get('/find')
    second.get('/find')
    assert len(stores) == 1
    first.post('/find/run', data=csrf_data({'surface': 'search', 'query': 'shared search'}))
    request = stores[0].compile_request
    assert request is not None
    assert request.query == 'shared search'
    second.post('/find/reset', data=csrf_data({'surface': 'search'}))
    assert stores[0].snapshot_value.state == 'idle'


def test_traces_cookie_is_secure_for_https_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('ORQ_API_KEY', raising=False)
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))
    client = TestClient(build_app(roots=[tmp_path]), base_url='https://testserver')

    response = client.get('/traces')

    assert response.status_code == 200
    assert 'secure' in response.headers['set-cookie'].lower()


def test_find_without_api_key_renders_empty_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv('ORQ_API_KEY', raising=False)
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))
    client = TestClient(build_app(roots=[tmp_path]), raise_server_exceptions=True)
    response = client.get('/find')
    assert response.status_code == 200
    assert 'Set ORQ_API_KEY to load traces' in response.text
    assert '<textarea name="query"' in response.text and 'disabled' in response.text.split('<textarea name="query"', 1)[1].split('>', 1)[0]
    assert 'id="explorer-results-slot"' not in response.text
    assert '<span class="finder-key-hint"' not in response.text


def test_find_run_starts_polling_and_completed_poll_shows_matches(setup_finder) -> None:
    store, client = setup_finder
    response = client.post('/find/run', data=csrf_data({'surface': 'search', 'query': 'frustrated customers', 'mode': 'immediate'}))
    assert response.status_code == 200
    assert 'hx-trigger="every 1s"' in response.text
    assert store.compile_wait is False

    store.complete()
    poll = client.get('/find/poll?surface=search')
    assert poll.status_code == 200
    assert 'frustrated' in poll.text
    assert 'included' in poll.text
    assert 'trace-1' in poll.text


def test_full_page_polling_timer_disappears_on_terminal_fragment(setup_finder) -> None:
    store, client = setup_finder
    client.post('/find/run', data=csrf_data({'surface': 'search', 'query': 'frustrated customers', 'mode': 'immediate'}))
    page = client.get('/find').text
    assert page.count('hx-get="/find/poll?surface=search"') == 1
    assert '<div id="finder-body"><div class="finder-body-fragment" hx-get="/find/poll?surface=search"' in page

    store.complete()
    poll = client.get('/find/poll?surface=search').text
    assert 'hx-get="/find/poll?surface=search"' not in poll


def test_score_review_keeps_generated_threshold_and_colors_scores() -> None:
    compiled = CompiledQuery(
        task=ClassifyQuestion(kind='score', instructions='Score the trace.', criteria=['Low', 'High'], state={}),
        selection=ThresholdSelection(kind='threshold', operator='gte', value=0.65),
    )
    review = finder_views.task_panel((compiled,), editable=True)
    assert '<option value="gte:0.65" selected>gte 0.65</option>' in review
    assert 'Question asked of each trace' in review
    assert 'Raw plan' not in review

    result = TraceClassification(
        trace_id='trace-1', span_id='span-1', answers=(DimensionAnswer(value=0.8, matched=True),), matched=True,
        raw_result={},
    )
    snapshot = RunSnapshot(
        state='completed', dimensions=(compiled,), trace_ids=('trace-1',), traces=(_trace(),),
        results={'trace-1': result}, total=1, completed=1, matched=1,
    )
    assert 'color-mix(in srgb, var(--chart-5) 80%, var(--chart-2))' in finder_views.matrix(snapshot)
    legend = finder_views.legend(snapshot)
    assert '0.0 → 1.0 <b>1</b>' in legend
    assert 'gte 0.65 <b>1</b>' in legend


def test_completed_table_renders_one_verdict_column_per_dimension() -> None:
    """A two-dimension run names each result column after its dimension, not the old fixed Verdict/Conf. pair."""
    sentiment = CompiledQuery(
        name='Sentiment',
        task=ClassifyQuestion(
            kind='choice', instructions="Judge the customer's tone.",
            criteria={'frustrated': 'Annoyed.', 'neutral': 'Calm.'}, state={},
        ),
        selection=ValueSelection(kind='values', values=('frustrated',)),
    )
    escalated = CompiledQuery(
        name='Escalated',
        task=ClassifyQuestion(kind='noul', instructions='Was this escalated?', noul_threshold=0.5, state={}),
        selection=ValueSelection(kind='values', values=(True,)),
    )
    trace = _trace()
    result = TraceClassification(
        trace_id=trace.trace_id,
        span_id=trace.span_id,
        answers=(
            DimensionAnswer(value='frustrated', matched=True),
            DimensionAnswer(value=True, matched=True),
        ),
        matched=True,
        raw_result={},
    )
    snapshot = RunSnapshot(
        state='completed', dimensions=(sentiment, escalated), trace_ids=(trace.trace_id,), traces=(trace,),
        results={trace.trace_id: result}, total=1, completed=1, matched=1,
    )

    html = finder_views.table(snapshot)

    assert '<th>Sentiment</th>' in html
    assert '<th>Escalated</th>' in html
    assert '<th>Verdict</th>' not in html
    assert '<th>Conf.</th>' not in html


def test_review_form_with_two_dimensions_posts_dimension_prefixed_fields(setup_finder) -> None:
    """Each dimension's edited fields carry its own ``d{index}_`` prefix, and store.start gets both dimensions."""
    store, client = setup_finder
    review = client.post('/find/run', data=csrf_data({'query': 'frustrated customers', 'mode': 'review'}))
    assert review.status_code == 200

    sentiment = CompiledQuery(
        name='Sentiment',
        task=ClassifyQuestion(kind='noul', instructions='Is the user frustrated?', noul_threshold=0.5, state={}),
        selection=ValueSelection(kind='values', values=(True,)),
    )
    escalated = CompiledQuery(
        name='Escalated',
        task=ClassifyQuestion(kind='noul', instructions='Was this escalated?', noul_threshold=0.5, state={}),
        selection=ValueSelection(kind='values', values=(True,)),
    )
    store.snapshot_value = replace(store.snapshot_value, dimensions=(sentiment, escalated))
    review = client.get('/find').text
    assert 'name="d0_instructions"' in review
    assert 'name="d1_instructions"' in review

    started = client.post(
        '/find/start',
        data=csrf_data({
            'd0_instructions': 'Is the user upset?',
            'd0_selection_value': 'true',
            'd1_instructions': 'Did this reach a human?',
            'd1_selection_value': 'true',
        }),
    )

    assert started.status_code == 200
    assert store.started_dimensions is not None
    assert len(store.started_dimensions) == 2
    assert store.started_dimensions[0].task.instructions == 'Is the user upset?'
    assert store.started_dimensions[1].task.instructions == 'Did this reach a human?'


def test_filter_output_panel_shows_structured_llm_reply_and_escaped_values() -> None:
    snapshot = RunSnapshot(
        generated_filters=FacetSelection(project=frozenset({'Demos'})),
        filter_response=ClassifyResponse(
            model='jev-latest', answers={'project': ClassifyAnswer(type='choice', choice='<script>')}
        ),
    )

    html = finder_views.filter_output_panel(snapshot)

    assert 'Filter selection' in html
    assert '1 chosen' in html
    assert 'Demos' in html
    assert 'View structured LLM output' in html
    assert '&lt;script&gt;' in html
    assert '<script>' not in html


def test_filter_output_panel_reports_selection_failure() -> None:
    html = finder_views.filter_output_panel(RunSnapshot(filter_selection_error='facet catalogue unavailable'))

    assert 'role="alert"' in html
    assert 'facet catalogue unavailable' in html
    assert 'View structured LLM output' not in html


def test_noul_task_panel_omits_empty_label_and_criteria_sections() -> None:
    html = finder_views.task_panel((_compiled().model_copy(update={
        'task': ClassifyQuestion(kind='noul', instructions='Does this match?', state={}),
        'selection': ValueSelection(kind='values', values=(True,)),
    }),), editable=False)

    assert 'Yes / no' in html
    assert 'Confidence needed for yes' in html
    assert '0 labels' not in html
    assert 'Verdict labels' not in html
    assert 'Raw plan' not in html


def test_trace_span_link_rejects_filter_delimiters(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_WORKSPACE', 'workspace')
    assert trace_span_url('trace-1', 'span-1') is not None
    assert trace_span_url('trace-1)//trace:other', 'span-1') is None
    assert trace_span_url('trace-1', 'span-1//trace:other') is None


@pytest.mark.parametrize('window_days', ['0', '91', str(10**12)])
def test_find_run_rejects_invalid_windows(setup_finder, window_days: str) -> None:
    store, client = setup_finder
    response = client.post(
        '/find/run',
        data=csrf_data({'query': 'frustrated customers', 'mode': 'immediate', 'window_days': window_days}),
    )

    assert response.status_code == 422
    assert store.compile_request is None


def test_find_run_rejects_blank_query_before_starting_work(setup_finder) -> None:
    store, client = setup_finder
    response = client.post('/find/run', data=csrf_data({'query': '   ', 'mode': 'immediate'}))
    assert response.status_code == 422
    assert store.compile_request is None


def test_find_run_passes_numeric_filter_and_renders_chip(setup_finder) -> None:
    store, client = setup_finder
    response = client.post(
        '/find/run', data=csrf_data({'query': 'long traces', 'mode': 'immediate', 'tokens_min': '5000'})
    )
    assert response.status_code == 200
    assert store.compile_request is not None
    assert store.compile_request.population.numeric.tokens_min == 5000
    assert '<span class="chip"><b>total tokens</b><span class="v">≥ 5,000</span>' in response.text
    assert 'name="tokens_min"' in response.text

    store.complete()
    # The last poll only swaps the run status; the filter row is never redrawn by a poll.
    poll = client.get('/find/poll')
    assert poll.status_code == 286
    assert 'id="finder-controls"' not in poll.text
    page = client.get('/traces').text
    assert 'class="chip is-editable" data-chip-name="tokens_min"' in page
    assert 'data-chip-open="tokens"' in page
    assert 'data-finder-remove="tokens_min"' in page


def test_poll_that_reaches_review_redraws_the_whole_body(setup_finder) -> None:
    store, client = setup_finder
    client.post('/find/run', data=csrf_data({'query': 'frustrated customers', 'mode': 'review'}))
    poll = client.get('/find/poll')
    # Review needs its own controls, so this poll swaps #finder-body instead of the run status.
    assert poll.headers['HX-Retarget'] == '#finder-body'
    assert poll.headers['HX-Reswap'] == 'innerHTML'
    assert 'id="finder-controls"' in poll.text


def test_find_review_start_transitions_to_classification(setup_finder) -> None:
    store, client = setup_finder
    review = client.post('/find/run', data=csrf_data({'query': 'frustrated customers', 'mode': 'review'}))
    assert review.status_code == 200
    assert 'Review the plan before running per-trace classification.' in review.text
    assert 'name="d0_instructions"' in review.text

    started = client.post(
        '/find/start',
        data=csrf_data({
            'd0_instructions': 'Edited instructions',
            'd0_criteria_label_0': 'frustrated',
            'd0_criteria_description_0': 'Changed criterion',
            'd0_criteria_label_1': 'neutral',
            'd0_criteria_description_1': 'Transactional.',
            'd0_selection_value': 'frustrated',
            'window_days': '14',
            'limit': '12',
            'parallelism': '3',
            'facet_project': 'support-agent',
            'tokens_min': '900',
        }),
    )
    assert started.status_code == 200
    assert store.started
    assert store.start_wait is False
    assert store.started_request is not None
    assert store.started_request.population.start is not None
    assert store.started_request.population.end is not None
    assert (store.started_request.population.end - store.started_request.population.start).days == 7
    assert store.started_request.population.limit == 12
    assert store.started_request.parallelism == 3
    assert store.started_request.population.facets.project == frozenset({'support-agent'})
    assert store.started_request.population.numeric.tokens_min == 900
    assert store.compile_request is not None
    assert store.started_request.population.end == store.compile_request.population.end
    assert store.started_dimensions is not None
    assert store.started_dimensions[0].task.instructions == 'Edited instructions'
    assert store.started_dimensions[0].task.criteria['frustrated'] == 'Changed criterion'
    assert 'name="kind"' not in review.text


def test_find_start_rejects_a_stale_review_form(setup_finder) -> None:
    store, client = setup_finder
    assert client.post('/find/run', data=csrf_data({'query': 'question', 'mode': 'immediate'})).status_code == 200
    response = client.post('/find/start', data=csrf_data({'instructions': 'stale'}))
    assert response.status_code == 409
    assert store.started is False


def test_finder_state_changing_posts_require_csrf_and_same_origin(setup_finder) -> None:
    _store, client = setup_finder
    missing = client.post('/find/run', data={'query': 'anything'})
    wrong = client.post('/find/run', data={CSRF_FIELD: 'wrong', 'query': 'anything'})
    cross_site = client.post(
        '/find/run',
        data=csrf_data({'query': 'anything'}),
        headers={'sec-fetch-site': 'cross-site'},
    )

    assert missing.status_code == 403
    assert wrong.status_code == 403
    assert cross_site.status_code == 403
    assert 'request rejected' in cross_site.text.lower()


def test_find_failed_snapshot_renders_escaped_error(setup_finder) -> None:
    store, client = setup_finder
    store.snapshot_value = RunSnapshot(state='failed', error='<compiler failed>')
    response = client.get('/find/poll')
    # 286 tells htmx to stop polling a run that has settled.
    assert response.status_code == 286
    assert '&lt;compiler failed&gt;' in response.text
    assert '<compiler failed>' not in response.text


def test_find_trace_drawer_renders_thread_and_classifier_input(setup_finder, monkeypatch: pytest.MonkeyPatch) -> None:
    store, client = setup_finder
    client.post('/find/run', data=csrf_data({'query': 'frustrated customers', 'mode': 'immediate'}))
    store.complete()
    monkeypatch.setenv('ORQ_WORKSPACE', 'workspace')
    drawer = client.get('/find/trace/trace-1')
    assert drawer.status_code == 200
    assert 'rt-drawer' in drawer.text
    assert 'Full thread' in drawer.text
    assert 'Classifier input' in drawer.text
    assert 'Raw result' in drawer.text
    assert '/workspace/traces/(trace:trace-1//span:span-1)' in drawer.text
    assert 'This is the third time' in drawer.text
    assert drawer.text.count('<details class="fd-msg') == 2
    assert '<summary><span class="role"><b>User</b></span><em class="fd-msg-meta">#1 · ~' in drawer.text
    assert '<span class="fd-msg-preview">This is the third time I am asking.</span>' in drawer.text
    assert '<div class="fd-msg-content">This is the third time I am asking.</div>' in drawer.text
    assert 'eqFinderTab(this' in drawer.text
    assert 'navigator.clipboard.writeText' in drawer.text


def test_find_export_is_404_until_completed_then_downloads_json(
    setup_finder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    caller_thread = threading.current_thread()
    save_threads: list[threading.Thread] = []
    save_export = finder_routes._save_finder_export

    def track_save_thread(*args: Any) -> None:
        save_threads.append(threading.current_thread())
        save_export(*args)

    monkeypatch.setattr(finder_routes, '_save_finder_export', track_save_thread)
    store, client = setup_finder
    assert client.get('/find/export.json').status_code == 404
    client.post('/find/run', data=csrf_data({'query': 'frustrated customers', 'mode': 'immediate'}))
    store.complete()
    response = client.get('/find/export.json')
    assert response.status_code == 200
    filename = response.headers['content-disposition'].split('filename="', 1)[1].rstrip('"')
    assert filename.startswith('trace-finder-1-') and filename.endswith('.json')
    assert filename in client.get('/traces').text
    assert json.loads(response.text)['counts']['matched'] == 1
    assert (tmp_path / 'finder-exports' / filename).read_text() == response.text
    assert save_threads and save_threads[0] is not caller_thread


def test_find_export_link_rejects_a_newer_run_instead_of_downloading_it(
    setup_finder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.trace_finder.export import export_filename

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    store, client = setup_finder
    client.post('/find/run', data=csrf_data({'query': 'frustrated customers', 'mode': 'immediate'}))
    store.complete()
    old_name = export_filename(store.snapshot_value)
    assert f'href="/find/export.json?export={old_name}"' in client.get('/traces').text

    store.snapshot_value = replace(store.snapshot_value, generation=store.snapshot_value.generation + 1)
    response = client.get(f'/find/export.json?export={old_name}')

    assert response.status_code == 409
    assert 'Refresh the page' in response.text
    assert not (tmp_path / 'finder-exports' / old_name).exists()


def test_find_export_save_failure_is_visible(
    setup_finder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    store, client = setup_finder
    client.post('/find/run', data=csrf_data({'query': 'frustrated customers', 'mode': 'immediate'}))
    store.complete()

    def fail_save(*args, **kwargs):
        raise OSError('read-only store')

    monkeypatch.setattr(finder_routes.tempfile, 'NamedTemporaryFile', fail_save)
    response = client.get('/find/export.json')
    assert response.status_code == 500
    assert 'Could not save Finder export for Insights.' in response.text


def test_find_export_survives_non_object_manifest_during_retention(
    setup_finder, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.dashboard.insights_launch import (
        ensure_private_finder_reference_dir,
        finder_export_reference_path,
    )

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    store, client = setup_finder
    client.post('/find/run', data=csrf_data({'query': 'frustrated customers', 'mode': 'immediate'}))
    store.complete()

    runs_dir = tmp_path / 'insights-runs'
    manifest_dir = runs_dir / '.manifests'
    manifest_dir.mkdir(parents=True)
    (manifest_dir / 'malformed-run.json').write_text('[]', encoding='utf-8')
    marker = finder_export_reference_path(runs_dir, 'malformed-run')
    ensure_private_finder_reference_dir(marker.parent)
    marker.write_text(json.dumps({'finder_export': 'trace-finder-old.json'}), encoding='utf-8')
    marker.chmod(0o600)

    response = client.get('/find/export.json')

    assert response.status_code == 200
    assert json.loads(response.text)['counts']['matched'] == 1
    assert marker.exists()


def test_finder_export_lock_uses_windows_interprocess_lock_when_fcntl_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    operations: list[int] = []

    def locking(descriptor: int, mode: int, length: int) -> None:
        assert length == 1
        assert os.lseek(descriptor, 0, os.SEEK_CUR) == 0
        operations.append(mode)

    monkeypatch.setitem(sys.modules, 'fcntl', None)
    monkeypatch.setitem(
        sys.modules,
        'msvcrt',
        SimpleNamespace(LK_LOCK=1, LK_UNLCK=2, locking=locking),
    )
    export_dir = tmp_path / 'finder-exports'

    with finder_routes._finder_export_lock(export_dir):
        lock_path = export_dir / '.finder-export.lock'
        assert lock_path.stat().st_size == 1
        assert operations == [1]

    assert operations == [1, 2]


def test_finder_export_lock_fails_closed_when_windows_lock_cannot_be_acquired(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_lock(descriptor: int, mode: int, length: int) -> None:
        raise OSError('lock failed')

    monkeypatch.setitem(sys.modules, 'fcntl', None)
    monkeypatch.setitem(
        sys.modules,
        'msvcrt',
        SimpleNamespace(LK_LOCK=1, LK_UNLCK=2, locking=fail_lock),
    )

    with pytest.raises(OSError, match='lock failed'):
        with finder_routes._finder_export_lock(tmp_path / 'finder-exports'):
            pytest.fail('export mutation must not run without the inter-process lock')


def test_finder_pruning_skips_when_a_lease_is_malformed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from evaluatorq.dashboard.insights_launch import (
        ensure_private_finder_reference_dir,
        finder_export_reference_path,
    )

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    export_dir = tmp_path / 'finder-exports'
    export_dir.mkdir()
    exports = [export_dir / f'trace-finder-{index}.json' for index in range(51)]
    for index, path in enumerate(exports):
        path.write_text('{}', encoding='utf-8')
        # Windows filesystem timestamps have coarser resolution than these
        # nanosecond increments; whole-second values keep the ordering stable.
        os.utime(path, (index + 1, index + 1))
    runs_dir = tmp_path / 'insights-runs'
    marker = finder_export_reference_path(runs_dir, 'malformed-lease')
    ensure_private_finder_reference_dir(marker.parent)
    marker.write_text('{invalid JSON', encoding='utf-8')
    marker.chmod(0o600)

    finder_routes._prune_finder_exports(export_dir)

    assert all(path.exists() for path in exports)
    assert marker.exists()


def test_finder_pruning_recovers_after_malformed_lease_expires(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from evaluatorq.dashboard.insights_launch import ensure_private_finder_reference_dir, finder_export_reference_path

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    export_dir = tmp_path / 'finder-exports'
    export_dir.mkdir()
    exports = [export_dir / f'trace-finder-{index}.json' for index in range(51)]
    for index, path in enumerate(exports):
        path.write_text('{}', encoding='utf-8')
        os.utime(path, (index + 1, index + 1))
    marker = finder_export_reference_path(tmp_path / 'insights-runs', 'malformed-lease')
    ensure_private_finder_reference_dir(marker.parent)
    marker.write_text('{invalid JSON', encoding='utf-8')
    marker.chmod(0o600)
    os.utime(marker, (1, 1))

    finder_routes._prune_finder_exports(export_dir)

    assert not marker.exists()
    assert not exports[0].exists()
    assert all(path.exists() for path in exports[1:])


@pytest.mark.skipif(os.name == 'nt', reason='chmod does not configure Windows directory ACLs')
def test_finder_pruning_skips_when_lease_directory_is_unsafe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from evaluatorq.dashboard.insights_launch import ensure_private_finder_reference_dir

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    export_dir = tmp_path / 'finder-exports'
    export_dir.mkdir()
    exports = [export_dir / f'trace-finder-{index}.json' for index in range(51)]
    for index, path in enumerate(exports):
        path.write_text('{}', encoding='utf-8')
        os.utime(path, (index + 1, index + 1))
    leases = tmp_path / 'insights-runs' / '.finder-export-leases'
    ensure_private_finder_reference_dir(leases)
    leases.chmod(0o755)

    finder_routes._prune_finder_exports(export_dir)

    assert all(path.exists() for path in exports)


def test_finder_pruning_is_normal_when_no_lease_directory_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    export_dir = tmp_path / 'finder-exports'
    export_dir.mkdir()
    exports = [export_dir / f'trace-finder-{index}.json' for index in range(51)]
    for index, path in enumerate(exports):
        path.write_text('{}', encoding='utf-8')
        os.utime(path, (index + 1, index + 1))

    finder_routes._prune_finder_exports(export_dir)

    assert exports[0].exists() is False
    assert all(path.exists() for path in exports[1:])


def test_finder_pruning_skips_reference_scan_within_retention_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    export_dir = tmp_path / 'finder-exports'
    export_dir.mkdir()
    for index in range(finder_routes._FINDER_EXPORT_RETENTION):
        (export_dir / f'trace-finder-{index}.json').write_text('{}', encoding='utf-8')

    monkeypatch.setattr(
        finder_routes,
        '_referenced_finder_exports',
        lambda _export_dir: pytest.fail('reference scan is unnecessary below the retention limit'),
    )

    finder_routes._prune_finder_exports(export_dir)

    assert len(list(export_dir.glob('trace-finder-*.json'))) == finder_routes._FINDER_EXPORT_RETENTION


def test_finder_export_retention_keeps_recent_and_saved_insights_references(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    export_dir = tmp_path / 'finder-exports'
    export_dir.mkdir()
    exports = [export_dir / f'trace-finder-{index}.json' for index in range(52)]
    for index, path in enumerate(exports):
        path.write_text('{}', encoding='utf-8')
        path.touch()
        os.utime(path, (index + 1, index + 1))
    old_referenced = exports[0]
    runs_dir = tmp_path / 'insights-runs'
    runs_dir.mkdir()
    (runs_dir / 'insights_saved.json').write_text(
        json.dumps({'population': {'finder_export': str(old_referenced)}}), encoding='utf-8'
    )

    finder_routes._prune_finder_exports(export_dir)

    remaining = {path.name for path in export_dir.glob('trace-finder-*.json')}
    assert len(remaining) == 51
    assert old_referenced.name in remaining
    assert {path.name for path in exports[-50:]} <= remaining


def test_finder_export_retention_keeps_recent_handoff_exports_over_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    export_dir = tmp_path / 'finder-exports'
    export_dir.mkdir()
    exports = [export_dir / f'trace-finder-{index}.json' for index in range(51)]
    for path in exports:
        path.write_text('{}', encoding='utf-8')

    finder_routes._prune_finder_exports(export_dir)

    assert {path.name for path in export_dir.glob('trace-finder-*.json')} == {path.name for path in exports}


def test_finder_export_pruning_cannot_delete_a_concurrent_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    export_dir = tmp_path / 'finder-exports'
    export_dir.mkdir()
    exports = [export_dir / f'trace-finder-{index}.json' for index in range(51)]
    for index, path in enumerate(exports):
        path.write_text('{}', encoding='utf-8')
        os.utime(path, (index + 1, index + 1))
    target = exports[0]
    unlink_started = threading.Event()
    continue_unlink = threading.Event()
    save_started = threading.Event()
    save_finished = threading.Event()
    original_unlink = Path.unlink

    def pause_target_unlink(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == target:
            unlink_started.set()
            assert continue_unlink.wait(timeout=5)
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', pause_target_unlink)
    prune_thread = threading.Thread(target=finder_routes._prune_finder_exports, args=(export_dir,))
    prune_thread.start()
    assert unlink_started.wait(timeout=5)

    def save_replacement() -> None:
        save_started.set()
        finder_routes._save_finder_export(export_dir, target.name, '{"replacement": true}')
        save_finished.set()

    save_thread = threading.Thread(target=save_replacement)
    save_thread.start()
    assert save_started.wait(timeout=5)
    assert not save_finished.wait(timeout=0.1)
    continue_unlink.set()
    prune_thread.join(timeout=5)
    save_thread.join(timeout=5)

    assert not prune_thread.is_alive()
    assert not save_thread.is_alive()
    assert json.loads(target.read_text(encoding='utf-8')) == {'replacement': True}


def test_finder_export_retention_pins_in_flight_insights_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.common.run_manifest import start_manifest
    from evaluatorq.dashboard.insights_launch import (
        ensure_private_finder_reference_dir,
        finder_export_reference_path,
    )

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    export_dir = tmp_path / 'finder-exports'
    export_dir.mkdir()
    exports = [export_dir / f'trace-finder-{index}.json' for index in range(51)]
    for index, path in enumerate(exports):
        path.write_text('{}', encoding='utf-8')
        os.utime(path, (index + 1, index + 1))
    runs_dir = tmp_path / 'insights-runs'
    writer = start_manifest(run_id='active-run', surface='insights', run_name='active', runs_dir=runs_dir)
    marker = finder_export_reference_path(runs_dir, 'active-run')
    ensure_private_finder_reference_dir(marker.parent)
    marker.write_text(json.dumps({'finder_export': exports[0].name}), encoding='utf-8')
    marker.chmod(0o600)

    finder_routes._prune_finder_exports(export_dir)

    assert exports[0].exists()
    assert marker.exists()

    writer.complete()
    finder_routes._prune_finder_exports(export_dir)

    assert not exports[0].exists()
    assert not marker.exists()


def test_finder_export_retention_expires_abandoned_in_flight_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from evaluatorq.common.run_manifest import start_manifest
    from evaluatorq.dashboard.insights_launch import (
        ensure_private_finder_reference_dir,
        finder_export_reference_path,
    )

    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path))
    export_dir = tmp_path / 'finder-exports'
    export_dir.mkdir()
    exports = [export_dir / f'trace-finder-{index}.json' for index in range(51)]
    for index, path in enumerate(exports):
        path.write_text('{}', encoding='utf-8')
        os.utime(path, (index + 1, index + 1))
    runs_dir = tmp_path / 'insights-runs'
    writer = start_manifest(run_id='abandoned-run', surface='insights', run_name='abandoned', runs_dir=runs_dir)
    marker = finder_export_reference_path(runs_dir, 'abandoned-run')
    ensure_private_finder_reference_dir(marker.parent)
    marker.write_text(json.dumps({'finder_export': str(exports[0])}), encoding='utf-8')
    marker.chmod(0o600)
    old = datetime.now(timezone.utc).timestamp() - timedelta(days=31).total_seconds()
    os.utime(writer.path, (old, old))

    finder_routes._prune_finder_exports(export_dir)

    assert not marker.exists()
    assert not exports[0].exists()


def test_find_export_filename_survives_generation_restart_without_overwriting() -> None:
    from evaluatorq.trace_finder.export import export_filename

    snapshot = RunSnapshot(
        generation=1,
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        finished_at=datetime(2026, 9, 1, 0, 1, tzinfo=timezone.utc),
    )
    first = export_filename(snapshot)
    second = export_filename(replace(snapshot, created_at=datetime(2026, 9, 2, tzinfo=timezone.utc)))

    assert first != second
    assert export_filename(snapshot) == first


def test_find_facets_menu_reports_unavailable_catalogue(setup_finder, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, client = setup_finder

    async def load_catalogue(app: Any, window_days: int | None = None) -> FacetCatalogue | None:
        return None

    monkeypatch.setattr(finder_routes, '_load_catalogue', load_catalogue)
    response = client.get('/find/facets')
    assert response.status_code == 200
    assert 'Facet values are unavailable' in response.text


def test_oauth_facet_catalogue_closes_both_clients(setup_finder, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock

    from evaluatorq.dashboard.auth import DashboardAuth

    _store, client = setup_finder
    client.get('/find')
    app = client.app
    app.state.finder_catalogue_cache = None
    closed: list[str] = []

    class Llm:
        async def close(self) -> None:
            closed.append('llm')

    monkeypatch.setattr(
        finder_routes,
        'selected_dashboard_auth',
        lambda _app: DashboardAuth('cli_oauth', None, 'https://my.orq.ai'),
    )
    monkeypatch.setattr(finder_routes, 'build_auth_clients', lambda *_args, **_kwargs: (object(), Llm()))
    monkeypatch.setattr(finder_routes, 'load_facet_catalogue', AsyncMock(return_value=FacetCatalogue()))
    async def close_orq(_orq: object) -> None:
        closed.append('orq')

    monkeypatch.setattr(finder_routes, 'close_orq_client', close_orq)

    result = asyncio.run(finder_routes._load_catalogue(app))

    assert result == FacetCatalogue()
    assert closed == ['orq', 'llm']


def test_find_facets_menu_renders_from_the_submitted_controls(setup_finder, monkeypatch: pytest.MonkeyPatch) -> None:
    """The refetched menu mirrors what the page currently holds, not what the store last saw."""
    _store, client = setup_finder

    async def load_catalogue(app: Any, window_days: int | None = None) -> FacetCatalogue:
        return FacetCatalogue(project=('support-agent', 'docs-agent'), model=('gpt-5.6-luna',))

    monkeypatch.setattr(finder_routes, '_load_catalogue', load_catalogue)
    response = client.get('/find/facets?facet_project=support-agent&tokens_min=5000')
    assert response.status_code == 200
    assert 'name="facet_project" value="support-agent" checked' in response.text
    assert 'name="facet_project" value="docs-agent">' in response.text
    assert 'name="tokens_min" type="number" min="0" placeholder="min" value="5000"' in response.text
    assert '<div class="facet-sub" data-facet-sub="project" hidden>' in response.text
    assert 'is-active' not in response.text


def test_find_facets_preserves_valid_numeric_filter_when_another_is_invalid(
    setup_finder, monkeypatch: pytest.MonkeyPatch
) -> None:
    _store, client = setup_finder
    async def load_catalogue(app: Any, window_days: int | None = None) -> FacetCatalogue:
        return FacetCatalogue()
    monkeypatch.setattr(finder_routes, '_load_catalogue', load_catalogue)
    response = client.get('/find/facets?tokens_min=5000&duration_ms_min=not-a-number')
    assert response.status_code == 200
    assert 'name="tokens_min" type="number" min="0" placeholder="min" value="5000"' in response.text
    assert 'name="duration_ms_min" type="number" min="0" placeholder="min" value=""' in response.text


def test_find_facets_rejects_inverted_range_without_server_error(setup_finder, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, client = setup_finder
    async def load_catalogue(app: Any, window_days: int | None = None) -> FacetCatalogue:
        return FacetCatalogue()
    monkeypatch.setattr(finder_routes, '_load_catalogue', load_catalogue)
    response = client.get('/find/facets?tokens_min=10&tokens_max=5')
    assert response.status_code == 200
    assert 'name="tokens_min" type="number" min="0" placeholder="min" value="10"' in response.text
    assert 'name="tokens_max" type="number" min="0" placeholder="max" value=""' in response.text


def test_find_facets_menu_lists_catalogue_values(setup_finder, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, client = setup_finder
    async def load_catalogue(app: Any, window_days: int | None = None) -> FacetCatalogue:
        assert window_days == 7
        return FacetCatalogue(project=('support-agent',), model=('gpt-5.6-luna',))

    monkeypatch.setattr(finder_routes, '_load_catalogue', load_catalogue)
    response = client.get('/find/facets')
    assert response.status_code == 200
    assert 'support-agent' in response.text
    assert 'gpt-5.6-luna' in response.text


def test_find_facets_menu_searches_returned_values_and_marks_overflow(
    setup_finder, monkeypatch: pytest.MonkeyPatch
) -> None:
    _store, client = setup_finder

    async def load_catalogue(app: Any, window_days: int | None = None) -> FacetCatalogue:
        return FacetCatalogue(model=('popular-model', 'rare-model'), truncated_facets=frozenset({'model'}))

    monkeypatch.setattr(finder_routes, '_load_catalogue', load_catalogue)
    response = client.get('/find/facets?facet_model=older-selection')

    assert response.status_code == 200
    assert 'aria-label="Search model values"' in response.text
    assert response.text.index('popular-model') < response.text.index('rare-model')
    assert 'name="facet_model" value="older-selection" checked' in response.text
    assert 'More values exist in Orq' in response.text
    assert 'class="facet-values"' in response.text


def test_find_facets_uses_submitted_window_without_a_query(setup_finder, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, client = setup_finder
    windows: list[int | None] = []

    async def load_catalogue(app: Any, window_days: int | None = None) -> FacetCatalogue:
        windows.append(window_days)
        return FacetCatalogue()

    monkeypatch.setattr(finder_routes, '_load_catalogue', load_catalogue)

    assert client.get('/find/facets?window_days=14').status_code == 200
    assert windows == [14]


def test_find_trace_drawer_renders_genai_parts_from_responses_spans() -> None:
    from evaluatorq.dashboard.trace_finder.views import _message_text

    user = {'role': 'user', 'parts': [{'type': 'text', 'content': 'Review this diff'}]}
    reasoning = {'role': 'assistant', 'parts': [{'type': 'reasoning', 'content': '[encrypted]'}]}
    call = {'role': 'assistant', 'parts': [{'type': 'tool_call', 'name': 'task_done', 'arguments': {'state': 'DONE'}}]}
    chat_call = {
        'role': 'assistant',
        'content': None,
        'tool_calls': [{'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"id": 1}'}}],
    }
    assert _message_text(user) == 'Review this diff'
    assert _message_text(reasoning) == ''
    assert _message_text(call) == '→ task_done({"state": "DONE"})'
    assert _message_text(chat_call) == '→ lookup({"id": 1})'


def test_find_trace_drawer_explains_a_trace_missing_from_the_current_run(setup_finder) -> None:
    _, client = setup_finder
    drawer = client.get('/find/trace/not-in-run')
    # 200, not 404: htmx discards a 4xx body, so the click would otherwise do nothing visible.
    assert drawer.status_code == 200
    assert 'not part of the current run' in drawer.text


@pytest.mark.parametrize(
    ('snapshot', 'label', 'kind'),
    [
        (RunSnapshot(), 'Idle', 'idle'),
        (RunSnapshot(state='compiling', phase='planning'), 'Planning search', 'busy'),
        (RunSnapshot(state='compiling', phase='loading_traces'), 'Loading traces', 'busy'),
        (RunSnapshot(state='compiling', phase='starting_classification'), 'Starting classification', 'busy'),
        (RunSnapshot(state='classifying', completed=3, total=10), 'Labelling data · 3/10', 'busy'),
        (RunSnapshot(state='completed'), 'Done', 'done'),
        (RunSnapshot(state='failed', error='boom'), 'Failed', 'failed'),
    ],
)
def test_status_indicator_names_each_run_phase(snapshot: RunSnapshot, label: str, kind: str) -> None:
    from evaluatorq.dashboard.trace_finder.views import status_indicator

    html = status_indicator(snapshot)
    assert f'finder-status {kind}' in html
    assert label in html


@pytest.mark.parametrize(
    ('phase', 'heading', 'detail'),
    [
        ('planning', 'Planning the search', 'compiling the question and selecting metadata filters'),
        ('loading_traces', 'Loading traces', 'loading selected traces'),
        ('starting_classification', 'Starting classification', 'preparing the reviewed task'),
    ],
)
def test_working_phases_show_matching_field_and_progress(
    phase: Literal['planning', 'loading_traces', 'starting_classification'], heading: str, detail: str
) -> None:
    from evaluatorq.dashboard.trace_finder.views import field

    html = field(RunSnapshot(state='compiling', phase=phase))
    assert heading in html
    assert detail in html
    assert 'finder-pulse' in html
    assert '0.0s' not in html


def test_progress_line_counts_loaded_traces_and_draws_a_bar() -> None:
    from evaluatorq.dashboard.trace_finder.views import progress

    loading = progress(RunSnapshot(state='compiling', phase='loading_traces', loaded=40, to_load=200, within_results=True))
    assert 'fetching conversations <b>40 / 200</b>' in loading
    assert 'style="width:20.0%"' in loading

    classifying = progress(RunSnapshot(state='classifying', completed=3, total=4))
    assert 'style="width:75.0%"' in classifying

    emptied = progress(RunSnapshot(state='failed', loaded=190, error='None of the 190 loaded traces have at least 50,001 tokens'))
    assert '190</b> traces loaded, none kept' in emptied
    assert 'Stopped before traces were loaded' not in emptied


def test_completed_within_progress_reports_run_counts() -> None:
    from evaluatorq.dashboard.trace_finder.views import progress

    matched = TraceClassification(trace_id='trace-1', span_id='span-1', matched=True, raw_result={})
    missed = TraceClassification(trace_id='trace-2', span_id='span-2', matched=False, raw_result={})
    snapshot = RunSnapshot(
        state='completed', within_results=True, results={'trace-1': matched, 'trace-2': missed}, total=2, matched=1
    )
    html = progress(snapshot)
    assert '<b>1</b> of 2 traces match' in html
    assert 'judged traces in view' not in html
    assert 'when asked' not in html


def test_progress_parts_keep_a_separator_before_each_part() -> None:
    from evaluatorq.dashboard.trace_finder.views import progress

    html = progress(RunSnapshot(state='completed', total=5, matched=2))
    assert html.count('<span class="part"><span class="sep">·</span>') >= 2


def test_completed_answer_sentence_offers_show_only_and_names_model() -> None:
    from evaluatorq.dashboard.trace_finder.views import progress

    from evaluatorq.trace_finder import PopulationRequest, RunRequest

    request = RunRequest(query='angry customer', mode='immediate', population=PopulationRequest())
    snapshot = RunSnapshot(state='completed', request=request, total=5, matched=2)
    html = progress(snapshot, classifier_model='acme/judge', show_only_url='/find/rows?quick_view=matches')
    assert '<b>2</b> of 5 traces match' in html
    assert '“angry customer”' in html
    assert 'Show only these' in html
    assert 'Uses acme/judge' in html
    assert 'Show only these' not in progress(replace(snapshot, matched=0), show_only_url='/x')
    assert 'Reading <b>3</b> of 5' in progress(replace(snapshot, state='classifying', completed=3))


def test_unsupported_question_reads_as_cannot_answer_not_error() -> None:
    from evaluatorq.dashboard.trace_finder.views import progress, status_indicator

    warning = 'Ask AI finds traces; it cannot compute totals, averages or rankings. Costs are sums.'
    snapshot = RunSnapshot(state='cancelled', plan_warning=warning, dimensions=())
    html = progress(snapshot)
    assert warning in html
    assert 'role="alert"' not in html
    assert 'compute totals or rankings' not in html
    assert 'Can&#x27;t answer' in status_indicator(snapshot)
    assert 'Filters answer this question' not in finder_views.fragment(
        snapshot, DashboardSettings(window_days=7, limit=500, parallelism=100)
    )
    assert 'Cancelled' in status_indicator(RunSnapshot(state='cancelled'))


def test_matched_answer_does_not_show_prose_reason_without_classifier_score() -> None:
    from evaluatorq.dashboard.trace_finder.views import _answer_cells

    tone = CompiledQuery(
        name='Tone',
        task=ClassifyQuestion(
            kind='choice', instructions='Tone?', criteria={'frustrated': 'Annoyed.', 'neutral': 'Calm.'}, state={}
        ),
        selection=ValueSelection(kind='values', values=('frustrated',)),
    )

    def result(matched: bool) -> TraceClassification:
        answer = DimensionAnswer(value='frustrated', matched=matched, summary='The customer repeats the request.')
        return TraceClassification(trace_id='t', span_id='s', answers=(answer,), matched=matched, raw_result={})

    assert 'The customer repeats the request.' not in _answer_cells(result(True), (tone,))
    assert 'The customer repeats the request.' not in _answer_cells(result(False), (tone,))


def _yes_no_snapshot(summary: str | None, confidence: float | None = 0.93):
    dimension = CompiledQuery(
        task=ClassifyQuestion(kind='noul', instructions='Is it about traces?', noul_threshold=0.5, state={}),
        selection=ValueSelection(kind='values', values=(True,)),
    )
    answer = DimensionAnswer(value=True, matched=True, confidence=confidence, summary=summary)
    result = TraceClassification(trace_id='t', span_id='s', answers=(answer,), matched=True, raw_result={})
    snapshot = RunSnapshot(state='completed', results={'t': result}, dimensions=(dimension,), within_results=True)
    return dimension, result, snapshot


def test_yes_no_cell_shows_only_classifier_score_when_prose_exists() -> None:
    from types import SimpleNamespace

    from evaluatorq.dashboard.trace_finder.explorer_views import _match_cells

    _, _, snapshot = _yes_no_snapshot('The user asks how to filter traces.')
    html = _match_cells(SimpleNamespace(trace_id='t'), snapshot)  # type: ignore[arg-type]

    assert 'xr-yn yes' in html
    assert 'Classifier score 93%' in html
    assert 'The user asks how to filter traces.' not in html


def test_prose_reason_is_not_rendered_in_table_or_drawer() -> None:
    from types import SimpleNamespace

    from evaluatorq.dashboard.trace_finder.explorer_views import _match_cells
    from evaluatorq.dashboard.trace_finder.views import _drawer_reason

    _, result, snapshot = _yes_no_snapshot('Asked <script>alert("x")</script> & explained.')
    html = _match_cells(SimpleNamespace(trace_id='t'), snapshot)  # type: ignore[arg-type]
    drawer = _drawer_reason(result.answers[0])

    assert 'Classifier score 93%' in html
    assert 'alert' not in html
    assert '<script>' not in html
    assert 'Classifier score 93%' in drawer
    assert 'alert' not in drawer
    assert '<script>' not in drawer


def test_score_only_summary_renders_neutral_classifier_score() -> None:
    from types import SimpleNamespace

    from evaluatorq.dashboard.trace_finder.explorer_views import _match_cells
    from evaluatorq.dashboard.trace_finder.views import _answer_cells, _drawer_reason

    score_text = 'noul=0.93 (threshold 0.5)'
    dimension, result, snapshot = _yes_no_snapshot(score_text)

    assert 'Classifier score 93%' in _match_cells(
        SimpleNamespace(trace_id='t'), snapshot
    )  # type: ignore[arg-type]
    assert 'Classifier score 93%' in _answer_cells(result, (dimension,))
    drawer = _drawer_reason(result.answers[0])
    assert score_text not in drawer
    assert 'Classifier score 93%' in drawer
    assert 'Reason' not in drawer


def test_matched_answer_without_confidence_shows_no_reason_or_score() -> None:
    from types import SimpleNamespace

    from evaluatorq.dashboard.trace_finder.explorer_views import _match_cells
    from evaluatorq.dashboard.trace_finder.views import _answer_cells, _drawer_reason

    dimension, result, snapshot = _yes_no_snapshot('noul=0.93 (threshold 0.5)', confidence=None)
    table = _match_cells(SimpleNamespace(trace_id='t'), snapshot)  # type: ignore[arg-type]
    cells = _answer_cells(result, (dimension,))
    drawer = _drawer_reason(result.answers[0])

    assert 'xr-reason' not in table
    assert 'classifier score' not in table
    assert 'xr-reason' not in cells
    assert 'classifier score' not in cells
    assert drawer == ''


def test_unmatched_or_failed_score_answer_has_no_score_callout() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from evaluatorq.dashboard.trace_finder.explorer_views import _match_cells
    from evaluatorq.dashboard.trace_finder.views import _drawer_reason

    _, result, snapshot = _yes_no_snapshot('noul=0.93 (threshold 0.5)')
    unmatched = result.answers[0].model_copy(update={'matched': False})
    failed = result.answers[0].model_copy(update={'error': 'classifier failed'})
    unmatched_result = result.model_copy(update={'answers': (unmatched,), 'matched': False})
    failed_result = result.model_copy(update={'answers': (failed,), 'error': 'classifier failed'})

    assert 'Classifier score' not in _drawer_reason(unmatched)
    assert 'Classifier score' not in _drawer_reason(failed)
    assert 'Classifier score' not in _match_cells(
        SimpleNamespace(trace_id='t'), replace(snapshot, results={'t': unmatched_result})
    )  # type: ignore[arg-type]
    assert 'Classifier score' not in _match_cells(
        SimpleNamespace(trace_id='t'), replace(snapshot, results={'t': failed_result})
    )  # type: ignore[arg-type]


def test_drawer_uses_score_only_when_classifier_returns_prose() -> None:
    from evaluatorq.dashboard.trace_finder.views import _drawer_reason

    _, result, _ = _yes_no_snapshot('The user asks how to filter traces.')

    drawer = _drawer_reason(result.answers[0])
    assert 'Classifier score 93%' in drawer
    assert 'The user asks how to filter traces.' not in drawer


def test_long_question_is_capped_in_the_result_line_with_full_text_in_title() -> None:
    from evaluatorq.dashboard.trace_finder.views import _judging_text

    question = 'Is the conversation about ' + 'searching and filtering traces ' * 8
    from evaluatorq.trace_finder import PopulationRequest, RunRequest

    request = RunRequest(query=question, mode='immediate', population=PopulationRequest())
    html = _judging_text(RunSnapshot(state='completed', total=10, matched=3, request=request, within_results=True))

    assert f'title="{question}"' in html
    assert question not in html.replace(f'title="{question}"', '')
    assert '…”' in html


def test_completed_empty_run_keeps_clear_without_download() -> None:
    from evaluatorq.dashboard.trace_finder.views import progress

    html = progress(RunSnapshot(state='completed', total=0))
    assert 'Clear AI results' in html
    assert 'Download results' not in html


def test_run_controls_are_preserved_across_polls_per_form(setup_finder) -> None:
    store, client = setup_finder
    run = client.post('/find/run', data=csrf_data({'query': 'frustrated customers', 'mode': 'immediate'})).text
    for name in ('window_days', 'parallelism'):
        assert f'id="finder-{name}-finder-query-form" hx-preserve' in run
    assert 'id="finder-limit-query"' in run
    assert 'hx-target="#finder-run-status" hx-swap="outerHTML"' in run
    poll = client.get('/find/poll')
    assert poll.status_code == 200
    assert poll.text.startswith('<div id="finder-run-status">')
    assert 'id="finder-controls"' not in poll.text


def test_traces_page_uses_compact_ai_strip_and_one_classification_surface(setup_finder) -> None:
    store, client = setup_finder
    html = client.get('/traces').text

    assert 'class="finder-title finder-command-title"' not in html
    assert 'class="finder-command-lede"' in html
    assert 'data-finder-example=' in html
    assert 'placeholder="Ask a question, e.g. Did any customers get frustrated?"' in html
    assert 'Search in' in html
    assert '<span>Within results</span>' in html
    assert '<span>New search</span>' in html
    assert 'answered by an AI model' in html
    assert 'Ask AI' in html
    assert '>Search<' in html
    assert 'href="/settings" title="Choose the models Ask AI uses" class="finder-command-gear"' in html
    assert 'aria-label="AI settings"' in html
    assert html.index('id="finder-scope"') < html.index('class="finder-command-gear"') < html.index('finder-command-search')
    assert 'value="within" form="finder-query-form"' in html
    assert 'value="new" form="finder-query-form"' in html
    assert 'Ask AI <span aria-hidden="true">↗</span>' not in html
    assert 'class="finder-hint' not in html
    assert 'type="hidden" name="mode" value="immediate"' in html
    assert 'type="hidden" form="finder-query-form" name="limit"' in html
    assert 'type="hidden" form="finder-query-form" name="parallelism"' in html
    assert 'name="limit" value="500"' in html
    assert 'name="parallelism" value="100"' in html
    assert 'id="finder-body"' in html
    assert 'id="explorer-results"' in html

    client.post('/find/run', data=csrf_data({'query': 'frustrated customers', 'mode': 'review'}))
    assert store.snapshot_value.request is not None
    assert store.snapshot_value.request.mode == 'review'
    assert store.snapshot_value.request.population.limit == 500
    assert store.snapshot_value.request.parallelism == 100
    store.complete()
    classified = client.get('/traces').text
    assert 'finder-progress' in classified
    assert 'finder-task-title">AI match<' in classified
    assert 'Filter selection' not in classified
    assert '<div class="finder-matrix' not in classified
    assert '<table class="finder-table' not in classified
    assert 'Included traces' not in classified

    store.snapshot_value = replace(store.snapshot_value, state='compiling', phase='planning')
    compiling = client.get('/traces').text
    # The step is named inline under Ask AI, not in the corner badge.
    assert 'compiling the question and selecting metadata filters' in compiling
    assert '<div class="finder-status' not in compiling
    assert '<div class="finder-matrix' not in compiling
    assert '<table class="finder-table' not in compiling


def test_find_keeps_legacy_search_hero_separate_from_traces(setup_finder) -> None:
    _store, client = setup_finder
    html = client.get('/find').text

    assert 'Find the signal.' in html
    assert 'Find traces' in html
    assert 'AI settings</a>' not in html
    assert 'id="explorer-results"' not in html
