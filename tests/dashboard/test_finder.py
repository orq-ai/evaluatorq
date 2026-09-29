from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.trace_finder import routes as finder_routes, views as finder_views
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.security import CSRF_FIELD, _CSRF_TOKEN
from evaluatorq.dashboard.trace_links import trace_span_url
from evaluatorq.trace_finder import (
    CompiledQuery,
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

    async def compile(self, request: Any, *, wait: bool = True, table: Any = None) -> RunSnapshot:
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
    assert 'hx-get="/find/facets?form_id=finder-query-form" hx-trigger="load" hx-include="#finder-controls"' in page
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
    app = build_app(roots=[tmp_path])
    client = TestClient(app, raise_server_exceptions=True)
    assert client.get('/traces').status_code == 200
    assert client.get('/find').status_code == 200
    assert len(stores) == 2
    assert app.state.finder_store is stores[0]
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
    poll = client.get('/find/poll').text
    assert 'class="chip is-editable" data-chip-name="tokens_min"' in poll
    assert 'data-chip-open="tokens"' in poll
    assert 'data-finder-remove="tokens_min"' in poll


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
    assert response.status_code == 200
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
    assert 'This is the third time' in drawer.text
    assert drawer.text.count('<details class="fd-msg') == 2
    assert '<summary><span class="role"><b>User</b></span><em class="fd-msg-meta">#1 · ~' in drawer.text
    assert '<span class="fd-msg-preview">This is the third time I am asking.</span>' in drawer.text
    assert '<div class="fd-msg-content">This is the third time I am asking.</div>' in drawer.text
    assert 'eqFinderTab(this' in drawer.text
    assert 'navigator.clipboard.writeText' in drawer.text


def test_find_export_is_404_until_completed_then_downloads_json(setup_finder) -> None:
    store, client = setup_finder
    assert client.get('/find/export.json').status_code == 404
    client.post('/find/run', data=csrf_data({'query': 'frustrated customers', 'mode': 'immediate'}))
    store.complete()
    response = client.get('/find/export.json')
    assert response.status_code == 200
    assert response.headers['content-disposition'] == 'attachment; filename="trace-finder-1.json"'
    assert json.loads(response.text)['counts']['matched'] == 1


def test_find_facets_menu_reports_unavailable_catalogue(setup_finder, monkeypatch: pytest.MonkeyPatch) -> None:
    _store, client = setup_finder

    async def load_catalogue(app: Any, window_days: int | None = None) -> FacetCatalogue | None:
        return None

    monkeypatch.setattr(finder_routes, '_load_catalogue', load_catalogue)
    response = client.get('/find/facets')
    assert response.status_code == 200
    assert 'Facet values are unavailable' in response.text


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
    assert '<b>1</b> of 2 judged traces match' in html
    assert 'judged traces in view' not in html
    assert 'when asked' not in html


def test_completed_empty_run_keeps_edit_and_clear_without_download() -> None:
    from evaluatorq.dashboard.trace_finder.views import progress

    html = progress(RunSnapshot(state='completed', total=0))
    assert 'Edit question' in html
    assert 'Clear AI results' in html
    assert 'Download results' not in html


def test_run_controls_are_preserved_across_polls_per_form(setup_finder) -> None:
    store, client = setup_finder
    client.post('/find/run', data=csrf_data({'query': 'frustrated customers', 'mode': 'immediate'}))
    poll = client.get('/find/poll').text
    for name in ('window_days', 'limit', 'parallelism'):
        assert f'id="finder-{name}-finder-query-form" hx-preserve' in poll


def test_traces_page_uses_compact_ai_strip_and_one_classification_surface(setup_finder) -> None:
    store, client = setup_finder
    html = client.get('/traces').text

    assert '<h2 class="finder-title finder-command-title">Traces</h2>' in html
    assert 'Ask AI' in html
    assert '>Search<' in html
    assert 'href="/settings" aria-label="AI configuration"' in html
    assert 'class="finder-command-gear" style="min-width:42px;min-height:42px"' in html
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
    assert 'href="/settings" aria-label="AI configuration"' not in html
    assert 'id="explorer-results"' not in html
