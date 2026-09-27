from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.trace_finder import explorer_views
from evaluatorq.trace_finder.columns import resolve_columns
from evaluatorq.trace_finder.explorer import ExplorerView
from evaluatorq.trace_finder.models import TraceRecord
from evaluatorq.trace_finder.rows import TraceRow

from evaluatorq.dashboard.trace_finder import routes as finder_routes
from evaluatorq.dashboard.app import build_app
from evaluatorq.trace_finder.explorer import ExplorerStore
from tests.dashboard.test_finder import FakeStore, csrf_data


def test_parse_range_applies_browser_offset() -> None:
    start, end = finder_routes.parse_range('2026-09-27T10:00:05', '2026-09-27T12:00:00', '-120')
    assert start == datetime(2026, 9, 27, 8, 0, 5, tzinfo=timezone.utc)
    assert end == datetime(2026, 9, 27, 10, 0, 0, tzinfo=timezone.utc)


def test_parse_range_accepts_minutes_only() -> None:
    start, _ = finder_routes.parse_range('2026-09-27T10:00', '2026-09-27T11:00', '0')
    assert start == datetime(2026, 9, 27, 10, 0, tzinfo=timezone.utc)


def test_parse_range_rejects_inverted() -> None:
    with pytest.raises(ValueError, match='From must be before To'):
        finder_routes.parse_range('2026-09-27T12:00', '2026-09-27T11:00', '0')


def test_parse_range_rejects_over_30_days() -> None:
    with pytest.raises(ValueError, match='30 days'):
        finder_routes.parse_range('2026-08-01T00:00', '2026-09-27T00:00', '0')


def test_parse_range_treats_bad_offset_as_utc() -> None:
    start, _ = finder_routes.parse_range('2026-09-27T10:00', '2026-09-27T11:00', 'abc')
    assert start.tzinfo == timezone.utc


def test_calendar_date_and_time_values_keep_seconds_and_timezone() -> None:
    values = finder_routes._range_values({
        'from': '2026-09-27',
        'from_time': '10:00:05',
        'to': '2026-09-27',
        'to_time': '11:30:09',
    })
    assert values == ('2026-09-27T10:00:05', '2026-09-27T11:30:09')
    start, end = finder_routes.parse_range(*values, '-120')
    assert start == datetime(2026, 9, 27, 8, 0, 5, tzinfo=timezone.utc)
    assert end == datetime(2026, 9, 27, 9, 30, 9, tzinfo=timezone.utc)


def test_calendar_range_rejects_missing_time() -> None:
    values = finder_routes._range_values({'from': '2026-09-27', 'from_time': '', 'to': '2026-09-27', 'to_time': '11:00:00'})
    with pytest.raises(ValueError, match='full dates and times'):
        finder_routes.parse_range(*values, '0')


def _rows(n: int) -> tuple[TraceRow, ...]:
    return tuple(
        TraceRow(
            trace_id=f'trace-{i:04d}',
            status='ok',
            tokens_in=100 + i,
            tokens_out=10,
            cached_tokens=50,
            agent_name='support',
            models=('gpt-5.6-luna',),
        )
        for i in range(n)
    )


def test_range_inputs_include_local_time_fields_and_default_rows() -> None:
    html = explorer_views.range_inputs(
        datetime(2026, 9, 26, tzinfo=timezone.utc), datetime(2026, 9, 27, tzinfo=timezone.utc), 7
    )
    assert 'name="from" type="date"' in html
    assert 'name="from_time" type="time" step="1"' in html
    assert 'name="to" type="date"' in html
    assert 'name="to_time" type="time" step="1"' in html
    assert 'name="tz_offset" form="explorer-load-form"' in html
    assert 'data-explorer-preset="900"' in html
    assert 'name="rows" type="number"' in html
    assert 'max="5000"' in html
    assert 'name="csrf"' in html
    assert 'value="200"' in html
    assert 'form="explorer-load-form">Load</button>' in html


def test_idle_results_render_an_empty_state() -> None:
    html = explorer_views.results(ExplorerView(), resolve_columns(None), records=None, snapshot=None)
    assert 'id="explorer-results"' in html
    assert 'Load traces' in html
    assert 'class="xr-empty"' in html
    assert 'finder-hint' not in html


def test_zero_rows_render_an_empty_state() -> None:
    html = explorer_views.results(ExplorerView(state='loaded'), resolve_columns(None), records=None, snapshot=None)
    assert 'No traces match' in html


def test_zero_columns_render_an_empty_state() -> None:
    html = explorer_views.results(ExplorerView(state='loaded', rows=_rows(1)), (), records=None, snapshot=None)
    assert 'No columns selected.' in html


def test_table_renders_one_row_per_page_row_with_drawer_links() -> None:
    view = ExplorerView(state='loaded', rows=_rows(150))
    html = explorer_views.results(view, resolve_columns(None), records=None, snapshot=None)
    assert html.count('hx-get="/find/trace/') == 100
    assert 'Page 1 of 2' in html
    assert '<th' in html and 'Tokens in' in html
    assert 'hx-sync="#explorer-results:replace"' in html


def test_failed_load_shows_banner_and_rows() -> None:
    view = ExplorerView(state='failed', rows=_rows(3), error='page 2 timed out')
    html = explorer_views.results(view, resolve_columns(None), records=None, snapshot=None)
    assert 'page 2 timed out' in html
    assert '3 rows loaded' in html
    assert html.count('hx-get="/find/trace/') == 3


def test_loading_results_poll() -> None:
    html = explorer_views.results(
        ExplorerView(state='loading', limit=1000, rows=_rows(400)), resolve_columns(None), records=None, snapshot=None
    )
    assert 'hx-trigger="every 1s"' in html
    assert '400 / 1000' in html


def test_trajectories_draws_a_row_for_a_failed_hydration() -> None:
    rows = _rows(2)
    record = TraceRecord(
        schema_version=1,
        trace_id=rows[0].trace_id,
        span_id='s',
        timestamp=datetime(2026, 9, 27, tzinfo=timezone.utc),
        project='p',
        model='gpt-5.6-luna',
        provider='openai',
        status='ok',
        product='chat',
        trace_type='agent',
        messages=({'role': 'user', 'content': 'hello there'}, {'role': 'assistant', 'content': 'hi'}),
    )
    view = ExplorerView(state='loaded', rows=rows, view='trajectories')
    html = explorer_views.results(
        view, resolve_columns(None), records={rows[0].trace_id: record, rows[1].trace_id: None}, snapshot=None
    )
    assert html.count('data-tv-row=') == 2
    assert 'tv-nomsg' in html
    assert 'data-tv-msg="1"' in html
    assert '2 msgs' in html


class FakeRowSource:
    def __init__(self, rows: tuple[TraceRow, ...]) -> None:
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    async def search(
        self, start: Any, end: Any, limit: int, *, facets: Any, numeric: Any, on_page: Any = None
    ) -> tuple[TraceRow, ...]:
        self.calls.append({'start': start, 'end': end, 'limit': limit, 'facets': facets, 'numeric': numeric})
        if on_page is not None:
            on_page(self.rows[:limit])
        return self.rows[:limit]

    async def hydrate_rows(self, rows: Any) -> dict[str, Any]:
        return {row.trace_id: None for row in rows}


@pytest.fixture
def explorer_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv('ORQ_API_KEY', 'test-key')
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))
    store = FakeStore()
    source = FakeRowSource(_rows(250))
    store.explorer = ExplorerStore(search=source.search, hydrate=source.hydrate_rows)
    store.snapshot_for_render = store.snapshot
    store.reset_calls = 0

    async def reset() -> Any:
        store.reset_calls += 1
        return store.snapshot_value

    store.reset = reset

    async def build_store(_app: Any) -> FakeStore:
        return store

    monkeypatch.setattr(finder_routes, '_build_store', build_store)
    return store, source, TestClient(build_app(roots=[tmp_path]), raise_server_exceptions=True)


def _load(client: TestClient, **extra: str) -> Any:
    return client.post(
        '/find/load',
        data=csrf_data({
            'from': '2026-09-27',
            'from_time': '10:00:00',
            'to': '2026-09-27',
            'to_time': '11:00:00',
            'tz_offset': '0',
            'rows': '250',
            **extra,
        }),
    )


def test_find_page_loads_last_seven_days_without_filters_or_ai(explorer_client) -> None:
    store, source, client = explorer_client
    before = datetime.now(timezone.utc)
    response = client.get('/find')
    after = datetime.now(timezone.utc)

    assert response.status_code == 200
    assert len(source.calls) == 1
    call = source.calls[0]
    assert before <= call['end'] <= after
    assert call['end'] - call['start'] == timedelta(days=7)
    assert call['limit'] == explorer_views.DEFAULT_EXPLORER_ROWS
    assert call['facets'].model_dump(exclude_none=True, exclude_defaults=True) == {}
    assert call['numeric'].model_dump(exclude_none=True) == {}
    assert store.reset_calls == 0
    assert store.compile_request is None
    assert not store.started
    assert f'data-utc="{call["start"].strftime("%Y-%m-%dT%H:%M:%S")}"' in response.text
    assert f'data-utc="{call["end"].strftime("%Y-%m-%dT%H:%M:%S")}"' in response.text

    client.get('/find')
    assert len(source.calls) == 1


def test_find_page_keeps_manually_loaded_range(explorer_client) -> None:
    _, source, client = explorer_client
    _load(client)
    response = client.get('/find')
    assert len(source.calls) == 1
    assert 'data-utc="2026-09-27T10:00:00"' in response.text
    assert 'data-utc="2026-09-27T11:00:00"' in response.text


def test_load_then_rows_then_sort_then_page(explorer_client) -> None:
    store, source, client = explorer_client
    response = _load(client)
    assert response.status_code == 200
    assert store.reset_calls == 1
    assert source.calls[0]['limit'] == 250
    assert 'finder-body' in response.text
    html = client.get('/find/rows').text
    assert 'Page 1 of 3' in html
    html = client.get('/find/rows?sort=tokens_in&dir=desc').text
    assert html.index('trace-0249') < html.index('trace-0248')
    html = client.get('/find/rows?page=2').text
    assert 'Page 3 of 3' in html


def test_load_rejects_an_inverted_range(explorer_client) -> None:
    _, source, client = explorer_client
    response = client.post(
        '/find/load',
        data=csrf_data({'from': '2026-09-27T12:00:00', 'to': '2026-09-27T11:00:00', 'tz_offset': '0', 'rows': '10'}),
    )
    assert response.status_code == 200
    assert 'From must be before To' in response.text
    assert source.calls == []


def test_load_rejects_rows_out_of_range(explorer_client) -> None:
    _, _, client = explorer_client
    assert _load(client, rows='6000').status_code == 200


def test_load_requires_csrf(explorer_client) -> None:
    _, _, client = explorer_client
    response = client.post('/find/load', data={'from': '2026-09-27T10:00:00', 'to': '2026-09-27T11:00:00', 'rows': '5'})
    assert response.status_code == 200
    assert 'id="explorer-results"' in response.text


def test_columns_post_requires_csrf(explorer_client) -> None:
    _, _, client = explorer_client
    response = client.post('/find/columns', data={'columns': ['model', 'cost']})
    assert response.status_code == 200
    assert 'id="explorer-results"' in response.text


def test_columns_post_saves_choice(explorer_client, tmp_path: Path) -> None:
    _, _, client = explorer_client
    _load(client)
    html = client.post('/find/columns', data=csrf_data({'columns': ['model', 'cost']})).text
    thead = html.split('<thead>')[1].split('</thead>')[0]
    assert 'Model' in thead and 'Cost' in thead
    assert 'Tokens in' not in thead
    assert json.loads((tmp_path / 'settings.json').read_text())['explorer_columns'] == ['model', 'cost']


def test_trajectories_view_hydrates_only_the_visible_page(explorer_client) -> None:
    store, _, client = explorer_client
    _load(client)
    calls: list[int] = []
    original = store.explorer._hydrate  # pyright: ignore[reportPrivateUsage]

    async def counting(rows: Any) -> Any:
        calls.append(len(rows))
        return await original(rows)

    store.explorer._hydrate = counting  # pyright: ignore[reportPrivateUsage]
    html = client.get('/find/rows?view=trajectories').text
    assert calls == [100]
    assert html.count('data-tv-row=') == 100


def test_new_search_results_do_not_change_trajectory_hydration_page(explorer_client) -> None:
    from evaluatorq.trace_finder import TraceClassification

    store, _, client = explorer_client
    _load(client)
    store.snapshot_value = replace(
        store.snapshot_value,
        results={'trace-0249': TraceClassification(trace_id='trace-0249', span_id='s', matched=True, raw_result={})},
        within_results=False,
    )
    calls: list[tuple[str, ...]] = []
    original = store.explorer._hydrate  # pyright: ignore[reportPrivateUsage]

    async def counting(rows: Any) -> Any:
        calls.append(tuple(row.trace_id for row in rows))
        return await original(rows)

    store.explorer._hydrate = counting  # pyright: ignore[reportPrivateUsage]
    html = client.get('/find/rows?view=trajectories').text
    assert len(calls) == 1 and len(calls[0]) == 100
    assert calls[0][0] == 'trace-0000'
    assert html.count('data-tv-row=') == 100


def test_poll_appends_explorer_results_during_classification(explorer_client) -> None:
    store, _, client = explorer_client
    _load(client)
    store.snapshot_value = replace(store.snapshot_value, state='classifying')
    html = client.get('/find/poll').text
    assert 'id="explorer-results"' in html
    assert 'hx-swap-oob="true"' in html


def test_load_keeps_filter_values_and_enables_within_scope(explorer_client) -> None:
    _, _, client = explorer_client
    response = _load(client, facet_model='gpt-x', tokens_min='100')
    assert 'name="facet_model" value="gpt-x"' in response.text
    assert 'name="tokens_min"' in response.text
    assert 'gpt-x' in response.text
    assert 'id="finder-scope" hx-swap-oob="innerHTML"' in response.text
    poll = client.get('/find/rows').text
    assert 'name="scope" value="within" form="finder-query-form" checked' in poll


def test_error_responses_swap_as_html(explorer_client) -> None:
    _, _, client = explorer_client
    response = _load(client, **{'from': '2026-09-27T12:00:00', 'to': '2026-09-27T11:00:00'})
    assert response.status_code == 200
    assert 'From must be before To.' in response.text


def test_load_preserves_exact_review_range() -> None:
    from types import SimpleNamespace

    from evaluatorq.trace_finder import PopulationRequest

    start = datetime(2026, 9, 27, 10, tzinfo=timezone.utc)
    end = datetime(2026, 9, 27, 11, tzinfo=timezone.utc)
    request = finder_routes._run_request(
        {'query': 'q', 'mode': 'review', 'window_days': '1', 'limit': '2', 'parallelism': '1'},
        SimpleNamespace(window_days=7, limit=200, parallelism=10, orq_project_id=None),
        anchor=PopulationRequest(start=start, end=end),
    )
    assert request.population.start == start
    assert request.population.end == end


def test_rows_controls_share_sync_and_scope_oob(explorer_client) -> None:
    _, _, client = explorer_client
    _load(client)
    html = client.get('/find/rows?view=trajectories').text
    assert 'hx-sync="#explorer-results:replace"' in html
    assert 'id="finder-scope" hx-swap-oob="innerHTML"' in html


def test_new_search_results_do_not_dim_unclassified_explorer_rows() -> None:
    from evaluatorq.trace_finder import TraceClassification

    view = ExplorerView(state='loaded', rows=_rows(1), view='trajectories')
    classified_other = TraceClassification(trace_id='other', span_id='s', matched=True, raw_result={})
    from evaluatorq.trace_finder import RunSnapshot

    snapshot = RunSnapshot(results={'other': classified_other}, within_results=False)
    html = explorer_views.trajectories(view, {}, snapshot)
    assert 'tv-r nomatch' not in html


def test_terminal_poll_includes_last_explorer_results(explorer_client) -> None:
    store, _, client = explorer_client
    _load(client)
    from evaluatorq.trace_finder import TraceClassification

    result = TraceClassification(trace_id='trace-0000', span_id='s', value=True, matched=True, raw_result={})
    store.snapshot_value = replace(
        store.snapshot_value, state='completed', results={'trace-0000': result}, within_results=True
    )
    html = client.get('/find/poll').text
    assert 'hx-swap-oob="true"' in html


def test_explorer_error_preserves_rows_and_escapes_message() -> None:
    html = explorer_views.results(
        ExplorerView(state='loaded', rows=_rows(1)),
        resolve_columns(None),
        records=None,
        snapshot=None,
        error='<script>bad</script>',
    )
    assert '<section id="explorer-results"' in html
    assert '&lt;script&gt;bad&lt;/script&gt;' in html
    assert 'hx-get="/find/trace/trace-0000"' in html


def test_invalid_load_keeps_existing_rows(explorer_client) -> None:
    _, _, client = explorer_client
    _load(client)
    response = _load(client, **{'from': '2026-09-27T12:00:00', 'to': '2026-09-27T11:00:00'})
    assert 'From must be before To.' in response.text
    assert 'hx-get="/find/trace/trace-0000"' in response.text


def test_drawer_opens_at_the_requested_message() -> None:
    from evaluatorq.dashboard.trace_finder.views import drawer
    from evaluatorq.trace_finder import TraceDetail

    trace = TraceRecord(
        schema_version=1,
        trace_id='t1',
        span_id='s',
        timestamp=datetime(2026, 9, 27, tzinfo=timezone.utc),
        project='p',
        model='gpt-5.6-luna',
        provider='openai',
        status='ok',
        product='chat',
        trace_type='agent',
        messages=(
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello'},
            {'role': 'user', 'content': 'bye'},
        ),
    )
    row = TraceRow(trace_id='t1', status='failed', cost_total=1.25, currency='EUR&')
    html = drawer(TraceDetail(trace=trace, projection=None, classification=None), msg=2, row=row)
    assert 'id="msg-2"' in html
    assert html.count('<details class="fd-msg') == 3
    assert 'fd-msg k-assistant on" id="msg-2" data-msg="2" open' in html
    assert 'class="fd-mini"' in html
    assert html.count('data-mini-msg=') == 3
    assert 'tv dot err' in html
    assert '1.2500 EUR&amp;' in html
    assert 'EUR&amp;amp;' not in html
    assert 'fd-msg-meta' in html


def test_missing_trace_reason_is_escaped_and_wrapped() -> None:
    from evaluatorq.dashboard.trace_finder.views import missing_trace_drawer

    html = missing_trace_drawer('trace', reason='<script>alert(1)</script>')
    assert '<p class="finder-empty">&lt;script&gt;alert(1)&lt;/script&gt;</p>' in html


def test_drawer_message_meta_and_segment_click_browser_fixes() -> None:
    js = Path('src/evaluatorq/dashboard/static/dashboard.js').read_text()
    styles = Path('src/evaluatorq/dashboard/styles.py').read_text()
    assert "if (tip) tip.hidden = true;" in js
    assert '.fd-msg summary .fd-msg-meta { white-space:nowrap;' in styles
    assert 'background:var(--surface-sunken)' in styles


def test_trace_route_falls_back_to_explorer_rows(explorer_client) -> None:
    store, source, client = explorer_client
    _load(client)
    trace_id = source.rows[0].trace_id

    async def records(ids: Any) -> dict[str, Any]:
        return {
            trace_id: TraceRecord(
                schema_version=1,
                trace_id=trace_id,
                span_id='s',
                timestamp=datetime(2026, 9, 27, tzinfo=timezone.utc),
                project='p',
                model='gpt-5.6-luna',
                provider='openai',
                status='ok',
                product='chat',
                trace_type='agent',
                messages=({'role': 'user', 'content': 'hi'},),
            )
        }

    store.explorer.records = records
    store.trace_detail = lambda _id: _none()
    html = client.get(f'/find/trace/{trace_id}?msg=1').text
    assert 'id="msg-1"' in html
    assert 'support' in html


def test_explorer_drawer_ignores_an_unrelated_new_search_result(explorer_client) -> None:
    from evaluatorq.trace_finder import TraceClassification

    store, source, client = explorer_client
    _load(client)
    trace_id = source.rows[0].trace_id
    store.snapshot_value = replace(
        store.snapshot_value,
        results={trace_id: TraceClassification(trace_id=trace_id, span_id='s', matched=True, raw_result={})},
        within_results=False,
    )

    async def unrelated_detail(_trace_id: str) -> Any:
        raise AssertionError('The unrelated search result must not supply the explorer drawer.')

    store.trace_detail = unrelated_detail
    html = client.get(f'/find/trace/{trace_id}').text
    assert 'messages could not be loaded' in html


def test_trace_route_explains_missing_messages(explorer_client) -> None:
    store, source, client = explorer_client
    _load(client)
    store.trace_detail = lambda _id: _none()
    html = client.get(f'/find/trace/{source.rows[0].trace_id}').text
    assert 'messages could not be loaded' in html


async def _none() -> None:
    return None


def test_ask_within_results_keeps_filters_and_uses_loaded_rows(explorer_client) -> None:
    store, source, client = explorer_client
    _load(client, facet_model='gpt-5.6-luna')
    captured: dict[str, Any] = {}

    async def compile(request: Any, *, wait: bool = True, traces: Any = None) -> Any:
        captured['request'] = request
        captured['traces'] = traces
        return store.snapshot_value

    store.compile = compile
    response = client.post(
        '/find/run',
        data=csrf_data({
            'query': 'frustrated users',
            'scope': 'within',
            'mode': 'review',
            'window_days': '7',
            'limit': '200',
            'parallelism': '10',
        }),
    )

    assert response.status_code == 200
    assert captured['traces'] is not None
    view = asyncio.run(store.explorer.view())
    population = captured['request'].population
    assert (population.start, population.end) == (view.start, view.end)
    assert population.facets == view.facets
    assert population.facets.model == frozenset({'gpt-5.6-luna'})
    assert population.numeric == view.numeric
    assert population.limit == 250
    assert len(source.calls) == 1
    assert [row.trace_id for row in view.rows] == [row.trace_id for row in source.rows]


def test_ask_new_search_uses_calendar_range(explorer_client) -> None:
    store, _, client = explorer_client
    captured: dict[str, Any] = {}

    async def compile(request: Any, *, wait: bool = True, traces: Any = None) -> Any:
        captured['request'] = request
        return store.snapshot_value

    store.compile = compile
    response = client.post(
        '/find/run',
        data=csrf_data({
            'query': 'frustrated users',
            'scope': 'new',
            'mode': 'review',
            'from': '2026-09-27',
            'from_time': '10:00:05',
            'to': '2026-09-27',
            'to_time': '11:30:09',
            'tz_offset': '-120',
            'window_days': '7',
            'limit': '200',
            'parallelism': '10',
        }),
    )

    assert response.status_code == 200
    assert captured['request'].population.start == datetime(2026, 9, 27, 8, 0, 5, tzinfo=timezone.utc)
    assert captured['request'].population.end == datetime(2026, 9, 27, 9, 30, 9, tzinfo=timezone.utc)


def test_ask_within_results_without_rows_explains(explorer_client) -> None:
    _, _, client = explorer_client
    from evaluatorq.trace_finder import FacetCatalogue

    client.app.state.finder_catalogue_cache = (
        datetime.now(timezone.utc) + timedelta(minutes=2),
        7,
        FacetCatalogue(model=('catalog-model',)),
    )
    response = client.post(
        '/find/run',
        data=csrf_data({
            'query': 'x',
            'scope': 'within',
            'mode': 'review',
            'window_days': '7',
            'limit': '200',
            'parallelism': '10',
        }),
    )
    assert 'Load traces first' in response.text
    assert 'catalog-model' in response.text


def test_hero_has_the_scope_toggle_and_defaults_to_review() -> None:
    from evaluatorq.dashboard.trace_finder.views import hero

    html = hero('', 'review', api_available=True, has_rows=True)
    assert 'name="scope" value="within" form="finder-query-form" checked' in html
    assert 'value="review" form="finder-query-form" checked' in html


def test_large_within_run_is_forced_through_review(explorer_client) -> None:
    store, source, client = explorer_client
    source.rows = _rows(600)
    _load(client, rows='600')
    captured: dict[str, Any] = {}

    async def compile(request: Any, *, wait: bool = True, traces: Any = None) -> Any:
        captured['request'] = request
        return store.snapshot_value

    store.compile = compile
    client.post(
        '/find/run',
        data=csrf_data({
            'query': 'x',
            'scope': 'within',
            'mode': 'immediate',
            'window_days': '7',
            'limit': '200',
            'parallelism': '10',
        }),
    )
    assert captured['request'].mode == 'review'
    assert captured['request'].population.limit == 600


def test_match_column_appears_only_with_results_and_can_be_hidden(explorer_client) -> None:
    import dataclasses

    from evaluatorq.dashboard.trace_finder.explorer_views import table
    from evaluatorq.trace_finder.columns import resolve_columns
    from evaluatorq.trace_finder.models import TraceClassification

    store, _, client = explorer_client
    _load(client)
    view = asyncio.run(store.explorer.view())
    assert '>Match<' not in table(view, resolve_columns(None), None)
    first = view.rows[0].trace_id
    snapshot = dataclasses.replace(
        store.snapshot_value,
        results={first: TraceClassification(trace_id=first, span_id='s', matched=True, raw_result={})},
        within_results=True,
    )
    assert '>Match<' in table(view, resolve_columns(None), snapshot)
    assert '>Match<' not in table(view, resolve_columns(['status', 'model']), snapshot)


def test_generated_filter_chip_carries_ai_badge() -> None:
    from evaluatorq.dashboard.trace_finder.views import _facet_chips
    from evaluatorq.trace_finder.models import FacetSelection

    html = _facet_chips(
        FacetSelection(model=frozenset({'gpt-5.6-luna', 'claude-sonnet-5'})),
        generated=FacetSelection(model=frozenset({'gpt-5.6-luna'})),
    )
    assert html.count('ai-badge') == 1
