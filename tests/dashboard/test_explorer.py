from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from evaluatorq.dashboard.trace_finder import explorer_views
from evaluatorq.trace_finder.columns import resolve_columns
from evaluatorq.trace_finder.explorer import ExplorerView
from evaluatorq.trace_finder.models import FacetCatalogue, FacetSelection, PopulationRequest, TraceRecord
from evaluatorq.trace_finder.rows import TraceRow

from evaluatorq.dashboard.trace_finder import routes as finder_routes
from evaluatorq.dashboard.app import build_app
from evaluatorq.trace_finder.explorer import ExplorerStore
from tests.dashboard.test_finder import FakeStore, csrf_data


def test_parse_range_applies_browser_offset() -> None:
    start, end = finder_routes.parse_range('2026-09-27T10:00:05', '2026-09-27T12:00:00', '-120')
    assert start == datetime(2026, 9, 27, 8, 0, 5, tzinfo=timezone.utc)
    assert end == datetime(2026, 9, 27, 10, 0, 0, tzinfo=timezone.utc)


def test_parse_range_uses_each_endpoint_offset_across_dst() -> None:
    # Amsterdam leaves summer time on 25 October 2026: getTimezoneOffset is
    # -120 before the change and -60 after it.
    start, end = finder_routes.parse_range(
        '2026-10-24T22:00', '2026-10-25T04:00', '-120', '-120', '-60'
    )
    assert start == datetime(2026, 10, 24, 20, 0, tzinfo=timezone.utc)
    assert end == datetime(2026, 10, 25, 3, 0, tzinfo=timezone.utc)


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
    assert 'name="range_mode" value="relative"' in html
    assert '<details class="xr-exact">' in html
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
    assert 'Trace / agent' in html
    assert 'Time' in html
    assert '<small>support · trace-00</small>' in html
    assert 'Success' in html
    assert 'hx-sync="#explorer-results:replace"' in html

    all_view = explorer_views.results(
        ExplorerView(state='loaded', rows=_rows(2)), resolve_columns(None), records=None, snapshot=None
    )
    conversation_rows = tuple(row.model_copy(update={'thread_id': 'thread-1'}) for row in _rows(2))
    conversation_view = explorer_views.results(
        ExplorerView(
            state='loaded',
            rows=conversation_rows,
            quick_view='conv_longest',
            message_counts={row.trace_id: 1 for row in conversation_rows},
        ),
        resolve_columns(None),
        records=None,
        snapshot=None,
    )
    assert 'data-conv=' not in all_view
    assert 'data-conv=' in conversation_view


def test_loading_rows_show_skeletons_and_explain_a_reached_fetch_cap() -> None:
    loading_html = explorer_views.results(
        ExplorerView(state='loading', rows=_rows(2), limit=2), resolve_columns(None), records=None, snapshot=None
    )
    cap_html = explorer_views.results(
        ExplorerView(state='loaded', rows=_rows(2), limit=2), resolve_columns(None), records=None, snapshot=None
    )

    assert loading_html.count('class="xr-skeleton"') == 4
    assert 'Showing 2 of 2 loaded traces' in cap_html
    assert 'requested cap 2' in cap_html


def test_trajectory_rows_show_duration_status_and_label_missing_messages() -> None:
    row = _rows(1)[0].model_copy(update={'duration_ms': 1000, 'status': 'failed'})
    html = explorer_views.trajectories(ExplorerView(state='loaded', rows=(row,)), {}, None)

    assert 'No messages available' in html
    assert '1.0s' in html
    assert 'tv-status err">Error' in html
    cancelled = explorer_views.trajectories(
        ExplorerView(state='loaded', rows=(_rows(1)[0].model_copy(update={'status': 'cancelled'}),)), {}, None
    )
    unknown = explorer_views.trajectories(
        ExplorerView(state='loaded', rows=(_rows(1)[0].model_copy(update={'status': None}),)), {}, None
    )
    assert 'tv-status other">Cancelled' in cancelled
    assert 'tv-status other">Unknown' in unknown



def test_table_column_widths_are_fixed_per_header_and_stable_across_pages() -> None:
    columns = resolve_columns(['status', 'trace'])
    long_rows = tuple(
        row.model_copy(update={'name': 'a very long trace name that should widen this column quite a bit'})
        for row in _rows(150)
    )
    view = ExplorerView(state='loaded', rows=long_rows)
    html = explorer_views.table(view, columns, None)

    assert html.count('<colgroup>') == 1
    cols = re.findall(r'<col(?: style="width:(\d+)ch")?>', html)
    headers = re.findall(r'<th(?=[ >])[^>]*>', html)
    assert len(cols) == len(headers) == len(columns)

    # The trace column is the flex column: it carries no fixed width, so it absorbs spare room.
    status_width, trace_width = cols
    assert status_width and not trace_width
    assert f'min-width:{status_width}' not in html
    assert re.search(r'min-width:(\d+)ch', html)

    page_two = replace(view, page=1)
    html_page_two = explorer_views.table(page_two, columns, None)
    assert re.findall(r'<col[^>]*>', html_page_two) == re.findall(r'<col[^>]*>', html)


def test_table_shows_escaped_ai_match_text_and_aligns_numeric_columns() -> None:
    from evaluatorq.trace_finder import CompiledQuery, DimensionAnswer, RunSnapshot, TraceClassification, ValueSelection
    from evaluatorq.common.judge import ClassifyQuestion

    dimension = CompiledQuery(
        name='Refund<&',
        task=ClassifyQuestion(
            kind='choice', instructions='Judge it.', criteria={'Refund <&>': 'x', 'other': 'y'}, state={}
        ),
        selection=ValueSelection(kind='values', values=('Refund <&>',)),
    )
    row = TraceRow(trace_id='trace<&', name='refund<&', agent_name='support<&', status='ok', tokens_in=1000)
    result = TraceClassification(
        trace_id=row.trace_id,
        span_id='s',
        answers=(DimensionAnswer(value='Refund <&>', matched=True),),
        matched=True,
        raw_result={},
    )
    html = explorer_views.results(
        ExplorerView(state='loaded', rows=(row,)), resolve_columns(None), records=None,
        snapshot=RunSnapshot(results={row.trace_id: result}, within_results=True, dimensions=(dimension,)),
    )
    assert 'refund&lt;&amp;' in html
    assert 'support&lt;&amp; · trace&lt;&amp;' in html
    assert 'Refund &lt;&amp;&gt;' in html
    assert '<th class="num">' in html
    assert '<td class="num">1.0k</td>' in html


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
    assert '<span class="tv-end">2 msgs · —</span>' in html
    assert '<span class="tv-end">No messages available · —</span>' in html
    assert 'Cache reads' in html
    assert '<small>in</small>' in html
    assert '<small>out</small>' in html


class FakeRowSource:
    def __init__(self, rows: tuple[TraceRow, ...]) -> None:
        self.rows = rows
        self.calls: list[dict[str, Any]] = []
        self.hydrate_calls: list[tuple[str, ...]] = []

    async def search(
        self, start: Any, end: Any, limit: int, *, facets: Any, numeric: Any, on_page: Any = None
    ) -> tuple[TraceRow, ...]:
        self.calls.append({'start': start, 'end': end, 'limit': limit, 'facets': facets, 'numeric': numeric})
        if on_page is not None:
            on_page(self.rows[:limit])
        return self.rows[:limit]

    async def hydrate_rows(self, rows: Any) -> dict[str, Any]:
        self.hydrate_calls.append(tuple(row.trace_id for row in rows))
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
    response = client.get('/traces')
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

    client.get('/traces')
    assert len(source.calls) == 1


def test_initial_traces_respect_saved_project(explorer_client) -> None:
    _, source, client = explorer_client
    client.app.state.finder_settings = client.app.state.finder_settings.model_copy(
        update={'orq_project_id': 'project-selected'}
    )

    assert client.get('/traces').status_code == 200
    assert len(source.calls) == 1
    assert source.calls[0]['facets'].project_id == 'project-selected'


def test_server_warms_traces_and_facets_before_the_first_find_request(explorer_client, monkeypatch: pytest.MonkeyPatch) -> None:
    store, source, client = explorer_client
    started = threading.Event()
    release = threading.Event()
    calls: list[int | None] = []

    async def load_catalogue(_app: Any, window_days: int | None = None) -> FacetCatalogue:
        calls.append(window_days)
        started.set()
        await asyncio.to_thread(release.wait)
        return FacetCatalogue()

    monkeypatch.setattr(store, 'close', store.explorer.close, raising=False)
    monkeypatch.setattr(finder_routes, '_load_catalogue', load_catalogue)
    with client:
        try:
            assert started.wait(timeout=2)
            assert client.get('/traces').status_code == 200
            assert len(source.calls) == 1
            assert calls == [7]
        finally:
            release.set()


@pytest.mark.asyncio
async def test_find_page_warms_facets_without_waiting_or_fetching_twice(explorer_client, monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, client = explorer_client
    app = client.app
    app.state.finder_settings = app.state.finder_settings.model_copy(update={'window_days': 14})
    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[int | None] = []

    async def load_catalogue(runtime: Any, window_days: int | None = None) -> FacetCatalogue:
        calls.append(window_days)
        started.set()
        await release.wait()
        catalogue = FacetCatalogue(project=('warmed-project',))
        runtime.state.finder_catalogue_cache = (
            datetime.now(timezone.utc) + timedelta(minutes=5), window_days, catalogue
        )
        return catalogue

    monkeypatch.setattr(finder_routes, '_load_catalogue', load_catalogue)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://testserver') as http:
        page = await http.get('/traces')
        await asyncio.wait_for(started.wait(), timeout=2)
        assert page.status_code == 200
        assert 'name="window_days" value="7"' in page.text
        assert calls == [7]

        await http.get('/traces')
        assert calls == [7]
        menu_task = asyncio.create_task(http.get('/find/facets?window_days=7'))
        await asyncio.sleep(0)
        assert not menu_task.done()
        assert calls == [7]

        release.set()
        menu = await asyncio.wait_for(menu_task, timeout=2)
        assert menu.status_code == 200
        assert 'warmed-project' in menu.text
        assert calls == [7]

        loaded = await http.post('/find/load', data=csrf_data({
            'from': '2026-09-13',
            'from_time': '10:00:00',
            'to': '2026-09-27',
            'to_time': '10:00:00',
            'tz_offset': '0',
            'rows': '10',
        }))
        await asyncio.sleep(0)
        assert loaded.status_code == 200
        assert 'name="window_days" value="14"' in loaded.text
        assert calls == [7, 14]


def test_find_page_keeps_manually_loaded_range(explorer_client) -> None:
    _, source, client = explorer_client
    _load(client)
    response = client.get('/traces')
    assert len(source.calls) == 1
    assert 'data-utc="2026-09-27T10:00:00"' in response.text
    assert 'data-utc="2026-09-27T11:00:00"' in response.text


def test_load_then_rows_then_sort_then_page(explorer_client) -> None:
    store, source, client = explorer_client
    response = _load(client)
    assert response.status_code == 200
    assert store.reset_calls == 0
    assert source.calls[0]['limit'] == 250
    assert 'finder-body' in response.text
    html = client.get('/find/rows').text
    assert 'Page 1 of 3' in html
    html = client.get('/find/rows?sort=tokens_in&dir=desc').text
    assert html.index('trace-0249') < html.index('trace-0248')
    html = client.get('/find/rows?page=2').text
    assert 'Page 3 of 3' in html


def test_load_keeps_ai_results_and_renders_answers_for_loaded_rows(explorer_client) -> None:
    from evaluatorq.common.judge import ClassifyQuestion
    from evaluatorq.trace_finder import CompiledQuery, DimensionAnswer, RunSnapshot, TraceClassification, ValueSelection

    store, _, client = explorer_client
    result = TraceClassification(
        trace_id='trace-0000',
        span_id='s',
        answers=(DimensionAnswer(value=True, matched=True, summary='Asked for a refund.'),),
        matched=True,
        raw_result={},
    )
    store.snapshot_value = RunSnapshot(
        state='completed',
        results={'trace-0000': result},
        completed=1,
        matched=1,
        within_results=True,
        dimensions=(CompiledQuery(
            task=ClassifyQuestion(kind='noul', instructions='Judge.', state={}),
            selection=ValueSelection(kind='values', values=(True,)),
        ),),
    )

    response = _load(client)
    rows_html = client.get('/find/rows').text

    assert response.status_code == 200
    assert store.reset_calls == 0
    assert store.snapshot_value.results == {'trace-0000': result}
    assert '>yes</span>' in rows_html
    assert '249 not judged' in rows_html
    assert 'outside the rows the question ran on' in rows_html
    assert 'over the AI trace limit' in rows_html
    assert 'no conversation to read, such as embedding calls' in rows_html
    assert '(no conversation)' not in rows_html


def test_rows_route_updates_quick_view_state_and_toolbar(explorer_client) -> None:
    _, _, client = explorer_client
    _load(client)
    html = client.get('/find/rows?quick_view=errors').text
    assert 'aria-pressed="true" hx-get="/find/rows?quick_view=errors"' in html
    assert '<summary class="">Views ▾</summary>' in html
    assert 'No errors found.' in html


def test_reset_refreshes_explorer_results_out_of_band(explorer_client) -> None:
    store, _, client = explorer_client
    store.snapshot_value = _judged_snapshot(matched=1, verdicts={'trace-0000': True})
    _load(client)
    matches = client.get('/find/rows?quick_view=matches').text
    assert 'aria-pressed="true" hx-get="/find/rows?quick_view=matches"' in matches
    response = client.post('/find/reset', data=csrf_data())
    assert response.status_code == 200
    assert '<section id="explorer-results" class="xr" hx-sync="this:replace" hx-include="#finder-scope" hx-swap-oob="true"' in response.text
    assert 'aria-pressed="true" hx-get="/find/rows?quick_view=all"' in response.text

def _judged_snapshot(**kwargs: Any) -> Any:
    """A completed within-results run whose classifier judged the given trace ids (a: match, b: no match)."""
    from evaluatorq.common.judge import ClassifyQuestion
    from evaluatorq.trace_finder import (
        CompiledQuery,
        DimensionAnswer,
        RunSnapshot,
        TraceClassification,
        ValueSelection,
    )

    results = {
        trace_id: TraceClassification(
            trace_id=trace_id,
            span_id=f's-{trace_id}',
            answers=(DimensionAnswer(value='yes' if matched else 'no', matched=matched),),
            matched=matched,
            raw_result={},
        )
        for trace_id, matched in kwargs.pop('verdicts').items()
    }
    dimension = CompiledQuery(
        task=ClassifyQuestion(kind='choice', instructions='Judge.', criteria={'yes': 'x', 'no': 'y'}, state={}),
        selection=ValueSelection(kind='values', values=('yes',)),
    )
    return RunSnapshot(state='completed', within_results=True, results=results, dimensions=(dimension,), **kwargs)

def test_status_line_reports_judged_and_matches_for_current_rows() -> None:
    rows = (TraceRow(trace_id='a'), TraceRow(trace_id='b'), TraceRow(trace_id='c'))
    snapshot = _judged_snapshot(total=2, matched=1, verdicts={'a': True, 'b': False})
    first = explorer_views.results(
        ExplorerView(state='loaded', rows=rows), resolve_columns(None), records=None, snapshot=snapshot
    )
    assert 'Showing 3 of 3 loaded traces · 2 judged, 1 match · ' in first
    assert '1 not judged' in first

    changed = explorer_views.results(
        ExplorerView(state='loaded', rows=(rows[0],)), resolve_columns(None), records=None, snapshot=snapshot
    )
    assert 'Showing 1 of 1 loaded traces · 1 judged, 1 match' in changed


def test_filter_only_run_does_not_report_judged_or_ai_matches() -> None:
    from evaluatorq.trace_finder import RunSnapshot, TraceClassification

    rows = (TraceRow(trace_id='a'), TraceRow(trace_id='b'))
    snapshot = RunSnapshot(
        state='completed',
        within_results=True,
        total=2,
        matched=2,
        results={r.trace_id: TraceClassification(trace_id=r.trace_id, span_id='s', matched=True, raw_result={}) for r in rows},
    )
    view = ExplorerView(state='loaded', rows=rows)
    html = explorer_views.results(view, resolve_columns(None), records=None, snapshot=snapshot)
    assert 'judged' not in html
    assert 'Kept by filters' in html
    assert 'AI matches' not in html


def test_narrowed_status_reports_judged_and_matches_before_not_judged() -> None:
    snapshot = _judged_snapshot(loaded=5, total=2, matched=1, verdicts={'a': True, 'b': False})
    view = ExplorerView(state='loaded', rows=(TraceRow(trace_id='a'), TraceRow(trace_id='b')), narrowed_from=5)
    html = explorer_views.results(view, resolve_columns(None), records=None, snapshot=snapshot)
    assert '2 of 5 loaded traces match the filters · 2 judged, 1 match' in html


def test_rows_route_renders_costly_conversation_view_empty_state(explorer_client) -> None:
    _, _, client = explorer_client
    _load(client)

    html = client.get('/find/rows?quick_view=conv_costly').text

    assert 'aria-pressed="true"' in html
    assert 'hx-get="/find/rows?quick_view=conv_costly"' in html
    assert '<summary class="on">Costliest conversations ▾</summary>' in html
    for key in ('slow', 'costly', 'tokens', 'conv_costly', 'conv_tokens', 'conv_longest'):
        assert f'hx-get="/find/rows?quick_view={key}"' in html
    assert 'Top 10% of conversations by cost · 0 conversations, 0 of 250 loaded traces' in html
    assert 'No conversations in the loaded traces.' in html
    assert 'These views group traces by thread or session id, and none of the loaded traces carry one.' in html


def test_rows_route_hydrates_longest_conversation_view_and_shows_message_status(explorer_client) -> None:
    _, source, client = explorer_client
    _load(client)

    html = client.get('/find/rows?quick_view=conv_longest').text

    # The load warmed the first page; the conversation view hydrates only the rest.
    hydrated = [trace_id for call in source.hydrate_calls for trace_id in call]
    assert hydrated == [row.trace_id for row in source.rows]
    assert 'Top 10% of conversations by messages' in html


def test_rows_route_reports_message_count_hydration_failure(explorer_client) -> None:
    store, _, client = explorer_client

    async def fail(_rows: Any) -> dict[str, Any]:
        raise RuntimeError('hydrator offline')

    store.explorer._hydrate = fail
    _load(client)
    response = client.get('/find/rows?quick_view=conv_longest')

    assert response.status_code == 200
    assert 'Could not count messages: hydrator offline' in response.text


def test_table_omits_conversation_markers_when_sorted() -> None:
    rows = tuple(row.model_copy(update={'thread_id': 'thread-1'}) for row in _rows(2))
    html = explorer_views.results(
        ExplorerView(state='loaded', rows=rows, quick_view='conv_longest', sort='tokens_in', message_counts={r.trace_id: 1 for r in rows}),
        resolve_columns(None), records=None, snapshot=None,
    )
    assert 'data-conv=' not in html


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


def test_match_column_choice_survives_a_columns_edit_made_before_any_ai_run(explorer_client) -> None:
    from evaluatorq.common.judge import ClassifyQuestion
    from evaluatorq.trace_finder import CompiledQuery, DimensionAnswer, RunSnapshot, TraceClassification, ValueSelection

    store, _, client = explorer_client
    _load(client)

    # Before any Ask AI run, the match checkbox is hidden but still present, so a columns
    # edit (as the browser would submit it) keeps 'match' in the posted values.
    menu_html = client.get('/find/rows').text
    assert '<label hidden><input type="checkbox" name="columns" value="match" checked>' in menu_html

    client.post('/find/columns', data=csrf_data({'columns': ['started', 'trace', 'status', 'model', 'match']}))

    # An Ask AI run now produces results.
    result = TraceClassification(
        trace_id='trace-0000',
        span_id='s',
        answers=(DimensionAnswer(value=True, matched=True, summary='Asked for a refund.'),),
        matched=True,
        raw_result={},
    )
    store.snapshot_value = RunSnapshot(
        state='completed',
        results={'trace-0000': result},
        completed=1,
        matched=1,
        within_results=True,
        dimensions=(CompiledQuery(
            task=ClassifyQuestion(kind='noul', instructions='Judge.', state={}),
            selection=ValueSelection(kind='values', values=(True,)),
        ),),
    )

    rows_html = client.get('/find/rows').text
    thead = rows_html.split('<thead>')[1].split('</thead>')[0]
    assert 'AI match' in thead


def test_trajectories_view_hydrates_only_the_visible_page(explorer_client) -> None:
    store, _, client = explorer_client
    calls: list[int] = []
    original = store.explorer._hydrate  # pyright: ignore[reportPrivateUsage]

    async def counting(rows: Any) -> Any:
        calls.append(len(rows))
        return await original(rows)

    store.explorer._hydrate = counting  # pyright: ignore[reportPrivateUsage]
    # The load warms the first page's trajectories, so opening Trajectories hydrates nothing more.
    _load(client)
    html = client.get('/find/rows?view=trajectories').text
    assert calls == [100]
    assert html.count('data-tv-row=') == 100


def test_new_search_results_do_not_change_trajectory_hydration_page(explorer_client) -> None:
    from evaluatorq.trace_finder import TraceClassification

    store, _, client = explorer_client
    calls: list[tuple[str, ...]] = []
    original = store.explorer._hydrate  # pyright: ignore[reportPrivateUsage]

    async def counting(rows: Any) -> Any:
        calls.append(tuple(row.trace_id for row in rows))
        return await original(rows)

    store.explorer._hydrate = counting  # pyright: ignore[reportPrivateUsage]
    _load(client)
    store.snapshot_value = replace(
        store.snapshot_value,
        results={'trace-0249': TraceClassification(trace_id='trace-0249', span_id='s', matched=True, raw_result={})},
        within_results=False,
    )
    html = client.get('/find/rows?view=trajectories').text
    assert len(calls) == 1 and len(calls[0]) == 100
    assert calls[0][0] == 'trace-0000'
    assert html.count('data-tv-row=') == 100


def test_poll_appends_explorer_results_during_classification(explorer_client) -> None:
    store, _, client = explorer_client
    _load(client, facet_model='poll-model')
    store.snapshot_value = replace(store.snapshot_value, state='classifying')
    html = client.get('/find/poll').text
    assert 'id="explorer-results"' in html
    assert 'hx-swap-oob="true"' in html
    assert html.count('id="finder-controls"') == 1
    assert 'name="facet_model" value="poll-model"' in html


def test_traces_filters_button_opens_the_facet_menu_directly(explorer_client) -> None:
    from evaluatorq.dashboard import styles

    _, _, client = explorer_client
    html = client.get('/traces').text
    css = styles._FINDER_CSS  # pyright: ignore[reportPrivateUsage]
    js = Path('src/evaluatorq/dashboard/static/dashboard.js').read_text()
    assert html.count('id="finder-controls"') == 1
    assert 'data-explorer-filters aria-haspopup="true" aria-expanded="false">Filters</button>' in html
    assert '.finder-command ~ #finder-body .finder-controls .add { display:none; }' in css
    assert 'else if (menu) {' in js
    assert 'openFacetMenu(menu, filtersButton);' in js
    assert 'if (filtersDirty) { filtersDirty = false; loadExplorer(); }' in js
    assert "window.htmx.trigger(menu, 'refreshFacets')" in js


def test_traces_facet_menu_submits_and_counts_the_current_loaded_rows(
    explorer_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def load_catalogue(_app: Any, _window_days: int | None = None) -> FacetCatalogue:
        return FacetCatalogue(model=('gpt-5.6-luna',))

    monkeypatch.setattr(finder_routes, '_load_catalogue', load_catalogue)
    _, source, client = explorer_client
    source.rows = _rows(2)
    _load(client)

    html = client.get('/traces').text

    assert 'Counts are of the 2 rows currently loaded' in html
    assert 'Loading facet values…' not in html
    assert 'form="explorer-load-form" type="checkbox" name="facet_model" value="gpt-5.6-luna"' in html
    assert '<span class="facet-n" title="Loaded traces">2</span>' in html
    assert 'id="finder-query-form" class="finder-query finder-command-query" hx-post="/find/run" hx-target="#finder-body" hx-swap="innerHTML" hx-include="#finder-controls"' in html

    refreshed_menu = client.get('/find/facets?form_id=explorer-load-form&counts=loaded').text
    assert 'Counts are of the 2 rows currently loaded' in refreshed_menu
    assert 'Loading facet values…' not in refreshed_menu
    refreshed_checkbox = 'form="explorer-load-form" type="checkbox" name="facet_model" value="gpt-5.6-luna"'
    assert refreshed_checkbox in refreshed_menu
    assert '<span class="facet-n" title="Loaded traces">2</span>' in refreshed_menu
    assert 'form="finder-query-form" type="checkbox" name="facet_model" value="gpt-5.6-luna"' not in refreshed_menu

    unscoped_menu = client.get('/find/facets?form_id=explorer-load-form').text
    assert 'form="finder-query-form" type="checkbox" name="facet_model" value="gpt-5.6-luna"' in unscoped_menu

    _load(client, facet_model='gpt-5.6-luna')
    assert source.calls[-1]['facets'].model == frozenset({'gpt-5.6-luna'})


def test_pending_traces_facet_menu_refreshes_when_loaded_rows_arrive(
    explorer_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def load_catalogue(_app: Any, _window_days: int | None = None) -> FacetCatalogue:
        return FacetCatalogue(model=('gpt-5.6-luna',))

    monkeypatch.setattr(finder_routes, '_load_catalogue', load_catalogue)
    _, source, client = explorer_client
    source.rows = ()
    _load(client)

    pending_menu = client.get('/find/facets?form_id=explorer-load-form&counts=loaded').text
    assert 'Counts are of the 0 rows currently loaded' in pending_menu
    assert 'hx-trigger="refreshFacets"' in pending_menu
    assert 'data-refresh-on-open' in pending_menu

    source.rows = _rows(2)
    _load(client)
    refreshed_menu = client.get(
        '/find/facets?form_id=explorer-load-form&counts=loaded&open=1&facet_model=gpt-5.6-luna'
    ).text

    assert 'Counts are of the 2 rows currently loaded' in refreshed_menu
    assert 'class="finder-facets open"' in refreshed_menu
    assert 'form="explorer-load-form" type="checkbox" name="facet_model" value="gpt-5.6-luna" checked' in refreshed_menu
    assert '<span class="facet-n" title="Loaded traces">2</span>' in refreshed_menu


def test_toolbar_shows_removable_filter_chips_and_count(explorer_client) -> None:
    _, _, client = explorer_client
    html = _load(client, facet_model='gpt-x', tokens_min='100').text
    toolbar = html[html.index('class="xr-toolbar"') :]
    assert '>Filters · 2</button>' in toolbar
    assert 'class="xr-chips"><span class="chip is-editable" data-chip-name="facet_model" data-finder-value="gpt-x">' in toolbar
    assert 'data-finder-remove="tokens_min"' in toolbar


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
    assert 'hx-include="#finder-scope"' in html


def test_loading_traces_preserves_new_search_scope(explorer_client) -> None:
    _, _, client = explorer_client
    loaded = _load(client, scope='new').text
    assert 'name="scope" value="new" form="finder-query-form" checked' in loaded

    polled = client.get('/find/rows?scope=new').text
    assert 'name="scope" value="new" form="finder-query-form" checked' in polled
    assert 'name="scope" value="within" form="finder-query-form" checked' not in polled


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

    result = TraceClassification(trace_id='trace-0000', span_id='s', matched=True, raw_result={})
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
    assert 'hx-get="/find/trace/trace-0000?surface=traces"' in response.text


def test_shared_toolbar_keeps_columns_before_load_and_view_switch_in_both_modes() -> None:
    columns = resolve_columns(None)
    for mode, context in (('table', 'Columns'), ('trajectories', 'Sort')):
        html = explorer_views.results(
            ExplorerView(state='loaded', rows=_rows(1), view=mode), columns, records=None, snapshot=None
        )
        assert html.index('Filters') < html.index('All') < html.index('Errors')
        assert 'AI matches' not in html
        assert html.index(context) < html.index('form="explorer-load-form">Load') < html.index('aria-label="View"')


@pytest.mark.parametrize('state', ['idle', 'loading', 'loaded'])
def test_shared_toolbar_keeps_load_and_view_switch_visible_in_all_states(state: str) -> None:
    html = explorer_views.results(
        ExplorerView(state=state, rows=_rows(1) if state == 'loaded' else (), limit=5),
        resolve_columns(None),
        records=None,
        snapshot=None,
    )
    assert 'form="explorer-load-form">Load' in html
    assert 'aria-label="View"' in html
    assert html.count('class="xr-toolbar"') == 1
    assert html.count('id="explorer-load-form"') == 1
    assert html.index('class="xr-status"') < html.index('class="xr-toolbar"') < html.index('>Filters</button>')


def test_quick_views_filter_only_loaded_population_and_render_empty_state() -> None:
    from evaluatorq.trace_finder import RunSnapshot

    rows = _rows(3)
    rows = (rows[0].model_copy(update={'status': 'error'}), rows[1], rows[2])
    snapshot = _judged_snapshot(verdicts={rows[1].trace_id: True, 'not-loaded': True})
    error_html = explorer_views.results(
        ExplorerView(state='loaded', rows=rows, quick_view='errors'),
        resolve_columns(None),
        records=None,
        snapshot=snapshot,
    )
    assert error_html.count('data-tv-row=') == 1
    match_html = explorer_views.results(
        ExplorerView(state='loaded', rows=rows, quick_view='matches'),
        resolve_columns(None),
        records=None,
        snapshot=snapshot,
    )
    assert 'trace-0001' in match_html
    assert 'trace-0000' not in match_html and 'not-loaded' not in match_html
    assert 'not judged' not in match_html
    empty_html = explorer_views.results(
        ExplorerView(state='loaded', rows=rows, quick_view='matches'),
        resolve_columns(None),
        records=None,
        snapshot=RunSnapshot(results={}, within_results=True),
    )
    assert 'No AI matches found.' in empty_html


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
    row = TraceRow(trace_id='t1', status='failed', cost_total=1.25, currency='EUR&', reasoning_tokens=148)
    html = drawer(TraceDetail(trace=trace, projection=None, classification=None), msg=2, row=row, traces_layout=True)
    assert 'id="msg-2"' in html
    assert html.count('<details class="fd-msg') == 3
    assert 'fd-msg k-assistant on" id="msg-2" data-msg="2" open' in html
    assert 'class="fd-mini"' in html
    assert html.count('data-mini-msg=') == 3
    assert 'tv dot err' in html
    assert '1.2500 EUR&amp;' in html
    assert 'EUR&amp;amp;' not in html
    assert 'fd-msg-meta' in html
    assert '148 reasoning' in html
    assert 'Trace conversation' in html
    assert 'Trace t1' not in html
    assert 'fd-row-head' in html
    assert html.index('fd-verdict') < html.index('fd-thread') < html.index('fd-technical')
    assert '<details class="fd-technical"><summary>Technical details</summary>' in html
    assert '<dt>trace</dt>' in html[html.index('fd-technical') :]
    assert 'Copy trace id' in html



def test_find_drawer_keeps_legacy_layout() -> None:
    from evaluatorq.dashboard.trace_finder.views import drawer
    from evaluatorq.trace_finder import TraceDetail

    trace = TraceRecord(
        schema_version=1,
        trace_id='legacy',
        span_id='span',
        timestamp=datetime(2026, 9, 27, tzinfo=timezone.utc),
        project='p',
        model='gpt-5.6-luna',
        provider='openai',
        status='ok',
        product='chat',
        trace_type='agent',
        messages=({'role': 'user', 'content': 'Hello'},),
    )
    html = drawer(TraceDetail(trace=trace, projection=None, classification=None))

    assert 'Trace legacy' in html
    assert 'Full thread' in html
    assert 'Technical details' not in html
    assert 'fd-traces' not in html


def test_drawer_labels_empty_conversation_state() -> None:
    from evaluatorq.dashboard.trace_finder.views import drawer
    from evaluatorq.trace_finder import TraceDetail

    trace = TraceRecord(
        schema_version=1,
        trace_id='empty',
        span_id='span',
        timestamp=datetime(2026, 9, 27, tzinfo=timezone.utc),
        project='p',
        model='gpt-5.6-luna',
        provider='openai',
        status='ok',
        product='chat',
        trace_type='agent',
        messages=({'role': 'user', 'content': ''},),
    )
    html = drawer(TraceDetail(trace=trace, projection=None, classification=None), traces_layout=True)

    assert 'class="fd-no-messages" role="status"' in html
    assert '<b>No messages available</b>' in html




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
    trace_id = source.rows[0].trace_id

    async def hydrate(rows: Any) -> dict[str, Any]:
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
            for row in rows
            if row.trace_id == trace_id
        }

    store.explorer._hydrate = hydrate
    _load(client)
    store.trace_detail = lambda _id: _none()
    store.snapshot_value = replace(store.snapshot_value, within_results=True)
    view = asyncio.run(store.explorer.view())
    cached = asyncio.run(store.explorer.records([trace_id]))
    assert cached[trace_id] is not None
    asyncio.run(store.explorer.narrow(view.generation, set(), facets=view.facets, numeric=view.numeric))
    html = client.get(f'/find/trace/{trace_id}?msg=1').text
    assert 'id="msg-1"' in html
    assert '<div class="fd-msg-content">hi</div>' in html


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
    trace_id = source.rows[0].trace_id

    async def hydrate(rows: Any) -> dict[str, None]:
        return {row.trace_id: None for row in rows}

    store.explorer._hydrate = hydrate
    asyncio.run(store.explorer.records([trace_id]))
    view = asyncio.run(store.explorer.view())
    asyncio.run(store.explorer.narrow(view.generation, set(), facets=view.facets, numeric=view.numeric))
    store.snapshot_value = replace(store.snapshot_value, within_results=True)
    store.trace_detail = lambda _id: _none()
    html = client.get(f'/find/trace/{trace_id}').text
    assert 'messages could not be loaded' in html


async def _none() -> None:
    return None


def test_ask_within_results_keeps_filters_and_uses_loaded_rows(explorer_client) -> None:
    store, source, client = explorer_client
    _load(client, facet_model='gpt-5.6-luna')
    captured: dict[str, Any] = {}

    async def compile(request: Any, *, wait: bool = True, traces: Any = None, table: Any = None) -> Any:
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
    # The AI trace limit from the form caps the judged rows, not the 250 loaded ones.
    assert population.limit == 200
    assert len(source.calls) == 1
    assert [row.trace_id for row in view.rows] == [row.trace_id for row in source.rows]


def test_ask_new_search_reloads_the_table_with_the_searched_population(explorer_client) -> None:
    store, source, client = explorer_client
    captured: dict[str, Any] = {}

    async def compile(request: Any, *, wait: bool = True, traces: Any = None, table: Any = None) -> Any:
        captured['table'] = table
        return store.snapshot_value

    store.compile = compile
    client.post('/find/run', data=csrf_data({'query': 'jev traces', 'scope': 'new', 'window_days': '7', 'limit': '20', 'parallelism': '10'}))

    assert captured['table'] is not None
    population = PopulationRequest(facets=FacetSelection(model=frozenset({'jev-latest'})), limit=20)
    asyncio.run(captured['table'](population))
    view = asyncio.run(store.explorer.view())
    assert source.calls[-1]['facets'] == population.facets
    assert source.calls[-1]['limit'] == 20
    assert view.facets == population.facets
    assert len(view.rows) == 20


def test_ask_new_search_uses_calendar_range(explorer_client) -> None:
    store, _, client = explorer_client
    captured: dict[str, Any] = {}

    async def compile(request: Any, *, wait: bool = True, traces: Any = None, table: Any = None) -> Any:
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

    async def compile(request: Any, *, wait: bool = True, traces: Any = None, table: Any = None) -> Any:
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
            'limit': '600',
            'parallelism': '10',
        }),
    )
    assert captured['request'].mode == 'review'
    assert captured['request'].population.limit == 600


def test_ai_match_column_appears_after_results_and_can_be_hidden(explorer_client) -> None:
    import dataclasses

    from evaluatorq.dashboard.trace_finder.explorer_views import table
    from evaluatorq.trace_finder.columns import resolve_columns
    from evaluatorq.trace_finder.models import TraceClassification

    store, _, client = explorer_client
    _load(client)
    view = asyncio.run(store.explorer.view())
    initial = table(view, resolve_columns(None), None)
    assert 'AI match' not in initial
    assert 'xr-ai-sparkle' not in initial
    first = view.rows[0].trace_id
    snapshot = dataclasses.replace(
        store.snapshot_value,
        results={first: TraceClassification(trace_id=first, span_id='s', matched=True, raw_result={})},
        within_results=True,
    )
    assert '>✦</span>AI match<' in table(view, resolve_columns(None), snapshot)
    assert '>✦</span>AI match<' not in table(view, resolve_columns(['status', 'model']), snapshot)


def test_generated_filter_chip_carries_ai_badge() -> None:
    from evaluatorq.dashboard.trace_finder.views import _facet_chips
    from evaluatorq.trace_finder.models import FacetSelection

    html = _facet_chips(
        FacetSelection(model=frozenset({'gpt-5.6-luna', 'claude-sonnet-5'})),
        generated=FacetSelection(model=frozenset({'gpt-5.6-luna'})),
    )
    assert html.count('ai-badge') == 1


def test_numeric_chip_formats_total_tokens_with_grouping() -> None:
    from evaluatorq.dashboard.trace_finder.views import _facet_chips
    from evaluatorq.trace_finder.models import NumericFilters

    html = _facet_chips(FacetSelection(), NumericFilters(tokens_min=20_001), removable=True)
    assert '<b>total tokens</b><span class="v">≥ 20,001</span>' in html
    assert 'aria-label="Edit total tokens ≥ 20,001"' in html


def test_narrowed_empty_state_and_status_explain_loaded_row_scope() -> None:
    view = ExplorerView(state='loaded', narrowed_from=200)
    html = explorer_views.results(view, resolve_columns(None), records=None, snapshot=None)
    assert '0 of 200 loaded traces match the filters' in html
    assert 'No loaded traces match.' in html
    assert 'None of the 200 loaded traces meet the filters. Remove a filter, or use New search to look beyond the loaded rows.' in html


def test_facet_menu_calls_numeric_token_bound_total_tokens() -> None:
    from evaluatorq.dashboard.trace_finder.views import facet_menu

    html = facet_menu()
    assert '<span>total tokens</span>' in html
    assert '<div class="hd">total tokens</div>' in html


def test_narrowed_status_accounts_for_traces_without_conversation() -> None:
    from evaluatorq.trace_finder import RunSnapshot

    view = ExplorerView(state='loaded', narrowed_from=200)
    snapshot = RunSnapshot(within_results=True, state='completed', loaded=190)
    html = explorer_views.results(view, resolve_columns(None), records=None, snapshot=snapshot)
    assert '0 of 200 loaded traces match the filters · 10 of the 200 have no conversation' in html


def test_ai_matches_chip_shows_once_the_classifier_has_judged_rows() -> None:
    snapshot = _judged_snapshot(matched=1, verdicts={'a': True, 'b': False})
    html = explorer_views.results(
        ExplorerView(state='loaded', rows=(TraceRow(trace_id='a'), TraceRow(trace_id='b'))),
        resolve_columns(None),
        records=None,
        snapshot=snapshot,
    )
    assert 'AI matches' in html


def test_totals_sum_rows_and_use_inclusive_cache_share() -> None:
    from evaluatorq.trace_finder.explorer import totals
    from evaluatorq.trace_finder.rows import TraceRow

    rows = [
        TraceRow(trace_id='a', status='ok', tokens_in=100, tokens_out=10, cached_tokens=50, cost_total=0.5, duration_ms=100),
        TraceRow(trace_id='b', status='error', tokens_in=300, tokens_out=20, cached_tokens=0, cost_total=1.0, duration_ms=900),
        TraceRow(trace_id='c', tokens_in=None, tokens_out=None, cost_total=None, duration_ms=None),
    ]
    t = totals(rows)
    assert (t.traces, t.errors, t.cost, t.tokens_in, t.tokens_out) == (3, 1, 1.5, 400, 30)
    assert t.cache_share == 50 / 400
    assert (t.p50_ms, t.p95_ms) == (100, 900)


def test_totals_of_no_rows_do_not_divide_by_zero() -> None:
    from evaluatorq.trace_finder.explorer import totals

    t = totals([])
    assert (t.traces, t.cost, t.cache_share, t.p50_ms) == (0, None, None, None)


def test_facet_menu_counts_values_from_loaded_rows_only() -> None:
    from evaluatorq.dashboard.trace_finder.views import facet_menu

    rows = [
        TraceRow(trace_id='a', models=('gpt-6-luna',), providers=('openai',), status='Success'),
        TraceRow(trace_id='b', models=('gpt-6-luna',), providers=('openai',), status='Success'),
        TraceRow(trace_id='c', models=('claude-x',), providers=('anthropic',), status='error'),
    ]
    catalogue = FacetCatalogue(model=('claude-x', 'gpt-6-luna', 'unused'))

    html = facet_menu(catalogue, loaded_rows=rows)

    assert 'Counts are of the 3 rows currently loaded' in html
    # Most common value first, zero for a catalogue value no loaded row carries.
    assert html.index('gpt-6-luna') < html.index('claude-x') < html.index('unused')
    assert '<span class="facet-n" title="Loaded traces">2</span>' in html
    assert '<span class="facet-n" title="Loaded traces">0</span>' in html
    # Without rows the menu is unchanged (the /find surface).
    assert 'class="facet-n"' not in facet_menu(catalogue)


def test_failed_load_offers_retry_and_plain_words_for_an_orq_outage() -> None:
    outage = explorer_views.results(
        ExplorerView(state='failed', error='503: no healthy upstream'), resolve_columns(None), records=None, snapshot=None
    )
    assert 'Orq is temporarily unavailable. Try again.' in outage
    assert '503: no healthy upstream' in outage
    assert 'form="explorer-load-form">Retry</button>' in outage

    other = explorer_views.results(
        ExplorerView(state='failed', error='bad range'), resolve_columns(None), records=None, snapshot=None
    )
    assert 'Load failed: bad range' in other
    assert 'Retry</button>' in other


def test_toolbar_is_split_out_of_the_results_section() -> None:
    html = explorer_views.results(ExplorerView(state='loaded'), resolve_columns(None), records=None, snapshot=None)
    assert 'id="explorer-toolbar" hx-swap-oob="true"' in html
    toolbar, section = explorer_views.split_toolbar(html)
    assert toolbar.startswith('<div id="explorer-toolbar">')
    assert 'xr-toolbar' in toolbar
    assert 'xr-toolbar' not in section


def test_traces_drawer_spans_tab_loads_lazy_and_span_tree_handles_errors_orphans_and_links() -> None:
    from evaluatorq.dashboard.trace_finder.views import drawer, span_tree
    from evaluatorq.trace_finder import TraceDetail

    trace = TraceRecord(
        schema_version=1,
        trace_id='trace-1',
        span_id='root',
        timestamp=datetime(2026, 9, 27, tzinfo=timezone.utc),
        project='p',
        model='gpt-5.6-luna',
        provider='openai',
        status='ok',
        product='chat',
        trace_type='agent',
        messages=({'role': 'user', 'content': 'hello'},),
    )
    html = drawer(TraceDetail(trace=trace, projection=None, classification=None), traces_layout=True)
    assert 'Spans</button>' in html and 'hx-get="/find/trace-spans?trace_id=trace-1"' in html
    assert 'Open Spans to load span details.' in html

    tree = span_tree(
        'trace-1',
        [
            {'span_id': 'root', 'name': 'root call', 'type': 'span.agent', 'status': 'ok', 'duration_ms': 80, 'usage': {'total_tokens': 20}},
            {'span_id': 'bad', 'parent_span_id': 'root', 'name': 'tool call', 'operation': 'search', 'status': 'error', 'status_message': '<failed>', 'duration_ms': 10},
            {'span_id': 'orphan', 'parent_span_id': 'missing', 'name': 'orphan', 'status': 'unknown'},
        ],
        'https://orq.example/workspace/experiments/e',
    )
    assert 'fd-span-first-error' in tree and 'fd-span-message' in tree and '&lt;failed&gt;' in tree
    assert '20 tokens' in tree and '80 ms' in tree and 'orphan' in tree
    assert 'href="https://orq.example/workspace/traces/(trace:trace-1//span:bad)"' in tree
    assert 'onclick="event.stopPropagation()"' in tree
    assert tree.index('</summary><div class="fd-span-children">') < tree.index('tool call')
    assert '<details class="fd-span-node" open>' in tree


def test_span_tree_missing_status_and_unsafe_ids_are_honest() -> None:
    from evaluatorq.dashboard.trace_finder.views import span_tree
    from orq_ai_sdk.models.spansummary import SpanSummary

    tree = span_tree('bad/id', [SpanSummary(span_id='span/id', status='error')])
    assert 'No status message available.' in tree
    assert 'unknown kind' in tree and 'Unnamed span' in tree and 'duration unknown' in tree and 'tokens unknown' in tree
    assert 'Open in Orq' not in tree


def test_span_tree_renders_lazy_raw_error_message_escaped() -> None:
    from evaluatorq.dashboard.trace_finder.views import span_tree

    tree = span_tree('trace', [{'span_id': 'failure', 'status': 'error'}], first_error_message='<raw & detail>')

    assert '&lt;raw &amp; detail&gt;' in tree
    assert '<raw & detail>' not in tree


def test_traces_missing_conversation_drawer_keeps_lazy_spans_tab_and_trace_actions(monkeypatch) -> None:
    from evaluatorq.dashboard.trace_finder.views import missing_trace_drawer

    monkeypatch.setenv('ORQ_WORKSPACE', 'workspace')
    monkeypatch.setenv('ORQ_UI_BASE_URL', 'https://orq.example')
    monkeypatch.setattr('evaluatorq.dashboard.orq_workspace.resolve_slug', lambda: 'workspace')
    html = missing_trace_drawer(
        'trace/with slash',
        reason='<Messages did not hydrate.>',
        traces_layout=True,
    )

    assert 'Conversation unavailable' in html
    assert '&lt;Messages did not hydrate.&gt;' in html
    assert 'No messages available' not in html
    assert 'Spans</button>' in html
    assert 'hx-get="/find/trace-spans?trace_id=trace%2Fwith%20slash"' in html
    assert 'href="https://orq.example/workspace/traces?query=trace_id%3Ais%3Atrace%2Fwith%20slash"' in html
    assert 'Copy trace id' in html and 'data-trace-id="trace/with slash"' in html


def test_span_tree_handles_cycles_and_maximum_pager_depth_without_recursion() -> None:
    from evaluatorq.dashboard.trace_finder.views import span_tree

    chain = [
        {'span_id': f'span-{index}', 'parent_span_id': f'span-{index - 1}' if index else None}
        for index in range(2000)
    ]
    cycle = [
        {'span_id': 'cycle-a', 'parent_span_id': 'cycle-b'},
        {'span_id': 'cycle-b', 'parent_span_id': 'cycle-a'},
    ]
    tree = span_tree('trace', [*chain, *cycle])
    assert tree.count('fd-span-row') == 2002
    assert 'span-1999' in tree and 'cycle-a' in tree and 'cycle-b' in tree
