from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from evaluatorq.dashboard.trace_finder import explorer_views
from evaluatorq.trace_finder.columns import resolve_columns
from evaluatorq.trace_finder.explorer import ExplorerView
from evaluatorq.trace_finder.models import TraceRecord
from evaluatorq.trace_finder.rows import TraceRow

from evaluatorq.dashboard.trace_finder import routes as finder_routes


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

def _rows(n: int) -> tuple[TraceRow, ...]:
    return tuple(TraceRow(trace_id=f'trace-{i:04d}', status='ok', tokens_in=100 + i, tokens_out=10, cached_tokens=50, agent_name='support', models=('gpt-5.6-luna',)) for i in range(n))


def test_range_inputs_include_local_time_fields_and_default_rows() -> None:
    html = explorer_views.range_inputs(datetime(2026, 9, 26, tzinfo=timezone.utc), datetime(2026, 9, 27, tzinfo=timezone.utc), 7)
    assert 'name="from" type="datetime-local"' in html
    assert 'name="to" type="datetime-local"' in html
    assert 'name="tz_offset" form="explorer-load-form"' in html
    assert 'data-explorer-preset="900"' in html
    assert 'name="rows" type="number"' in html
    assert 'value="200"' in html
    assert 'form="explorer-load-form">Load</button>' in html


def test_idle_results_render_an_empty_state() -> None:
    html = explorer_views.results(ExplorerView(), resolve_columns(None), records=None, snapshot=None)
    assert 'id="explorer-results"' in html
    assert 'Load traces' in html


def test_zero_rows_render_an_empty_state() -> None:
    html = explorer_views.results(ExplorerView(state='loaded'), resolve_columns(None), records=None, snapshot=None)
    assert 'No traces match' in html


def test_table_renders_one_row_per_page_row_with_drawer_links() -> None:
    view = ExplorerView(state='loaded', rows=_rows(150))
    html = explorer_views.results(view, resolve_columns(None), records=None, snapshot=None)
    assert html.count('hx-get="/find/trace/') == 100
    assert 'Page 1 of 2' in html
    assert '<th' in html and 'Tokens in' in html


def test_failed_load_shows_banner_and_rows() -> None:
    view = ExplorerView(state='failed', rows=_rows(3), error='page 2 timed out')
    html = explorer_views.results(view, resolve_columns(None), records=None, snapshot=None)
    assert 'page 2 timed out' in html
    assert '3 rows loaded' in html
    assert html.count('hx-get="/find/trace/') == 3


def test_loading_results_poll() -> None:
    html = explorer_views.results(ExplorerView(state='loading', limit=1000, rows=_rows(400)), resolve_columns(None), records=None, snapshot=None)
    assert 'hx-trigger="every 1s"' in html
    assert '400 / 1000' in html


def test_trajectories_draws_a_row_for_a_failed_hydration() -> None:
    rows = _rows(2)
    record = TraceRecord(schema_version=1, trace_id=rows[0].trace_id, span_id='s', timestamp=datetime(2026, 9, 27, tzinfo=timezone.utc),
                         project='p', model='gpt-5.6-luna', provider='openai', status='ok', product='chat', trace_type='agent',
                         messages=({'role': 'user', 'content': 'hello there'}, {'role': 'assistant', 'content': 'hi'}))
    view = ExplorerView(state='loaded', rows=rows, view='trajectories')
    html = explorer_views.results(view, resolve_columns(None), records={rows[0].trace_id: record, rows[1].trace_id: None}, snapshot=None)
    assert html.count('data-tv-row=') == 2
    assert 'tv-nomsg' in html
    assert 'data-tv-msg="1"' in html
    assert '2 msgs' in html
