from __future__ import annotations

from datetime import datetime, timezone

import pytest

from evaluatorq.trace_finder.columns import COLUMNS, DEFAULT_COLUMNS, MATCH, resolve_columns, sort_rows
from evaluatorq.trace_finder.models import TraceClassification
from evaluatorq.trace_finder.rows import TraceRow


def test_defaults_are_the_agreed_set() -> None:
    assert DEFAULT_COLUMNS == ('status', 'started', 'agent', 'model', 'tokens_in', 'tokens_out', 'cache_pct', 'cost', 'duration', 'match')
    assert 'name' not in DEFAULT_COLUMNS
    assert 'provider' not in DEFAULT_COLUMNS


def test_registry_is_frozen() -> None:
    with pytest.raises(TypeError):
        COLUMNS['x'] = COLUMNS['status']  # pyright: ignore[reportIndexIssue]


def test_every_column_renders_on_a_sparse_row() -> None:
    row = TraceRow(trace_id='t')
    for column in COLUMNS.values():
        html = column.render(row)
        assert isinstance(html, str)
    assert COLUMNS['tokens_in'].render(row) == '—'
    assert COLUMNS['cache_pct'].render(row) == '—'
    assert COLUMNS['cost'].render(row) == '—'


def test_resolve_drops_unknown_keys() -> None:
    assert [column.key for column in resolve_columns(['model', 'nope'])] == ['model']
    assert [column.key for column in resolve_columns(None)] == list(DEFAULT_COLUMNS)


def test_sort_puts_missing_values_last_both_directions() -> None:
    rows = [TraceRow(trace_id='a', tokens_in=5), TraceRow(trace_id='b'), TraceRow(trace_id='c', tokens_in=9)]
    assert [row.trace_id for row in sort_rows(rows, 'tokens_in', descending=True)] == ['c', 'a', 'b']
    assert [row.trace_id for row in sort_rows(rows, 'tokens_in', descending=False)] == ['a', 'c', 'b']


def test_sort_by_match_puts_matches_first_and_unclassified_last() -> None:
    rows = [TraceRow(trace_id='a'), TraceRow(trace_id='b'), TraceRow(trace_id='c')]
    results = {
        'a': TraceClassification(trace_id='a', span_id='s', matched=False, raw_result={}),
        'c': TraceClassification(trace_id='c', span_id='s', matched=True, raw_result={}),
    }
    assert [row.trace_id for row in sort_rows(rows, MATCH, descending=True, results=results)] == ['c', 'a', 'b']
    assert [row.trace_id for row in sort_rows(rows, MATCH, descending=True)] == ['a', 'b', 'c']


def test_sort_by_time() -> None:
    early = TraceRow(trace_id='e', started_at=datetime(2026, 9, 1, tzinfo=timezone.utc))
    late = TraceRow(trace_id='l', started_at=datetime(2026, 9, 2, tzinfo=timezone.utc))
    assert [row.trace_id for row in sort_rows([early, late], 'started', descending=True)] == ['l', 'e']
