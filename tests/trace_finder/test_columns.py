from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

import pytest

from evaluatorq.trace_finder.columns import COLUMNS, DEFAULT_COLUMNS, MATCH, resolve_columns, sort_rows
from evaluatorq.trace_finder.models import TraceClassification
from evaluatorq.trace_finder.rows import TraceRow


def test_defaults_are_the_agreed_set() -> None:
    assert DEFAULT_COLUMNS == ('started', 'status', 'trace', 'model', 'tokens_in', 'tokens_out', 'cache_pct', 'cost', 'duration', 'match')
    assert 'name' not in DEFAULT_COLUMNS
    assert 'provider' not in DEFAULT_COLUMNS


def test_registry_is_frozen() -> None:
    with pytest.raises(TypeError):
        cast('Any', COLUMNS)['x'] = COLUMNS['status']


def test_trace_column_combines_name_and_agent_and_escapes_text() -> None:
    row = TraceRow(trace_id='trace<&', name='refund<&', agent_name='support<&')
    rendered = COLUMNS['trace'].render(row)
    assert 'refund&lt;&amp;' in rendered
    assert 'support&lt;&amp;' in rendered
    assert 'trace&lt;&amp;' in rendered
    assert '<small>support<&</small>' not in rendered


def test_model_column_shows_served_model_before_summary_model() -> None:
    row = TraceRow(
        trace_id='served', models=('gpt-5.6-terra',), response_models=('openai/gpt-5.6-terra',)
    )
    column = COLUMNS['model']

    assert column.render(row) == 'openai/gpt-5.6-terra'
    assert column.value(row) == 'openai/gpt-5.6-terra'
    assert row.models == ('gpt-5.6-terra',)


def test_provider_column_uses_model_prefix_when_summary_omits_provider() -> None:
    row = TraceRow(trace_id='anthropic', models=('anthropic/claude-sonnet-5',))
    column = COLUMNS['provider']

    assert column.render(row) == 'anthropic'
    assert column.value(row) == 'anthropic'
    assert row.providers == ()

    explicit = row.model_copy(update={'providers': ('router',)})
    assert column.render(explicit) == 'router'


def test_every_column_renders_on_a_sparse_row() -> None:
    row = TraceRow(trace_id='t')
    for column in COLUMNS.values():
        html = column.render(row)
        assert isinstance(html, str)
    assert COLUMNS[MATCH].value(row) is None
    assert COLUMNS['tokens_in'].render(row) == '—'
    assert COLUMNS['cache_pct'].render(row) == '—'
    assert COLUMNS['cost'].render(row) == '—'


def test_cache_column_names_and_displays_cache_reads() -> None:
    row = TraceRow(trace_id='t', tokens_in=1000, cached_tokens=250, cache_write_tokens=700)
    assert COLUMNS['cache_pct'].label == 'Cache read %'
    assert '250 cache-read tokens / 1,000 input tokens' in COLUMNS['cache_pct'].render(row)
    assert 'width:25%' in COLUMNS['cache_pct'].render(row)


def test_optional_error_message_column_is_honest_without_summary_message() -> None:
    assert 'error_message' not in DEFAULT_COLUMNS
    error = TraceRow(trace_id='failed', status='error')
    success = TraceRow(trace_id='ok', status='ok')

    assert COLUMNS['error_message'].label == 'Error details'
    assert COLUMNS['error_message'].render(error) == (
        '<span title="Open this trace, then choose Spans">Open Spans for details</span>'
    )
    assert COLUMNS['error_message'].render(success) == '—'
    assert COLUMNS['error_message'].value(error) is None


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
