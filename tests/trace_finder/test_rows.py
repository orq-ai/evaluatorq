"""Token normalisation and table rows from Orq trace-search summaries."""

from __future__ import annotations

from datetime import datetime, timezone

from orq_ai_sdk.models import TraceUsage

from evaluatorq.trace_finder.rows import TraceRow, normalise_usage, row_from_summary


def test_inclusive_router_usage_on_cache_write() -> None:
    usage = normalise_usage(TraceUsage(prompt_tokens=9121, completion_tokens=40, total_tokens=9161, prompt_cache_creation_tokens=9119, prompt_cached_tokens=0))
    assert usage.tokens_in == 9121
    assert usage.cached == 0
    assert usage.exclusive is False


def test_inclusive_router_usage_on_cache_read() -> None:
    usage = normalise_usage(TraceUsage(prompt_tokens=9116, completion_tokens=40, total_tokens=9156, prompt_cached_tokens=9114))
    assert usage.tokens_in == 9116
    assert usage.cached == 9114


def test_native_exclusive_usage_adds_cache_tokens_to_input() -> None:
    usage = normalise_usage(TraceUsage(prompt_tokens=1150, completion_tokens=900, total_tokens=2050, prompt_cached_tokens=65_600_000, prompt_cache_creation_tokens=120_000))
    assert usage.tokens_in == 1150 + 65_600_000 + 120_000
    assert usage.exclusive is True


def test_missing_usage_is_unknown_not_zero() -> None:
    usage = normalise_usage(None)
    assert usage.tokens_in is None
    assert usage.tokens_out is None
    assert usage.cached is None


def test_row_from_raw_summary_reads_every_table_field() -> None:
    raw = {
        'trace_id': 'abc123def456',
        'name': 'chat',
        'operation': 'responses',
        'status': 'ok',
        'started_at': '2026-09-27T10:00:01+00:00',
        'duration_ms': 1234,
        'providers': ['anthropic'],
        'models': ['claude-sonnet-5'],
        'agent': {'id': 'a1', 'name': 'support'},
        'usage': {'prompt_tokens': 1000, 'completion_tokens': 50, 'prompt_cached_tokens': 800},
        'cost': {'total': 0.0123, 'currency': 'USD'},
        'session_id': 's1',
    }
    row = row_from_summary(None, raw)
    assert row is not None
    assert row.trace_id == 'abc123def456'
    assert row.started_at == datetime(2026, 9, 27, 10, 0, 1, tzinfo=timezone.utc)
    assert row.agent_name == 'support'
    assert row.models == ('claude-sonnet-5',)
    assert row.tokens_in == 1000
    assert row.cache_pct == 0.8
    assert row.cost_total == 0.0123
    assert row.currency == 'USD'
    assert row.raw['session_id'] == 's1'


def test_row_without_usage_or_cost_has_unknown_values() -> None:
    row = row_from_summary(None, {'trace_id': 't', 'started_at': '2026-09-27T10:00:00Z'})
    assert row is not None
    assert row.tokens_in is None
    assert row.cache_pct is None
    assert row.cost_total is None
    assert row.started_at is not None


def test_row_without_trace_id_is_dropped() -> None:
    assert row_from_summary(None, {'name': 'orphan'}) is None


def test_trace_row_is_frozen() -> None:
    row = TraceRow(trace_id='t')
    try:
        row.trace_id = 'x'  # pyright: ignore[reportAttributeAccessIssue]
    except Exception:  # noqa: BLE001
        return
    raise AssertionError('TraceRow must be frozen')
