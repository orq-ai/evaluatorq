"""The explorer table's column registry. Adding a column means adding one entry here."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.common.reports import esc
from evaluatorq.common.reports import fmt_cost as shared_fmt_cost

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from datetime import datetime

    from .models import TraceClassification
    from .rows import TraceRow

DASH = '—'


def fmt_tokens(value: int | None) -> str:
    if value is None:
        return DASH
    return (
        f'{value / 1_000_000:.1f}M' if value >= 1_000_000 else f'{value / 1000:.1f}k' if value >= 1000 else str(value)
    )


def fmt_cost(value: float | None, currency: str | None) -> str:
    if value is None:
        return DASH
    if currency and currency.upper() != 'USD':
        return f'{value:,.4f} {esc(currency)}'
    return shared_fmt_cost(value)


def fmt_duration(value: int | None) -> str:
    if value is None:
        return DASH
    return f'{value / 1000:.1f}s' if value >= 1000 else f'{value}ms'


def fmt_time(value: datetime | None) -> str:
    return value.strftime('%Y-%m-%d %H:%M:%S') if value is not None else DASH


def _text(value: str | None) -> str:
    return esc(value) if value else DASH


def _dash(_row: TraceRow) -> str:
    return DASH


def _status(row: TraceRow) -> str:
    if row.status is None:
        return DASH
    kind = 'err' if row.is_error else 'ok'
    label = 'Error' if row.is_error else 'Success' if row.status.lower() in {'ok', 'success'} else row.status
    return f'<span class="status-label {kind}"><span class="dot {kind}" aria-hidden="true"></span>{esc(label)}</span>'


def _time(row: TraceRow) -> str:
    if row.started_at is None:
        return DASH
    return f'{row.started_at:%H:%M:%S}<small>{row.started_at:%d %b %Y}</small>'


def _trace(row: TraceRow) -> str:
    name = row.name or row.trace_id
    agent = row.agent_name or '—'
    return f'<span class="trace-name">{esc(name)}</span><small>{esc(agent)} · {esc(row.trace_id[:8])}</small>'


def _cache(row: TraceRow) -> str:
    pct = row.cache_pct
    if pct is None:
        return DASH
    return (
        f'<span class="cache-value" title="{row.cached_tokens:,} cache-read tokens / {row.tokens_in:,} input tokens">'
        f'<span class="cachebar"><i style="width:{pct * 100:.0f}%"></i></span>'
        f'<span class="cache-percent">{pct * 100:.0f}%</span></span>'
    )


def _error_message(row: TraceRow) -> str:
    if not row.is_error:
        return DASH
    return '<span title="Open this trace, then choose Spans">Open Spans for details</span>'


@dataclass(frozen=True)
class Column:
    key: str
    label: str
    value: Callable[[TraceRow], object]
    render: Callable[[TraceRow], str]
    numeric: bool = False
    default: bool = False


MATCH = 'match'

_ENTRIES = (
    Column('started', 'Time', lambda r: r.started_at, _time, default=True),
    Column('trace', 'Trace / agent', lambda r: r.name or r.trace_id, _trace, default=True),
    Column('status', 'Status', lambda r: r.status, _status, default=True),
    Column('error_message', 'Error details', lambda _r: None, _error_message),
    Column('name', 'Name', lambda r: r.name, lambda r: _text(r.name)),
    Column('agent', 'Agent', lambda r: r.agent_name, lambda r: _text(r.agent_name)),
    Column(
        'model',
        'Model',
        lambda r: ', '.join(r.display_models) or None,
        lambda r: _text(', '.join(r.display_models)),
        default=True,
    ),
    Column(
        'tokens_in', 'Tokens in', lambda r: r.tokens_in, lambda r: fmt_tokens(r.tokens_in), numeric=True, default=True
    ),
    Column(
        'tokens_out',
        'Tokens out',
        lambda r: r.tokens_out,
        lambda r: fmt_tokens(r.tokens_out),
        numeric=True,
        default=True,
    ),
    Column('cache_pct', 'Cache read %', lambda r: r.cache_pct, _cache, numeric=True, default=True),
    Column(
        'cost', 'Cost', lambda r: r.cost_total, lambda r: fmt_cost(r.cost_total, r.currency), numeric=True, default=True
    ),
    Column(
        'duration',
        'Duration',
        lambda r: r.duration_ms,
        lambda r: fmt_duration(r.duration_ms),
        numeric=True,
        default=True,
    ),
    Column(
        'provider',
        'Provider',
        lambda r: ', '.join(r.display_providers) or None,
        lambda r: _text(', '.join(r.display_providers)),
    ),
    Column('product', 'Product', lambda r: r.product, lambda r: _text(r.product)),
    Column('operation', 'Operation', lambda r: r.operation, lambda r: _text(r.operation)),
    Column(
        'reasoning_tokens',
        'Reasoning tokens',
        lambda r: r.reasoning_tokens,
        lambda r: fmt_tokens(r.reasoning_tokens),
        numeric=True,
    ),
    Column(
        'cache_write_tokens',
        'Cache writes',
        lambda r: r.cache_write_tokens,
        lambda r: fmt_tokens(r.cache_write_tokens),
        numeric=True,
    ),
    Column('session', 'Session', lambda r: r.session_id, lambda r: _text(r.session_id)),
    Column('thread', 'Thread', lambda r: r.thread_id, lambda r: _text(r.thread_id)),
    Column('trace_id', 'Trace ID', lambda r: r.trace_id, lambda r: f'<span class="mono">{esc(r.trace_id)}</span>'),
    Column(MATCH, 'AI match', lambda r: None, _dash, default=True),  # noqa: ARG005
)
COLUMNS: Mapping[str, Column] = MappingProxyType({column.key: column for column in _ENTRIES})
DEFAULT_COLUMNS: tuple[str, ...] = tuple(column.key for column in _ENTRIES if column.default)
if len(COLUMNS) != len(_ENTRIES):
    raise RuntimeError('explorer column keys must be unique')


def resolve_columns(keys: Sequence[str] | None) -> tuple[Column, ...]:
    """Return chosen columns in order; ``None`` selects the defaults."""
    chosen = DEFAULT_COLUMNS if keys is None else keys
    unknown = [key for key in chosen if key not in COLUMNS]
    if unknown:
        logger.warning('Ignoring unknown explorer column(s) {}', ', '.join(unknown))
    return tuple(COLUMNS[key] for key in chosen if key in COLUMNS)


def _match_value(result: TraceClassification | None) -> object:
    if result is None or result.error:
        return None
    return (result.matched, tuple(str(answer.value) for answer in result.answers))


def sort_rows(
    rows: Sequence[TraceRow],
    key: str,
    *,
    descending: bool,
    results: Mapping[str, TraceClassification] | None = None,
) -> tuple[TraceRow, ...]:
    """Sort rows by a column, keeping missing values last in either direction.

    Match sorting uses ``results`` and puts matches first when descending. Without results, input
    order is preserved.
    """
    column = COLUMNS.get(key)
    if column is None or (key == MATCH and not results):
        return tuple(rows)
    value = (lambda row: _match_value(results.get(row.trace_id))) if key == MATCH and results else column.value
    present = [row for row in rows if value(row) is not None]
    missing = [row for row in rows if value(row) is None]
    present.sort(key=value, reverse=descending)  # pyright: ignore[reportArgumentType, reportCallIssue]
    return (*present, *missing)
