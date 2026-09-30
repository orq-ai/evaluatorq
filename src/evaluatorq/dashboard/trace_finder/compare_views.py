"""HTML fragments for comparing two currently visible trace summaries."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import quote

from evaluatorq.common.reports import esc
from evaluatorq.trace_finder.columns import fmt_cost, fmt_duration, fmt_time, fmt_tokens

if TYPE_CHECKING:
    from evaluatorq.trace_finder.rows import TraceRow


def _status(row: TraceRow) -> str:
    if row.is_error:
        detail = f' ({esc(row.status)})' if row.status else ''
        return f'Error{detail}'
    if row.status is None:
        return '—'
    if row.status.lower() in {'ok', 'success'}:
        return 'Success'
    return esc(row.status)


def _model(row: TraceRow) -> str:
    model = ', '.join(row.models)
    return esc(model) if model else '—'


def _drawer_link(row: TraceRow) -> str:
    trace_id = quote(row.trace_id, safe='')
    label = row.name or row.trace_id
    return (
        f'<a class="tv-compare-open" href="/find/trace/{trace_id}?surface=traces" '
        f'hx-get="/find/trace/{trace_id}?surface=traces" hx-target="#finder-drawer" '
        f'hx-swap="innerHTML" hx-indicator="#finder-drawer-loading">Open {esc(label)} drawer</a>'
    )


def comparison_panel(first: TraceRow, second: TraceRow) -> str:
    """Render the requested pair in a side-by-side summary with links to each drawer."""
    metrics = (
        ('Status / error', _status(first), _status(second)),
        ('Started', esc(fmt_time(first.started_at)), esc(fmt_time(second.started_at))),
        ('Duration', esc(fmt_duration(first.duration_ms)), esc(fmt_duration(second.duration_ms))),
        ('Model', _model(first), _model(second)),
        ('Input tokens', esc(fmt_tokens(first.tokens_in)), esc(fmt_tokens(second.tokens_in))),
        ('Output tokens', esc(fmt_tokens(first.tokens_out)), esc(fmt_tokens(second.tokens_out))),
        ('Cache reads', esc(fmt_tokens(first.cached_tokens)), esc(fmt_tokens(second.cached_tokens))),
        ('Cost', fmt_cost(first.cost_total, first.currency), fmt_cost(second.cost_total, second.currency)),
    )
    rows = ''.join(
        f'<tr><th scope="row">{label}</th><td>{left}</td><td>{right}</td></tr>' for label, left, right in metrics
    )
    return (
        '<section class="tv-compare" aria-labelledby="tv-compare-title">'
        '<h3 id="tv-compare-title">Trace comparison</h3>'
        '<table><thead><tr><th scope="col">Metric</th>'
        f'<th scope="col">{_drawer_link(first)}</th><th scope="col">{_drawer_link(second)}</th>'
        f'</tr></thead><tbody>{rows}</tbody></table></section>'
    )


def comparison_feedback(message: str) -> str:
    """Render a short selection error as a safe status message."""
    return f'<p class="tv-compare-feedback" role="status">{esc(message)}</p>'
