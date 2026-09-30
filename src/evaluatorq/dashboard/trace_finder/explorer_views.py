"""Explorer results: the trace table, the trajectory bars, and their toolbar and pager."""

from __future__ import annotations

import math
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from urllib.parse import quote

from loguru import logger

from evaluatorq.common.reports import esc
from evaluatorq.trace_finder.columns import COLUMNS, MATCH, Column, fmt_cost, fmt_duration, fmt_time, fmt_tokens
from evaluatorq.trace_finder.explorer import (
    CONVERSATION_METRICS,
    PAGE_ROWS,
    TOP_METRICS,
    ai_matched,
    conversation_key,
    totals,
)
from evaluatorq.trace_finder.orq_source import MAX_LIVE_TRACES
from evaluatorq.trace_finder.trajectory import KIND_LABELS, segments

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from evaluatorq.trace_finder import RunSnapshot
    from evaluatorq.trace_finder.explorer import ExplorerView
    from evaluatorq.trace_finder.models import DimensionAnswer, TraceClassification, TraceRecord
    from evaluatorq.trace_finder.rows import TraceRow
    from evaluatorq.trace_finder.trajectory import Segment

DEFAULT_EXPLORER_ROWS = 200
TRAJECTORIES_HELP = 'Shows each trace as a bar of its messages, sized by estimated tokens'
COLUMN_HELP = {
    'cache_pct': 'Share of input tokens the provider reused from an earlier request, which is cheaper and faster.',
    'tokens_in': 'Tokens sent to the model, including any served from the cache',
    'tokens_out': 'Tokens the model wrote back',
}
TOTAL_HELP = {
    'in': COLUMN_HELP['tokens_in'],
    'out': COLUMN_HELP['tokens_out'],
    'cache read': COLUMN_HELP['cache_pct'],
    'p50': 'Half of the shown traces finished faster than this',
    'p95': 'Nineteen in twenty of the shown traces finished faster than this',
}
PRESETS = (
    ('15m', timedelta(minutes=15)),
    ('1h', timedelta(hours=1)),
    ('24h', timedelta(days=1)),
    ('7d', timedelta(days=7)),
    ('30d', timedelta(days=30)),
)
TOP_QUICK_VIEW_LABELS = {
    'slow': 'Slowest traces',
    'costly': 'Costliest traces',
    'tokens': 'Token-heavy traces',
    'context': 'Largest context',
    'conv_costly': 'Costliest conversations',
    'conv_tokens': 'Token-heavy conversations',
    'conv_longest': 'Longest conversations',
}


def _local_value(value: datetime) -> str:
    # The server renders UTC; dashboard.js rewrites the date and time controls to browser-local time.
    return value.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')


def range_inputs(start: datetime | None, end: datetime | None, window_days: int, *, include_load: bool = True) -> str:
    end = end or datetime.now(timezone.utc)
    start = start or end - timedelta(days=window_days)
    presets = ''.join(
        f'<button type="button" class="xr-preset" data-explorer-preset="{int(span.total_seconds())}" '
        f'aria-pressed="{str(span == timedelta(days=window_days)).lower()}">{label}</button>'
        for label, span in PRESETS
    )
    controls = (
        f'<form id="explorer-load-form" hidden hx-post="/find/load" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace" hx-include="#finder-controls, #finder-scope">{_csrf()}</form>'
        '<input id="explorer-range-mode" type="hidden" name="range_mode" value="relative" form="explorer-load-form" hx-preserve data-explorer-range-mode>'
        f'<input id="explorer-range-seconds" type="hidden" name="range_seconds" value="{window_days * 86400}" form="explorer-load-form" hx-preserve data-explorer-range-seconds>'
        '<input type="hidden" name="tz_offset" form="explorer-load-form" data-explorer-tz>'
        f'<details class="xr-time-menu"><summary><span class="xr-time-label" data-explorer-range-label>Last {window_days} days</span></summary><div class="xr-time-options"><span class="xr-presets" role="group" aria-label="Relative range">{presets}</span>'
        '<details class="xr-exact"><summary>Custom range</summary>'
        f'<span class="xr-range"><b>From</b><input id="explorer-from" class="xr-date" aria-label="From date" hx-preserve form="explorer-load-form" name="from" type="date" required data-utc="{_local_value(start)}" value="{_local_value(start)[:10]}"><input id="explorer-from-time" class="xr-time" aria-label="From time" hx-preserve form="explorer-load-form" name="from_time" type="time" step="1" required data-utc="{_local_value(start)}" value="{_local_value(start)[11:]}"></span>'
        f'<span class="xr-range"><b>To</b><input id="explorer-to" class="xr-date" aria-label="To date" hx-preserve form="explorer-load-form" name="to" type="date" required data-utc="{_local_value(end)}" value="{_local_value(end)[:10]}"><input id="explorer-to-time" class="xr-time" aria-label="To time" hx-preserve form="explorer-load-form" name="to_time" type="time" step="1" required data-utc="{_local_value(end)}" value="{_local_value(end)[11:]}"></span>'
        '<small class="xr-tz" data-explorer-tz-label>Local time</small>'
        '<button type="button" class="xr-apply" data-explorer-apply>Apply range</button></details></div></details>'
        f'<span class="quiet"><b>Rows</b><input id="explorer-rows" hx-preserve form="explorer-load-form" name="rows" type="number" min="1" max="{MAX_LIVE_TRACES}" value="{DEFAULT_EXPLORER_ROWS}" style="width:72px"></span>'
    )
    load = '<button class="btn-secondary" type="submit" form="explorer-load-form">Load</button>' if include_load else ''
    return controls + load


_UNAVAILABLE_MARKERS = ('503', 'no healthy upstream', 'service unavailable')


def _load_failed(error: str | None) -> str:
    """The failed-load panel: plain words for an Orq outage, the raw detail kept, and a Retry that re-submits the Load form."""
    raw = error or 'unknown error'
    unavailable = any(marker in raw.lower() for marker in _UNAVAILABLE_MARKERS)
    message = 'Orq is temporarily unavailable. Try again.' if unavailable else f'Load failed: {raw}'
    detail = f'<small class="finder-error-detail">{esc(raw)}</small>' if unavailable else ''
    return (
        f'<div class="finder-review finder-form-error" role="alert" title="{esc(raw)}">{esc(message)} {detail}'
        '<button class="btn-secondary xr-retry" type="submit" form="explorer-load-form">Retry</button></div>'
    )


def _empty(title: str, body: str) -> str:
    paragraph = f'<p>{esc(body)}</p>' if body else ''
    return f'<div class="xr-empty"><h4>{esc(title)}</h4>{paragraph}</div>'


def _match_names(snapshot: RunSnapshot | None, fallback: str) -> tuple[str, ...]:
    """Header labels for the AI match column: one per classifier dimension, or the column's own label."""
    names = tuple(dimension.name for dimension in (snapshot.dimensions or ())) if snapshot is not None else ()
    return names or (fallback,)


def _match_cells(row: TraceRow, snapshot: RunSnapshot | None) -> str:
    from evaluatorq.dashboard.trace_finder.views import _answer_cells, _result_color

    result = snapshot.results.get(row.trace_id) if snapshot is not None else None
    dimensions = (snapshot.dimensions or ()) if snapshot is not None else ()
    if result is None:
        if snapshot is not None and snapshot.state in {'compiling', 'classifying'}:
            return '<td class="muted xr-pending" title="Waiting for the AI">…</td>' * max(1, len(dimensions))
        if snapshot is not None and snapshot.state == 'completed' and dimensions:
            cell = '<td class="muted" title="Not judged: this trace has no conversation to read.">not judged</td>'
            return cell * len(dimensions)
        return '<td class="muted">—</td>' * max(1, len(dimensions))
    if not result.error and dimensions and all(d.task.kind == 'noul' for d in dimensions):
        return ''.join(
            _yes_no_cell(result.answers[index] if index < len(result.answers) else None)
            for index in range(len(dimensions))
        )
    if result.error or not dimensions:
        label = (
            'Judgment failed'
            if result.error
            else 'Kept by filters'
            if not dimensions and result.matched
            else 'Included'
            if result.matched
            else 'Not included'
        )
        cell = f'<td><span class="verdict xr-match"><span class="sw" style="background:{esc(_result_color(result, dimensions))}"></span>{esc(label)}</span></td>'
        return cell * max(1, len(dimensions))
    return _answer_cells(result, dimensions)


def _yes_no_cell(answer: DimensionAnswer | None) -> str:
    """A compact yes/no verdict; a match carries the classifier's reason when it gave prose."""
    from evaluatorq.dashboard.trace_finder.views import _reason_line

    if answer is None:
        return '<td>—</td>'
    if answer.error:
        return '<td><span class="verdict xr-yn failed">Judgment failed</span></td>'
    title = f' title="AI confidence {answer.confidence:.0%}"' if answer.confidence is not None else ''
    text = 'yes' if answer.value is True else 'no' if answer.value is False else esc(str(answer.value))
    return f'<td><span class="verdict xr-yn {"yes" if answer.matched else "no"}"{title}>{text}</span>{_reason_line(answer)}</td>'


def _within_snapshot(snapshot: RunSnapshot | None) -> RunSnapshot | None:
    return snapshot if snapshot is not None and snapshot.within_results else None


def _ai_has_run(snapshot: RunSnapshot | None) -> bool:
    return snapshot is not None and snapshot.within_results and bool(snapshot.results)


def _classified(snapshot: RunSnapshot | None) -> bool:
    """True when the run asked the classifier something; a filter-only run keeps rows without judging them."""
    return snapshot is not None and bool(snapshot.dimensions)


def _visible_columns(columns: Sequence[Column], snapshot: RunSnapshot | None) -> tuple[Column, ...]:
    if _ai_has_run(snapshot):
        return tuple(columns)
    return tuple(column for column in columns if column.key != MATCH)


def _drawer_attrs(trace_id: str, msg: int | None = None, *, traces_layout: bool = False) -> str:
    params = []
    if msg is not None:
        params.append(f'msg={msg}')
    if traces_layout:
        params.append('surface=traces')
    query = f'?{"&".join(params)}' if params else ''
    return f'hx-get="/find/trace/{quote(trace_id, safe="")}{query}" hx-target="#finder-drawer" hx-swap="innerHTML" hx-indicator="#finder-drawer-loading"'


_MIN_COL_CH = 6
_CELL_PAD_CH = 6  # 11px cell padding either side plus the status dot and bar glyphs
_MAX_COL_CH = 48
# Header text is uppercase and letter-spaced at a smaller font than the body cells, so it runs
# wider per character; this factor approximates that without measuring real glyph metrics.
_HEADER_CHAR_FACTOR = 1.15
_HEADER_ARROW_CH = 3  # room for the sort arrow / ✦ sparkle appended to the header label
_TAG_RE = re.compile(r'<[^>]+>')
_SMALL_SPLIT_RE = re.compile(r'<small[^>]*>|</small>')


def _visible_chars(cell_html: str) -> int:
    """Longest visible line in a rendered cell, HTML tags stripped.

    A ``<small>`` second line (e.g. the date under a time, or the agent under a trace name) is
    treated as its own line so it doesn't just add to the first line's width.
    """
    lines = _SMALL_SPLIT_RE.split(cell_html)
    return max((len(_TAG_RE.sub('', line)) for line in lines), default=0)


def _clamp_ch(chars: float) -> int:
    return max(_MIN_COL_CH, min(_MAX_COL_CH, round(chars)))


def _header_widths(
    columns: Sequence[Column], rows: Sequence[TraceRow], snapshot: RunSnapshot | None
) -> tuple[int, ...]:
    """A fixed width in ``ch`` per rendered header (MATCH can expand into several).

    Widths are computed from every loaded row (``rows``, not just the current page) so paging
    never changes them, and are clamped to a sane range so one long value can't blow up the table.
    """
    widths: list[int] = []
    for column in columns:
        labels = _match_names(snapshot, column.label) if column.key == MATCH else (column.label,)
        if column.key == MATCH:
            per_label_chars = [0] * len(labels)
            for row in rows:
                cells = [cell for cell in _match_cells(row, snapshot).split('</td>') if cell.strip()]
                for index in range(len(labels)):
                    cell = cells[index] if index < len(cells) else (cells[-1] if cells else '')
                    per_label_chars[index] = max(per_label_chars[index], _visible_chars(cell))
            cell_chars = per_label_chars
        else:
            max_chars = max((_visible_chars(column.render(row)) for row in rows), default=0)
            cell_chars = [max_chars] * len(labels)
        for label, chars in zip(labels, cell_chars, strict=True):
            header_chars = len(label) * _HEADER_CHAR_FACTOR + _HEADER_ARROW_CH
            widths.append(_clamp_ch(max(header_chars, chars + _CELL_PAD_CH)))
    return tuple(widths)


def table(
    view: ExplorerView, columns: Sequence[Column], snapshot: RunSnapshot | None, *, traces_layout: bool = False
) -> str:
    snapshot = _within_snapshot(snapshot)
    columns = _visible_columns(columns, snapshot)
    results = snapshot.results if snapshot is not None and snapshot.results else None
    heads = ''
    for column in columns:
        sorted_here = view.sort == column.key
        arrow = (
            f'<span class="xr-sort-arrow" aria-hidden="true">{"↓" if view.descending else "↑"}</span>'
            if sorted_here
            else ''
        )
        aria_sort = f' aria-sort="{"descending" if view.descending else "ascending"}"' if sorted_here else ''
        direction = 'asc' if sorted_here and view.descending else 'desc'
        labels = _match_names(snapshot, column.label) if column.key == MATCH else (column.label,)
        sparkle = '<span class="xr-ai-sparkle" aria-hidden="true">✦</span>' if column.key == MATCH else ''
        help_text = COLUMN_HELP.get(column.key)
        title = f' title="{esc(help_text)}"' if help_text else ''
        help_mark = '<span class="xr-help" aria-hidden="true">?</span>' if help_text else ''
        heads += ''.join(
            f'<th class="{"num" if column.numeric else ""}"{aria_sort}><button type="button" class="link" data-xr-sort="{column.key}"{title} '
            f'hx-get="/find/rows?sort={column.key}&dir={direction}" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">'
            f'{sparkle}{esc(label)}{help_mark}{arrow}</button></th>'
            for label in labels
        )
    if not columns:
        return _empty('No columns selected.', 'Choose at least one column from Columns to show trace details.')
    widths = _header_widths(columns, view.rows, snapshot)
    # One text column takes no fixed width so it absorbs the card's spare room; the table's
    # min-width keeps every column at least as wide as its measured content.
    rendered_keys = [
        c.key for c in columns for _ in (_match_names(snapshot, c.label) if c.key == MATCH else (c.label,))
    ]
    flex_key = next((key for key in ('trace', 'name', 'agent') if key in rendered_keys), None)
    colgroup = ''.join(
        '<col>' if key == flex_key else f'<col style="width:{width}ch">'
        for key, width in zip(rendered_keys, widths, strict=True)
    )
    conversation = None
    body = ''
    for row in view.page_rows(results):
        key = conversation_key(row)
        marker = (
            f' data-conv="{esc(key)}"'
            if view.quick_view in CONVERSATION_METRICS and view.sort is None and key is not None and key != conversation
            else ''
        )
        row_classes = (['xr-err'] if row.is_error else []) + (
            ['xr-hit']
            if results is not None and _classified(snapshot) and ai_matched(results.get(row.trace_id))
            else []
        )
        row_class = f' class="{" ".join(row_classes)}"' if row_classes else ''
        body += (
            f'<tr data-tv-row="{esc(row.trace_id)}"{marker}{row_class} tabindex="0" {_drawer_attrs(row.trace_id, traces_layout=traces_layout)}>'
            + ''.join(
                _match_cells(row, snapshot)
                if c.key == MATCH
                else f'<td class="{"num" if c.numeric else ""}">{c.render(row)}</td>'
                for c in columns
            )
            + '</tr>'
        )
        conversation = key
    if view.state == 'loading':
        body += ''.join(
            '<tr class="xr-skeleton" aria-hidden="true">' + ''.join('<td><i></i></td>' for _ in rendered_keys) + '</tr>'
            for _ in range(4)
        )
    return (
        f'<table class="finder-table xr-table" style="min-width:{sum(widths)}ch"><colgroup>{colgroup}</colgroup>'
        f'<thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table>'
    )


def _tip_attrs(segment: Segment, position: int, count: int) -> str:
    label = f' data-tv-tool="{esc(segment.label)}"' if segment.label and segment.kind in {'call', 'result'} else ''
    return (
        f'data-tv-msg="{segment.index}" data-tv-kind="{esc(KIND_LABELS[segment.kind])}" data-tv-n="{position} / {count}" '
        f'data-tv-tok="~{segment.tokens:,} tokens" data-tv-p="{esc(segment.preview)}"{label}'
    )


def _identity(row: TraceRow, snapshot: RunSnapshot | None) -> str:
    raw_status = (row.status or 'unknown').casefold()
    status = 'err' if row.is_error else 'ok' if raw_status in {'ok', 'success'} else 'other'
    status_label = (
        'Error'
        if row.is_error
        else 'Success'
        if status == 'ok'
        else (row.status or 'Unknown').replace('_', ' ').title()
    )
    model = f'<span class="pill">{esc(row.models[0])}</span>' if row.models else ''
    tick = ''
    result = snapshot.results.get(row.trace_id) if snapshot is not None else None
    if result is not None:
        from evaluatorq.dashboard.trace_finder.views import _result_color

        tick = f'<span class="tv-tick" style="background:{esc(_result_color(result, snapshot.dimensions if snapshot else None))}"></span>'
    return (
        f'<div class="tv-id"><span class="dot {status}"></span>{tick}<div class="t">'
        f'<div class="a">{esc(row.agent_name or row.name or "—")}{model}'
        f'<span class="tv-status {status}">{esc(status_label)}</span></div>'
        f'<div class="s mono">{esc(fmt_time(row.started_at))} · {esc(row.trace_id[:8])}</div></div></div>'
    )


def _metrics(row: TraceRow) -> str:
    pct = row.cache_pct
    cache = (
        f'<span class="cc" title="{row.cached_tokens:,} cache-read tokens / {row.tokens_in:,} input tokens">'
        f'<b>{pct * 100:.0f}%</b><span class="cache-track"><i style="width:{pct * 100:.0f}%"></i></span></span>'
        if pct is not None
        else '<span class="cc muted">—</span>'
    )
    return (
        f'<div class="tv-m"><span class="io"><span><b>{fmt_tokens(row.tokens_in)}</b><small>in</small></span>'
        f'<span><b>{fmt_tokens(row.tokens_out)}</b><small>out</small></span></span>'
        f'{cache}<span class="tv-cost">{fmt_cost(row.cost_total, row.currency)}</span></div>'
    )


def trajectories(
    view: ExplorerView,
    records: Mapping[str, TraceRecord | None],
    snapshot: RunSnapshot | None,
    *,
    loading: bool = False,
    traces_layout: bool = False,
) -> str:
    snapshot = _within_snapshot(snapshot)
    page = view.page_rows(snapshot.results if snapshot is not None and snapshot.results else None)
    bars = {row.trace_id: segments(record.messages) if (record := records.get(row.trace_id)) else None for row in page}
    other = sum(1 for segs in bars.values() if segs for s in segs if s.kind == 'other')
    if other:
        logger.warning('Trajectory view drew {} message part(s) of unknown type as "other"', other)
    # Input usage is available for every loaded summary row, even when message
    # hydration is limited to the current page. Use its nearest-rank p95 so the
    # axis does not jump as the user pages through the loaded population.
    loaded_inputs = sorted(row.tokens_in for row in view.rows if row.tokens_in is not None)
    scale = loaded_inputs[max(0, math.ceil(len(loaded_inputs) * 0.95) - 1)] if loaded_inputs else 1
    scale = max(scale, 1)
    totals: dict[str, int] = {}
    for segs in bars.values():
        for s in segs or ():
            totals[s.kind] = totals.get(s.kind, 0) + s.tokens
    grand = sum(totals.values()) or 1
    legend = ''.join(
        f'<span><i class="k-{kind}"></i>{esc(label.lower())} <em>{totals.get(kind, 0) * 100 // grand}%</em></span>'
        for kind, label in KIND_LABELS.items()
        if kind in totals
    )
    scale_title = (
        f'Scale: p95 of provider-reported input across all loaded rows ({scale:,} tokens)'
        if loaded_inputs
        else 'No loaded rows report provider input; bars use a 1-token fallback scale'
    )
    ticks = ''.join(f'<b style="left:{q * 25}%">{fmt_tokens(scale * q // 4) if q else "0"}</b>' for q in range(5))
    classified = _classified(snapshot) and snapshot is not None and bool(snapshot.results)
    rows_html = ''
    for row in page:
        segs = bars[row.trace_id]
        result = snapshot.results.get(row.trace_id) if classified and snapshot is not None else None
        dim = (' hit' if ai_matched(result) else ' nomatch') if result is not None else ''
        dim += ' row-err' if row.is_error else ''
        has_conversation = bool(segs) and any(segment.preview.strip() or segment.label for segment in segs or ())
        if has_conversation and segs:
            estimated = sum(s.tokens for s in segs)
            reported = row.tokens_in
            unattributed = max(0, reported - estimated) if reported is not None else 0
            total = max(estimated, reported or 0)
            width = min(total / scale * 100, 100)
            inner = ''.join(
                f'<i class="k-{s.kind}" style="flex-grow:{s.tokens}" {_tip_attrs(s, n, len(segs))}></i>'
                for n, s in enumerate(segs, start=1)
            )
            if unattributed:
                inner += (
                    f'<i class="tv-unattributed" style="flex-grow:{unattributed}" '
                    f'title="Unattributed: {unattributed:,} tokens, the difference between provider-reported trace input and estimated captured messages; may include other spans, hidden configuration, or formatting differences"></i>'
                )
            accounting = (
                f'Provider input {reported:,}; estimated captured messages {estimated:,}; '
                f'unattributed difference {unattributed:,}.'
                if reported is not None and estimated <= reported
                else f'Estimated captured messages {estimated:,} exceed provider-reported trace input {reported:,} by {estimated - reported:,}.'
                if reported is not None
                else f'Provider input unavailable; captured message estimate {estimated:,}.'
            )
            overflow = total > scale
            overflow_marker = (
                f'<span class="tv-overflow" title="Total {total:,} tokens exceeds the loaded-population p95 scale of {scale:,}" aria-label="Bar exceeds scale">!</span>'
                if overflow
                else ''
            )
            count = len({s.index for s in segs})
            bar = f'<div class="tv-segs" title="{esc(accounting)}" style="width:{width:.2f}%">{inner}{overflow_marker}</div>'
            row_title = f' title="{esc(accounting)}"'
            message_count = f'{count} msgs'
        else:
            bar = '<div class="tv-segs tv-nomsg" style="width:100%"></div>'
            row_title = ''
            message_count = 'Loading messages…' if loading and row.trace_id not in records else 'No messages available'
        row_summary = f'{message_count} · {fmt_duration(row.duration_ms)}'
        rows_html += (
            f'<div class="tv-r{dim}" data-tv-row="{esc(row.trace_id)}" tabindex="0"{row_title} {_drawer_attrs(row.trace_id, traces_layout=traces_layout)}>'
            f'{_identity(row, snapshot)}<div class="tv-bar"><div class="tv-plot"><div class="tv-track"></div>{bar}</div>'
            f'<span class="tv-end">{row_summary}</span></div>{_metrics(row)}</div>'
        )
    if loading:
        rows_html += ''.join(
            '<div class="tv-r tv-skeleton" aria-hidden="true"><i></i><i></i><i></i></div>' for _ in range(4)
        )
    return (
        f'<div class="tv"><div class="tv-lg">{legend}<span class="tv-legend-unattributed"><i></i>unattributed</span><span class="tv-note" title="{esc(TRAJECTORIES_HELP)}">Estimated captured message tokens (text length ÷ 4); tool definitions are omitted. Unattributed is the difference from provider input and may include hidden configuration, formatting differences, missing content, or other spans. It does not identify a specific source.</span></div>'
        f'<div class="tv-hd"><span>Trace</span><div class="ax"><div class="tv-scale" title="{esc(scale_title)}">{ticks}</div><span>Estimated message tokens</span></div>'
        f'<div class="tv-mh"><span title="{esc(COLUMN_HELP["tokens_in"])}; and tokens the model wrote back">Input / output</span>'
        f'<span title="{esc(COLUMN_HELP["cache_pct"])}">Cache reads</span><span>Cost</span></div></div>'
        f'<div class="tv-rows">{rows_html}</div><div class="tv-tip" role="tooltip" hidden></div></div>'
    )


def _toolbar(
    view: ExplorerView,
    columns: Sequence[Column],
    *,
    has_results: bool,
    ai_has_run: bool = False,
    window_days: int = 7,
    counts: Mapping[str, int] | None = None,
) -> str:
    chosen = {c.key for c in columns}
    counts = counts or {}
    chips = (('all', 'All'), ('errors', 'Errors')) + ((('matches', 'AI matches'),) if 'matches' in counts else ())
    quick_views = (
        ''.join(
            f'<button type="button" class="xr-qv-{key}{" has-errors" if key == "errors" and counts.get(key) else ""}{" on" if view.quick_view == key else ""}" aria-pressed="{str(view.quick_view == key).lower()}" '
            f'hx-get="/find/rows?quick_view={key}" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">'
            f'{label}{f" <span class={chr(34)}xr-count{chr(34)}>{counts[key]}</span>" if key in counts else ""}</button>'
            for key, label in chips
        )
        + '<span class="xr-qv-sep" aria-hidden="true"></span>'
        + '<details class="xr-top-menu"><summary class="'
        + ('on' if view.quick_view in TOP_METRICS or view.quick_view in CONVERSATION_METRICS else '')
        + '">'
        + (TOP_QUICK_VIEW_LABELS.get(view.quick_view, 'Views') + ' ▾')
        + '</summary><div class="xr-top-list" role="group" aria-label="Views">'
        + f'<button type="button" class="{"on" if view.quick_view == "all" else ""}" aria-pressed="{str(view.quick_view == "all").lower()}" hx-get="/find/rows?quick_view=all" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">All traces</button>'
        + '<span class="xr-top-head">Traces</span>'
        + ''.join(
            f'<button type="button" class="{"on" if view.quick_view == key else ""}" aria-pressed="{str(view.quick_view == key).lower()}" title="Top 10% of loaded traces by {TOP_METRICS[key][0]}" hx-get="/find/rows?quick_view={key}" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">{TOP_QUICK_VIEW_LABELS[key]}</button>'
            for key in ('slow', 'costly', 'tokens', 'context')
        )
        + '<span class="xr-top-head">Conversations</span>'
        + ''.join(
            f'<button type="button" class="{"on" if view.quick_view == key else ""}" aria-pressed="{str(view.quick_view == key).lower()}" title="Top 10% of conversations by {CONVERSATION_METRICS[key][0]}" hx-get="/find/rows?quick_view={key}" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">{TOP_QUICK_VIEW_LABELS[key]}</button>'
            for key in ('conv_costly', 'conv_tokens', 'conv_longest')
        )
        + '</div></details>'
    )
    boxes = ''.join(
        f'<label{"" if ai_has_run or c.key != MATCH else " hidden"}>'
        f'<input type="checkbox" name="columns" value="{c.key}"{" checked" if c.key in chosen else ""}>{esc(c.label)}</label>'
        for c in COLUMNS.values()
    )
    switch = ''.join(
        f'<button type="button" class="{"on" if view.view == mode else ""}" aria-pressed="{str(view.view == mode).lower()}" '
        f'hx-get="/find/rows?view={mode}" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">{label}</button>'
        for mode, label in (('table', 'Table'), ('trajectories', 'Trajectories'))
    )
    switch = switch.replace(
        '>Trajectories</button>',
        f' title="{esc(TRAJECTORIES_HELP)}">Trajectories<span class="xr-help" aria-hidden="true">?</span></button>',
    )
    sort = (
        '<details class="xr-sort"><summary>Sort</summary><button type="button" class="link" hx-get="/find/rows?sort=started&dir=desc" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">time</button>'
        ' · <button type="button" class="link" hx-get="/find/rows?sort=tokens_in&dir=desc" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">tokens in</button>'
        + (
            ' · <button type="button" class="link" hx-get="/find/rows?sort=match&dir=desc" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">match</button>'
            if has_results
            else ''
        )
        + '</details>'
        if view.view == 'trajectories'
        else ''
    )
    columns_menu = (
        '<details class="xr-cols" id="explorer-cols"><summary>Columns ▾</summary>'
        f'<form hx-post="/find/columns" hx-trigger="change" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">{_csrf()}{boxes}</form></details>'
        if view.view == 'table'
        else ''
    )
    date_and_rows = range_inputs(view.start, view.end, window_days, include_load=False)
    context_menu = sort if view.view == 'trajectories' else columns_menu
    load = '<button class="btn-secondary" type="submit" form="explorer-load-form">Load</button>'
    from evaluatorq.dashboard.trace_finder.views import _facet_chips  # pyright: ignore[reportPrivateUsage]

    chips = _facet_chips(view.facets, view.numeric, removable=True)
    active = chips.count('class="chip ')
    label = f'Filters · {active}' if active else 'Filters'
    return (
        f'<div class="xr-toolbar"><button type="button" class="xr-filter" data-explorer-filters aria-haspopup="true" aria-expanded="false">{label}</button>'
        f'<span class="xr-chips">{chips}</span>'
        f'<span class="xr-quickviews" role="group" aria-label="Quick views">{quick_views}</span>'
        f'<span class="spacer"></span><span class="xr-toolbar-right">{date_and_rows}{context_menu}{load}'
        f'<span class="finder-seg xr-switch" role="group" aria-label="View">{switch}</span></span></div>'
    )


def _csrf() -> str:
    from evaluatorq.dashboard.security import csrf_field

    return csrf_field()


def _pager(view: ExplorerView, results: Mapping[str, TraceClassification] | None) -> str:
    pages = view.page_count(results)
    if pages <= 1:
        return ''
    page = min(view.page, pages - 1)
    prev = (
        f'<button type="button" class="link" hx-get="/find/rows?page={page - 1}" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">← Prev</button>'
        if page
        else ''
    )
    nxt = (
        f'<button type="button" class="link" hx-get="/find/rows?page={page + 1}" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">Next →</button>'
        if page < pages - 1
        else ''
    )
    return f'<div class="xr-pager">{prev}<span>Page {page + 1} of {pages} · {PAGE_ROWS} per page</span>{nxt}</div>'


def _model_totals(rows: Sequence[TraceRow]) -> str:
    """Summarize traces and safely attributable cost for each model in the shown rows."""
    grouped: dict[str, list[TraceRow]] = {}
    for row in rows:
        for model in set(row.models) or {'Unknown model'}:
            grouped.setdefault(model, []).append(row)

    items: list[str] = []
    for model, model_rows in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0].casefold())):
        currencies = {row.currency for row in model_rows}
        can_sum_cost = (
            all(len(row.models) == 1 and row.cost_total is not None and row.currency for row in model_rows)
            and len(currencies) == 1
        )
        cost = sum(row.cost_total or 0 for row in model_rows) if can_sum_cost else None
        currency = next(iter(currencies)) if can_sum_cost else None
        items.append(
            f'<li><span class="xr-model-name">{esc(model)}</span> '
            f'<span>{len(model_rows):,} {"trace" if len(model_rows) == 1 else "traces"}</span> '
            f'<span>{fmt_cost(cost, currency)}</span></li>'
        )
    if not items:
        return '<div class="xr-model-groups" aria-label="Trace count and cost by model"><span>No traces by model</span></div>'
    return (
        '<div class="xr-model-groups" aria-label="Trace count and cost by model"><span class="xr-model-heading">By model</span><ul>'
        + ''.join(items)
        + '</ul></div>'
    )


def _totals_strip(rows: Sequence[TraceRow], *, traces_layout: bool = False) -> str:
    """One line of sums over the rows the table shows, so a filter visibly moves the numbers."""
    t = totals(rows)
    share = f'{t.cache_share:.0%}' if t.cache_share is not None else fmt_tokens(None)
    parts = (
        ('traces', f'{t.traces:,}'),
        ('errors', f'{t.errors:,}'),
        ('cost', fmt_cost(t.cost, t.currency)),
        ('in', fmt_tokens(t.tokens_in)),
        ('out', fmt_tokens(t.tokens_out)),
        ('cache read', share),
        ('p50', fmt_duration(t.p50_ms)),
        ('p95', fmt_duration(t.p95_ms)),
    )
    cells = ''.join(
        f'<span class="xr-total"{f" title={chr(34)}{esc(TOTAL_HELP[label])}{chr(34)}" if label in TOTAL_HELP else ""}>'
        f'<span class="xr-total-label">{label}</span> {value}</span>'
        for label, value in parts
    )
    grouped = _model_totals(rows) if traces_layout else ''
    return f'<div class="xr-totals" aria-label="Totals for the traces shown">{cells}</div>{grouped}'


def results(
    view: ExplorerView,
    columns: Sequence[Column],
    *,
    records: Mapping[str, TraceRecord | None] | None,
    snapshot: RunSnapshot | None,
    oob: bool = False,
    error: str | None = None,
    window_days: int = 7,
    traces_layout: bool = False,
) -> str:
    snapshot = _within_snapshot(snapshot)
    if (
        snapshot is not None
        and snapshot.state == 'completed'
        and snapshot.dimensions
        and view.quick_view == 'all'
        and view.sort is None
    ):
        # Keep All selected while surfacing positive classifier results first. An explicit
        # column sort and the chosen table/trajectory mode remain user-controlled.
        matched = tuple(row for row in view.rows if ai_matched(snapshot.results.get(row.trace_id)))
        unmatched = tuple(row for row in view.rows if not ai_matched(snapshot.results.get(row.trace_id)))
        view = replace(view, rows=(*matched, *unmatched))
    oob_attr = ' hx-swap-oob="true"' if oob else ''
    poll = ' hx-get="/find/rows" hx-trigger="every 1s" hx-swap="outerHTML"' if view.state == 'loading' else ''
    judged = snapshot.results if snapshot is not None and snapshot.results else None
    counts: dict[str, int] = {'errors': sum(1 for row in view.rows if row.is_error)} if view.rows else {}
    classified = bool(judged) and _classified(snapshot)
    if judged and classified:
        counts['matches'] = sum(1 for row in view.rows if ai_matched(judged.get(row.trace_id)))
    toolbar = _toolbar(
        view,
        columns,
        has_results=bool(judged),
        ai_has_run=_ai_has_run(snapshot),
        window_days=window_days,
        counts=counts,
    )
    judged_count = sum(1 for row in view.rows if row.trace_id in judged) if judged else 0
    judged_in_view = judged_count
    matched_in_view = sum(1 for row in view.rows if ai_matched(judged.get(row.trace_id))) if judged else 0
    judged_status = (
        f' · {judged_in_view} judged, {matched_in_view} match'
        if classified and view.quick_view != 'matches' and view.state != 'loading'
        else ''
    )
    reason = (
        '<span title="Possible reasons: outside the rows the question ran on, over the AI trace limit, '
        'or no conversation to read, such as embedding calls.">'
        f'{len(view.rows) - judged_count} not judged</span>'
    )
    not_judged = (
        f' · {reason}'
        if classified and snapshot is not None and snapshot.state == 'completed' and judged_count < len(view.rows)
        else ''
    )
    missing_conversations = (
        f' · {view.narrowed_from - snapshot.loaded} of the {view.narrowed_from} have no conversation'
        if view.narrowed_from is not None
        and snapshot is not None
        and snapshot.within_results
        and snapshot.state == 'completed'
        and snapshot.loaded < view.narrowed_from
        else ''
    )
    shown = len(view.visible_rows(judged))
    at_fetch_cap = bool(view.limit and len(view.rows) >= view.limit)
    shown_loaded = (
        f'Showing {shown} of {len(view.rows)} loaded traces · requested cap {view.limit:,}; '
        'more traces may exist, so raise Rows or narrow the range'
        if at_fetch_cap
        else f'Showing {shown} of {len(view.rows)} loaded traces'
    )
    status = (
        f'<div class="xr-status" role="status">Loading traces · {len(view.rows)} / {view.limit}</div>'
        if view.state == 'loading'
        else f'<div class="xr-status" role="status">{counts["matches"]} AI matches</div>'
        if view.quick_view == 'matches' and 'matches' in counts
        else f'<div class="xr-status" role="status">Top 10% by {TOP_METRICS[view.quick_view][0]} · '
        f'{len(view.visible_rows(judged))} of {len(view.rows)} loaded traces, heaviest first</div>'
        if view.quick_view in TOP_METRICS
        else f'<div class="xr-status" role="status">Top 10% of conversations by {CONVERSATION_METRICS[view.quick_view][0]} · {len({conversation_key(row) for row in view.visible_rows(judged) if conversation_key(row) is not None})} conversation{"s" if len({conversation_key(row) for row in view.visible_rows(judged) if conversation_key(row) is not None}) != 1 else ""}, {len(view.visible_rows(judged))} of {len(view.rows)} loaded traces</div>'
        if view.quick_view in CONVERSATION_METRICS
        else f'<div class="xr-status" role="status">Showing {len(view.rows)} of {view.narrowed_from} loaded traces match the filters{judged_status}{missing_conversations}{not_judged}'
        f'{f" · {view.limit:,}-trace fetch cap reached" if at_fetch_cap else ""}</div>'
        if view.narrowed_from is not None
        else f'<div class="xr-status" role="status">{shown_loaded}{judged_status}{not_judged}</div>'
    )
    if view.state == 'idle':
        inner = _empty('Load traces to start.', 'Pick a time range and filters, then press Load. Loading uses no AI.')
    elif not view.rows and view.state == 'loaded':
        inner = (
            _empty(
                'No loaded traces match.',
                f'None of the {view.narrowed_from} loaded traces meet the filters. Remove a filter, or use New search to look beyond the loaded rows.',
            )
            if view.narrowed_from is not None
            else _empty(
                'No traces match.',
                'Nothing in this time range matches the filters. Widen the range or remove a filter.',
            )
        )
    elif not view.rows and view.state == 'failed':
        inner = _load_failed(view.error)
    else:
        banner = (
            f'<div class="finder-review finder-form-error" role="alert">Load stopped: {esc(view.error or "unknown error")}. {len(view.rows)} rows loaded.</div>'
            if view.state == 'failed'
            else ''
        )
        results = snapshot.results if snapshot is not None and snapshot.within_results else None
        visible_rows = view.visible_rows(results)
        body = (
            trajectories(view, records or {}, snapshot, loading=view.state == 'loading', traces_layout=traces_layout)
            if view.view == 'trajectories'
            else f'<div class="xr-table-wrap">{table(view, columns, snapshot, traces_layout=traces_layout)}</div>'
        )
        if not visible_rows:
            label = (
                f'No loaded traces report {TOP_METRICS[view.quick_view][0]}.'
                if view.quick_view in TOP_METRICS
                else f'No loaded conversations report {CONVERSATION_METRICS[view.quick_view][0]}.'
                if view.quick_view in CONVERSATION_METRICS and any(row.thread_id or row.session_id for row in view.rows)
                else 'No conversations in the loaded traces.'
                if view.quick_view in CONVERSATION_METRICS
                else {'errors': 'No errors found.', 'matches': 'No AI matches found.'}.get(view.quick_view)
            )
            if view.quick_view == 'matches' and snapshot is not None and snapshot.state == 'completed':
                body = (
                    '<div class="xr-empty"><h4>No AI matches</h4>'
                    f'<p>None of the {judged_count} judged traces match “{esc(snapshot.request.query if snapshot.request else "")}”. '
                    'Rephrase the question, or pick New search to look beyond the loaded rows.</p></div>'
                )
            elif label:
                body = _empty(
                    label,
                    ''
                    if view.quick_view in CONVERSATION_METRICS
                    and any(row.thread_id or row.session_id for row in view.rows)
                    else 'These views group traces by thread or session id, and none of the loaded traces carry one.'
                    if view.quick_view in CONVERSATION_METRICS
                    else 'Choose All to return to the full loaded trace population.',
                )
        if results and view.matched_only and not view.visible_rows(results):
            body = _empty(
                'No matches yet.', 'None of the loaded traces match so far. Turn off Matches only to see every row.'
            )
        inner = f'{banner}{body}{_pager(view, results)}'
    totals_strip = (
        _totals_strip(view.visible_rows(judged), traces_layout=traces_layout)
        if view.rows or view.state == 'loaded'
        else ''
    )
    inner = f'{status}{totals_strip}{inner}'
    error_html = f'<div class="finder-review finder-form-error" role="alert">{esc(error)}</div>' if error else ''
    # The toolbar travels beside the section, out of band, so the page can keep it ahead of the Ask AI band in the DOM.
    return (
        f'<section id="explorer-results" class="xr" hx-sync="this:replace" hx-include="#finder-scope"{oob_attr}{poll}>{error_html}{inner}</section>'
        f'<div id="{TOOLBAR_SLOT_ID}" hx-swap-oob="true">{toolbar}</div>'
    )


TOOLBAR_SLOT_ID = 'explorer-toolbar'


def split_toolbar(html: str) -> tuple[str, str]:
    """Split ``results()`` output into the toolbar slot (in place, no OOB attribute) and the results section."""
    marker = f'<div id="{TOOLBAR_SLOT_ID}" hx-swap-oob="true">'
    head, sep, tail = html.partition(marker)
    if not sep:
        return '', html
    return marker.replace(' hx-swap-oob="true"', '') + tail, head
