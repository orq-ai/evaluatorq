"""Explorer results: the trace table, the trajectory bars, and their toolbar and pager."""

from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from html import unescape
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
    matches_first_view,
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
TRAJECTORY_FALLBACK_SCALE = 1_000
COLUMN_HELP = {
    'cache_pct': 'Share of input tokens the provider reused from an earlier request, which is cheaper and faster.',
    'tokens_in': 'Tokens sent to the model, including any served from the cache',
    'tokens_out': 'Tokens the model wrote back',
}
TOTAL_HELP = {
    'cost': 'Unavailable when cost is missing or the shown traces use different currencies.',
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


def range_inputs(
    start: datetime | None,
    end: datetime | None,
    window_days: int,
    *,
    include_load: bool = True,
    row_limit: int = DEFAULT_EXPLORER_ROWS,
) -> str:
    end = end or datetime.now(timezone.utc)
    start = start or end - timedelta(days=window_days)
    range_seconds = max(1, int((end - start).total_seconds()))
    preset_seconds = {int(span.total_seconds()) for _, span in PRESETS}
    range_mode = 'relative' if range_seconds in preset_seconds else 'exact'
    presets = ''.join(
        f'<button type="button" class="xr-preset" data-explorer-preset="{int(span.total_seconds())}" '
        f'aria-pressed="{str(int(span.total_seconds()) == range_seconds).lower()}">{label}</button>'
        for label, span in PRESETS
    )
    controls = (
        f'<form id="explorer-load-form" hidden hx-post="/find/load" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace" hx-include="#finder-controls, #finder-scope">{_csrf()}</form>'
        f'<input id="explorer-range-mode" type="hidden" name="range_mode" value="{range_mode}" form="explorer-load-form" hx-preserve data-explorer-range-mode>'
        f'<input id="explorer-range-seconds" type="hidden" name="range_seconds" value="{range_seconds}" form="explorer-load-form" hx-preserve data-explorer-range-seconds>'
        '<input type="hidden" name="tz_offset" form="explorer-load-form" data-explorer-tz>'
        f'<details class="xr-time-menu"><summary><span class="xr-time-label" data-explorer-range-label>Custom range</span></summary><div class="xr-time-options"><span class="xr-presets" role="group" aria-label="Relative range">{presets}</span>'
        '<details class="xr-exact"><summary>Custom range</summary>'
        f'<span class="xr-range"><b>From</b><input id="explorer-from" class="xr-date" aria-label="From date" hx-preserve form="explorer-load-form" name="from" type="date" required data-utc="{_local_value(start)}" value="{_local_value(start)[:10]}"><input id="explorer-from-time" class="xr-time" aria-label="From time" hx-preserve form="explorer-load-form" name="from_time" type="time" step="1" required data-utc="{_local_value(start)}" value="{_local_value(start)[11:]}"></span>'
        f'<span class="xr-range"><b>To</b><input id="explorer-to" class="xr-date" aria-label="To date" hx-preserve form="explorer-load-form" name="to" type="date" required data-utc="{_local_value(end)}" value="{_local_value(end)[:10]}"><input id="explorer-to-time" class="xr-time" aria-label="To time" hx-preserve form="explorer-load-form" name="to_time" type="time" step="1" required data-utc="{_local_value(end)}" value="{_local_value(end)[11:]}"></span>'
        '<small class="xr-tz" data-explorer-tz-label>Local time</small>'
        '<button type="button" class="xr-apply" data-explorer-apply>Apply range</button></details></div></details>'
        f'<label class="quiet"><b>Rows</b><input id="explorer-rows" hx-preserve form="explorer-load-form" name="rows" type="number" min="1" max="{MAX_LIVE_TRACES}" value="{row_limit}" style="width:72px"></label>'
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
    """A compact yes/no verdict with the classifier score when available."""
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
    return (
        snapshot is not None and snapshot.within_results and (bool(snapshot.results) or snapshot.state == 'classifying')
    )


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
    return max((len(unescape(_TAG_RE.sub('', line))) for line in lines), default=0)


def _clamp_ch(chars: float) -> int:
    return max(_MIN_COL_CH, min(_MAX_COL_CH, round(chars)))


@dataclass
class _Measured:
    """The widest cell seen so far for one column of one load; it only grows, so widths never jitter."""

    seen: set[str] = field(default_factory=set)
    chars: list[int] = field(default_factory=list)
    key: tuple[object, ...] = ()


_MEASURED: dict[tuple[int, str], _Measured] = {}
_MEASURED_LOADS = 32


def _measured(generation: int, column_key: str) -> _Measured:
    # Generation 0 is a view that was never loaded (tests, empty states): measure it fresh every time.
    if not generation:
        return _Measured()
    entry = _MEASURED.get((generation, column_key))
    if entry is None:
        entry = _MEASURED[generation, column_key] = _Measured()
        live = sorted({load for load, _ in _MEASURED})
        for stale in [key for key in _MEASURED if key[0] in live[:-_MEASURED_LOADS]]:
            del _MEASURED[stale]
    return entry


def _header_widths(
    columns: Sequence[Column], rows: Sequence[TraceRow], snapshot: RunSnapshot | None, *, generation: int = 0
) -> tuple[int, ...]:
    """A fixed width in ``ch`` per rendered header (MATCH can expand into several).

    Widths come from every loaded row (``rows``, not just the current page) so paging never changes
    them, and are clamped to a sane range so one long value can't blow up the table. Per load
    (``generation``) each row is measured once and a width only grows, so polls stay cheap and the
    columns hold still while rows stream in.
    """
    widths: list[int] = []
    for column in columns:
        labels = _match_names(snapshot, column.label) if column.key == MATCH else (column.label,)
        measured = _measured(generation, column.key)
        if column.key == MATCH:
            # AI answers arrive for rows already seen, so MATCH re-measures when the answers change.
            key = (labels, len(rows), len(snapshot.results) if snapshot is not None else 0)
            if key != measured.key:
                per_label = [0] * len(labels)
                for row in rows:
                    cells = [cell for cell in _match_cells(row, snapshot).split('</td>') if cell.strip()]
                    for index in range(len(labels)):
                        cell = cells[index] if index < len(cells) else (cells[-1] if cells else '')
                        per_label[index] = max(per_label[index], _visible_chars(cell))
                grow = bool(measured.key) and measured.key[0] == labels
                measured.chars = (
                    [max(pair) for pair in zip(per_label, measured.chars, strict=True)] if grow else per_label
                )
                measured.key = key
            cell_chars = measured.chars
        else:
            fresh = [row for row in rows if row.trace_id not in measured.seen]
            widest = max((_visible_chars(column.render(row)) for row in fresh), default=0)
            measured.chars = [max([widest, *measured.chars])]
            measured.seen.update(row.trace_id for row in fresh)
            cell_chars = measured.chars * len(labels)
        for label, chars in zip(labels, cell_chars, strict=True):
            header_chars = len(label) * _HEADER_CHAR_FACTOR + _HEADER_ARROW_CH
            widths.append(_clamp_ch(max(header_chars, chars + _CELL_PAD_CH)))
    return tuple(widths)


def table(
    view: ExplorerView, columns: Sequence[Column], snapshot: RunSnapshot | None, *, traces_layout: bool = False
) -> str:
    snapshot = _within_snapshot(snapshot)
    requested_columns = columns
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
        heads += ''.join(
            f'<th class="{"num" if column.numeric else ""}"{aria_sort if index == 0 else ""}><button type="button" class="link" data-xr-sort="{column.key}"{title} '
            f'hx-get="/find/rows?sort={column.key}&dir={direction}" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">'
            f'{sparkle}{esc(label)}{arrow if index == 0 else ""}</button></th>'
            for index, label in enumerate(labels)
        )
    if not columns:
        if any(column.key == MATCH for column in requested_columns) and not _ai_has_run(snapshot):
            return _empty('AI columns are not ready yet.', 'AI match columns appear after an AI classification starts.')
        return _empty('No columns selected.', 'Choose at least one column from Columns to show trace details.')
    rendered_keys = [
        c.key for c in columns for _ in (_match_names(snapshot, c.label) if c.key == MATCH else (c.label,))
    ]
    widths = _header_widths(columns, view.rows, snapshot, generation=view.generation)
    if traces_layout:
        widths = tuple(
            max(width, 16) if key == 'duration' else width for key, width in zip(rendered_keys, widths, strict=True)
        )
    duration_rows = view.visible_rows(results) if traces_layout else ()
    duration_totals = totals(duration_rows) if traces_layout else None
    duration_scale = max((row.duration_ms or 0 for row in duration_rows), default=0)
    duration_p95 = duration_totals.p95_ms if duration_totals is not None else None
    # One text column takes no fixed width so it absorbs the card's spare room; the table's
    # min-width keeps every column at least as wide as its measured content.
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
                else _duration_cell(row.duration_ms, duration_scale, duration_p95)
                if traces_layout and c.key == 'duration'
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


def _duration_cell(value: int | None, scale: int, p95: int | None) -> str:
    """Render numeric duration plus an accessible magnitude cue and visible-set p95 tint."""
    label = fmt_duration(value)
    if value is None:
        return f'<td class="num xr-duration"><span>{esc(label)}</span></td>'
    width = min(100, max(0, round(value / scale * 100))) if scale else 0
    magnitude = (
        f'<span class="xr-duration-bar" role="img" aria-label="Duration magnitude: {esc(label)}; '
        f'{width}% of the longest visible trace" title="{esc(label)} relative to the longest visible trace">'
        f'<span style="width:{width}%"></span></span>'
    )
    at_p95 = p95 is not None and value >= p95
    p95_class = ' xr-duration-p95' if at_p95 else ''
    p95_title = f' title="At or above visible-set p95 ({esc(fmt_duration(p95))})"' if at_p95 else ''
    p95_cue = '<span class="xr-duration-accessible">At or above p95</span>' if at_p95 else ''
    return f'<td class="num xr-duration{p95_class}"{p95_title}><span>{esc(label)}</span>{p95_cue}{magnitude}</td>'


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
    display_models = row.display_models
    primary = row.agent_name or (display_models[0] if display_models else None) or row.name or row.operation or '—'
    model = f'<span class="tv-model">{esc(display_models[0])}</span>' if row.agent_name and display_models else ''
    tick = ''
    result = snapshot.results.get(row.trace_id) if snapshot is not None else None
    if result is not None:
        from evaluatorq.dashboard.trace_finder.views import _result_color

        tick = f'<span class="tv-tick" style="background:{esc(_result_color(result, snapshot.dimensions if snapshot else None))}"></span>'
    return (
        f'<div class="tv-id"><span class="dot {status}"></span>{tick}<div class="t">'
        f'<div class="a" title="{esc(primary)}">{esc(primary)}</div>{model}'
        f'<div class="s mono">{esc(fmt_time(row.started_at))}</div></div></div>'
        f'<span class="tv-status {status}">{esc(status_label)}</span>'
    )


def _is_embedding(row: TraceRow) -> bool:
    """Embedding requests have no conversation to visualize."""
    return any(
        value.casefold().split('.', 1)[0] in {'embedding', 'embeddings'}
        for value in (row.product, row.operation, row.name)
        if value
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
    summary_loading: bool | None = None,
    traces_layout: bool = False,
) -> str:
    if summary_loading is None:
        summary_loading = loading
    snapshot = _within_snapshot(snapshot)
    page = view.page_rows(snapshot.results if snapshot is not None and snapshot.results else None)
    bars = {
        row.trace_id: segments(record.messages)
        if not _is_embedding(row) and (record := records.get(row.trace_id))
        else None
        for row in page
    }
    other = sum(1 for segs in bars.values() if segs for s in segs if s.kind == 'other')
    if other:
        logger.warning('Trajectory view drew {} message part(s) of unknown type as "other"', other)
    tool_sizes = {
        row.trace_id: record.tool_definition_tokens
        if view.show_tool_definitions and not _is_embedding(row) and (record := records.get(row.trace_id))
        else 0
        for row in page
    }
    captured_estimates = sorted(
        sum(segment.tokens for segment in bars[row.trace_id] or ()) + tool_sizes[row.trace_id]
        for row in page
        if bars[row.trace_id] or tool_sizes[row.trace_id]
    )
    scale = (
        captured_estimates[max(0, math.ceil(len(captured_estimates) * 0.95) - 1)]
        if captured_estimates
        else TRAJECTORY_FALLBACK_SCALE
    )
    scale = max(scale, 1)
    totals: dict[str, int] = {}
    for segs in bars.values():
        for s in segs or ():
            totals[s.kind] = totals.get(s.kind, 0) + s.tokens
    tool_total = sum(tool_sizes.values())
    grand = sum(totals.values()) + tool_total or 1
    legend = ''.join(
        f'<span><i class="k-{kind}"></i>{esc(label.lower())} <em>{totals.get(kind, 0) * 100 // grand}%</em></span>'
        for kind, label in KIND_LABELS.items()
        if kind in totals
    )
    if tool_total:
        legend += f'<span><i class="k-tools"></i>tool definitions <em>{tool_total * 100 // grand}%</em></span>'
    toggle = (
        f'<button type="button" class="tv-tool-toggle" aria-pressed="{str(view.show_tool_definitions).lower()}" '
        f'hx-get="/find/rows?show_tool_definitions={0 if view.show_tool_definitions else 1}" '
        'hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">'
        'Tool definitions</button>'
    )
    if captured_estimates:
        source_label = (
            'captured messages and tool definitions' if view.show_tool_definitions else 'captured message estimates'
        )
        scale_title = f'Scale: p95 of {source_label} on this page ({scale:,} tokens)'
    else:
        source_label = 'captured messages or tool definitions' if view.show_tool_definitions else 'captured messages'
        scale_title = f'No {source_label} on this page; fixed {scale:,}-token scale'
    context_label = 'Captured context' if view.show_tool_definitions else 'Captured messages'
    ticks = ''.join(f'<b style="left:{q * 25}%">{fmt_tokens(scale * q // 4) if q else "0"}</b>' for q in range(5))
    classified = _classified(snapshot) and snapshot is not None and bool(snapshot.results)
    rows_html = ''
    for row in page:
        segs = bars[row.trace_id]
        result = snapshot.results.get(row.trace_id) if classified and snapshot is not None else None
        dim = (' hit' if ai_matched(result) else ' nomatch') if result is not None else ''
        dim += ' row-err' if row.is_error else ''
        if _is_embedding(row):
            rows_html += (
                f'<div class="tv-r{dim}" data-tv-row="{esc(row.trace_id)}" tabindex="0" '
                f'{_drawer_attrs(row.trace_id, traces_layout=traces_layout)}>'
                f'{_identity(row, snapshot)}<div class="tv-bar tv-embedding">Embedding call</div>{_metrics(row)}</div>'
            )
            continue
        has_conversation = bool(segs) and any(segment.preview.strip() or segment.label for segment in segs or ())
        tool_tokens = tool_sizes[row.trace_id]
        if has_conversation or tool_tokens:
            estimated = sum(s.tokens for s in segs or ()) + tool_tokens
            width = min(estimated / scale * 100, 100)
            record = records.get(row.trace_id)
            tool_block = (
                f'<i class="k-tools" style="flex-grow:{tool_tokens}" data-tv-tools="1" '
                f'data-tv-kind="Tool definitions" data-tv-n="{record.tool_definition_count:,} definitions" '
                f'data-tv-tok="~{tool_tokens:,} tokens" data-tv-p="Estimated from captured tool schema JSON." '
                f'role="img" aria-label="{record.tool_definition_count:,} tool definitions, about {tool_tokens:,} tokens"></i>'
                if tool_tokens and record is not None
                else ''
            )
            inner = ''.join(
                f'<i class="k-{s.kind}" style="flex-grow:{s.tokens}" {_tip_attrs(s, n, len(segs))}></i>'
                for n, s in enumerate(segs or (), start=1)
            )
            message_tokens = estimated - tool_tokens
            accounting = f'Estimated captured messages: {message_tokens:,} token{"s" if message_tokens != 1 else ""}.'
            if tool_tokens:
                accounting += f' Tool definitions: {tool_tokens:,} estimated tokens.'
            overflow = estimated > scale
            overflow_marker = (
                f'<span class="tv-overflow" title="Captured messages exceed the page p95 scale of {scale:,} tokens" aria-label="Bar exceeds scale">!</span>'
                if overflow
                else ''
            )
            count = len({s.index for s in segs or ()})
            bar = f'<div class="tv-segs" style="width:{width:.2f}%">{tool_block}{inner}{overflow_marker}</div>'
            row_title = f' aria-description="{esc(accounting)}"'
            message_count = f'{count} msgs' if count else 'No messages available'
        else:
            bar = '<div class="tv-segs tv-nomsg" style="width:100%"></div>'
            row_title = ''
            message_count = 'Loading messages…' if loading and row.trace_id not in records else 'No messages available'
        row_summary = f'<span>{esc(message_count)}</span><span>{esc(fmt_duration(row.duration_ms))}</span>'
        rows_html += (
            f'<div class="tv-r{dim}" data-tv-row="{esc(row.trace_id)}" tabindex="0"{row_title} {_drawer_attrs(row.trace_id, traces_layout=traces_layout)}>'
            f'{_identity(row, snapshot)}<div class="tv-bar"><div class="tv-plot"><div class="tv-track"></div>{bar}</div>'
            f'<span class="tv-end">{row_summary}</span></div>{_metrics(row)}</div>'
        )
    if summary_loading:
        rows_html += ''.join(
            '<div class="tv-r tv-skeleton" aria-hidden="true"><i></i><i></i><i></i><i></i></div>' for _ in range(4)
        )
    return (
        f'<div class="tv"><div class="tv-lg">{legend}{toggle}</div>'
        f'<div class="tv-hd"><span>Model / agent</span><span>Status</span><div class="ax"><div class="tv-scale" title="{esc(scale_title)}">{ticks}</div><span>{context_label}</span></div>'
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
    traces_layout: bool = False,
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
    table_icon = '<svg viewBox="0 0 16 16" focusable="false"><rect x="1.5" y="2" width="13" height="12" rx="1"/><path d="M1.5 6h13M6 2v12"/></svg>'
    trajectories_icon = '<svg viewBox="0 0 16 16" focusable="false"><path d="M2 3h12M2 8h12M2 13h12"/><circle cx="5" cy="3" r="1.5"/><circle cx="10" cy="8" r="1.5"/><circle cx="7" cy="13" r="1.5"/></svg>'
    view_options = (('table', 'Table', table_icon), ('trajectories', 'Trajectories', trajectories_icon))
    switch = ''.join(
        f'<button type="button" class="{"on" if view.view == mode else ""}" aria-pressed="{str(view.view == mode).lower()}" '
        f'hx-get="/find/rows?view={mode}" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">'
        f'<span class="xr-view-icon" aria-hidden="true">{icon}</span><span>{label}</span></button>'
        for mode, label, icon in view_options
    )
    sort_options = (('started', 'Time'), ('tokens_in', 'Input tokens'))
    if has_results:
        sort_options += (('match', 'Match'),)
    sort = (
        '<details class="xr-sort"><summary>Sort ▾</summary><div class="xr-sort-options" role="group" aria-label="Sort trajectories">'
        + ''.join(
            f'<button type="button" hx-get="/find/rows?sort={key}&dir=desc" hx-target="#explorer-results" '
            f'hx-swap="outerHTML" hx-sync="#explorer-results:replace">{label}</button>'
            for key, label in sort_options
        )
        + '</div></details>'
        if view.view == 'trajectories'
        else ''
    )
    columns_menu = (
        '<details class="xr-cols" id="explorer-cols"><summary>Columns ▾</summary>'
        f'<form hx-post="/find/columns" hx-trigger="change" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">{_csrf()}{boxes}</form></details>'
        if view.view == 'table'
        else ''
    )
    export = (
        '<a class="btn-secondary xr-export" href="/traces/export.csv">Export</a>'
        if traces_layout and view.view == 'table'
        else ''
    )
    date_and_rows = range_inputs(
        view.start, view.end, window_days, include_load=False, row_limit=view.limit or DEFAULT_EXPLORER_ROWS
    )
    context_menu = sort if view.view == 'trajectories' else columns_menu
    load = '<button class="btn-secondary xr-load" type="submit" form="explorer-load-form">Load</button>'
    from evaluatorq.dashboard.trace_finder.views import _facet_chips  # pyright: ignore[reportPrivateUsage]

    chips = _facet_chips(view.facets, view.numeric, removable=True)
    active = chips.count('class="chip ')
    label = f'Filters · {active}' if active else 'Filters'
    return (
        f'<div class="xr-toolbar"><button type="button" class="xr-filter" data-explorer-filters aria-haspopup="true" aria-expanded="false">{label}</button>'
        f'<span class="xr-chips">{chips}</span>'
        f'<span class="xr-quickviews" role="group" aria-label="Quick views">{quick_views}</span>'
        f'<span class="spacer"></span><span class="xr-toolbar-right">{export}{date_and_rows}{context_menu}{load}'
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
        ('traces', 'Traces', '▦', f'{t.traces:,}', True),
        ('errors', 'Errors', '!', f'{t.errors:,}', True),
        ('cost', 'Cost', '$', fmt_cost(t.cost, t.currency), t.cost is not None),
        ('in', 'Input', '↓', fmt_tokens(t.tokens_in), t.tokens_in is not None),
        ('out', 'Output', '↑', fmt_tokens(t.tokens_out), t.tokens_out is not None),
        ('cache read', 'Cache read', '↻', share, t.cache_share is not None),
        ('p50', 'P50', '◷', fmt_duration(t.p50_ms), t.p50_ms is not None),
        ('p95', 'P95', '◷', fmt_duration(t.p95_ms), t.p95_ms is not None),
    )
    cells = ''.join(
        f'<span class="xr-total xr-total-{label.replace(" ", "-")}'
        f'{" has-errors" if label == "errors" and t.errors else ""}{" unavailable" if not available else ""}"'
        f'{f" title={chr(34)}{esc(TOTAL_HELP[label])}{chr(34)}" if label in TOTAL_HELP else ""}>'
        f'<span class="xr-total-mark" aria-hidden="true">{icon}</span>'
        f'<span class="xr-total-copy"><span class="xr-total-label">{display}</span>'
        f'<b class="xr-total-value">{value if available else "Unavailable"}</b></span></span>'
        for label, display, icon, value, available in parts
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
    poll: bool = False,
) -> str:
    """Render the results section plus its out-of-band toolbar; ``poll`` marks a timer-driven render.

    Both elements carry ``data-view-version`` and ``data-render-key`` so dashboard.js can drop a poll
    render that is older than the table on screen or identical to it (see ``find_poll`` in routes).
    """
    snapshot = _within_snapshot(snapshot)
    # Keep All selected while surfacing positive classifier results first. An explicit
    # column sort and the chosen table/trajectory mode remain user-controlled.
    view = matches_first_view(view, snapshot)
    oob_attr = ' hx-swap-oob="true"' if oob else ''
    # The self-poll swaps nothing itself: /find/rows answers it with out-of-band renders.
    poll_attrs = (
        ' hx-get="/find/rows" hx-trigger="every 1s" hx-swap="none"'
        if view.state == 'loading' or view.trajectory_warming
        else ''
    )
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
        traces_layout=traces_layout,
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
            trajectories(
                view,
                records or {},
                snapshot,
                loading=view.state == 'loading' or view.trajectory_warming,
                summary_loading=view.state == 'loading',
                traces_layout=traces_layout,
            )
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
            body = _empty('No matches.', 'No loaded traces match this filter.')
        inner = f'{banner}{body}{_pager(view, results)}'
    totals_strip = (
        _totals_strip(view.visible_rows(judged), traces_layout=traces_layout)
        if view.rows or view.state == 'loaded'
        else ''
    )
    inner = f'{status}{totals_strip}{inner}'
    error_html = f'<div class="finder-review finder-form-error" role="alert">{esc(error)}</div>' if error else ''
    section_body = f'{error_html}{inner}'
    marks = f' data-view-version="{view.version}"' + (' data-poll' if poll else '')
    # The toolbar travels beside the section, out of band, so the page can keep it ahead of the Ask AI band in the DOM.
    return (
        f'<section id="explorer-results" class="xr" hx-sync="this:replace" hx-include="#finder-scope"{oob_attr}{poll_attrs}'
        f'{marks} data-render-key="{_render_key(poll_attrs + section_body)}">{section_body}</section>'
        f'<div id="{TOOLBAR_SLOT_ID}" hx-swap-oob="true"{marks} data-render-key="{_render_key(toolbar)}">{toolbar}</div>'
    )


def _render_key(markup: str) -> str:
    return format(zlib.crc32(markup.encode()), '08x')


TOOLBAR_SLOT_ID = 'explorer-toolbar'


def split_toolbar(html: str) -> tuple[str, str]:
    """Split ``results()`` output into the toolbar slot (in place, no OOB attribute) and the results section."""
    marker = f'<div id="{TOOLBAR_SLOT_ID}" hx-swap-oob="true"'
    head, sep, tail = html.partition(marker)
    if not sep:
        return '', html
    return f'<div id="{TOOLBAR_SLOT_ID}"' + tail, head
