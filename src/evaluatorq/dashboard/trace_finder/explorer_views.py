"""Explorer results: the trace table, the trajectory bars, and their toolbar and pager."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from urllib.parse import quote

from loguru import logger

from evaluatorq.common.reports import esc
from evaluatorq.trace_finder.columns import COLUMNS, MATCH, Column, fmt_cost, fmt_time, fmt_tokens
from evaluatorq.trace_finder.explorer import PAGE_ROWS
from evaluatorq.trace_finder.trajectory import KIND_LABELS, segments

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from evaluatorq.trace_finder import RunSnapshot
    from evaluatorq.trace_finder.explorer import ExplorerView
    from evaluatorq.trace_finder.models import TraceClassification, TraceRecord
    from evaluatorq.trace_finder.rows import TraceRow
    from evaluatorq.trace_finder.trajectory import Segment

DEFAULT_EXPLORER_ROWS = 200
PRESETS = (
    ('15m', timedelta(minutes=15)),
    ('1h', timedelta(hours=1)),
    ('24h', timedelta(days=1)),
    ('7d', timedelta(days=7)),
    ('30d', timedelta(days=30)),
)


def _local_value(value: datetime) -> str:
    # The server renders UTC; dashboard.js rewrites both inputs to browser-local time on load (data-utc).
    return value.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')


def range_inputs(start: datetime | None, end: datetime | None, window_days: int) -> str:
    end = end or datetime.now(timezone.utc)
    start = start or end - timedelta(days=window_days)
    presets = ''.join(
        f'<button type="button" class="link" data-explorer-preset="{int(span.total_seconds())}">{label}</button>'
        for label, span in PRESETS
    )
    return (
        '<form id="explorer-load-form" hx-post="/find/load" hx-target="#explorer-results" hx-swap="outerHTML" hx-include="#finder-controls"></form>'
        '<input type="hidden" name="tz_offset" form="explorer-load-form" data-explorer-tz>'
        f'<span class="quiet"><b>From</b><input id="explorer-from" hx-preserve form="explorer-load-form" name="from" type="datetime-local" step="1" required data-utc="{_local_value(start)}" value="{_local_value(start)}"></span>'
        f'<span class="quiet"><b>To</b><input id="explorer-to" hx-preserve form="explorer-load-form" name="to" type="datetime-local" step="1" required data-utc="{_local_value(end)}" value="{_local_value(end)}"></span>'
        f'<span class="xr-presets">{presets}</span>'
        f'<span class="quiet"><b>Rows</b><input id="explorer-rows" hx-preserve form="explorer-load-form" name="rows" type="number" min="1" max="5000" value="{DEFAULT_EXPLORER_ROWS}" style="width:72px"></span>'
        '<button class="btn-primary" type="submit" form="explorer-load-form">Load</button>'
    )


def _empty(title: str, body: str) -> str:
    return f'<div class="finder-hint xr-empty"><div class="inner"><h4>{esc(title)}</h4><p>{esc(body)}</p></div></div>'


def _match_cell(row: TraceRow, snapshot: RunSnapshot | None) -> str:
    from evaluatorq.dashboard.trace_finder.views import _result_color, _value_text

    result = snapshot.results.get(row.trace_id) if snapshot is not None else None
    if result is None:
        return '<td class="muted">—</td>'
    label = 'failed' if result.error else _value_text(result.value)
    return f'<td><span class="verdict"><span class="sw" style="background:{esc(_result_color(result, snapshot.compiled if snapshot else None))}"></span>{esc(label)}</span></td>'


def _drawer_attrs(trace_id: str, msg: int | None = None) -> str:
    query = f'?msg={msg}' if msg is not None else ''
    return f'hx-get="/find/trace/{quote(trace_id, safe="")}{query}" hx-target="#finder-drawer" hx-swap="innerHTML" hx-indicator="#finder-drawer-loading"'


def table(view: ExplorerView, columns: Sequence[Column], snapshot: RunSnapshot | None) -> str:
    results = snapshot.results if snapshot is not None and snapshot.results else None
    columns = [c for c in columns if results or not c.needs_results]
    heads = ''
    for column in columns:
        arrow = (' ↓' if view.descending else ' ↑') if view.sort == column.key else ''
        direction = 'asc' if view.sort == column.key and view.descending else 'desc'
        heads += (
            f'<th class="{"num" if column.numeric else ""}"><button type="button" class="link" '
            f'hx-get="/find/rows?sort={column.key}&dir={direction}" hx-target="#explorer-results" hx-swap="outerHTML">{esc(column.label)}{arrow}</button></th>'
        )
    body = ''.join(
        f'<tr data-tv-row="{esc(row.trace_id)}" {_drawer_attrs(row.trace_id)}>'
        + ''.join(
            _match_cell(row, snapshot)
            if c.key == MATCH
            else f'<td class="{"num" if c.numeric else ""}">{c.render(row)}</td>'
            for c in columns
        )
        + '</tr>'
        for row in view.page_rows(results)
    )
    return f'<table class="finder-table xr-table"><thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table>'


def _tip_attrs(segment: Segment, position: int, count: int) -> str:
    label = f' data-tv-tool="{esc(segment.label)}"' if segment.label and segment.kind in {'call', 'result'} else ''
    return (
        f'data-tv-msg="{segment.index}" data-tv-kind="{esc(KIND_LABELS[segment.kind])}" data-tv-n="{position} / {count}" '
        f'data-tv-tok="~{segment.tokens:,} tokens" data-tv-p="{esc(segment.preview)}"{label}'
    )


def _identity(row: TraceRow, snapshot: RunSnapshot | None) -> str:
    status = 'err' if (row.status or '').lower() in {'error', 'failed'} else 'ok'
    model = f'<span class="pill">{esc(row.models[0])}</span>' if row.models else ''
    tick = ''
    result = snapshot.results.get(row.trace_id) if snapshot is not None else None
    if result is not None:
        from evaluatorq.dashboard.trace_finder.views import _result_color

        tick = f'<span class="tv-tick" style="background:{esc(_result_color(result, snapshot.compiled if snapshot else None))}"></span>'
    return (
        f'<div class="tv-id"><span class="dot {status}"></span>{tick}<div class="t">'
        f'<div class="a">{esc(row.agent_name or row.name or "—")}{model}</div>'
        f'<div class="s mono">{esc(fmt_time(row.started_at))} · {esc(row.trace_id[:8])}</div></div></div>'
    )


def _metrics(row: TraceRow) -> str:
    pct = row.cache_pct
    ring = (
        f'<span class="cc"><span class="ring" style="background:conic-gradient(var(--traj-assistant) {pct * 360:.0f}deg,#ebe9e4 0)"></span>{pct * 100:.0f}%</span>'
        if pct is not None
        else '<span class="cc">—</span>'
    )
    return (
        f'<div class="tv-m"><span class="io"><b>{fmt_tokens(row.tokens_in)}</b> <span>→</span> {fmt_tokens(row.tokens_out)}</span>'
        f'{ring}<span>{fmt_cost(row.cost_total, row.currency)}</span></div>'
    )


def trajectories(view: ExplorerView, records: Mapping[str, TraceRecord | None], snapshot: RunSnapshot | None) -> str:
    page = view.page_rows(snapshot.results if snapshot is not None and snapshot.results else None)
    bars = {row.trace_id: segments(record.messages) if (record := records.get(row.trace_id)) else None for row in page}
    other = sum(1 for segs in bars.values() if segs for s in segs if s.kind == 'other')
    if other:
        logger.warning('Trajectory view drew {} message part(s) of unknown type as "other"', other)
    widest = max((sum(s.tokens for s in segs) for segs in bars.values() if segs), default=1)
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
    ticks = ''.join(f'<b style="left:{q * 25}%">{fmt_tokens(widest * q // 4) if q else "0"}</b>' for q in range(5))
    matched = snapshot is not None and bool(snapshot.results)
    rows_html = ''
    for row in page:
        segs = bars[row.trace_id]
        result = snapshot.results.get(row.trace_id) if matched and snapshot is not None else None
        dim = ' nomatch' if matched and (result is None or not result.matched) else ''
        if segs:
            width = sum(s.tokens for s in segs) / widest * 100
            inner = ''.join(
                f'<i class="k-{s.kind}" style="flex-grow:{s.tokens}" {_tip_attrs(s, n, len(segs))}></i>'
                for n, s in enumerate(segs, start=1)
            )
            count = len({s.index for s in segs})
            bar = f'<div class="tv-segs" style="width:{width:.2f}%">{inner}</div><span class="tv-end" style="left:{width:.2f}%">{count} msgs</span>'
        else:
            bar = '<div class="tv-segs tv-nomsg" style="width:100%"><i class="k-other" style="flex-grow:1"></i></div><span class="tv-end" style="left:0">no messages</span>'
        rows_html += (
            f'<div class="tv-r{dim}" data-tv-row="{esc(row.trace_id)}" {_drawer_attrs(row.trace_id)}>'
            f'{_identity(row, snapshot)}<div class="tv-bar"><div class="tv-track"></div>{bar}</div>{_metrics(row)}</div>'
        )
    return (
        f'<div class="tv"><div class="tv-lg">{legend}</div>'
        f'<div class="tv-hd"><span>Trace</span><div class="ax">{ticks}</div><span class="m">In → out · cache · cost</span></div>'
        f'<div class="tv-rows">{rows_html}</div><div class="tv-tip" role="tooltip" hidden></div></div>'
    )


def _toolbar(view: ExplorerView, columns: Sequence[Column], *, has_results: bool) -> str:
    chosen = {c.key for c in columns}
    matched_only = (
        f'<button type="button" class="link{" on" if view.matched_only else ""}" hx-get="/find/rows?matched_only={0 if view.matched_only else 1}" '
        f'hx-target="#explorer-results" hx-swap="outerHTML">Matches only</button>'
        if has_results
        else ''
    )
    boxes = ''.join(
        f'<label><input type="checkbox" name="columns" value="{c.key}"{" checked" if c.key in chosen else ""}>{esc(c.label)}</label>'
        for c in COLUMNS.values()
    )
    switch = ''.join(
        f'<button type="button" class="{"on" if view.view == mode else ""}" hx-get="/find/rows?view={mode}" hx-target="#explorer-results" hx-swap="outerHTML">{label}</button>'
        for mode, label in (('table', 'Table'), ('trajectories', 'Trajectories'))
    )
    sort = (
        '<span class="xr-sort">Sort <button type="button" class="link" hx-get="/find/rows?sort=started&dir=desc" hx-target="#explorer-results" hx-swap="outerHTML">time</button>'
        ' · <button type="button" class="link" hx-get="/find/rows?sort=tokens_in&dir=desc" hx-target="#explorer-results" hx-swap="outerHTML">tokens in</button>'
        + (
            ' · <button type="button" class="link" hx-get="/find/rows?sort=match&dir=desc" hx-target="#explorer-results" hx-swap="outerHTML">match</button>'
            if has_results
            else ''
        )
        + '</span>'
        if view.view == 'trajectories'
        else ''
    )
    columns_menu = (
        '<details class="xr-cols"><summary>Columns ▾</summary>'
        f'<form hx-post="/find/columns" hx-trigger="change" hx-target="#explorer-results" hx-swap="outerHTML">{_csrf()}{boxes}</form></details>'
        if view.view == 'table'
        else ''
    )
    progress = (
        f'<span>loading {len(view.rows)} / {view.limit}</span>'
        if view.state == 'loading'
        else f'<span><b>{len(view.rows)}</b> traces</span>'
    )
    return f'<div class="xr-toolbar">{progress}<span class="spacer"></span>{matched_only}{sort}<span class="finder-seg xr-switch">{switch}</span>{columns_menu}</div>'


def _csrf() -> str:
    from evaluatorq.dashboard.security import csrf_field

    return csrf_field()


def _pager(view: ExplorerView, results: Mapping[str, TraceClassification] | None) -> str:
    pages = view.page_count(results)
    if pages <= 1:
        return ''
    page = min(view.page, pages - 1)
    prev = (
        f'<button type="button" class="link" hx-get="/find/rows?page={page - 1}" hx-target="#explorer-results" hx-swap="outerHTML">← Prev</button>'
        if page
        else ''
    )
    nxt = (
        f'<button type="button" class="link" hx-get="/find/rows?page={page + 1}" hx-target="#explorer-results" hx-swap="outerHTML">Next →</button>'
        if page < pages - 1
        else ''
    )
    return f'<div class="xr-pager">{prev}<span>Page {page + 1} of {pages} · {PAGE_ROWS} per page</span>{nxt}</div>'


def results(
    view: ExplorerView,
    columns: Sequence[Column],
    *,
    records: Mapping[str, TraceRecord | None] | None,
    snapshot: RunSnapshot | None,
    oob: bool = False,
) -> str:
    oob_attr = ' hx-swap-oob="true"' if oob else ''
    poll = ' hx-get="/find/rows" hx-trigger="every 1s" hx-swap="outerHTML"' if view.state == 'loading' else ''
    if view.state == 'idle':
        inner = _empty('Load traces to start.', 'Pick a time range and filters, then press Load. Loading uses no AI.')
    elif not view.rows and view.state == 'loaded':
        inner = _empty(
            'No traces match.', 'Nothing in this time range matches the filters. Widen the range or remove a filter.'
        )
    elif not view.rows and view.state == 'failed':
        inner = f'<div class="finder-review finder-form-error" role="alert">Load failed: {esc(view.error or "unknown error")}</div>'
    else:
        banner = (
            f'<div class="finder-review finder-form-error" role="alert">Load stopped: {esc(view.error or "unknown error")}. {len(view.rows)} rows loaded.</div>'
            if view.state == 'failed'
            else ''
        )
        results = snapshot.results if snapshot is not None and snapshot.results else None
        body = (
            trajectories(view, records or {}, snapshot)
            if view.view == 'trajectories'
            else table(view, columns, snapshot)
        )
        if results and view.matched_only and not view.visible_rows(results):
            body = _empty(
                'No matches yet.', 'None of the loaded traces match so far. Turn off Matches only to see every row.'
            )
        inner = f'{banner}{_toolbar(view, columns, has_results=bool(results))}{body}{_pager(view, results)}'
    return f'<section id="explorer-results" class="xr"{oob_attr}{poll}>{inner}</section>'
