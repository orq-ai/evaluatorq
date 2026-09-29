"""Explorer results: the trace table, the trajectory bars, and their toolbar and pager."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from urllib.parse import quote

from loguru import logger

from evaluatorq.common.reports import esc
from evaluatorq.trace_finder.columns import COLUMNS, MATCH, Column, fmt_cost, fmt_time, fmt_tokens
from evaluatorq.trace_finder.explorer import CONVERSATION_METRICS, PAGE_ROWS, TOP_METRICS, conversation_key
from evaluatorq.trace_finder.orq_source import MAX_LIVE_TRACES
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
        f'<form id="explorer-load-form" hidden hx-post="/find/load" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace" hx-include="#finder-controls">{_csrf()}</form>'
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
            '<td>—</td>'
            if index >= len(result.answers)
            else '<td><span class="verdict xr-yn failed">Judgment failed</span></td>'
            if result.answers[index].error
            else f'<td><span class="verdict xr-yn {"yes" if result.answers[index].matched else "no"}"'
            f'{f" title={chr(34)}AI confidence {result.answers[index].confidence:.0%}{chr(34)}" if result.answers[index].confidence is not None else ""}>'
            f'{"yes" if result.answers[index].value is True else "no" if result.answers[index].value is False else esc(str(result.answers[index].value))}</span></td>'
            for index in range(len(dimensions))
        )
    if result.error or not dimensions:
        label = 'Judgment failed' if result.error else 'Included' if result.matched else 'Not included'
        cell = f'<td><span class="verdict xr-match"><span class="sw" style="background:{esc(_result_color(result, dimensions))}"></span>{esc(label)}</span></td>'
        return cell * max(1, len(dimensions))
    return _answer_cells(result, dimensions)


def _within_snapshot(snapshot: RunSnapshot | None) -> RunSnapshot | None:
    return snapshot if snapshot is not None and snapshot.within_results else None


def _drawer_attrs(trace_id: str, msg: int | None = None) -> str:
    query = f'?msg={msg}' if msg is not None else ''
    return f'hx-get="/find/trace/{quote(trace_id, safe="")}{query}" hx-target="#finder-drawer" hx-swap="innerHTML" hx-indicator="#finder-drawer-loading"'


def table(view: ExplorerView, columns: Sequence[Column], snapshot: RunSnapshot | None) -> str:
    snapshot = _within_snapshot(snapshot)
    results = snapshot.results if snapshot is not None and snapshot.results else None
    heads = ''
    for column in columns:
        arrow = (' ↓' if view.descending else ' ↑') if view.sort == column.key else ''
        direction = 'asc' if view.sort == column.key and view.descending else 'desc'
        labels = _match_names(snapshot, column.label) if column.key == MATCH else (column.label,)
        heads += ''.join(
            f'<th class="{"num" if column.numeric else ""}"><button type="button" class="link" '
            f'hx-get="/find/rows?sort={column.key}&dir={direction}" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">{esc(label)}{arrow}</button></th>'
            for label in labels
        )
    if not columns:
        return _empty('No columns selected.', 'Choose at least one column from Columns to show trace details.')
    conversation = None
    body = ''
    for row in view.page_rows(results):
        key = conversation_key(row)
        marker = (
            f' data-conv="{esc(key)}"'
            if view.quick_view in CONVERSATION_METRICS and view.sort is None and key is not None and key != conversation
            else ''
        )
        body += (
            f'<tr data-tv-row="{esc(row.trace_id)}"{marker} {_drawer_attrs(row.trace_id)}>'
            + ''.join(
                _match_cells(row, snapshot)
                if c.key == MATCH
                else f'<td class="{"num" if c.numeric else ""}">{c.render(row)}</td>'
                for c in columns
            )
            + '</tr>'
        )
        conversation = key
    return f'<table class="finder-table xr-table"><thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table>'


def _tip_attrs(segment: Segment, position: int, count: int) -> str:
    label = f' data-tv-tool="{esc(segment.label)}"' if segment.label and segment.kind in {'call', 'result'} else ''
    return (
        f'data-tv-msg="{segment.index}" data-tv-kind="{esc(KIND_LABELS[segment.kind])}" data-tv-n="{position} / {count}" '
        f'data-tv-tok="~{segment.tokens:,} tokens" data-tv-p="{esc(segment.preview)}"{label}'
    )


def _identity(row: TraceRow, snapshot: RunSnapshot | None) -> str:
    status = 'err' if row.is_error else 'ok'
    model = f'<span class="pill">{esc(row.models[0])}</span>' if row.models else ''
    tick = ''
    result = snapshot.results.get(row.trace_id) if snapshot is not None else None
    if result is not None:
        from evaluatorq.dashboard.trace_finder.views import _result_color

        tick = f'<span class="tv-tick" style="background:{esc(_result_color(result, snapshot.dimensions if snapshot else None))}"></span>'
    return (
        f'<div class="tv-id"><span class="dot {status}"></span>{tick}<div class="t">'
        f'<div class="a">{esc(row.agent_name or row.name or "—")}{model}</div>'
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


def trajectories(view: ExplorerView, records: Mapping[str, TraceRecord | None], snapshot: RunSnapshot | None) -> str:
    snapshot = _within_snapshot(snapshot)
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
        dim = ' nomatch' if result is not None and not result.matched else ''
        if segs:
            width = sum(s.tokens for s in segs) / widest * 100
            inner = ''.join(
                f'<i class="k-{s.kind}" style="flex-grow:{s.tokens}" {_tip_attrs(s, n, len(segs))}></i>'
                for n, s in enumerate(segs, start=1)
            )
            count = len({s.index for s in segs})
            bar = f'<div class="tv-segs" style="width:{width:.2f}%">{inner}</div>'
            message_count = f'{count} msgs'
        else:
            bar = '<div class="tv-segs tv-nomsg" style="width:100%"><i class="k-other" style="flex-grow:1"></i></div>'
            message_count = 'no messages'
        rows_html += (
            f'<div class="tv-r{dim}" data-tv-row="{esc(row.trace_id)}" {_drawer_attrs(row.trace_id)}>'
            f'{_identity(row, snapshot)}<div class="tv-bar"><div class="tv-plot"><div class="tv-track"></div>{bar}</div>'
            f'<span class="tv-end">{message_count}</span></div>{_metrics(row)}</div>'
        )
    return (
        f'<div class="tv"><div class="tv-lg">{legend}</div>'
        f'<div class="tv-hd"><span>Trace</span><div class="ax"><div class="tv-scale">{ticks}</div><span>Messages</span></div>'
        '<div class="tv-mh"><span>Input / output</span><span>Cache reads</span><span>Cost</span></div></div>'
        f'<div class="tv-rows">{rows_html}</div><div class="tv-tip" role="tooltip" hidden></div></div>'
    )


def _toolbar(
    view: ExplorerView,
    columns: Sequence[Column],
    *,
    has_results: bool,
    window_days: int = 7,
    counts: Mapping[str, int] | None = None,
) -> str:
    chosen = {c.key for c in columns}
    counts = counts or {}
    quick_views = (
        ''.join(
            f'<button type="button" class="{"on" if view.quick_view == key else ""}" aria-pressed="{str(view.quick_view == key).lower()}" '
            f'hx-get="/find/rows?quick_view={key}" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">'
            f'{label}{f" <span class={chr(34)}xr-count{chr(34)}>{counts[key]}</span>" if key in counts else ""}</button>'
            for key, label in (('all', 'All'), ('errors', 'Errors'), ('matches', 'AI matches'))
        )
        + '<span class="xr-qv-sep" aria-hidden="true"></span>'
        + '<details class="xr-top-menu"><summary class="'
        + ('on' if view.quick_view in TOP_METRICS or view.quick_view in CONVERSATION_METRICS else '')
        + '">'
        + (TOP_QUICK_VIEW_LABELS.get(view.quick_view, 'Top 10%') + ' ▾')
        + '</summary><div class="xr-top-list" role="group" aria-label="Top 10% views">'
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
        f'<label><input type="checkbox" name="columns" value="{c.key}"{" checked" if c.key in chosen else ""}>{esc(c.label)}</label>'
        for c in COLUMNS.values()
    )
    switch = ''.join(
        f'<button type="button" class="{"on" if view.view == mode else ""}" aria-pressed="{str(view.view == mode).lower()}" '
        f'hx-get="/find/rows?view={mode}" hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">{label}</button>'
        for mode, label in (('table', 'Table'), ('trajectories', 'Trajectories'))
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
        '<details class="xr-cols"><summary>Columns ▾</summary>'
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


def results(
    view: ExplorerView,
    columns: Sequence[Column],
    *,
    records: Mapping[str, TraceRecord | None] | None,
    snapshot: RunSnapshot | None,
    oob: bool = False,
    error: str | None = None,
    window_days: int = 7,
) -> str:
    snapshot = _within_snapshot(snapshot)
    oob_attr = ' hx-swap-oob="true"' if oob else ''
    poll = ' hx-get="/find/rows" hx-trigger="every 1s" hx-swap="outerHTML"' if view.state == 'loading' else ''
    judged = snapshot.results if snapshot is not None and snapshot.results else None
    counts: dict[str, int] = {'errors': sum(1 for row in view.rows if row.is_error)} if view.rows else {}
    if judged:
        counts['matches'] = sum(1 for row in view.rows if (hit := judged.get(row.trace_id)) is not None and hit.matched)
    toolbar = _toolbar(view, columns, has_results=bool(judged), window_days=window_days, counts=counts)
    judged_count = sum(1 for row in view.rows if row.trace_id in judged) if judged else 0
    judged_in_view = judged_count
    matched_in_view = (
        sum(1 for row in view.rows if (result := judged.get(row.trace_id)) is not None and result.matched)
        if judged
        else 0
    )
    judged_status = (
        f' · {judged_in_view} judged, {matched_in_view} match'
        if judged and view.quick_view != 'matches' and view.state != 'loading'
        else ''
    )
    reason = (
        '<span title="Possible reasons: outside the rows the question ran on, over the AI trace limit, '
        'or no conversation to read, such as embedding calls.">'
        f'{len(view.rows) - judged_count} not judged</span>'
    )
    not_judged = (
        f' · {reason}'
        if judged and snapshot is not None and snapshot.state == 'completed' and judged_count < len(view.rows)
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
        else f'<div class="xr-status" role="status">{len(view.rows)} of {view.narrowed_from} loaded traces match the filters{judged_status}{missing_conversations}{not_judged}</div>'
        if view.narrowed_from is not None
        else f'<div class="xr-status" role="status">{len(view.rows)} traces loaded{judged_status}{not_judged}</div>'
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
        inner = f'<div class="finder-review finder-form-error" role="alert">Load failed: {esc(view.error or "unknown error")}</div>'
    else:
        banner = (
            f'<div class="finder-review finder-form-error" role="alert">Load stopped: {esc(view.error or "unknown error")}. {len(view.rows)} rows loaded.</div>'
            if view.state == 'failed'
            else ''
        )
        results = snapshot.results if snapshot is not None and snapshot.within_results else None
        visible_rows = view.visible_rows(results)
        body = (
            trajectories(view, records or {}, snapshot)
            if view.view == 'trajectories'
            else f'<div class="xr-table-wrap">{table(view, columns, snapshot)}</div>'
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
                    'Rephrase the question, or pick New search to look beyond the loaded rows.</p>'
                    '<button type="button" class="btn-secondary" data-finder-edit>Edit question</button></div>'
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
    inner = f'{status}{toolbar}{inner}'
    error_html = f'<div class="finder-review finder-form-error" role="alert">{esc(error)}</div>' if error else ''
    return f'<section id="explorer-results" class="xr" hx-sync="this:replace"{oob_attr}{poll}>{error_html}{inner}</section>'
