"""HTML builders for the JEV trace-finder dashboard surface."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from urllib.parse import quote

from evaluatorq.common.reports import esc
from evaluatorq.dashboard.apply_ui import drawer as drawer_shell
from evaluatorq.dashboard.security import csrf_field
from evaluatorq.dashboard.shell import page
from evaluatorq.dashboard.trace_links import trace_link_button, trace_span_url
from evaluatorq.trace_finder import classification_legend
from evaluatorq.trace_finder.models import FACET_NAMES

if TYPE_CHECKING:
    from evaluatorq.trace_finder import (
        CompiledQuery,
        DashboardSettings,
        FacetCatalogue,
        FacetSelection,
        RunRequest,
        RunSnapshot,
        TraceClassification,
        TraceDetail,
    )


SAMPLES = (
    'Frustrated customers in the support agent on production this week.',
    'Errored traces on claude-sonnet-5 where the user was blocked by a failed tool.',
    'Conversations over 20k tokens that should have been escalated to a human.',
    'Responses from the docs agent that make unsupported claims.',
)
FACET_LABELS = (
    ('project', 'project'),
    ('agent_name', 'agent'),
    ('model', 'model'),
    ('provider', 'provider'),
    ('status', 'status'),
    ('product', 'product'),
    ('trace_type', 'trace type'),
    ('tool_name', 'tool'),
    ('tokens', 'tokens'),
    ('duration_ms', 'duration'),
)


def icon_search() -> str:
    return (
        '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/>'
        '<path d="m20 20-3.5-3.5"/></svg>'
    )


def examples() -> str:
    items = ''.join(
        f'<button type="button" data-finder-example="{esc(sample)}">{esc(sample)}</button>' for sample in SAMPLES
    )
    return f'<div class="finder-examples"><div class="hd">Examples</div>{items}</div>'


def hero(query: str, mode: str, *, api_available: bool, error: str | None = None) -> str:
    disabled = '' if api_available else ' disabled'
    error_html = f'<div class="finder-form-error" role="alert">{esc(error)}</div>' if error else ''
    selected_immediate = ' selected' if mode == 'immediate' else ''
    selected_review = ' selected' if mode == 'review' else ''
    return (
        '<section class="finder-hero"><div class="finder-hero-bg"></div>'
        '<div class="finder-kicker">Trace finder</div>'
        '<h2 class="finder-title">Ask <em>JEV</em> to find the conversations you care about.</h2>'
        '<form id="finder-query-form" class="finder-query" hx-post="/find/run" hx-target="#finder-body" '
        'hx-swap="innerHTML" hx-include="#finder-controls">'
        f'{csrf_field()}{icon_search()}<div class="col"><textarea name="query" placeholder="Describe the conversations you want to find…" '
        f'required{disabled}>{esc(query)}</textarea></div><div class="finder-run">'
        f'<button class="rt-apply-btn" type="submit"{disabled}>Run</button>'
        '</div></form>'
        '<div class="finder-below"><label class="finder-mode"><span>Mode</span>'
        f'<select name="mode" form="finder-query-form"><option value="immediate"{selected_immediate}>Immediate</option>'
        f'<option value="review"{selected_review}>Review first</option></select></label>'
        f'<span class="ex"><span class="link">Examples ▾</span>{examples()}</span>'
        '<span class="spacer"></span><span class="hint"><kbd>⌘</kbd> <kbd>↵</kbd> to run</span></div>'
        f'{error_html}</section>'
    )


def _facet_values(catalogue: FacetCatalogue | None, name: str) -> tuple[str, ...]:
    if catalogue is None or name in {'tokens', 'duration_ms'}:
        return ()
    return tuple(getattr(catalogue, name, ()))


def facet_menu(
    catalogue: FacetCatalogue | None = None,
    *,
    numeric: object | None = None,
    open_: bool = False,
    form_id: str = 'finder-query-form',
    selection: FacetSelection | None = None,
) -> str:
    rows: list[str] = []
    for name, label in FACET_LABELS:
        kind = '≥ / ≤' if name in {'tokens', 'duration_ms'} else 'any of ▾'
        values = _facet_values(catalogue, name)
        if name in {'tokens', 'duration_ms'}:
            minimum = getattr(numeric, f'{name}_min', None) if numeric is not None else None
            maximum = getattr(numeric, f'{name}_max', None) if numeric is not None else None
            rows.append(
                f'<div class="facet-group"><div><span>{esc(label)}</span><span class="kind">{kind}</span></div>'
                f'<label><span>≥</span><input form="{form_id}" name="{esc(name)}_min" type="number" min="0" '
                f'placeholder="min" value="{esc(str(minimum)) if minimum is not None else ""}"></label>'
                f'<label><span>≤</span><input form="{form_id}" name="{esc(name)}_max" type="number" min="0" '
                f'placeholder="max" value="{esc(str(maximum)) if maximum is not None else ""}"></label></div>'
            )
        elif values:
            selected_values = getattr(selection, name, frozenset()) if selection is not None else frozenset()
            options = ''.join(
                f'<label><input form="{form_id}" type="checkbox" name="facet_{esc(name)}" value="{esc(value)}"{(" checked" if value in selected_values else "")}>{esc(value)}</label>'
                for value in values
            )
            rows.append(
                f'<div class="facet-group"><div><span>{esc(label)}</span><span class="kind">{kind}</span></div>{options}</div>'
            )
        else:
            rows.append(
                f'<div class="facet-group"><div><span>{esc(label)}</span><span class="kind">{kind}</span></div></div>'
            )
    if catalogue is None:
        rows.append('<p class="finder-empty">Facet values are unavailable; check the Orq connection and reopen.</p>')
    elif not any(_facet_values(catalogue, name) for name, _ in FACET_LABELS):
        rows.append('<p class="finder-empty">No facet values available.</p>')
    return f'<div class="finder-facets{" open" if open_ else ""}">{"".join(rows)}</div>'


def _facet_chips(
    selection: FacetSelection,
    generated: FacetSelection,
    numeric: object | None = None,
    generated_numeric: object | None = None,
    *,
    removable: bool = False,
) -> str:
    chips: list[str] = []
    for name, label in FACET_LABELS[:8]:
        values = getattr(selection, name, frozenset())
        if not values:
            continue
        generated_values = getattr(generated, name, frozenset())
        for value in sorted(values):
            jev = ' jev' if value in generated_values else ''
            tag = '<span class="tag">jev</span>' if jev else ''
            remove = (
                f'<button type="button" class="finder-chip-remove" data-finder-remove="facet_{name}" '
                f'data-finder-value="{esc(value)}" aria-label="Remove {esc(label)} {esc(value)}">✕</button>'
                if removable
                else '<i>✕</i>'
            )
            chips.append(f'<span class="chip{jev}"><b>{esc(label)}</b>{esc(value)}{tag}{remove}</span>')
    for name, label, operator in (
        ('tokens_min', 'tokens', '≥'),
        ('tokens_max', 'tokens', '≤'),
        ('duration_ms_min', 'duration', '≥'),
        ('duration_ms_max', 'duration', '≤'),
    ):
        value = getattr(numeric, name, None) if numeric is not None else None
        if value is None:
            continue
        generated_value = getattr(generated_numeric, name, None) if generated_numeric is not None else None
        jev = ' jev' if generated_value == value else ''
        tag = '<span class="tag">jev</span>' if jev else ''
        remove = (
            f'<button type="button" class="finder-chip-remove" data-finder-remove="{name}" '
            f'aria-label="Remove {esc(label)} {operator} {value}">✕</button>'
            if removable
            else '<i>✕</i>'
        )
        chips.append(f'<span class="chip{jev}"><b>{label}</b>{operator} {value}{tag}{remove}</span>')
    return ''.join(chips)


def controls(snapshot: RunSnapshot, settings: DashboardSettings, catalogue: FacetCatalogue | None = None) -> str:
    request = snapshot.request
    population = request.population if request is not None else None
    selection = population.facets if population is not None else None
    facets = selection or _empty_facets()
    review = snapshot.state == 'awaiting_review'
    form_id = 'finder-start-form' if review else 'finder-query-form'
    window_days = settings.window_days
    if population is not None and population.start is not None and population.end is not None:
        window_days = max(1, round((population.end - population.start).total_seconds() / 86400))
    values = {
        'window_days': window_days,
        'limit': population.limit if population is not None else settings.limit,
        'parallelism': request.parallelism if request is not None else settings.parallelism,
    }
    numeric = population.numeric if population is not None else None
    count_html = f'<span class="count">{esc(str(snapshot.total))} traces selected</span>' if snapshot.total else ''
    hidden_facets = ''.join(
        f'<input type="hidden" form="{form_id}" name="facet_{name}" value="{esc(value)}">'
        for name in FACET_NAMES
        for value in sorted(getattr(facets, name))
    )
    return (
        '<div class="finder-controls" id="finder-controls">'
        f'{hidden_facets}{_facet_chips(facets, snapshot.generated_filters, numeric, snapshot.generated_numeric, removable=review)}'
        f'<span class="addwrap"><button class="add" type="button" hx-get="/find/facets?form_id={form_id}" hx-include="#finder-window" hx-target=".finder-facets" '
        f'hx-swap="outerHTML">+ Filter</button>{facet_menu(catalogue, numeric=numeric, form_id=form_id, selection=facets)}</span><span class="spacer"></span>'
        f'<span class="quiet"><b>Window</b><input id="finder-window" form="{form_id}" name="window_days" type="number" min="1" max="90" value="{values["window_days"]}" style="width:52px"></span>'
        f'<span class="quiet"><b>Limit</b><input form="{form_id}" name="limit" type="number" min="1" max="500" value="{values["limit"]}" style="width:52px"></span>'
        f'<span class="quiet"><b>Parallel</b><input form="{form_id}" name="parallelism" type="number" min="1" max="200" value="{values["parallelism"]}" style="width:48px"></span>'
        f'{count_html}</div>'
    )


def _empty_facets() -> FacetSelection:
    from evaluatorq.trace_finder import FacetSelection

    return FacetSelection()


def _value_text(value: object) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        return f'{value:g}'
    return str(value)


def _result_color(result: TraceClassification | None, compiled: CompiledQuery | None) -> str:
    if result is None or result.error:
        return 'var(--red-600)' if result and result.error else '#3d3c4a'
    if compiled is None:
        return 'var(--chart-5)' if result.matched else 'var(--chart-2)'
    for item in classification_legend(compiled):
        if item.label == _value_text(result.value):
            return item.color
    return 'var(--chart-5)' if result.matched else 'var(--chart-2)'


def matrix(snapshot: RunSnapshot) -> str:
    traces = snapshot.traces
    if not traces:
        idle_dots = ''.join('<i class="idle"></i>' for _ in range(500))
        return f'<div class="finder-matrix idle">{idle_dots}</div>'
    unresolved = [trace.trace_id for trace in traces if trace.trace_id not in snapshot.results]
    active = set(unresolved[: snapshot.active]) if snapshot.state == 'classifying' else set()
    dots: list[str] = []
    for trace in traces[:500]:
        result = snapshot.results.get(trace.trace_id)
        if result is not None:
            state = 'failed' if result.error else 'match' if result.matched else 'unmatched'
            title = (
                result.error
                or f'{_value_text(result.value)} · {_value_text(result.confidence) if result.confidence is not None else "no confidence"}'
            )
        elif trace.trace_id in active:
            state, title = 'active', 'classifying'
        else:
            state, title = 'pending', 'pending'
        color = _result_color(result, snapshot.compiled)
        trace_id = quote(trace.trace_id, safe='')
        dots.append(
            f'<i class="{state}" style="--c:{esc(color)}" title="{esc(title)}" '
            f'hx-get="/find/trace/{trace_id}" hx-target="#finder-drawer" hx-swap="innerHTML"></i>'
        )
    return f'<div class="finder-matrix">{"".join(dots)}</div>'


def legend(snapshot: RunSnapshot) -> str:
    counts: dict[str, int] = {}
    for result in snapshot.results.values():
        if result.error is None:
            label = _value_text(result.value)
            counts[label] = counts.get(label, 0) + 1
    items = ''.join(
        f'<span><span class="sw" style="background:{esc(item.color)}"></span>{esc(item.label)} '
        f'<b>{counts.get(item.label, 0)}</b></span>'
        for item in (classification_legend(snapshot.compiled) if snapshot.compiled else ())
    )
    return (
        '<div class="finder-legend">'
        f'<span><span class="sw match"></span><span class="num hot">{snapshot.matched}</span> included</span>'
        f'<span class="muted">|</span>{items}'
        f'<span><span class="sw failed"></span>failed <b>{snapshot.failed}</b></span>'
        f'<span class="muted">|</span><span class="muted">{snapshot.completed} judged</span></div>'
    )


def progress(snapshot: RunSnapshot) -> str:
    running = snapshot.state in {'compiling', 'classifying'}
    state = 'classifying' if snapshot.state == 'classifying' else snapshot.state.replace('_', ' ')
    reset = (
        f'<form hx-post="/find/reset" hx-target="#finder-body" hx-swap="innerHTML">{csrf_field()}<button class="btn-secondary" type="submit">Reset</button></form>'
        if snapshot.state != 'idle'
        else ''
    )
    action = (
        f'<form hx-post="/find/cancel" hx-target="#finder-body" hx-swap="innerHTML">{csrf_field()}<button class="btn-secondary" type="submit">Cancel</button></form>'
        if running
        else f'<a class="btn-secondary" href="/find/export.json">Download JSON</a>{reset}'
        if snapshot.state == 'completed'
        else reset
    )
    live_html = '<span class="live"></span>' if running else ''
    error_html = (
        f'<span class="finder-progress-error finder-review" role="alert">{esc(snapshot.error)}</span>'
        if snapshot.error
        else ''
    )
    return (
        f'<div class="finder-progress">{live_html}<span class="state">{esc(state)}</span>'
        f'<span class="sep">·</span><span><b>{snapshot.completed} / {snapshot.total}</b> judged</span>'
        f'<span class="sep">·</span><span><b>{snapshot.failed}</b> failed</span><span class="sep">·</span>'
        f'<span><b>{snapshot.rate:.1f}</b>/s</span><span class="sep">·</span><span>{snapshot.elapsed:.1f}s</span>{error_html}{action}</div>'
    )


def field(snapshot: RunSnapshot, *, api_available: bool = True) -> str:
    body = matrix(snapshot)
    if snapshot.state == 'idle':
        if api_available:
            body += '<div class="finder-hint"><div class="inner"><h4>Every dot is a trace. Ask a question to light them up.</h4>'
            body += '<p>JEV picks filters from your wording, judges recent traces, and marks the ones that match.</p></div></div>'
        else:
            body += '<div class="finder-hint"><div class="inner"><h4>Set ORQ_API_KEY to load traces</h4>'
            body += '<p>Trace finding is unavailable until the Orq API key is configured.</p></div></div>'
    elif not snapshot.traces:
        body += '<div class="finder-hint"><div class="inner"><h4>No traces loaded.</h4><p>No traces match the current window and filters.</p></div></div>'
    unavailable = ' unavailable' if snapshot.state == 'idle' and not api_available else ''
    return f'<div class="finder-field{unavailable}">{progress(snapshot) if snapshot.state != "idle" else ""}{body}{legend(snapshot) if snapshot.compiled else ""}</div>'


def table(snapshot: RunSnapshot) -> str:
    traces_by_id = {trace.trace_id: trace for trace in snapshot.traces}
    matches = [
        (traces_by_id[trace_id], snapshot.results[trace_id])
        for trace_id in snapshot.trace_ids
        if trace_id in traces_by_id
        and trace_id in snapshot.results
        and snapshot.results[trace_id].matched
        and not snapshot.results[trace_id].error
    ]
    matches.sort(key=lambda pair: pair[0].timestamp, reverse=True)
    rows = ''.join(
        f'<tr hx-get="/find/trace/{quote(trace.trace_id, safe="")}" hx-target="#finder-drawer" hx-swap="innerHTML">'
        f'<td class="id">{esc(trace.trace_id)}</td><td><span class="verdict"><span class="sw" style="background:{esc(_result_color(result, snapshot.compiled))}"></span>{esc(_value_text(result.value))}</span></td>'
        f'<td class="conf">{esc(f"{result.confidence:.2f}" if result.confidence is not None else "—")}</td>'
        f'<td>{esc(trace.project)}</td><td>{esc(trace.model)}</td><td>{esc(trace.status)}</td><td>{esc(trace.timestamp.strftime("%Y-%m-%d %H:%M"))}</td></tr>'
        for trace, result in matches
    )
    empty = '<p class="finder-empty">No matches yet.</p>' if not rows else ''
    return (
        '<section class="finder-section"><h3 class="finder-section-title">Included traces</h3>'
        f'<p class="finder-section-sub">{snapshot.matched} of {snapshot.total or 0} judged as included, newest first. Click a row to inspect.</p>'
        '<table class="finder-table"><thead><tr><th>Trace</th><th>Verdict</th><th>Conf.</th><th>Project</th><th>Model</th><th>Status</th><th>Time</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>{empty}</section>'
    )


def _task_kind_select(kind: str, *, editable: bool) -> str:
    return f'<p class="finder-kind-badge">{esc(kind)}</p>'


def _selection_rule_html(compiled: CompiledQuery, *, editable: bool) -> str:
    from evaluatorq.trace_finder import ThresholdSelection

    task = compiled.task
    selection = compiled.selection
    if task.kind == 'choice':
        labels = tuple(task.criteria) if isinstance(task.criteria, dict) else ()
        selected = next(
            (value for value in getattr(selection, 'values', ()) if value in labels), labels[0] if labels else ''
        )
        if not editable:
            return f'<p>include verdict = <b>{esc(str(selected))}</b></p>'
        options = ''.join(
            f'<option value="{esc(str(label))}"{" selected" if label == selected else ""}>{esc(str(label))}</option>'
            for label in labels
        )
        return f'<select name="selection_value">{options}</select>'
    if task.kind == 'noul':
        selected_bool = next((value for value in getattr(selection, 'values', ()) if type(value) is bool), False)
        if not editable:
            return f'<p>include verdict = <b>{str(selected_bool).lower()}</b></p>'
        options = ''.join(
            f'<option value="{value}"{" selected" if selected_bool == (value == "true") else ""}>{value}</option>'
            for value in ('true', 'false')
        )
        return f'<select name="selection_value">{options}</select>'
    if not isinstance(selection, ThresholdSelection):
        operator, threshold = 'gte', 0.5
    else:
        operator, threshold = selection.operator, selection.value
    if not editable:
        return f'<p>include score {esc(operator)} <b>{threshold:g}</b></p>'
    options = ''.join(
        f'<option value="{op}:{value:g}"{" selected" if operator == op and threshold == value else ""}>{op} {value:g}</option>'
        for op, value in (('gte', 0.5), ('gte', 0.7), ('gte', 0.8), ('lte', 0.3), ('lte', 0.5))
    )
    return f'<select name="selection_rule">{options}</select>'


def _criterion_html(compiled: CompiledQuery, *, editable: bool) -> tuple[str, int]:
    task = compiled.task
    if task.kind == 'choice' and isinstance(task.criteria, dict):
        pairs = tuple(task.criteria.items())
        if editable:
            return (
                ''.join(
                    f'<div class="finder-crit-row"><input name="criteria_label_{index}" value="{esc(str(label))}" required>'
                    f'<input name="criteria_description_{index}" value="{esc(str(description or ""))}" required></div>'
                    for index, (label, description) in enumerate(pairs)
                ),
                len(pairs),
            )
        return (
            ''.join(
                f'<span class="lab"><span class="sw" style="background:var(--chart-{index + 1})"></span>{esc(str(label))}</span>'
                f'<span class="desc">{esc(str(description or ""))}</span>'
                for index, (label, description) in enumerate(pairs)
            ),
            len(pairs),
        )
    if task.kind == 'score' and isinstance(task.criteria, list):
        if editable:
            return (
                ''.join(
                    f'<div class="finder-crit-row"><span class="lab">Level {index + 1}</span>'
                    f'<input name="score_criteria_{index}" value="{esc(str(description))}" required></div>'
                    for index, description in enumerate(task.criteria)
                ),
                len(task.criteria),
            )
        return (
            ''.join(
                f'<span class="lab">Level {index + 1}</span><span class="desc">{esc(str(description))}</span>'
                for index, description in enumerate(task.criteria)
            ),
            len(task.criteria),
        )
    return '<p>Binary true / false judgment.</p>', 0


def task_panel(
    compiled: CompiledQuery,
    *,
    editable: bool,
    open_: bool = False,
    request: RunRequest | None = None,
) -> str:
    task = compiled.task
    criterion_html, criterion_count = _criterion_html(compiled, editable=editable)
    instruction = (
        f'<textarea name="instructions" required>{esc(task.instructions)}</textarea>'
        if editable
        else f'<p>{esc(task.instructions)}</p>'
    )
    threshold = (
        f'<div><h5>Noul threshold</h5><input name="noul_threshold" type="number" min="0" max="1" step="0.01" value="{task.noul_threshold:g}"></div>'
        if editable and task.kind == 'noul'
        else ''
    )
    if editable:
        form_open = '<form id="finder-start-form" hx-post="/find/start" hx-target="#finder-body" hx-swap="innerHTML">'
        hidden_request = ''
        if request is not None:
            hidden_request = (
                f'<input type="hidden" name="query" value="{esc(request.query)}">'
                f'<input type="hidden" name="mode" value="review">'
            )
        form_open += hidden_request + csrf_field()
        form_close = '<button class="rt-apply-btn" type="submit">Start classification</button></form>'
    else:
        form_open = ''
        form_close = ''
    return (
        f'<details class="finder-task"{" open" if open_ else ""}><summary><span class="chev">▸</span><span class="kind">{esc(task.kind)}</span>'
        f'<span>Generated JEV task · {criterion_count} labels</span></summary>{form_open}'
        f'<div class="finder-task-body"><div><h5>Kind</h5>{_task_kind_select(task.kind, editable=editable)}</div>'
        f'<div><h5>Inclusion rule</h5>{_selection_rule_html(compiled, editable=editable)}</div>{threshold}'
        f'<div class="full"><h5>Instructions</h5>{instruction}</div>'
        f'<div class="full"><h5>Criteria</h5><div class="finder-crit">{criterion_html}</div></div>'
        f'<div class="full"><h5>Raw plan</h5><pre>{esc(json.dumps(compiled.model_dump(mode="json"), indent=2))}</pre></div></div>{form_close}</details>'
    )


def body(
    snapshot: RunSnapshot,
    settings: DashboardSettings,
    *,
    catalogue: FacetCatalogue | None = None,
    api_available: bool = True,
) -> str:
    if snapshot.state == 'awaiting_review' and snapshot.compiled is not None:
        return (
            f'{controls(snapshot, settings, catalogue)}<div class="finder-review"><span>⏸</span><span><b>Review the plan before running per-trace JEV classification.</b> '
            'Edit the task, criteria or filters, then start.</span></div>'
            f'{task_panel(snapshot.compiled, editable=True, open_=True, request=snapshot.request)}'
        )
    if snapshot.state == 'idle':
        return f'{controls(snapshot, settings, catalogue)}{field(snapshot, api_available=api_available)}'
    return f'{controls(snapshot, settings, catalogue)}{field(snapshot, api_available=api_available)}{table(snapshot)}{task_panel(snapshot.compiled, editable=False) if snapshot.compiled else ""}'


def page_html(
    snapshot: RunSnapshot, settings: DashboardSettings, *, api_available: bool, error: str | None = None
) -> str:
    request = snapshot.request
    query = request.query if request is not None else ''
    mode = request.mode if request is not None else 'immediate'
    polling = (
        ' hx-get="/find/poll" hx-trigger="every 1s" hx-target="#finder-body" hx-swap="innerHTML"'
        if snapshot.state in {'compiling', 'classifying'}
        else ''
    )
    html = f'<div class="finder">{hero(query, mode, api_available=api_available, error=error)}<div id="finder-body"{polling}>{body(snapshot, settings, api_available=api_available)}</div><div id="finder-drawer"></div></div>'
    return page('Find', html, active_nav='find')


def fragment(
    snapshot: RunSnapshot,
    settings: DashboardSettings,
    *,
    error: str | None = None,
    api_available: bool = True,
) -> str:
    attrs = ''
    if snapshot.state in {'compiling', 'classifying'}:
        attrs = ' hx-get="/find/poll" hx-trigger="every 1s" hx-target="#finder-body" hx-swap="innerHTML"'
    error_html = f'<div class="finder-review finder-form-error" role="alert">{esc(error)}</div>' if error else ''
    return f'<div class="finder-body-fragment"{attrs}>{error_html}{body(snapshot, settings, api_available=api_available)}</div>'


def drawer(detail: TraceDetail, *, experiment_url: str | None = None) -> str:
    trace = detail.trace
    result = detail.classification
    result_html = (
        '<p>Not classified yet.</p>'
        if result is None
        else (
            f'<p><b>{"Included" if result.matched and not result.error else "Not included"}</b> · {esc(_value_text(result.value))}</p>'
            if not result.error
            else f'<p role="alert">Failed: {esc(result.error)}</p>'
        )
    )
    messages = ''.join(
        f'<div class="fd-msg {"user" if message.get("role") == "user" else "assistant"}"><div class="role">{esc(str(message.get("role", "unknown")))}</div>{esc(_message_text(message.get("content")))}</div>'
        for message in trace.messages
    )
    jev = json.dumps(detail.projection.payload if detail.projection else {}, indent=2, ensure_ascii=False)
    raw = json.dumps(result.raw_result if result else {}, indent=2, ensure_ascii=False)
    thread_html = messages or '<p class="finder-empty">No messages.</p>'
    body_html = (
        f'<dl class="fd-meta"><dt>trace</dt><dd>{esc(trace.trace_id)}</dd><dt>span</dt><dd>{esc(trace.span_id)}</dd>'
        f'<dt>project</dt><dd>{esc(trace.project)}</dd><dt>model</dt><dd>{esc(trace.model)}</dd><dt>time</dt><dd>{esc(trace.timestamp.isoformat())}</dd></dl>'
        f'<div class="fd-verdict"><span class="sw" style="background:{esc(_result_color(result, detail.compiled))}"></span>{result_html}</div>'
        '<div class="fd-tabs">'
        '<button type="button" class="on" data-panel="fd-thread" onclick="eqFinderTab(this,\'fd-thread\')">Full thread</button>'
        '<button type="button" data-panel="fd-jev" onclick="eqFinderTab(this,\'fd-jev\')">JEV input</button>'
        '<button type="button" data-panel="fd-raw" onclick="eqFinderTab(this,\'fd-raw\')">Raw result</button></div>'
        f'<div id="fd-thread" class="fd-panel"><div class="fd-panel-title">Full thread</div>{thread_html}</div>'
        f'<div id="fd-jev" class="fd-panel" hidden><div class="fd-panel-title">JEV input</div><pre>{esc(jev)}</pre></div>'
        f'<div id="fd-raw" class="fd-panel" hidden><div class="fd-panel-title">Raw result</div><pre>{esc(raw)}</pre></div>'
    )
    url = trace_span_url(trace.trace_id, trace.span_id, experiment_url)
    footer = (
        trace_link_button(url, 'Open in Orq ↗')
        + f'<button class="btn-secondary" type="button" data-trace-id="{esc(trace.trace_id)}" onclick="navigator.clipboard.writeText(this.dataset.traceId)">Copy trace id</button>'
    )
    return drawer_shell(
        f'Trace {esc(trace.trace_id)}', body_html, footer, dismiss_route='/find/dismiss', drawer_id='finder-drawer'
    )


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ''
    return json.dumps(content, ensure_ascii=False, default=str)
