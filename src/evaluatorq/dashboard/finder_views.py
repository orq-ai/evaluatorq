"""HTML builders for the classifier trace-finder dashboard surface."""

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
from evaluatorq.trace_finder.models import FACET_NAMES, NUMERIC_FACET_NAMES

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
if {name for name, _ in FACET_LABELS} != {*FACET_NAMES, *NUMERIC_FACET_NAMES}:
    raise RuntimeError('FACET_LABELS must mirror FACET_NAMES and NUMERIC_FACET_NAMES')


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
    checked_immediate = ' checked' if mode != 'review' else ''
    checked_review = ' checked' if mode == 'review' else ''
    return (
        '<section class="finder-hero"><div class="finder-hero-bg"></div>'
        '<h2 class="finder-title">Find the signal.</h2>'
        '<p class="finder-sub">Ask a question of your traces. Inspect every judgment.</p>'
        '<form id="finder-query-form" class="finder-query" hx-post="/find/run" hx-target="#finder-body" '
        'hx-swap="innerHTML" hx-include="#finder-controls" hx-disabled-elt="find button">'
        f'{csrf_field()}{icon_search()}<div class="col"><textarea name="query" rows="1" placeholder="Describe the conversations you want to find…" '
        f'required{disabled}>{esc(query)}</textarea></div>'
        f'<button class="finder-go" type="submit"{disabled}><span class="finder-go-idle">Find traces <span aria-hidden="true">↗</span></span>'
        '<span class="finder-go-working" role="status">Starting search…</span></button>'
        '</form>'
        '<div class="finder-below"><div class="finder-seg" role="radiogroup" aria-label="Mode">'
        f'<label><input type="radio" name="mode" value="immediate" form="finder-query-form"{checked_immediate} '
        'hx-post="/find/reset" hx-trigger="change[document.getElementById(\'finder-start-form\')]" '
        'hx-include="#finder-query-form" hx-target="#finder-body" hx-swap="innerHTML" hx-indicator="#finder-mode-working"><span>Immediate</span></label>'
        f'<label><input type="radio" name="mode" value="review" form="finder-query-form"{checked_review}><span>Review first</span></label></div>'
        '<span id="finder-mode-working" role="status">Resetting review…</span>'
        f'<span class="ex"><button type="button" class="link">Examples ▾</button>{examples()}</span>'
        '<span class="spacer"></span><span class="hint"><kbd>⌘</kbd> <kbd>↵</kbd> to run</span></div>'
        f'{error_html}</section>'
    )


def _facet_values(catalogue: FacetCatalogue | None, name: str) -> tuple[str, ...]:
    if catalogue is None or name in NUMERIC_FACET_NAMES:
        return ()
    return tuple(getattr(catalogue, name, ()))


def facet_menu(
    catalogue: FacetCatalogue | None = None,
    *,
    numeric: object | None = None,
    open_: bool = False,
    form_id: str = 'finder-query-form',
    selection: FacetSelection | None = None,
    pending: bool = False,
) -> str:
    """Two-level filter menu: a category list, and a value popout the client opens per category.

    ``pending`` renders the menu without values and has it fetch them itself as soon as it lands
    on the page, so a page render never waits on the Orq facet call.
    """
    items: list[str] = []
    subs: list[str] = []
    for name, label in FACET_LABELS:
        numeric_facet = name in NUMERIC_FACET_NAMES
        values = _facet_values(catalogue, name)
        selected_values = getattr(selection, name, frozenset()) if selection is not None else frozenset()
        if numeric_facet:
            minimum = getattr(numeric, f'{name}_min', None) if numeric is not None else None
            maximum = getattr(numeric, f'{name}_max', None) if numeric is not None else None
            count = sum(bound is not None for bound in (minimum, maximum))
            body = (
                f'<label><span>≥</span><input form="{form_id}" name="{esc(name)}_min" type="number" min="0" '
                f'placeholder="min" value="{esc(str(minimum)) if minimum is not None else ""}"></label>'
                f'<label><span>≤</span><input form="{form_id}" name="{esc(name)}_max" type="number" min="0" '
                f'placeholder="max" value="{esc(str(maximum)) if maximum is not None else ""}"></label>'
            )
        else:
            count = len(selected_values)
            body = (
                ''.join(
                    f'<label><input form="{form_id}" type="checkbox" name="facet_{esc(name)}" value="{esc(value)}"{(" checked" if value in selected_values else "")}><span>{esc(value)}</span></label>'
                    for value in values
                )
                or '<p class="finder-empty">No values in this window.</p>'
            )
        count_html = f'<span class="count">{count}</span>' if count else ''
        items.append(
            f'<button type="button" class="facet-item" data-facet="{esc(name)}" aria-haspopup="true" '
            f'aria-expanded="false"><span>{esc(label)}</span>{count_html}<span class="chev" aria-hidden="true">&rsaquo;</span></button>'
        )
        subs.append(
            f'<div class="facet-sub" data-facet-sub="{esc(name)}" hidden><div class="hd">{esc(label)}</div>{body}</div>'
        )
    if pending:
        note = '<p class="finder-empty">Loading facet values…</p>'
        loader = f' hx-get="/find/facets?form_id={form_id}" hx-trigger="load" hx-include="#finder-controls" hx-swap="outerHTML" hx-indicator=".finder-facet-loading"'
    else:
        note = (
            '<p class="finder-empty">Facet values are unavailable; check the Orq connection and reopen.</p>'
            if catalogue is None
            else ''
        )
        loader = ''
    return (
        f'<div class="finder-facets{" open" if open_ else ""}{" pending" if pending else ""}"{loader}><div class="facet-list">{"".join(items)}{note}</div>'
        f'{"".join(subs)}</div>'
    )


def _facet_chips(selection: FacetSelection, numeric: object | None = None, *, removable: bool = False) -> str:
    """Render one chip per active filter value. Editable chips open the already-rendered menu at their category."""

    def chip(facet: str, label: str, value_html: str, remove_name: str, remove_value: str | None, aria: str) -> str:
        if not removable:
            return f'<span class="chip"><b>{esc(label)}</b><span class="v">{value_html}</span></span>'
        value_attr = f' data-finder-value="{esc(remove_value)}"' if remove_value is not None else ''
        return (
            f'<span class="chip is-editable" data-chip-name="{esc(remove_name)}"{value_attr}>'
            f'<button type="button" class="chip-open" data-chip-open="{esc(facet)}" aria-label="Edit {aria}">'
            f'<b>{esc(label)}</b><span class="v">{value_html}</span></button>'
            f'<button type="button" class="finder-chip-remove" data-finder-remove="{esc(remove_name)}"{value_attr} '
            f'aria-label="Remove {aria}">✕</button></span>'
        )

    chips: list[str] = []
    for name, label in FACET_LABELS:
        if name in NUMERIC_FACET_NAMES:
            continue
        chips.extend(
            chip(name, label, esc(value), f'facet_{name}', value, f'{esc(label)} {esc(value)}')
            for value in sorted(getattr(selection, name, frozenset()))
        )
    for name, label, operator in (
        ('tokens_min', 'tokens', '≥'),
        ('tokens_max', 'tokens', '≤'),
        ('duration_ms_min', 'duration', '≥'),
        ('duration_ms_max', 'duration', '≤'),
    ):
        value = getattr(numeric, name, None) if numeric is not None else None
        if value is None:
            continue
        facet = name.rsplit('_', 1)[0]
        chips.append(chip(facet, label, f'{operator} {value}', name, None, f'{label} {operator} {value}'))
    return ''.join(chips)


def controls(
    snapshot: RunSnapshot,
    settings: DashboardSettings,
    catalogue: FacetCatalogue | None = None,
    *,
    pending: bool = False,
) -> str:
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
    # The 1s poll re-renders this row; hx-preserve keeps the live inputs so a value typed for the next run
    # survives. The id carries the form so a review-form input is never carried into the query form.
    keep = {name: f'id="finder-{name}-{form_id}" hx-preserve' for name in values}
    # A reviewed start reuses the whole population, classifier picks included. A fresh query only carries
    # the filters the user set themselves; the classifier's picks for the last question are not sticky.
    carried_facets = facets if review else snapshot.explicit_filters
    carried_numeric = numeric if review else snapshot.explicit_numeric
    count_html = f'<span class="count">{esc(str(snapshot.total))} traces selected</span>' if snapshot.total else ''
    scope_html = (
        f'<span class="quiet"><b>Project</b><a href="/settings">{esc(settings.orq_project_name or settings.orq_project_id)}</a></span>'
        if settings.orq_project_id
        else ''
    )
    hidden_facets = ''.join(
        f'<input type="hidden" form="{form_id}" name="facet_{name}" value="{esc(value)}">'
        for name in FACET_NAMES
        for value in sorted(getattr(carried_facets, name))
    )
    return (
        '<div class="finder-controls" id="finder-controls">'
        f'{hidden_facets}{scope_html}{_facet_chips(facets, numeric, removable=snapshot.state not in {"compiling", "classifying"})}'
        f'<span class="addwrap"><button class="add" type="button" aria-haspopup="true">+ Filter</button>'
        f'{facet_menu(catalogue, numeric=carried_numeric, form_id=form_id, selection=carried_facets, pending=pending)}'
        '<span class="finder-facet-loading" role="status">Loading filters…</span></span><span class="spacer"></span>'
        f'<span class="quiet"><b>Window</b><input {keep["window_days"]} form="{form_id}" name="window_days" type="number" min="1" max="90" value="{values["window_days"]}" style="width:64px" '
        f'hx-get="/find/facets?form_id={form_id}" hx-trigger="change" hx-include="#finder-controls" hx-target=".finder-facets" hx-swap="outerHTML" hx-indicator=".finder-facet-loading"></span>'
        f'<span class="quiet"><b>Limit</b><input {keep["limit"]} form="{form_id}" name="limit" type="number" min="1" max="5000" value="{values["limit"]}" style="width:72px"></span>'
        f'<span class="quiet"><b>Parallel</b><input {keep["parallelism"]} form="{form_id}" name="parallelism" type="number" min="1" max="200" value="{values["parallelism"]}" style="width:64px"></span>'
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
    if compiled.task.kind == 'score' and isinstance(result.value, (int, float)) and not isinstance(result.value, bool):
        score = max(0.0, min(1.0, result.value))
        return f'color-mix(in srgb, var(--chart-5) {score * 100:g}%, var(--chart-2))'
    for item in classification_legend(compiled):
        if item.label == _value_text(result.value):
            return item.color
    return 'var(--chart-5)' if result.matched else 'var(--chart-2)'


def matrix(snapshot: RunSnapshot) -> str:
    traces = snapshot.traces
    if not traces:
        # Decorative: a CSS dot pattern that tiles the whole field, edge to edge, at any width.
        return '<div class="finder-matrix idle" aria-hidden="true"></div>'
    unresolved = [trace.trace_id for trace in traces if trace.trace_id not in snapshot.results]
    active = set(unresolved[: snapshot.active]) if snapshot.state == 'classifying' else set()
    dots: list[str] = []
    for trace in traces:
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
            f'hx-get="/find/trace/{trace_id}" hx-target="#finder-drawer" hx-swap="innerHTML" hx-indicator="#finder-drawer-loading"></i>'
        )
    return f'<div class="finder-matrix">{"".join(dots)}</div>'


def legend(snapshot: RunSnapshot) -> str:
    counts: dict[str, int] = {}
    for result in snapshot.results.values():
        if result.error is None:
            label = _value_text(result.value)
            counts[label] = counts.get(label, 0) + 1
    if snapshot.compiled is not None and snapshot.compiled.task.kind == 'score':
        entries = classification_legend(snapshot.compiled)
        counts[entries[0].label] = sum(result.error is None for result in snapshot.results.values())
        counts[entries[1].label] = snapshot.matched
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
    state = {
        'planning': 'planning search',
        'loading_traces': 'loading traces',
        'starting_classification': 'starting classification',
    }.get(snapshot.phase or '', snapshot.state.replace('_', ' '))
    reset = (
        f'<form class="finder-progress-action" hx-post="/find/reset" hx-target="#finder-body" hx-swap="innerHTML" hx-disabled-elt="find button">{csrf_field()}<button class="btn-secondary" type="submit">Reset</button><span role="status">Resetting…</span></form>'
        if snapshot.state != 'idle'
        else ''
    )
    action = (
        f'<form class="finder-progress-action" hx-post="/find/cancel" hx-target="#finder-body" hx-swap="innerHTML" hx-disabled-elt="find button">{csrf_field()}<button class="btn-secondary" type="submit">Cancel</button><span role="status">Cancelling…</span></form>'
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
    counts = (
        '<span class="sep">·</span><span>'
        + {
            'planning': 'compiling the question and selecting metadata filters',
            'loading_traces': 'loading selected traces',
            'starting_classification': 'preparing the reviewed task',
        }.get(snapshot.phase or '', 'preparing the search')
        + '</span>'
        if snapshot.state == 'compiling'
        else '<span class="sep">·</span><span>Stopped before traces were loaded</span>'
        if snapshot.total == 0
        else f'<span class="sep">·</span><span><b>{snapshot.completed} / {snapshot.total}</b> judged</span>'
        f'<span class="sep">·</span><span><b>{snapshot.failed}</b> failed</span><span class="sep">·</span>'
        f'<span><b>{snapshot.rate:.1f}</b>/s</span>'
    )
    elapsed_html = (
        f'<span class="sep">·</span><span>{snapshot.elapsed:.1f}s</span>'
        if snapshot.state != 'compiling' and snapshot.total
        else ''
    )
    return (
        f'<div class="finder-progress">{live_html}<span class="state">{esc(state)}</span>{counts}'
        f'{elapsed_html}{error_html}{action}</div>'
    )


def field(snapshot: RunSnapshot, *, api_available: bool = True) -> str:
    body = matrix(snapshot)
    if snapshot.state == 'idle':
        if api_available:
            body += '<div class="finder-hint"><div class="inner"><h4>Every dot is a trace. Ask a question to light them up.</h4>'
            body += '<p>The classifier picks filters from your wording, judges recent traces, and marks the ones that match.</p></div></div>'
        else:
            body += '<div class="finder-hint"><div class="inner"><h4>Set ORQ_API_KEY to load traces</h4>'
            body += '<p>Trace finding is unavailable until the Orq API key is configured.</p></div></div>'
    elif snapshot.state == 'compiling':
        heading, description = {
            'planning': ('Planning the search…', 'Compiling your question and selecting metadata filters.'),
            'loading_traces': ('Loading traces…', 'Fetching the trace population selected by your filters.'),
            'starting_classification': ('Starting classification…', 'Preparing the reviewed task and selected traces.'),
        }.get(snapshot.phase or '', ('Preparing the search…', 'Getting the next step ready.'))
        body += (
            '<div class="finder-hint finder-compiling" role="status" aria-live="polite"><div class="inner">'
            '<span class="finder-pulse" aria-hidden="true"><i></i><i></i><i></i></span>'
            f'<h4>{esc(heading)}</h4><p>{esc(description)}</p></div></div>'
        )
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
        f'<tr hx-get="/find/trace/{quote(trace.trace_id, safe="")}" hx-target="#finder-drawer" hx-swap="innerHTML" hx-indicator="#finder-drawer-loading">'
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
            return f'<p>Verdict <b>{esc(str(selected))}</b></p>'
        options = ''.join(
            f'<option value="{esc(str(label))}"{" selected" if label == selected else ""}>{esc(str(label))}</option>'
            for label in labels
        )
        return f'<select name="selection_value">{options}</select>'
    if task.kind == 'noul':
        selected_bool = next((value for value in getattr(selection, 'values', ()) if type(value) is bool), False)
        if not editable:
            return f'<p>Verdict <b>{str(selected_bool).lower()}</b></p>'
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
        return f'<p>Score {esc(operator)} <b>{threshold:g}</b></p>'
    presets = (('gte', 0.5), ('gte', 0.7), ('gte', 0.8), ('lte', 0.3), ('lte', 0.5))
    options = (
        ''
        if (operator, threshold) in presets
        else (f'<option value="{operator}:{threshold:g}" selected>{operator} {threshold:g}</option>')
    )
    options += ''.join(
        f'<option value="{op}:{value:g}"{" selected" if operator == op and threshold == value else ""}>{op} {value:g}</option>'
        for op, value in presets
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
    kind_label = {'noul': 'Yes / no', 'choice': 'Choice', 'score': 'Score'}[task.kind]
    instruction = (
        f'<textarea name="instructions" required>{esc(task.instructions)}</textarea>'
        if editable
        else f'<p>{esc(task.instructions)}</p>'
    )
    threshold = ''
    if task.kind == 'noul':
        value = (
            f'<input name="noul_threshold" type="number" min="0" max="1" step="0.01" value="{task.noul_threshold:g}">'
            if editable
            else f'<p>{task.noul_threshold:g}</p>'
        )
        threshold = f'<div class="finder-task-stage"><h5>Yes threshold</h5>{value}</div>'
    criteria = (
        f'<div class="finder-task-criteria"><h5>Verdict labels</h5><div class="finder-crit">{criterion_html}</div></div>'
        if criterion_count
        else ''
    )
    count_html = f'<span class="finder-task-count">{criterion_count} labels</span>' if criterion_count else ''
    if editable:
        form_open = '<form id="finder-start-form" hx-post="/find/start" hx-target="#finder-body" hx-swap="innerHTML" hx-disabled-elt="find button">'
        hidden_request = ''
        if request is not None:
            hidden_request = (
                f'<input type="hidden" name="query" value="{esc(request.query)}">'
                f'<input type="hidden" name="mode" value="review">'
            )
        form_open += hidden_request + csrf_field()
        form_close = '<button class="rt-apply-btn" type="submit">Start classification</button><span class="finder-start-working" role="status">Starting classification…</span></form>'
    else:
        form_open = ''
        form_close = ''
    return (
        f'<details class="finder-task"{" open" if open_ else ""}><summary><span class="chev" aria-hidden="true">▸</span>'
        f'<span class="finder-task-title">Classifier task</span><span class="kind">{kind_label}</span>'
        f'{count_html}</summary>{form_open}'
        f'<div class="finder-task-body"><div class="finder-task-question"><h5>Question asked of each trace</h5>{instruction}</div>'
        f'<div class="finder-task-flow"><div class="finder-task-stage"><h5>Model returns</h5><strong>{kind_label}</strong></div>'
        '<span class="finder-task-arrow" aria-hidden="true">→</span>'
        f'<div class="finder-task-stage"><h5>Include when</h5>{_selection_rule_html(compiled, editable=editable)}</div>'
        f'{threshold}</div>{criteria}</div>{form_close}</details>'
    )


def filter_output_panel(snapshot: RunSnapshot) -> str:
    """Show the filter model's chosen values and its structured classify reply."""
    selected = [
        (name.replace('_', ' '), value)
        for name in FACET_NAMES
        for value in sorted(getattr(snapshot.generated_filters, name))
    ]
    chips = ''.join(
        f'<span class="finder-filter-chip"><b>{esc(name)}</b>{esc(value)}</span>' for name, value in selected
    )
    chips_html = chips or '<span class="finder-filter-none">No metadata filters selected.</span>'
    if snapshot.filter_selection_error:
        response = f'<p class="finder-filter-note" role="alert">Filter selection was unavailable: {esc(snapshot.filter_selection_error)}</p>'
    elif snapshot.filter_response is None:
        response = '<p class="finder-filter-note">No structured filter response was produced.</p>'
    else:
        structured = json.dumps(
            snapshot.filter_response.model_dump(mode='json', exclude_none=True), indent=2, ensure_ascii=False
        )
        response = (
            '<details class="finder-filter-json"><summary>View structured LLM output</summary>'
            f'<pre>{esc(structured)}</pre></details>'
        )
    return (
        '<section class="finder-filter-output"><div class="finder-filter-heading"><div>'
        '<h3>Filter selection</h3><p>The model proposes metadata filters before your choices take precedence.</p>'
        f'</div><span class="finder-filter-count">{len(selected)} chosen</span></div>'
        f'<div class="finder-filter-chips">{chips_html}</div>'
        f'{response}</section>'
    )


def status_indicator(snapshot: RunSnapshot) -> str:
    """Corner badge naming the current run phase: idle, a spinner per working step, or done."""
    state = snapshot.state
    if state == 'compiling':
        kind, label = (
            'busy',
            {
                'planning': 'Planning search',
                'loading_traces': 'Loading traces',
                'starting_classification': 'Starting classification',
            }.get(snapshot.phase or '', 'Preparing search'),
        )
    elif state == 'classifying':
        kind, label = 'busy', f'Labelling data · {snapshot.completed}/{snapshot.total}'
    else:
        kind, label = {
            'idle': ('idle', 'Idle'),
            'awaiting_review': ('paused', 'Waiting for review'),
            'completed': ('done', 'Done'),
            'failed': ('failed', 'Failed'),
            'cancelled': ('idle', 'Cancelled'),
        }[state]
    icon = {'busy': '<span class="spin"></span>', 'done': '✓', 'failed': '✕', 'paused': '⏸'}.get(kind, '')
    return (
        f'<div class="finder-status {kind}" role="status" aria-live="polite">'
        f'<span class="icon" aria-hidden="true">{icon}</span>{esc(label)}</div>'
    )


def body(
    snapshot: RunSnapshot,
    settings: DashboardSettings,
    *,
    catalogue: FacetCatalogue | None = None,
    pending: bool = False,
    api_available: bool = True,
) -> str:
    indicator = status_indicator(snapshot)
    if snapshot.state == 'awaiting_review' and snapshot.compiled is not None:
        return (
            f'{indicator}{controls(snapshot, settings, catalogue, pending=pending)}<div class="finder-review"><span>⏸</span><span><b>Review the plan before running per-trace classification.</b> '
            'Edit the task, criteria or filters, then start.</span></div>'
            f'{task_panel(snapshot.compiled, editable=True, open_=True, request=snapshot.request)}'
            f'{filter_output_panel(snapshot)}'
        )
    if snapshot.state == 'idle':
        return f'{indicator}{controls(snapshot, settings, catalogue, pending=pending)}{field(snapshot, api_available=api_available)}'
    return f'{indicator}{controls(snapshot, settings, catalogue, pending=pending)}{field(snapshot, api_available=api_available)}{table(snapshot)}{task_panel(snapshot.compiled, editable=False) + filter_output_panel(snapshot) if snapshot.compiled else ""}'


def page_html(
    snapshot: RunSnapshot,
    settings: DashboardSettings,
    *,
    api_available: bool,
    error: str | None = None,
    catalogue: FacetCatalogue | None = None,
    pending: bool = False,
) -> str:
    request = snapshot.request
    query = request.query if request is not None else ''
    mode = request.mode if request is not None else 'immediate'
    body_html = fragment(snapshot, settings, catalogue=catalogue, pending=pending, api_available=api_available)
    html = f'<div class="finder">{hero(query, mode, api_available=api_available, error=error)}<div id="finder-body">{body_html}</div><div id="finder-drawer"></div><div id="finder-drawer-loading" role="status">Loading trace…</div></div>'
    return page('Trace search', html, active_nav='find')


def fragment(
    snapshot: RunSnapshot,
    settings: DashboardSettings,
    *,
    error: str | None = None,
    api_available: bool = True,
    catalogue: FacetCatalogue | None = None,
    pending: bool = False,
) -> str:
    attrs = ''
    if snapshot.state in {'compiling', 'classifying'}:
        attrs = ' hx-get="/find/poll" hx-trigger="every 1s" hx-target="#finder-body" hx-swap="innerHTML"'
    error_html = f'<div class="finder-review finder-form-error" role="alert">{esc(error)}</div>' if error else ''
    return f'<div class="finder-body-fragment"{attrs}>{error_html}{body(snapshot, settings, catalogue=catalogue, pending=pending, api_available=api_available)}</div>'


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
    messages = ''.join(_thread_message(message, index) for index, message in enumerate(trace.messages, start=1))
    payload = json.dumps(detail.projection.payload if detail.projection else {}, indent=2, ensure_ascii=False)
    raw = json.dumps(result.raw_result if result else {}, indent=2, ensure_ascii=False)
    thread_html = messages or '<p class="finder-empty">No messages.</p>'
    body_html = (
        f'<dl class="fd-meta"><dt>trace</dt><dd>{esc(trace.trace_id)}</dd><dt>span</dt><dd>{esc(trace.span_id)}</dd>'
        f'<dt>project</dt><dd>{esc(trace.project)}</dd><dt>model</dt><dd>{esc(trace.model)}</dd><dt>time</dt><dd>{esc(trace.timestamp.isoformat())}</dd></dl>'
        f'<div class="fd-verdict"><span class="sw" style="background:{esc(_result_color(result, detail.compiled))}"></span>{result_html}</div>'
        '<div class="fd-tabs">'
        '<button type="button" class="on" data-panel="fd-thread" onclick="eqFinderTab(this,\'fd-thread\')">Full thread</button>'
        '<button type="button" data-panel="fd-input" onclick="eqFinderTab(this,\'fd-input\')">Classifier input</button>'
        '<button type="button" data-panel="fd-raw" onclick="eqFinderTab(this,\'fd-raw\')">Raw result</button></div>'
        f'<div id="fd-thread" class="fd-panel"><div class="fd-panel-title">Full thread</div>{thread_html}</div>'
        f'<div id="fd-input" class="fd-panel" hidden><div class="fd-panel-title">Classifier input</div><pre>{esc(payload)}</pre></div>'
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


def missing_trace_drawer(trace_id: str) -> str:
    body = (
        '<p class="finder-empty">This trace is not part of the current run. Results live in memory only, so a '
        'dashboard restart or a new search clears them. Run the search again to reopen it.</p>'
    )
    return drawer_shell(f'Trace {esc(trace_id)}', body, '', dismiss_route='/find/dismiss', drawer_id='finder-drawer')


def _message_text(message: dict[str, object]) -> str:
    """Render chat ``content``/``tool_calls`` or OTel GenAI ``parts`` (the Responses span shape)."""
    content = message.get('content')
    blocks = [content if content not in (None, '') else message.get('parts'), message.get('tool_calls')]
    return '\n'.join(text for block in blocks if (text := _block_text(block)))


def _thread_message(message: dict[str, object], index: int) -> str:
    role = str(message.get('role', 'unknown'))
    content = _message_text(message)
    preview = ' '.join(content.split())
    if len(preview) > 100:
        preview = preview[:100].rstrip() + '…'
    return (
        f'<details class="fd-msg {"user" if role == "user" else "assistant"}" open>'
        f'<summary><span class="role">{esc(role)} <span class="fd-msg-index">{index}</span></span>'
        f'<span class="fd-msg-preview">{esc(preview)}</span></summary>'
        f'<div class="fd-msg-content">{esc(content)}</div></details>'
    )


def _block_text(value: object) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return '\n'.join(text for item in value if (text := _block_text(item)))
    if isinstance(value, dict):
        kind = value.get('type')
        function = value.get('function')
        call = function if isinstance(function, dict) else value
        if kind in ('tool_call', 'function_call', 'function'):
            arguments = call.get('arguments')
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False, default=str)
            return f'→ {call.get("name")}({arguments})'
        if kind == 'reasoning':
            return ''
        for key in ('content', 'text', 'result', 'response', 'output'):
            if key in value:
                return _block_text(value[key])
    return json.dumps(value, ensure_ascii=False, default=str)
