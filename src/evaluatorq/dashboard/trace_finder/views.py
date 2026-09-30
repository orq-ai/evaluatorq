"""HTML builders for the classifier trace-finder dashboard surface."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING
from urllib.parse import quote

from evaluatorq.common.reports import esc
from evaluatorq.dashboard.apply_ui import drawer as drawer_shell
from evaluatorq.dashboard.security import csrf_field
from evaluatorq.dashboard.shell import page
from evaluatorq.dashboard.trace_finder import explorer_views
from evaluatorq.dashboard.trace_links import single_trace_url, trace_link_button, trace_span_url
from evaluatorq.trace_finder import classification_legend
from evaluatorq.trace_finder.columns import fmt_cost, fmt_time, fmt_tokens
from evaluatorq.trace_finder.models import FACET_NAMES, NUMERIC_FACET_NAMES
from evaluatorq.trace_finder.trajectory import KIND_LABELS, Kind, Segment, segments

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evaluatorq.trace_finder import (
        CompiledQuery,
        DashboardSettings,
        DimensionAnswer,
        FacetCatalogue,
        FacetSelection,
        RunRequest,
        RunSnapshot,
        TraceClassification,
        TraceDetail,
        TraceRow,
    )
    from evaluatorq.trace_finder.explorer import ExplorerView


SAMPLES = (
    'Frustrated customers in the support agent on production this week.',
    'Errored traces on claude-sonnet-5 where the user was blocked by a failed tool.',
    'Conversations over 20k tokens that should have been escalated to a human.',
    'Responses from the docs agent that make unsupported claims.',
)
COMMAND_LEDE = 'Recent traces from your agents, newest first: ask a question in plain words, or filter the list below.'
COMMAND_PLACEHOLDER = 'Ask a question, e.g. Did any customers get frustrated?'
COMMAND_EXAMPLES = (
    'Did any customers get frustrated?',
    'Which conversations mention refunds?',
    'Where did the agent fail to help someone?',
)
SCOPE_HELP_WITHIN = 'Loaded traces: Ask AI reads only the traces already loaded below, up to the row limit.'
SCOPE_HELP_NEW = 'All traces: Ask AI searches every trace in the chosen time range, then loads the matches.'
ASK_AI_COST_NOTE = 'Each question is answered by an AI model, so it takes a few seconds and costs a little.'
FACET_LABELS = (
    ('project', 'project'),
    ('agent_name', 'agent'),
    ('model', 'model'),
    ('provider', 'provider'),
    ('status', 'status'),
    ('product', 'product'),
    ('trace_type', 'trace type'),
    ('tool_name', 'tool'),
    ('tokens', 'total tokens'),
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


def hero(query: str, mode: str, *, api_available: bool, error: str | None = None, has_rows: bool = False) -> str:
    disabled = '' if api_available else ' disabled'
    error_html = f'<div class="finder-form-error" role="alert">{esc(error)}</div>' if error else ''
    checked_immediate = ' checked' if mode != 'review' else ''
    checked_review = ' checked' if mode == 'review' else ''
    return (
        '<section class="finder-hero"><div class="finder-hero-bg"></div>'
        '<h2 class="finder-title">Traces</h2>'
        '<p class="finder-sub">Ask AI about your traces. Inspect every judgment.</p>'
        '<form id="finder-query-form" class="finder-query" hx-post="/find/run" hx-target="#finder-body" '
        'hx-swap="innerHTML" hx-include="#finder-controls" hx-disabled-elt="find button">'
        f'{csrf_field()}{icon_search()}<div class="col"><textarea name="query" rows="1" placeholder="Ask AI about these traces…" '
        f'required{disabled}>{esc(query)}</textarea></div>'
        f'<button class="finder-go" type="submit"{disabled}><span class="finder-go-idle">Ask AI <span aria-hidden="true">↗</span></span>'
        '<span class="finder-go-working" role="status">Starting search…</span></button>'
        '</form>'
        f'<div id="finder-scope" class="finder-seg" role="radiogroup" aria-label="Ask AI scope">'
        f'<label><input type="radio" name="scope" value="within" form="finder-query-form"{" checked" if has_rows else ""}{"" if has_rows else " disabled"}><span>Within results</span></label>'
        f'<label><input type="radio" name="scope" value="new" form="finder-query-form"{"" if has_rows else " checked"}><span>New search</span></label></div>'
        '<div class="finder-below">'
        '<p class="finder-hint-line">Within results classifies all loaded traces.</p>'
        '<div class="finder-seg" role="radiogroup" aria-label="Mode">'
        f'<label><input type="radio" name="mode" value="immediate" form="finder-query-form"{checked_immediate} '
        'hx-post="/find/reset" hx-trigger="change[document.getElementById(\'finder-start-form\')]" '
        'hx-include="#finder-query-form" hx-target="#finder-body" hx-swap="innerHTML" hx-indicator="#finder-mode-working"><span>Immediate</span></label>'
        f'<label><input type="radio" name="mode" value="review" form="finder-query-form"{checked_review}><span>Review first</span></label></div>'
        '<span id="finder-mode-working" role="status">Resetting review…</span>'
        f'<span class="ex"><button type="button" class="link">Examples ▾</button>{examples()}</span>'
        '<span class="spacer"></span><span class="hint"><kbd>⌘</kbd> <kbd>↵</kbd> to run</span></div>'
        f'{error_html}</section>'
    )


def traces_command_strip(
    query: str, *, api_available: bool, error: str | None = None, has_rows: bool = False, mode: str = 'immediate'
) -> str:
    """Compact AI query strip used by /traces; /find keeps its own legacy search hero."""
    disabled = '' if api_available else ' disabled'
    error_html = f'<div class="finder-form-error" role="alert">{esc(error)}</div>' if error else ''
    chips = ''.join(
        f'<button type="button" data-finder-example="{esc(sample)}">{esc(sample)}</button>'
        for sample in COMMAND_EXAMPLES
    )
    return (
        '<section class="finder-hero finder-command">'
        f'<p class="finder-command-lede">{esc(COMMAND_LEDE)}</p>'
        '<form id="finder-query-form" class="finder-query finder-command-query" hx-post="/find/run" hx-target="#finder-body" '
        'hx-swap="innerHTML" hx-include="#finder-controls" hx-disabled-elt="find button">'
        f'{csrf_field()}<span class="finder-ai-icon" aria-hidden="true">✦</span><span class="finder-ai-label">Ask AI</span>'
        f'<div class="col"><textarea class="finder-command-textarea" name="query" rows="1" placeholder="{esc(COMMAND_PLACEHOLDER)}" required{disabled}>'
        f'{esc(query)}</textarea></div><input type="hidden" name="mode" value="{esc(mode)}">'
        '<span class="finder-scope-label" id="finder-scope-label">Search in</span>'
        '<div id="finder-scope" class="finder-seg" role="radiogroup" aria-labelledby="finder-scope-label">'
        f'{scope_toggle(has_rows=has_rows)}</div><a href="/settings" title="Choose the models Ask AI uses" '
        'class="finder-command-gear"><span aria-hidden="true">⚙</span> AI settings</a>'
        f'<button class="finder-go finder-command-search" type="submit"{disabled}><span class="finder-go-idle">Search</span>'
        '<span class="finder-go-working" role="status">Searching…</span></button>'
        '</form>'
        f'<p class="finder-command-help"><span class="scope-within">{esc(SCOPE_HELP_WITHIN)}</span>'
        f'<span class="scope-new">{esc(SCOPE_HELP_NEW)}</span> {esc(ASK_AI_COST_NOTE)}</p>'
        f'<div class="finder-command-examples"><span>Try</span>{chips}</div>'
        f'{error_html}</section>'
    )


def _loaded_tallies(rows: Sequence[TraceRow]) -> dict[str, dict[str, tuple[str, int]]]:
    """Tally loaded rows per facet: ``{facet: {casefolded value: (display value, count)}}``.

    Project, trace type and tool are not on a row (project values are shown by name, rows hold ids),
    so those menus carry no counts rather than a wrong one.
    """
    single = {'agent_name': 'agent_name', 'status': 'status', 'product': 'product'}
    multi = {'model': 'models', 'provider': 'providers'}
    tallies: dict[str, dict[str, tuple[str, int]]] = {name: {} for name in (*single, *multi)}

    def bump(facet: str, value: str) -> None:
        shown, count = tallies[facet].get(value.casefold(), (value, 0))
        tallies[facet][value.casefold()] = (shown, count + 1)

    for row in rows:
        for facet, attr in single.items():
            value = getattr(row, attr)
            if value:
                bump(facet, str(value))
        for facet, attr in multi.items():
            for value in dict.fromkeys(getattr(row, attr)):
                bump(facet, value)
    return tallies


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
    loaded_rows: Sequence[TraceRow] | None = None,
) -> str:
    """Two-level filter menu: a category list, and a value popout the client opens per category.

    ``pending`` renders the menu without values and has it fetch them itself as soon as it lands
    on the page, so a page render never waits on the Orq facet call.

    ``loaded_rows`` adds a count next to each value, tallied from the rows already on the page (no
    extra Orq call), so a count is of loaded traces, not of everything Orq holds.
    """
    tallies = _loaded_tallies(loaded_rows) if loaded_rows is not None else {}
    items: list[str] = []
    subs: list[str] = []
    for name, label in FACET_LABELS:
        numeric_facet = name in NUMERIC_FACET_NAMES
        selected_values = getattr(selection, name, frozenset()) if selection is not None else frozenset()
        values = tuple(dict.fromkeys((*_facet_values(catalogue, name), *sorted(selected_values))))
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
            tally = tallies.get(name)
            counts = {key: count for key, (_, count) in tally.items()} if tally is not None else None
            if tally is not None and counts is not None:
                # Catalogue values first, then values only the loaded rows carry (the catalogue may be
                # unavailable); most common first, and sorted() is stable so ties keep catalogue order.
                known = {value.casefold() for value in values}
                values = (*values, *(shown for key, (shown, _) in tally.items() if key not in known))
                by_count = counts
                values = tuple(sorted(values, key=lambda v: -by_count.get(v.casefold(), 0)))
            options = ''.join(
                f'<label><input form="{form_id}" type="checkbox" name="facet_{esc(name)}" value="{esc(value)}"{(" checked" if value in selected_values else "")}><span>{esc(value)}</span>'
                + (
                    f'<span class="facet-n" title="Loaded traces">{counts.get(value.casefold(), 0):,}</span>'
                    if counts is not None
                    else ''
                )
                + '</label>'
                for value in values
            )
            if options:
                overflow = (
                    '<p class="facet-note">More values exist in Orq; showing the returned values ranked by frequency.</p>'
                    if catalogue is not None and name in catalogue.truncated_facets
                    else ''
                )
                loaded_note = (
                    f'<p class="facet-scope">Counts are of the {len(loaded_rows):,} rows currently loaded, so they change as you filter.</p>'
                    if tally is not None and loaded_rows is not None
                    else ''
                )
                body = (
                    f'{loaded_note}<input class="facet-search" type="search" placeholder="Search values" aria-label="Search {esc(label)} values" autocomplete="off">'
                    f'<div class="facet-values">{options}</div>'
                    '<p class="facet-no-results" hidden>No matching values in this list.</p>'
                    f'{overflow}'
                )
            else:
                body = '<p class="finder-empty">No values in this window.</p>'
        count_html = f'<span class="count">{count}</span>' if count else ''
        items.append(
            f'<button type="button" class="facet-item" data-facet="{esc(name)}" aria-haspopup="true" '
            f'aria-expanded="false"><span>{esc(label)}</span>{count_html}<span class="chev" aria-hidden="true">&rsaquo;</span></button>'
        )
        subs.append(
            f'<div class="facet-sub" data-facet-sub="{esc(name)}" hidden><div class="hd">{esc(label)}</div>{body}</div>'
        )
    row_scoped = form_id == 'explorer-load-form'
    if pending or row_scoped:
        # The /traces facet menu already has counts from its loaded rows. Keep the
        # background catalogue refresh quiet there instead of implying those counts
        # are still waiting to load.
        note = '' if row_scoped and loaded_rows is not None else '<p class="finder-empty">Loading facet values…</p>'
        counts_param = '&counts=loaded' if loaded_rows is not None or row_scoped else ''
        triggers = 'load, refreshFacets' if pending else 'refreshFacets'
        loader = (
            f' hx-get="/find/facets?form_id={form_id}{counts_param}" hx-trigger="{triggers}" '
            'hx-include="#finder-controls" hx-swap="outerHTML" hx-indicator=".finder-facet-loading" '
            'hx-sync="#finder-controls:replace"'
        )
    else:
        note = (
            '<p class="finder-empty">Facet values are unavailable; check the Orq connection and reopen.</p>'
            if catalogue is None and loaded_rows is None
            else '<p class="facet-note">Orq\'s full value list is unavailable; showing values from the loaded traces.</p>'
            if catalogue is None
            else ''
        )
        loader = ''
    return (
        f'<div class="finder-facets{" open" if open_ else ""}{" pending" if pending else ""}"{" data-refresh-on-open" if row_scoped else ""}{loader}><div class="facet-list">{"".join(items)}{note}</div>'
        f'{"".join(subs)}</div>'
    )


def _facet_chips(
    selection: FacetSelection,
    numeric: object | None = None,
    *,
    removable: bool = False,
    generated: FacetSelection | None = None,
) -> str:
    """Render one chip per active filter value. Editable chips open the already-rendered menu at their category."""

    def chip(
        facet: str,
        label: str,
        value_html: str,
        remove_name: str,
        remove_value: str | None,
        aria: str,
        *,
        ai: bool = False,
    ) -> str:
        ai_class = ' ai' if ai else ''
        badge = '<b class="ai-badge">AI</b>' if ai else ''
        if not removable:
            return f'<span class="chip{ai_class}"><b>{esc(label)}</b>{badge}<span class="v">{value_html}</span></span>'
        value_attr = f' data-finder-value="{esc(remove_value)}"' if remove_value is not None else ''
        return (
            f'<span class="chip is-editable{ai_class}" data-chip-name="{esc(remove_name)}"{value_attr}>'
            f'<button type="button" class="chip-open" data-chip-open="{esc(facet)}" aria-label="Edit {aria}">'
            f'<b>{esc(label)}</b>{badge}<span class="v">{value_html}</span></button>'
            f'<button type="button" class="finder-chip-remove" data-finder-remove="{esc(remove_name)}"{value_attr} '
            f'aria-label="Remove {aria}">✕</button></span>'
        )

    chips: list[str] = []
    for name, label in FACET_LABELS:
        if name in NUMERIC_FACET_NAMES:
            continue
        chips.extend(
            chip(
                name,
                label,
                esc(value),
                f'facet_{name}',
                value,
                f'{esc(label)} {esc(value)}',
                ai=bool(generated and value in getattr(generated, name)),
            )
            for value in sorted(getattr(selection, name, frozenset()))
        )
    for name, label, operator in (
        ('tokens_min', 'total tokens', '≥'),
        ('tokens_max', 'total tokens', '≤'),
        ('duration_ms_min', 'duration', '≥'),
        ('duration_ms_max', 'duration', '≤'),
    ):
        value = getattr(numeric, name, None) if numeric is not None else None
        if value is None:
            continue
        facet = name.rsplit('_', 1)[0]
        formatted = f'{operator} {value:,}'
        chips.append(chip(facet, label, formatted, name, None, f'{label} {formatted}'))
    return ''.join(chips)


def controls(
    snapshot: RunSnapshot,
    settings: DashboardSettings,
    catalogue: FacetCatalogue | None = None,
    *,
    pending: bool = False,
    explorer_facets: FacetSelection | None = None,
    explorer_numeric: object | None = None,
    explorer_view: ExplorerView | None = None,
) -> str:
    request = snapshot.request
    population = request.population if request is not None else None
    selection = population.facets if population is not None else None
    facets = selection or _empty_facets()
    review = snapshot.state == 'awaiting_review'
    # On /traces the same facet menu is opened from the table toolbar and
    # closing it reloads the table. Keep its controls associated with that
    # form; hx-include still carries them into an Ask AI submission.
    form_id = (
        'finder-start-form' if review else 'explorer-load-form' if explorer_view is not None else 'finder-query-form'
    )
    window_days = settings.window_days
    if population is not None and population.start is not None and population.end is not None:
        window_days = max(1, round((population.end - population.start).total_seconds() / 86400))
    elif explorer_view is not None and explorer_view.start is not None and explorer_view.end is not None:
        window_days = max(1, round((explorer_view.end - explorer_view.start).total_seconds() / 86400))
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
    carried_facets = (
        facets
        if review
        else (
            explorer_facets
            or (explorer_view.facets if explorer_view is not None else None)
            or snapshot.explicit_filters
        )
    )
    carried_numeric = (
        numeric
        if review
        else (
            explorer_numeric
            or (explorer_view.numeric if explorer_view is not None else None)
            or snapshot.explicit_numeric
        )
    )
    if not review and population is None and (explorer_facets is not None or explorer_view is not None):
        facets = carried_facets
        numeric = carried_numeric
    generated_only = None
    if review:
        generated_only = snapshot.generated_filters.model_copy(
            update={
                name: getattr(snapshot.generated_filters, name) - getattr(snapshot.explicit_filters, name)
                for name in FACET_NAMES
            }
        )
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
        f'{hidden_facets}{scope_html}{_facet_chips(facets, numeric, removable=snapshot.state not in {"compiling", "classifying"}, generated=generated_only)}'
        f'<span class="addwrap"><button class="add" type="button" aria-haspopup="true">+ Filter</button>'
        f'{facet_menu(catalogue, numeric=carried_numeric, form_id=form_id, selection=carried_facets, pending=pending, loaded_rows=explorer_view.rows if explorer_view is not None else None)}'
        '<span class="finder-facet-loading" role="status">Loading filters…</span></span><span class="spacer"></span>'
        f'<input {keep["window_days"]} type="hidden" form="{form_id}" name="window_days" value="{values["window_days"]}">'
        f'<input {keep["limit"]} type="hidden" form="{form_id}" name="limit" value="{values["limit"]}">'
        f'<input {keep["parallelism"]} type="hidden" form="{form_id}" name="parallelism" value="{values["parallelism"]}">'
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


def _answer_color(answer: DimensionAnswer, dimension: CompiledQuery) -> str:
    if answer.error:
        return 'var(--red-600)'
    if dimension.task.kind == 'score' and isinstance(answer.value, (int, float)) and not isinstance(answer.value, bool):
        score = max(0.0, min(1.0, answer.value))
        return f'color-mix(in srgb, var(--chart-5) {score * 100:g}%, var(--chart-2))'
    for item in classification_legend(dimension):
        if item.label == _value_text(answer.value):
            return item.color
    return 'var(--chart-5)' if answer.matched else 'var(--chart-2)'


def _result_color(result: TraceClassification | None, dimensions: tuple[CompiledQuery, ...] | None) -> str:
    """Colour a trace by its only dimension's answer, or by the combined match when there are several."""

    if result is None or result.error:
        return 'var(--red-600)' if result and result.error else '#3d3c4a'
    if dimensions is not None and len(dimensions) == 1 and len(result.answers) == 1:
        return _answer_color(result.answers[0], dimensions[0])
    return 'var(--chart-5)' if result.matched else 'var(--chart-2)'


def _answers_text(result: TraceClassification, dimensions: tuple[CompiledQuery, ...] | None) -> str:
    """One-line summary of a trace's answers, named by dimension."""

    if not result.answers:
        return 'kept by filters, not judged by the AI'
    names = [dimension.name for dimension in dimensions or ()]
    return ' · '.join(
        f'{names[index] + ": " if index < len(names) else ""}{_value_text(answer.value)}'
        + (f' ({answer.confidence:.2f})' if answer.confidence is not None else '')
        for index, answer in enumerate(result.answers)
    )


_SCORE_TEXT_RE = re.compile(r'^\s*(?:noul|choice|score)\s*=')
_QUESTION_CAP = 80


def _reason_text(answer: DimensionAnswer) -> str | None:
    """The classifier's prose explanation, or None when it only returned score text.

    JEV returns a distribution and no rationale; ``run_classify`` synthesises ``noul=0.93 (threshold 0.5)``
    from the numbers. That is a score, not a reason, so it is never shown as one.
    """
    text = (answer.summary or '').strip()
    return None if not text or _SCORE_TEXT_RE.match(text) else text


def _reason_line(answer: DimensionAnswer) -> str:
    reason = _reason_text(answer) if answer.matched else None
    return f'<div class="xr-reason" title="{esc(reason)}">{esc(reason)}</div>' if reason else ''


def _capped(question: str, limit: int = _QUESTION_CAP) -> str:
    return question if len(question) <= limit else question[: limit - 1].rstrip() + '…'


def _drawer_reason(answer: DimensionAnswer) -> str:
    """A labelled prose reason, or a compact confidence when the classifier gave only numbers."""
    reason = _reason_text(answer)
    if reason:
        return f'<p class="fd-reason"><b>Reason</b> {esc(reason)}</p>'
    if answer.confidence is not None and not answer.error:
        verdict = 'yes' if answer.value is True else 'no' if answer.value is False else _value_text(answer.value)
        return f'<p class="fd-reason">{esc(verdict)} · {answer.confidence:.0%} confident</p>'
    return ''


def _answer_cells(result: TraceClassification, dimensions: tuple[CompiledQuery, ...]) -> str:
    """One verdict cell per dimension, each coloured by that dimension's legend."""

    cells: list[str] = []
    for index, dimension in enumerate(dimensions):
        answer = result.answers[index] if index < len(result.answers) else None
        if answer is None:
            cells.append('<td>—</td>')
            continue
        label = 'Judgment failed' if answer.error else _value_text(answer.value)
        confidence = f' <span class="conf">{answer.confidence:.2f}</span>' if answer.confidence is not None else ''
        reason = _reason_line(answer)
        cells.append(
            f'<td><span class="verdict"><span class="sw" style="background:{esc(_answer_color(answer, dimension))}"></span>'
            f'{esc(label)}</span>{confidence}{reason}</td>'
        )
    return ''.join(cells)


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
            title = result.error or _answers_text(result, snapshot.dimensions)
        elif trace.trace_id in active:
            state, title = 'active', 'classifying'
        else:
            state, title = 'pending', 'pending'
        color = _result_color(result, snapshot.dimensions)
        trace_id = quote(trace.trace_id, safe='')
        dots.append(
            f'<i class="{state}" style="--c:{esc(color)}" title="{esc(title)}" '
            f'hx-get="/find/trace/{trace_id}" hx-target="#finder-drawer" hx-swap="innerHTML" hx-indicator="#finder-drawer-loading"></i>'
        )
    return f'<div class="finder-matrix">{"".join(dots)}</div>'


def legend(snapshot: RunSnapshot) -> str:
    """Value counts for a single dimension; several dimensions colour by the combined match, so show only that."""

    items = ''
    if snapshot.dimensions is not None and len(snapshot.dimensions) == 1:
        dimension = snapshot.dimensions[0]
        counts: dict[str, int] = {}
        for result in snapshot.results.values():
            if result.error is None and result.answers:
                label = _value_text(result.answers[0].value)
                counts[label] = counts.get(label, 0) + 1
        entries = classification_legend(dimension)
        if dimension.task.kind == 'score':
            counts[entries[0].label] = sum(result.error is None for result in snapshot.results.values())
            counts[entries[1].label] = snapshot.matched
        items = '<span class="muted">|</span>' + ''.join(
            f'<span><span class="sw" style="background:{esc(item.color)}"></span>{esc(item.label)} '
            f'<b>{counts.get(item.label, 0)}</b></span>'
            for item in entries
        )
    return (
        '<div class="finder-legend">'
        f'<span><span class="sw match"></span><span class="num hot">{snapshot.matched}</span> included</span>'
        f'{items}'
        f'<span><span class="sw failed"></span>failed <b>{snapshot.failed}</b></span>'
        f'<span class="muted">|</span><span class="muted">{snapshot.completed} judged</span></div>'
    )


def _cannot_answer(snapshot: RunSnapshot) -> bool:
    """True when Ask AI stopped early because the question is not one it can answer (not a user cancel)."""
    return snapshot.state == 'cancelled' and snapshot.error is None and bool(snapshot.plan_warning)


def progress(
    snapshot: RunSnapshot,
    *,
    export_url: str = '/find/export.json',
    classifier_model: str | None = None,
    show_only_url: str | None = None,
) -> str:
    cannot_answer = _cannot_answer(snapshot)
    running = snapshot.state in {'compiling', 'classifying'}
    state = {
        'planning': 'planning search',
        'loading_traces': 'loading traces',
        'starting_classification': 'starting classification',
    }.get(snapshot.phase or '', "can't answer" if cannot_answer else snapshot.state.replace('_', ' '))
    reset = (
        f'<form class="finder-progress-action" hx-post="/find/reset" hx-target="#finder-body" hx-swap="innerHTML" hx-disabled-elt="find button">{csrf_field()}<button class="btn-secondary" type="submit">Clear AI results</button><span role="status">Clearing…</span></form>'
        if snapshot.state != 'idle'
        else ''
    )
    action = (
        f'<form class="finder-progress-action" hx-post="/find/cancel" hx-target="#finder-body" hx-swap="innerHTML" hx-disabled-elt="find button">{csrf_field()}<button class="btn-secondary" type="submit">Cancel</button><span role="status">Cancelling…</span></form>'
        if running
        else (
            (
                f'<a class="btn-secondary" href="{esc(export_url)}" title="Every judged trace with its answers, as JSON">Download results</a>'
                if snapshot.total
                else ''
            )
            + reset
        )
        if snapshot.state == 'completed'
        else reset
    )
    live_html = '<span class="live"></span>' if running else ''
    error_html = (
        f'<span class="finder-progress-error finder-review" role="alert">{esc(snapshot.error)}</span>'
        if snapshot.error
        else f'<span class="sep">·</span><span class="finder-progress-answer" role="status">{esc(snapshot.plan_warning)}</span>'
        if cannot_answer
        else f'<span class="finder-progress-error finder-progress-warning finder-review" role="status">{esc(snapshot.plan_warning)}</span>'
        if snapshot.plan_warning
        else ''
    )
    counts = (
        ''
        if cannot_answer
        else '<span class="sep">·</span><span>' + _compiling_text(snapshot) + '</span>'
        if snapshot.state == 'compiling'
        else f'<span class="sep">·</span><span><b>{snapshot.loaded}</b> traces loaded, none kept</span>'
        if snapshot.total == 0 and snapshot.loaded
        else '<span class="sep">·</span><span>Stopped before traces were loaded</span>'
        if snapshot.total == 0
        else '<span class="sep">·</span><span class="finder-progress-answer">This question did not map to any filter or AI check. '
        'Try asking which traces show something.</span>'
        if snapshot.dimensions == () and not _question_added_filters(snapshot)
        else f'<span class="sep">·</span><span><b>{snapshot.matched}</b> kept by filters</span>'
        '<span class="sep">·</span><span>no AI classification needed</span>'
        if snapshot.dimensions == ()
        else _judging_text(snapshot, classifier_model=classifier_model, show_only_url=show_only_url)
    )
    elapsed_html = (
        f'<span class="sep">·</span><span>{snapshot.elapsed:.1f}s</span>'
        if snapshot.state != 'compiling' and snapshot.total and snapshot.dimensions != ()
        else ''
    )
    return (
        f'<div class="finder-progress" data-state="{esc(snapshot.state)}">{live_html}<span class="state">{esc(state)}</span>{counts}'
        f'{elapsed_html}{error_html}{action}{_progress_bar(snapshot)}</div>'
    )


def _question_added_filters(snapshot: RunSnapshot) -> bool:
    """True when the compiled question narrowed the traces by any facet or numeric filter."""
    facets = any(getattr(snapshot.generated_filters, name) for name in FACET_NAMES)
    numeric = any(value is not None for value in snapshot.generated_numeric.model_dump().values())
    return facets or numeric


def _judging_text(
    snapshot: RunSnapshot, *, classifier_model: str | None = None, show_only_url: str | None = None
) -> str:
    """Lead with the answer as a sentence, then the work done; zero failures stay silent.

    The run keeps no usage or cost figure, so the model is named instead of a price.
    """
    done = snapshot.state == 'completed'
    question = snapshot.request.query if snapshot.request is not None else ''
    quoted = (
        f' <span class="finder-progress-q" title="{esc(question)}">“{esc(_capped(question))}”</span>'
        if question
        else ''
    )
    show_only = (
        f'<button type="button" class="btn-secondary finder-show-only" hx-get="{esc(show_only_url)}" '
        'hx-target="#explorer-results" hx-swap="outerHTML" hx-sync="#explorer-results:replace">Show only these</button>'
        if show_only_url and snapshot.matched
        else ''
    )
    parts = [
        f'<span class="finder-progress-answer"><b>{snapshot.matched}</b> of {snapshot.total} traces match{quoted}</span>{show_only}'
        if done
        else f'<span>Reading <b>{snapshot.completed}</b> of {snapshot.total}</span><span class="sep">·</span>'
        f'<span><b>{snapshot.matched}</b> matching so far</span>'
    ]
    if snapshot.failed:
        parts.append(f'<span class="finder-progress-failed"><b>{snapshot.failed}</b> failed</span>')
    parts.append(
        '<span class="nw">within the loaded rows</span>'
        if snapshot.within_results
        else '<span class="nw">new search</span>'
    )
    if classifier_model:
        parts.append(
            f'<span class="nw" title="Ask AI reads each trace with this model">Uses {esc(classifier_model)}</span>'
        )
    if not done:
        parts.append(f'<span>{snapshot.rate:.1f}/s</span>')
    return ''.join(f'<span class="part"><span class="sep">·</span>{part}</span>' for part in parts)


def _compiling_text(snapshot: RunSnapshot) -> str:
    """Describe the current planning step, with a live count while traces load."""
    if snapshot.phase == 'loading_traces' and snapshot.to_load:
        if snapshot.within_results:
            return f'fetching conversations <b>{snapshot.loaded} / {snapshot.to_load}</b>'
        return f'<b>{snapshot.loaded}</b> traces loaded, up to {snapshot.to_load}'
    if snapshot.phase == 'planning' and snapshot.within_results:
        return 'turning your question into a yes/no check'
    return {
        'planning': 'compiling the question and selecting metadata filters',
        'loading_traces': 'loading selected traces',
        'starting_classification': 'preparing the reviewed task',
    }.get(snapshot.phase or '', 'preparing the search')


def _progress_bar(snapshot: RunSnapshot) -> str:
    """A thin bar under the progress line for the step that has a known size."""
    if snapshot.state == 'compiling' and snapshot.phase == 'loading_traces' and snapshot.to_load:
        fraction = snapshot.loaded / snapshot.to_load
    elif snapshot.state == 'classifying' and snapshot.total:
        fraction = snapshot.completed / snapshot.total
    else:
        return ''
    return f'<span class="finder-progress-bar" aria-hidden="true"><i style="width:{min(fraction, 1.0) * 100:.1f}%"></i></span>'


def field(snapshot: RunSnapshot, *, api_available: bool = True, export_url: str = '/find/export.json') -> str:
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
    return f'<div class="finder-field{unavailable}">{progress(snapshot, export_url=export_url) if snapshot.state != "idle" else ""}{body}{legend(snapshot) if snapshot.dimensions is not None else ""}</div>'


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
    dimensions = snapshot.dimensions or ()
    rows = ''.join(
        f'<tr hx-get="/find/trace/{quote(trace.trace_id, safe="")}" hx-target="#finder-drawer" hx-swap="innerHTML" hx-indicator="#finder-drawer-loading">'
        f'<td class="id">{esc(trace.trace_id)}</td>{_answer_cells(result, dimensions)}'
        f'<td>{esc(trace.project)}</td><td>{esc(trace.model)}</td><td>{esc(trace.status)}</td><td>{esc(trace.timestamp.strftime("%Y-%m-%d %H:%M"))}</td></tr>'
        for trace, result in matches
    )
    empty = '<p class="finder-empty">No matches yet.</p>' if not rows else ''
    return (
        '<section class="finder-section"><h3 class="finder-section-title">Included traces</h3>'
        f'<p class="finder-section-sub">{snapshot.matched} of {snapshot.total or 0} judged as included, newest first. Click a row to inspect.</p>'
        '<table class="finder-table"><thead><tr><th>Trace</th>'
        + ''.join(f'<th>{esc(dimension.name)}</th>' for dimension in dimensions)
        + '<th>Project</th><th>Model</th><th>Status</th><th>Time</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>{empty}</section>'
    )


def _selection_rule_html(compiled: CompiledQuery, *, editable: bool, prefix: str = '') -> str:
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
        return f'<select name="{prefix}selection_value">{options}</select>'
    if task.kind == 'noul':
        selected_bool = next((value for value in getattr(selection, 'values', ()) if type(value) is bool), False)
        if not editable:
            return f'<p>Answer is <b>{"yes" if selected_bool else "no"}</b></p>'
        options = ''.join(
            f'<option value="{value}"{" selected" if selected_bool == (value == "true") else ""}>{value}</option>'
            for value in ('true', 'false')
        )
        return f'<select name="{prefix}selection_value">{options}</select>'
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
    return f'<select name="{prefix}selection_rule">{options}</select>'


def _criterion_html(compiled: CompiledQuery, *, editable: bool, prefix: str = '') -> tuple[str, int]:
    task = compiled.task
    if task.kind == 'choice' and isinstance(task.criteria, dict):
        pairs = tuple(task.criteria.items())
        if editable:
            return (
                ''.join(
                    f'<div class="finder-crit-row"><input name="{prefix}criteria_label_{index}" value="{esc(str(label))}" required>'
                    f'<input name="{prefix}criteria_description_{index}" value="{esc(str(description or ""))}" required></div>'
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
                    f'<input name="{prefix}score_criteria_{index}" value="{esc(str(description))}" required></div>'
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


def _dimension_card(compiled: CompiledQuery, *, editable: bool, open_: bool, prefix: str) -> str:
    """One classifier dimension: its question, what the model returns, and the rule that counts as a match."""
    task = compiled.task
    criterion_html, criterion_count = _criterion_html(compiled, editable=editable, prefix=prefix)
    kind_label = {'noul': 'Yes / no', 'choice': 'Choice', 'score': 'Score'}[task.kind]
    instruction = (
        f'<textarea name="{prefix}instructions" required>{esc(task.instructions)}</textarea>'
        if editable
        else f'<p>{esc(task.instructions)}</p>'
    )
    title = (
        f'<input class="finder-task-name" name="{prefix}name" value="{esc(compiled.name)}" maxlength="40" required aria-label="Dimension name">'
        if editable
        else f'<span class="finder-task-title">{esc(compiled.name)}</span>'
    )
    threshold = ''
    if task.kind == 'noul':
        value = (
            f'<input name="{prefix}noul_threshold" type="number" min="0" max="1" step="0.01" value="{task.noul_threshold:g}">'
            if editable
            else f'<p>{task.noul_threshold:.0%}</p>'
        )
        threshold = f'<div class="finder-task-stage"><h5>Confidence needed for yes</h5>{value}</div>'
    criteria = (
        f'<div class="finder-task-criteria"><h5>Verdict labels</h5><div class="finder-crit">{criterion_html}</div></div>'
        if criterion_count
        else ''
    )
    count_html = f'<span class="finder-task-count">{criterion_count} labels</span>' if criterion_count else ''
    return (
        f'<details class="finder-task"{" open" if open_ else ""}><summary><span class="chev" aria-hidden="true">▸</span>'
        f'{title}<span class="kind">{kind_label}</span>{count_html}'
        f'{"" if editable else f"<span class=finder-task-q title={chr(34)}{esc(task.instructions)}{chr(34)}>{esc(task.instructions)}</span>"}</summary>'
        f'<div class="finder-task-body"><div class="finder-task-question"><h5>Question asked of each trace</h5>{instruction}</div>'
        f'<div class="finder-task-flow"><div class="finder-task-stage"><h5>Model returns</h5><strong>{kind_label}</strong></div>'
        '<span class="finder-task-arrow" aria-hidden="true">→</span>'
        f'<div class="finder-task-stage"><h5>Include when</h5>{_selection_rule_html(compiled, editable=editable, prefix=prefix)}</div>'
        f'{threshold}</div>{criteria}</div></details>'
    )


def task_panel(
    dimensions: tuple[CompiledQuery, ...],
    *,
    editable: bool,
    open_: bool = False,
    request: RunRequest | None = None,
    total: int = 0,
) -> str:
    """One card per classifier dimension; a trace is included only when it matches every dimension."""
    cards = ''.join(
        _dimension_card(dimension, editable=editable, open_=open_, prefix=f'd{index}_')
        for index, dimension in enumerate(dimensions)
    )
    if not dimensions:
        cards = '<p class="finder-task-none">Filters answer this question, so no per-trace AI classification runs. Every trace the filters keep is included.</p>'
    elif len(dimensions) > 1:
        cards = f'<p class="finder-task-none">A trace is included only when it matches all {len(dimensions)} dimensions.</p>{cards}'
    if not editable:
        return f'<div class="finder-tasks">{cards}</div>'
    hidden_request = ''
    if request is not None:
        hidden_request = (
            f'<input type="hidden" name="query" value="{esc(request.query)}">'
            '<input type="hidden" name="mode" value="review">'
        )
    action = f'Apply + classify {total}' if dimensions else f'Apply + include {total}'
    return (
        '<form id="finder-start-form" class="finder-tasks" hx-post="/find/start" hx-target="#finder-body" hx-swap="innerHTML" hx-disabled-elt="find button">'
        f'{hidden_request}{csrf_field()}{cards}'
        f'<button class="rt-apply-btn" type="submit">{action}</button><span class="finder-start-working" role="status">Starting classification…</span></form>'
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
    warning = (
        f'<p class="finder-filter-note" role="status">{esc(snapshot.plan_warning)}</p>' if snapshot.plan_warning else ''
    )
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
        f'{warning}{response}</section>'
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
            'cancelled': ('idle', "Can't answer" if _cannot_answer(snapshot) else 'Cancelled'),
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
    explorer_facets: FacetSelection | None = None,
    explorer_numeric: object | None = None,
    explorer_view: ExplorerView | None = None,
) -> str:
    indicator = status_indicator(snapshot)
    if snapshot.state == 'awaiting_review' and snapshot.dimensions is not None:
        return (
            f'{indicator}{controls(snapshot, settings, catalogue, pending=pending)}<div class="finder-review"><span>⏸</span><span><b>Review the plan before running per-trace classification.</b> '
            'Edit the task, criteria or filters, then start.</span></div>'
            f'{task_panel(snapshot.dimensions, editable=True, open_=True, request=snapshot.request, total=snapshot.total)}'
            f'{filter_output_panel(snapshot)}'
            '<div class="finder-review-actions"><button class="btn-secondary" type="submit" form="explorer-load-form" '
            'hx-post="/find/load" hx-include="#finder-start-form, #finder-controls, #finder-scope" hx-target="#explorer-results" hx-swap="outerHTML">'
            'Apply filters only</button></div>'
        )
    if snapshot.state == 'idle':
        unavailable = field(snapshot, api_available=False) if not api_available else ''
        return f'{indicator}{controls(snapshot, settings, catalogue, pending=pending, explorer_facets=explorer_facets, explorer_numeric=explorer_numeric, explorer_view=explorer_view)}{unavailable}'
    controls_html = controls(
        snapshot,
        settings,
        catalogue,
        pending=pending,
        explorer_facets=explorer_facets,
        explorer_numeric=explorer_numeric,
        explorer_view=explorer_view,
    )
    details = (
        task_panel(snapshot.dimensions, editable=False)
        if snapshot.dimensions is not None and not _cannot_answer(snapshot)
        else ''
    )
    # The progress line under Ask AI names the running step, so the corner badge would repeat it.
    running = snapshot.state in {'compiling', 'classifying'}
    return f'{"" if running else indicator}{controls_html}{progress(snapshot, classifier_model=settings.classifier_model, show_only_url="/find/rows?quick_view=matches" if explorer_view is not None else None)}{details}'


def page_html(
    snapshot: RunSnapshot,
    settings: DashboardSettings,
    *,
    api_available: bool,
    error: str | None = None,
    catalogue: FacetCatalogue | None = None,
    pending: bool = False,
    explorer_html: str = '',
    explorer_view: ExplorerView | None = None,
    has_rows: bool = False,
) -> str:
    request = snapshot.request
    query = request.query if request is not None else ''
    body_html = fragment(
        snapshot,
        settings,
        catalogue=catalogue,
        pending=pending,
        api_available=api_available,
        explorer_view=explorer_view,
    )
    if not explorer_html:
        from evaluatorq.trace_finder.columns import resolve_columns
        from evaluatorq.trace_finder.explorer import ExplorerView

        explorer_html = explorer_views.results(ExplorerView(), resolve_columns(None), records=None, snapshot=None)
    toolbar_html, explorer_html = explorer_views.split_toolbar(explorer_html)
    html = f'<div class="finder">{traces_command_strip(query, api_available=api_available, error=error, has_rows=has_rows, mode=settings.ask_ai_mode)}{toolbar_html}<div id="finder-body">{body_html}</div><div id="explorer-results-slot">{explorer_html}</div><div id="finder-drawer"></div><div id="finder-drawer-loading" role="status">Loading trace…</div></div>'
    return page('Traces', html, active_nav='traces')


def fragment(
    snapshot: RunSnapshot,
    settings: DashboardSettings,
    *,
    error: str | None = None,
    api_available: bool = True,
    catalogue: FacetCatalogue | None = None,
    pending: bool = False,
    explorer_facets: FacetSelection | None = None,
    explorer_numeric: object | None = None,
    explorer_view: ExplorerView | None = None,
) -> str:
    attrs = ''
    if snapshot.state in {'compiling', 'classifying'}:
        attrs = ' hx-get="/find/poll" hx-trigger="every 1s" hx-target="#finder-body" hx-swap="innerHTML"'
    error_html = f'<div class="finder-review finder-form-error" role="alert">{esc(error)}</div>' if error else ''
    return f'<div class="finder-body-fragment"{attrs}>{error_html}{body(snapshot, settings, catalogue=catalogue, pending=pending, api_available=api_available, explorer_facets=explorer_facets, explorer_numeric=explorer_numeric, explorer_view=explorer_view)}</div>'


def scope_toggle(*, has_rows: bool, selected: str | None = None) -> str:
    selected = selected if selected in {'within', 'new'} else ('within' if has_rows else 'new')
    if selected == 'within' and not has_rows:
        selected = 'new'
    within = '' if has_rows else ' disabled'
    return (
        '<label><input type="radio" name="scope" value="within" form="finder-query-form"'
        + (' checked' if selected == 'within' else '')
        + within
        + '><span>Loaded traces</span></label>'
        + '<label><input type="radio" name="scope" value="new" form="finder-query-form"'
        + (' checked' if selected == 'new' else '')
        + '><span>All traces</span></label>'
    )


def drawer(
    detail: TraceDetail,
    *,
    experiment_url: str | None = None,
    msg: int | None = None,
    row: TraceRow | None = None,
    traces_layout: bool = False,
) -> str:
    trace = detail.trace
    result = detail.classification
    result_html = (
        '<p>Not classified yet.</p>'
        if result is None
        else (
            f'<p><b>{"Not included" if not result.matched else "Included" if result.answers else "Kept by filters"}</b> · {esc(_answers_text(result, detail.dimensions))}</p>'
            + ''.join(_drawer_reason(answer) for answer in result.answers)
            if not result.error
            else f'<p role="alert">Failed: {esc(result.error)}</p>'
        )
    )
    segs = segments(trace.messages)
    by_message: dict[int, list[Segment]] = {}
    for segment in segs:
        by_message.setdefault(segment.index, []).append(segment)
    selected = msg if msg is not None and 1 <= msg <= len(trace.messages) else 1
    messages = ''.join(
        _thread_message(
            message,
            index,
            selected=index == selected,
            kind=by_message[index][0].kind if by_message.get(index) else 'other',
            label=next((s.label for s in by_message.get(index, []) if s.label), None),
            tokens=sum(s.tokens for s in by_message.get(index, [])),
        )
        for index, message in enumerate(trace.messages, start=1)
    )
    mini = ''.join(
        f'<i class="k-{s.kind}{" on" if s.index == selected else ""}" style="flex-grow:{s.tokens}" data-mini-msg="{s.index}"></i>'
        for s in segs
    )
    mini_html = f'<div class="fd-mini">{mini}</div>' if segs else ''
    payload = json.dumps(detail.projection.payload if detail.projection else {}, indent=2, ensure_ascii=False)
    raw = json.dumps(result.raw_result if result else {}, indent=2, ensure_ascii=False)
    has_conversation = any(segment.preview.strip() or segment.label for segment in segs)
    thread_html = messages or '<p class="finder-empty">No messages.</p>'
    row_header = ''
    if row is not None:
        models = ''.join(f'<span class="tv pill">{esc(model)}</span>' for model in row.models)
        reasoning = f' ({esc(fmt_tokens(row.reasoning_tokens))} reasoning)' if row.reasoning_tokens else ''
        row_header = (
            f'<div class="fd-row-head"><span class="tv dot {"err" if row.is_error else "ok"}"></span>'
            f'<b>{esc(row.agent_name or row.name or "Unknown agent")}</b>{models}</div>'
            f'<div class="fd-row-meta">Started {esc(fmt_time(row.started_at))} · {esc(fmt_tokens(row.tokens_in))} in → '
            f'{esc(fmt_tokens(row.tokens_out))} out{reasoning} · {esc(fmt_tokens(row.cached_tokens))} cache · '
            f'{fmt_cost(row.cost_total, row.currency)}</div>'
        )
    technical_html = (
        f'{row_header}<dl class="fd-meta"><dt>trace</dt><dd>{esc(trace.trace_id)}</dd><dt>span</dt><dd>{esc(trace.span_id)}</dd>'
        f'<dt>project</dt><dd>{esc(trace.project)}</dd><dt>model</dt><dd>{esc(trace.model)}</dd><dt>time</dt><dd>{esc(trace.timestamp.isoformat())}</dd></dl>'
        f'<button class="btn-secondary" type="button" data-trace-id="{esc(trace.trace_id)}" onclick="navigator.clipboard.writeText(this.dataset.traceId)">Copy trace id</button>'
        '<div class="fd-tabs">'
        '<button type="button" data-panel="fd-input" onclick="eqFinderTab(this,\'fd-input\')">Classifier input</button>'
        '<button type="button" data-panel="fd-raw" onclick="eqFinderTab(this,\'fd-raw\')">Raw result</button></div>'
        f'<div id="fd-input" class="fd-panel" hidden><div class="fd-panel-title">Classifier input</div><pre>{esc(payload)}</pre></div>'
        f'<div id="fd-raw" class="fd-panel" hidden><div class="fd-panel-title">Raw result</div><pre>{esc(raw)}</pre></div>'
    )
    if traces_layout:
        identity_html = ''
        if row is not None:
            models = ''.join(f'<span class="tv pill">{esc(model)}</span>' for model in row.models)
            identity_html = (
                f'<div class="fd-row-head"><span class="tv dot {"err" if row.is_error else "ok"}"></span>'
                f'<b>{esc(row.agent_name or row.name or "Unknown agent")}</b>{models}</div>'
            )
        trace_thread = (
            messages
            if has_conversation
            else '<div class="fd-no-messages" role="status"><b>No messages available</b><span>This trace has no conversation text to display.</span></div>'
        )
        body_html = (
            '<div class="fd-traces">'
            f'{identity_html}'
            f'<div class="fd-verdict"><span class="sw" style="background:{esc(_result_color(result, detail.dimensions))}"></span>{result_html}</div>'
            f'{mini_html}<div class="fd-tabs"><button type="button" class="on" onclick="eqFinderTraceTab(this,\'fd-thread\')">Conversation</button>'
            '<button type="button" hx-get="/find/trace-spans?trace_id='
            + quote(trace.trace_id, safe='')
            + '" hx-target="#fd-spans" hx-swap="innerHTML" onclick="eqFinderTraceTab(this,\'fd-spans\')">Spans</button></div>'
            f'<div id="fd-thread" class="fd-panel">{trace_thread}</div><div id="fd-spans" class="fd-panel" hidden><p class="finder-empty">Open Spans to load span details.</p></div>'
            f'<details class="fd-technical"><summary>Technical details</summary>{technical_html}</details></div>'
        )
        title = 'Trace conversation'
        footer = (
            trace_link_button(trace_span_url(trace.trace_id, trace.span_id, experiment_url), 'Open in Orq ↗')
            + f'<button class="btn-secondary" type="button" data-trace-id="{esc(trace.trace_id)}" onclick="navigator.clipboard.writeText(this.dataset.traceId)">Copy trace id</button>'
        )
    else:
        body_html = (
            f'{row_header}{mini_html}'
            f'<dl class="fd-meta"><dt>trace</dt><dd>{esc(trace.trace_id)}</dd><dt>span</dt><dd>{esc(trace.span_id)}</dd>'
            f'<dt>project</dt><dd>{esc(trace.project)}</dd><dt>model</dt><dd>{esc(trace.model)}</dd><dt>time</dt><dd>{esc(trace.timestamp.isoformat())}</dd></dl>'
            f'<div class="fd-verdict"><span class="sw" style="background:{esc(_result_color(result, detail.dimensions))}"></span>{result_html}</div>'
            '<div class="fd-tabs"><button type="button" class="on" data-panel="fd-thread" onclick="eqFinderTab(this,\'fd-thread\')">Full thread</button>'
            '<button type="button" data-panel="fd-input" onclick="eqFinderTab(this,\'fd-input\')">Classifier input</button>'
            '<button type="button" data-panel="fd-raw" onclick="eqFinderTab(this,\'fd-raw\')">Raw result</button></div>'
            f'<div id="fd-thread" class="fd-panel"><div class="fd-panel-title">Full thread</div>{thread_html}</div>'
            f'<div id="fd-input" class="fd-panel" hidden><div class="fd-panel-title">Classifier input</div><pre>{esc(payload)}</pre></div>'
            f'<div id="fd-raw" class="fd-panel" hidden><div class="fd-panel-title">Raw result</div><pre>{esc(raw)}</pre></div>'
        )
        title = f'Trace {esc(trace.trace_id)}'
        footer = (
            trace_link_button(trace_span_url(trace.trace_id, trace.span_id, experiment_url), 'Open in Orq ↗')
            + f'<button class="btn-secondary" type="button" data-trace-id="{esc(trace.trace_id)}" onclick="navigator.clipboard.writeText(this.dataset.traceId)">Copy trace id</button>'
        )
    return drawer_shell(title, body_html, footer, dismiss_route='/find/dismiss', drawer_id='finder-drawer')


def missing_trace_drawer(trace_id: str, *, reason: str | None = None, traces_layout: bool = False) -> str:
    if traces_layout:
        failure_reason = reason or 'The conversation could not be loaded for this trace.'
        footer = (
            trace_link_button(single_trace_url(trace_id), 'Open in Orq ↗')
            + f'<button class="btn-secondary" type="button" data-trace-id="{esc(trace_id)}" onclick="navigator.clipboard.writeText(this.dataset.traceId)">Copy trace id</button>'
        )
        return drawer_shell(
            'Trace conversation',
            '<div class="fd-traces"><div class="fd-no-messages" role="status"><b>Conversation unavailable</b>'
            f'<span>{esc(failure_reason)}</span></div>'
            '<div class="fd-tabs"><button type="button" class="on" onclick="eqFinderTraceTab(this,\'fd-thread\')">'
            'Conversation</button><button type="button" hx-get="/find/trace-spans?trace_id='
            + quote(trace_id, safe='')
            + '" hx-target="#fd-spans" hx-swap="innerHTML" onclick="eqFinderTraceTab(this,\'fd-spans\')">Spans</button></div>'
            '<div id="fd-thread" class="fd-panel"><p class="fd-no-messages">Conversation content is unavailable.</p></div>'
            '<div id="fd-spans" class="fd-panel" hidden><p class="finder-empty">Open Spans to load span details.</p></div></div>',
            footer,
            dismiss_route='/find/dismiss',
            drawer_id='finder-drawer',
        )
    body = (
        f'<p class="finder-empty">{esc(reason)}</p>'
        if reason
        else (
            '<p class="finder-empty">This trace is not part of the current run. Results live in memory only, so a '
            'dashboard restart or a new search clears them. Run the search again to reopen it.</p>'
        )
    )
    return drawer_shell(f'Trace {esc(trace_id)}', body, '', dismiss_route='/find/dismiss', drawer_id='finder-drawer')


def span_tree(  # noqa: C901
    trace_id: str,
    spans: Sequence[object],
    experiment_url: str | None = None,
    *,
    first_error_message: str | None = None,
) -> str:
    """Render bounded span summaries as a defensive hierarchy, retaining orphan and cyclic entries."""

    def value(span: object, key: str) -> object | None:
        if isinstance(span, dict):
            return span.get(key)
        return getattr(span, key, None)

    def text(span: object, key: str) -> str:
        raw = value(span, key)
        return str(raw) if raw is not None else ''

    def status_text(span: object) -> str:
        status = value(span, 'status')
        if isinstance(status, str) and status:
            return status
        if status is not None:
            for key in ('code', 'status_code', 'state', 'name'):
                detail = value(status, key)
                if detail:
                    return str(detail)
        return text(span, 'status_code') or 'unknown'

    def is_error(span: object) -> bool:
        lowered = status_text(span).lower()
        return 'error' in lowered or 'failed' in lowered

    entries: dict[str, object] = {}
    order: list[str] = []
    for index, span in enumerate(spans):
        span_id = text(span, 'span_id') or text(span, 'id') or f'unknown-{index}'
        entries[span_id] = span
        order.append(span_id)
    children: dict[str, list[str]] = {span_id: [] for span_id in order}
    roots: list[str] = []
    for span_id in order:
        parent = text(entries[span_id], 'parent_span_id') or text(entries[span_id], 'parent_id')
        if parent in children and parent != span_id:
            children[parent].append(span_id)
        else:
            roots.append(span_id)
    errors = [span_id for span_id in order if is_error(entries[span_id])]
    first_error = errors[0] if errors else None
    ancestors: set[str] = set()
    if first_error is not None:
        parent_by_child = {child: parent for parent, child_ids in children.items() for child in child_ids}
        current = first_error
        while current in parent_by_child and current not in ancestors:
            current = parent_by_child[current]
            ancestors.add(current)
    durations: list[float] = []
    for span in spans:
        duration = value(span, 'duration_ms')
        if isinstance(duration, (int, float)) and duration >= 0:
            durations.append(float(duration))
    max_duration = max(durations, default=0.0)

    def row(span_id: str) -> str:
        span = entries[span_id]
        status = status_text(span)
        failed = is_error(span)
        kind = text(span, 'type') or text(span, 'operation') or 'unknown kind'
        name = text(span, 'name') or text(span, 'operation') or 'Unnamed span'
        duration = value(span, 'duration_ms')
        duration_label = f'{duration:g} ms' if isinstance(duration, (int, float)) else 'duration unknown'
        width = (
            max(2, min(100, int(float(duration) / max_duration * 100)))
            if isinstance(duration, (int, float)) and max_duration
            else 2
        )
        usage = value(span, 'usage')
        tokens = value(usage, 'total_tokens') if usage is not None else value(span, 'total_tokens')
        token_label = f'{tokens} tokens' if isinstance(tokens, (int, float)) else 'tokens unknown'
        status_obj = value(span, 'status')
        message = value(span, 'status_message') or value(span, 'statusMessage') or value(span, 'error_message')
        if not isinstance(message, str) and status_obj is not None and not isinstance(status_obj, str):
            message = value(status_obj, 'message') or value(status_obj, 'description')
        if span_id == first_error and isinstance(first_error_message, str) and first_error_message.strip():
            message = first_error_message
        if failed and not isinstance(message, str):
            message = 'No status message available.'
        link = trace_link_button(
            trace_span_url(trace_id, text(span, 'span_id') or text(span, 'id'), experiment_url),
            'Orq ↗',
            onclick='event.stopPropagation()',
        )
        return (
            f'<div class="fd-span-row{" fd-span-error" if failed else ""}{" fd-span-first-error" if span_id == first_error else ""}">'
            f'<span class="fd-span-kind">{esc(kind)}</span><b>{esc(name)}</b>'
            f'<span class="fd-span-duration"><i style="width:{width}%"></i>{esc(duration_label)}</span>'
            f'<span>{esc(token_label)}</span><span class="fd-span-status">{esc(status)}</span>{link}</div>'
            + (f'<p class="fd-span-message">{esc(str(message))}</p>' if failed else '')
        )

    if not spans:
        return '<p class="finder-empty">No spans are available for this trace.</p>'
    rendered: set[str] = set()
    output: list[str] = []
    for start in [*roots, *order]:
        if start in rendered:
            continue
        stack: list[tuple[str, str]] = [('node', start)]
        while stack:
            action, span_id = stack.pop()
            if action == 'close':
                output.append('</div></details>')
                continue
            if span_id in rendered:
                continue
            rendered.add(span_id)
            descendants = [child for child in children[span_id] if child not in rendered]
            row_html = row(span_id)
            if descendants:
                opened = (
                    ' open'
                    if span_id in ancestors or span_id == first_error or (first_error is None and span_id in roots)
                    else ''
                )
                output.append(
                    f'<details class="fd-span-node"{opened}><summary>{row_html}</summary><div class="fd-span-children">'
                )
                stack.append(('close', span_id))
                stack.extend(('node', child) for child in reversed(descendants))
            else:
                output.append(f'<div class="fd-span-node">{row_html}</div>')
    return '<div class="fd-span-tree">' + ''.join(output) + '</div>'


def _message_text(message: dict[str, object]) -> str:
    """Render chat ``content``/``tool_calls`` or OTel GenAI ``parts`` (the Responses span shape)."""
    content = message.get('content')
    blocks = [content if content not in (None, '') else message.get('parts'), message.get('tool_calls')]
    return '\n'.join(text for block in blocks if (text := _block_text(block)))


def _thread_message(
    message: dict[str, object], index: int, *, selected: bool, kind: Kind, label: str | None, tokens: int
) -> str:
    role = str(message.get('role', 'unknown'))
    content = _message_text(message)
    preview = ' '.join(content.split())
    if len(preview) > 100:
        preview = preview[:100].rstrip() + '…'
    name = KIND_LABELS.get(kind, role)
    tool = f' · <span class="mono">{esc(label)}</span>' if label else ''
    return (
        f'<details class="fd-msg k-{esc(kind)}{" on" if selected else ""}" id="msg-{index}" data-msg="{index}"{" open" if selected and len(content) <= 10_000 else ""}>'
        f'<summary><span class="role"><b>{esc(name)}</b>{tool}</span><em class="fd-msg-meta">#{index} · ~{tokens:,} tok</em>'
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
