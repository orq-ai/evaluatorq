"""HTML builders for the question-first Trace search page."""

from __future__ import annotations

from typing import TYPE_CHECKING

from evaluatorq.common.reports import esc
from evaluatorq.dashboard.facet_picker import render_facet_chips
from evaluatorq.dashboard.security import csrf_field
from evaluatorq.dashboard.shell import page
from evaluatorq.dashboard.trace_finder import views as shared

if TYPE_CHECKING:
    from evaluatorq.trace_finder import DashboardSettings, FacetCatalogue, RunSnapshot


def _hero(snapshot: RunSnapshot, *, api_available: bool, error: str | None = None) -> str:
    request = snapshot.request
    query = request.query if request else ''
    mode = request.mode if request else 'immediate'
    disabled = '' if api_available else ' disabled'
    error_html = f'<div class="finder-form-error" role="alert">{esc(error)}</div>' if error else ''
    examples = ''.join(
        f'<button type="button" data-finder-example="{esc(sample)}">{esc(sample)}</button>' for sample in shared.SAMPLES
    )
    return (
        '<section class="finder-hero"><div class="finder-hero-bg"></div>'
        '<h2 class="finder-title">Find the signal.</h2>'
        '<p class="finder-sub">Ask a question of your traces. Inspect every judgment.</p>'
        '<form id="finder-query-form" class="finder-query" hx-post="/find/run" hx-target="#finder-body" '
        'hx-swap="innerHTML" hx-include="#finder-controls" hx-disabled-elt="find button">'
        f'{csrf_field()}{shared.icon_search()}<div class="col"><textarea name="query" rows="1" required{disabled} '
        f'placeholder="Describe the traces you want to find…">{esc(query)}</textarea></div>'
        f'<button class="finder-go" type="submit"{disabled}><span class="finder-go-idle">Find traces <span aria-hidden="true">↗</span></span>'
        '<span class="finder-go-working" role="status">Starting search…</span></button></form>'
        '<div class="finder-below"><div class="finder-seg" role="radiogroup" aria-label="Mode">'
        f'<label><input type="radio" name="mode" value="immediate" form="finder-query-form"{" checked" if mode != "review" else ""} '
        'hx-post="/find/reset" hx-trigger="change[document.getElementById(\'finder-start-form\')]" '
        'hx-include="#finder-query-form" hx-target="#finder-body" hx-swap="innerHTML" hx-indicator="#finder-mode-working"><span>Immediate</span></label>'
        f'<label><input type="radio" name="mode" value="review" form="finder-query-form"{" checked" if mode == "review" else ""}><span>Review first</span></label>'
        f'</div><span id="finder-mode-working" role="status">Resetting review…</span><span class="ex"><button type="button" class="link">Examples ▾</button><span class="finder-examples"><span class="hd">Examples</span>{examples}</span></span>'
        f'<span class="spacer"></span><span class="hint"><kbd>⌘</kbd> <kbd>↵</kbd> to run</span></div>{error_html}</section>'
    )


def _controls(
    snapshot: RunSnapshot,
    settings: DashboardSettings,
    catalogue: FacetCatalogue | None,
    *,
    pending: bool,
) -> str:
    request = snapshot.request
    population = request.population if request else None
    review = snapshot.state == 'awaiting_review'
    form_id = 'finder-start-form' if review else 'finder-query-form'
    facets = population.facets if population else snapshot.explicit_filters
    numeric = population.numeric if population else snapshot.explicit_numeric
    window_days = settings.window_days
    if population and population.start and population.end:
        window_days = max(1, round((population.end - population.start).total_seconds() / 86400))
    values = {
        'window_days': window_days,
        'limit': population.limit if population else settings.limit,
        'parallelism': request.parallelism if request else settings.parallelism,
    }
    carried_facets = facets if review else snapshot.explicit_filters
    carried_numeric = numeric if review else snapshot.explicit_numeric
    hidden = ''.join(
        f'<input type="hidden" form="{form_id}" name="facet_{name}" value="{esc(value)}">'
        for name in shared.FACET_NAMES
        for value in sorted(getattr(carried_facets, name))
    )
    keep = {name: f'id="finder-{name}-{form_id}" hx-preserve' for name in values}
    count = f'<span class="count">{snapshot.total} traces selected</span>' if snapshot.total else ''
    scope = (
        f'<span class="quiet"><b>Project</b><a href="/settings">{esc(settings.orq_project_name or settings.orq_project_id)}</a></span>'
        if settings.orq_project_id
        else ''
    )
    return (
        f'<div class="finder-controls" id="finder-controls">{hidden}{scope}'
        f'{render_facet_chips(facets, numeric, removable=snapshot.state not in {"compiling", "classifying"})}'
        f'<span class="addwrap"><button class="add" type="button" aria-haspopup="true">+ Filter</button>'
        f'{shared.facet_menu(catalogue, numeric=carried_numeric, form_id=form_id, selection=carried_facets, pending=pending, window_days=window_days)}'
        '<span class="finder-facet-loading" role="status">Loading filters…</span></span><span class="spacer"></span>'
        f'<span class="quiet"><b>Window</b><input {keep["window_days"]} form="{form_id}" name="window_days" type="number" min="1" max="90" value="{values["window_days"]}" style="width:64px" '
        'hx-get="/find/facets?form_id='
        + form_id
        + '" hx-trigger="change" hx-include="#finder-controls" hx-target=".finder-facets" hx-swap="outerHTML" hx-indicator=".finder-facet-loading"></span>'
        f'<span class="quiet"><b>Limit</b><input {keep["limit"]} form="{form_id}" name="limit" type="number" min="1" max="5000" value="{values["limit"]}" style="width:72px"></span>'
        f'<span class="quiet"><b>Parallel</b><input {keep["parallelism"]} form="{form_id}" name="parallelism" type="number" min="1" max="200" value="{values["parallelism"]}" style="width:64px"></span>{count}</div>'
    )


def _body(
    snapshot: RunSnapshot,
    settings: DashboardSettings,
    *,
    api_available: bool,
    catalogue: FacetCatalogue | None,
    pending: bool,
) -> str:
    status = shared.status_indicator(snapshot)
    controls = _controls(snapshot, settings, catalogue, pending=pending)
    if snapshot.state == 'awaiting_review' and snapshot.dimensions is not None:
        return (
            f'{status}{controls}<div class="finder-review"><span>⏸</span><span><b>Review the plan before running per-trace classification.</b> Edit the task, criteria or filters, then start.</span></div>'
            f'{shared.task_panel(snapshot.dimensions, editable=True, open_=True, request=snapshot.request)}'
            f'{shared.filter_output_panel(snapshot)}'
        )
    result = f'{status}{controls}{shared.field(snapshot, api_available=api_available, export_url="/find/export.json?surface=search")}'
    if snapshot.state not in {'idle', 'awaiting_review'}:
        result += shared.table(snapshot)
        if snapshot.dimensions is not None:
            result += shared.task_panel(snapshot.dimensions, editable=False) + shared.filter_output_panel(snapshot)
    return result


def search_fragment(
    snapshot: RunSnapshot,
    settings: DashboardSettings,
    *,
    error: str | None = None,
    api_available: bool = True,
    catalogue: FacetCatalogue | None = None,
    pending: bool = False,
) -> str:
    """Render the replaceable /find body, including polling while a search runs."""
    polling = (
        ' hx-get="/find/poll?surface=search" hx-trigger="every 1s" hx-target="#finder-body" hx-swap="innerHTML"'
        if snapshot.state in {'compiling', 'classifying'}
        else ''
    )
    error_html = f'<div class="finder-review finder-form-error" role="alert">{esc(error)}</div>' if error else ''
    body = _body(snapshot, settings, api_available=api_available, catalogue=catalogue, pending=pending)
    return f'<div class="finder-body-fragment"{polling}>{error_html}{body}</div>'


def search_page_html(
    snapshot: RunSnapshot,
    settings: DashboardSettings,
    *,
    api_available: bool,
    error: str | None = None,
    catalogue: FacetCatalogue | None = None,
    pending: bool = False,
) -> str:
    """Render the standalone, question-first legacy Trace search page."""
    html = (
        f'<div class="finder" hx-vals=\'{{"surface":"search"}}\'>{_hero(snapshot, api_available=api_available, error=error)}'
        f'<div id="finder-body">{search_fragment(snapshot, settings, api_available=api_available, catalogue=catalogue, pending=pending)}</div>'
        '<div id="finder-drawer"></div><div id="finder-drawer-loading" role="status">Loading trace…</div></div>'
    )
    return page('Trace search', html, active_nav='find')
