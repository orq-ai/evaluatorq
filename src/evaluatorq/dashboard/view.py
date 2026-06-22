"""View helpers: HTML fragments and FastHTML header assets for the dashboard.

``head_assets()`` returns a tuple of FastHTML header elements (Script/Link
tags) for vendored JS (htmx, vega trio, dashboard.js).  The referenced paths
under ``/static/`` do not yet exist on disk — they will be vendored in a
later task.  The tags can be emitted now so routes are wired correctly.

``index_body(cards)`` and ``report_not_found()`` render pure-HTML fragments
that are injected into the shell via ``shell.page()``.

``render_filter_form(rid, surface, opts, selections)`` renders the HTMX
filter sidebar form.  ``filter_fragment(rid, surface, body_html, form_html)``
combines a re-rendered body and a re-rendered form into the HTMX swap target.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlencode

from fasthtml.common import Script

from evaluatorq.common.reports import esc

if TYPE_CHECKING:
    from evaluatorq.dashboard.library import ReportCard


def head_assets() -> tuple[Script, ...]:
    """Return FastHTML header elements for vendored JS assets.

    The ``/static/`` paths will be populated in a later task (Task 6/10).
    Returning Script tags now ensures routes that call ``head_assets()`` emit
    the correct ``<script src>`` markup without requiring the files to exist.
    """
    return (
        Script(src='/static/htmx.min.js'),
        Script(src='/static/vega.min.js'),
        Script(src='/static/vega-lite.min.js'),
        Script(src='/static/vega-embed.min.js'),
        Script(src='/static/dashboard.js'),
    )


def index_body(cards: list[ReportCard]) -> str:
    """Render the report-listing page body as an HTML fragment.

    Returns a ``<section>`` element containing either a grid of report cards
    or a friendly "no reports" message when *cards* is empty.
    """
    if not cards:
        return (
            '<section class="report-index">'
            '<h1>Reports</h1>'
            '<p class="empty-state">No reports found. Run a red team or simulation job to generate reports.</p>'
            '</section>'
        )

    items: list[str] = []
    for card in cards:
        error_badge = f'<span class="card-error" title="{esc(card.error)}">error</span>' if card.error else ''
        created = card.created_at.strftime('%Y-%m-%d %H:%M') if card.created_at else ''
        surface_label = card.surface.replace('redteam', 'Red Team').replace('sim', 'Simulation')
        items.append(
            f'<article class="report-card-item">'
            f'<a href="/r/{card.id}" class="report-card-link">'
            f'<div class="report-card-surface">{surface_label}</div>'
            f'<div class="report-card-name">{card.name}{error_badge}</div>'
            f'<div class="report-card-meta">{created}'
            f'{" · " + card.headline if card.headline else ""}'
            f'</div>'
            f'</a>'
            f'<a href="/r/{card.id}/export" class="report-card-export" title="Download standalone HTML">export</a>'
            f'</article>'
        )

    grid = f'<div class="report-grid">{"".join(items)}</div>'
    return f'<section class="report-index"><h1>Reports</h1>{grid}</section>'


def report_not_found(rid: str) -> str:
    """Render a 404-style body fragment for an unknown report ID."""
    return (
        '<section class="report-not-found">'
        '<h1>Report not found</h1>'
        f'<p>No report with id <code>{esc(rid)}</code> could be located.</p>'
        '<p><a href="/">Back to reports</a></p>'
        '</section>'
    )


def report_broken(rid: str, filename: str, detail: str) -> str:
    """Render an error page fragment for a report that sniffed OK but failed to load."""
    return (
        '<section class="report-broken">'
        '<h1>Report could not be loaded</h1>'
        f'<p class="report-broken-filename">File: <code>{esc(filename)}</code></p>'
        f'<p class="report-broken-detail">{esc(detail)}</p>'
        '<p><a href="/">Back to reports</a></p>'
        '</section>'
    )


# ---------------------------------------------------------------------------
# Filter form + HTMX fragment helpers
# ---------------------------------------------------------------------------

# Dimensions that behave as a radio (single-select).  All others are
# multiselect checkboxes.
_RADIO_DIMS: frozenset[str] = frozenset({'result', 'goal_outcome'})


def render_filter_form(
    rid: str,
    surface: str,
    opts: dict[str, list[str]],
    selections: dict[str, list[str]],
) -> str:
    """Render the HTMX filter sidebar form as an HTML fragment.

    The form POSTs the full set of dimensions on every change (``hx-trigger``
    on the containing form's ``change`` event).  Missing dimensions default to
    "all" inside the filter route handler.

    Args:
        rid:        Report ID (used to construct the POST URL).
        surface:    Surface key (``'redteam'`` | ``'sim'``).
        opts:       Option lists per dimension (from ``FilterDef.options`` or
                    ``FilterDef.recompute_options``).
        selections: Currently active selections per dimension.

    Returns:
        An HTML ``<form>`` fragment suitable for injection into the sidebar.
    """
    from evaluatorq.dashboard.filters import FILTERS

    filter_def = FILTERS.get(surface)
    if filter_def is None:
        return ''

    parts: list[str] = []
    for dim in filter_def.dimensions:
        dim_opts = opts.get(dim, [])
        if not dim_opts:
            continue

        sel = selections.get(dim, [])

        label = dim.replace('_', ' ').title()
        parts.extend([
            f'<div class="filter-group" data-dim="{esc(dim)}">',
            f'<label class="filter-label">{esc(label)}</label>',
        ])

        if dim in _RADIO_DIMS:
            # Radio buttons: single-select dimension
            current = sel[0] if sel else dim_opts[0]
            for opt in dim_opts:
                checked = ' checked' if opt == current else ''
                parts.append(
                    f'<label class="filter-radio">'
                    f'<input type="radio" name="{esc(dim)}" value="{esc(opt)}"{checked}>'
                    f' {esc(opt)}'
                    f'</label>'
                )
        else:
            # Checkboxes: multiselect dimension.  Default = all selected.
            selected_set = set(sel) if sel else set(dim_opts)
            for opt in dim_opts:
                checked = ' checked' if opt in selected_set else ''
                parts.append(
                    f'<label class="filter-checkbox">'
                    f'<input type="checkbox" name="{esc(dim)}" value="{esc(opt)}"{checked}>'
                    f' {esc(opt)}'
                    f'</label>'
                )

        parts.append('</div>')

    inner = ''.join(parts)
    return (
        f'<form id="filter-form" class="filter-form"'
        f' hx-post="/r/{esc(rid)}/filter"'
        f' hx-trigger="change"'
        f' hx-target="#filter-swap"'
        f' hx-swap="outerHTML">'
        f'<div class="filter-sidebar">'
        f'<h3 class="filter-title">Filters</h3>'
        f'{inner}'
        f'</div>'
        f'</form>'
    )


def filter_fragment(
    rid: str,
    surface: str,
    body_html: str,
    form_html: str,
) -> str:
    """Wrap the re-rendered body + filter form in the HTMX swap container.

    The outer ``<div id="filter-swap">`` is the element that HTMX replaces on
    each POST.  It contains both the filter form (sidebar) and the report body
    so that a single swap updates both simultaneously.

    Args:
        rid:       Report ID (for data-attribute bookkeeping).
        surface:   Surface key.
        body_html: Re-rendered report body HTML fragment.
        form_html: Re-rendered filter form HTML fragment.

    Returns:
        An HTML string starting with ``<div id="filter-swap">``.
    """
    return (
        f'<div id="filter-swap" class="filter-swap-container"'
        f' data-rid="{esc(rid)}" data-surface="{esc(surface)}">'
        f'{form_html}'
        f'<div class="report-body-area">'
        f'{body_html}'
        f'</div>'
        f'</div>'
    )


def report_view_with_filters(
    rid: str,
    surface: str,
    body_html: str,
    form_html: str,
) -> str:
    """Render the full report view including the filter swap container.

    Used by ``GET /r/{rid}`` to produce the initial page body that already
    contains the HTMX-wired filter form.
    """
    swap = filter_fragment(rid, surface, body_html, form_html)
    return f'<section class="report-view">{swap}</section>'


_DOWNLOAD_SIDEBAR_ID = 'download-sidebar'


def download_sidebar(
    rid: str,
    surface: str,
    *,
    selections: dict[str, list[str]] | None = None,
    filter_qs: str = '',
    has_markdown: bool = False,
    has_csv: bool = False,
    has_json: bool = True,
    oob: bool = False,
) -> str:
    """Render the download links sidebar for a report page.

    Generates a ``<section id="download-sidebar" class="download-sidebar">``
    containing links for the available export formats for *surface*.  CSV/JSON
    links carry the active filter query-string so the downloaded data reflects
    the currently filtered set.

    The sidebar has a stable ``id`` (``"download-sidebar"``) so it can be
    targeted by HTMX out-of-band swaps after filter POSTs.

    Args:
        rid:          Report ID (URL-safe).
        surface:      Surface key (``'redteam'`` | ``'sim'``).
        selections:   Active filter selections as ``dict[str, list[str]]``.
                      When provided, takes precedence over *filter_qs*.
        filter_qs:    Legacy: active filter query-string
                      (e.g. ``"result=Vulnerable"``).  Ignored when
                      *selections* is given.  Empty string → no filter suffix.
        has_markdown: Whether to include a Markdown download link.
        has_csv:      Whether to include a CSV download link.
        has_json:     Whether to include a JSON download link.
        oob:          When ``True``, add ``hx-swap-oob="true"`` so HTMX
                      replaces the sidebar in-place without it being inside
                      the primary swap target.

    Returns:
        An HTML ``<section>`` fragment.
    """
    safe_rid = esc(rid)

    # Build the query-string from selections (multi-value) using urlencode so
    # that & separators are NOT HTML-escaped.  Only the rid path segment is
    # escaped via esc().  The legacy filter_qs fallback is kept for callers
    # that still pass a raw string.
    if selections:
        # Flatten dict[str, list[str]] → list of (key, val) pairs for urlencode.
        pairs: list[tuple[str, str]] = [
            (k, v) for k, vals in selections.items() for v in vals
        ]
        qs = f'?{urlencode(pairs)}' if pairs else ''
    else:
        qs = f'?{filter_qs}' if filter_qs else ''

    oob_attr = ' hx-swap-oob="true"' if oob else ''

    links: list[str] = [
        f'<a class="download-link" href="/r/{safe_rid}/export.html">HTML</a>',
    ]
    if has_markdown:
        links.append(f'<a class="download-link" href="/r/{safe_rid}/export.md">Markdown</a>')
    if has_csv:
        links.append(f'<a class="download-link" href="/r/{safe_rid}/export.csv{qs}">CSV</a>')
    if has_json:
        links.append(f'<a class="download-link" href="/r/{safe_rid}/export.json{qs}">JSON</a>')

    inner = '\n'.join(links)
    return (
        f'<section id="{_DOWNLOAD_SIDEBAR_ID}" class="download-sidebar"{oob_attr}>'
        f'<h3 class="download-title">Downloads</h3>{inner}</section>'
    )


def sim_interactive_panels(rid: str, entries: list[dict]) -> str:
    """Render the interactive sim panels section (conversation list + transcript).

    Embeds the sim row list table with HTMX-wired transcript drill-down panel.
    Parity: Streamlit ``_render_transcripts`` (dashboard.py:316-390).

    Args:
        rid:     Report ID (URL-safe).
        entries: Individual-results entries from the section layer.

    Returns:
        An HTML ``<section class="sim-interactive-panels">`` fragment.
    """
    from evaluatorq.dashboard.sim_views import render_sim_row_list

    row_list = render_sim_row_list(rid, entries)
    return (
        f'<section class="sim-interactive-panels"><h1 class="sim-panels-title">Conversations</h1>{row_list}</section>'
    )


def redteam_interactive_panels(rid: str) -> str:
    """Render the interactive dashboard panels section for a redteam report.

    Returns an HTML ``<section>`` containing four HTMX-wired panels that load
    their content via ``GET /r/{rid}/view/*`` routes:

    - Interactive breakdown (group_by x stack_by bar chart)
    - Agent heatmap (dimension selector — multi-agent reports only)
    - Conversation viewer (per-row transcript drill-down)
    - Disagreement viewer (agent-pair side-by-side — multi-agent only)

    Each panel contains a placeholder ``<div>`` with ``hx-get`` + ``hx-trigger``
    ``load`` so the content is fetched on page load without blocking the initial
    render.  Task 6's ``dashboard.js`` re-embeds ``render_embed`` Vega charts
    after each HTMX swap.
    """
    safe_rid = esc(rid)

    breakdown = (
        f'<div class="rt-panel" id="panel-breakdown">'
        f'<h2 class="rt-panel-title">Interactive Breakdown</h2>'
        f'<div'
        f' hx-get="/r/{safe_rid}/view/breakdown?group_by=vulnerability&amp;stack_by=none"'
        f' hx-trigger="load"'
        f' hx-target="this"'
        f' hx-swap="outerHTML">'
        f'<p class="rt-panel-loading">Loading breakdown…</p>'
        f'</div>'
        f'</div>'
    )

    heatmap = (
        f'<div class="rt-panel" id="panel-agent-heatmap">'
        f'<h2 class="rt-panel-title">Agent Heatmap</h2>'
        f'<div'
        f' hx-get="/r/{safe_rid}/view/agent-heatmap?dim=vulnerability"'
        f' hx-trigger="load"'
        f' hx-target="this"'
        f' hx-swap="outerHTML">'
        f'<p class="rt-panel-loading">Loading heatmap…</p>'
        f'</div>'
        f'</div>'
    )

    conversation = (
        f'<div class="rt-panel" id="panel-conversation">'
        f'<h2 class="rt-panel-title">Conversation Viewer</h2>'
        f'<div'
        f' hx-get="/r/{safe_rid}/view/conversation?idx=0"'
        f' hx-trigger="load"'
        f' hx-target="this"'
        f' hx-swap="outerHTML">'
        f'<p class="rt-panel-loading">Loading conversation viewer…</p>'
        f'</div>'
        f'</div>'
    )

    disagreement = (
        f'<div class="rt-panel" id="panel-disagreement">'
        f'<h2 class="rt-panel-title">Disagreement Viewer</h2>'
        f'<div'
        f' hx-get="/r/{safe_rid}/view/disagreement?page=1"'
        f' hx-trigger="load"'
        f' hx-target="this"'
        f' hx-swap="outerHTML">'
        f'<p class="rt-panel-loading">Loading disagreement viewer…</p>'
        f'</div>'
        f'</div>'
    )

    return (
        f'<section class="rt-interactive-panels">'
        f'<h1 class="rt-panels-title">Interactive Analysis</h1>'
        f'{breakdown}'
        f'{heatmap}'
        f'{conversation}'
        f'{disagreement}'
        f'</section>'
    )
