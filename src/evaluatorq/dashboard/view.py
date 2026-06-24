"""View helpers: HTML fragments and FastHTML header assets for the dashboard.

``head_assets()`` returns a tuple of FastHTML header elements (Script/Link
tags) for the vendored JS (htmx, vega trio, dashboard.js) under ``/static/``,
served by the app from ``dashboard/static/``.

``index_body(cards)`` and ``report_not_found()`` render pure-HTML fragments
that are injected into the shell via ``shell.page()``.

``render_filter_form(rid, surface, opts, selections)`` renders the HTMX
filter sidebar form.  ``filter_fragment(rid, surface, body_html, form_html)``
combines a re-rendered body and a re-rendered form into the HTMX swap target.

``render_message_list(messages, *, role_labels, class_prefix)`` renders a
role-labeled message list as a series of ``<div>`` elements.  Shared by
``sim_views`` (and any other surface that uses the simple flat layout).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from fasthtml.common import Script

from evaluatorq.common.reports import esc
from evaluatorq.dashboard.surfaces import SURFACE_LABELS

if TYPE_CHECKING:
    from evaluatorq.dashboard.library import ReportCard


def head_assets() -> tuple[Script, ...]:
    """Return FastHTML header elements for vendored JS assets.

    The ``/static/`` files are vendored under ``dashboard/static/`` and served
    by the app; these Script tags wire them into every report page.
    """
    return (
        Script(src='/static/htmx.min.js'),
        Script(src='/static/vega.min.js'),
        Script(src='/static/vega-lite.min.js'),
        Script(src='/static/vega-embed.min.js'),
        Script(src='/static/dashboard.js'),
    )


def index_body(cards: list[ReportCard], *, active_surface: str | None = None) -> str:
    """Render the report-listing page body as an HTML fragment.

    When *active_surface* is given (``'redteam'`` | ``'sim'``), only cards
    matching that surface are shown.  Passing ``None`` (default) shows all.

    Returns a ``<section>`` element containing either a grid of report cards
    or a friendly "no reports" message when the (filtered) cards list is empty.
    """
    visible = [c for c in cards if active_surface is None or c.surface == active_surface]

    if not visible:
        return (
            '<section class="report-index">'
            '<h1>Reports</h1>'
            '<p class="empty-state">No reports found. Run a red team or simulation job to generate reports.</p>'
            '</section>'
        )

    items: list[str] = []
    for card in visible:
        error_badge = f'<span class="card-error" title="{esc(card.error)}">error</span>' if card.error else ''
        created = card.created_at.strftime('%Y-%m-%d %H:%M') if card.created_at else ''
        surface_label = SURFACE_LABELS.get(card.surface, card.surface)
        items.append(
            f'<article class="report-card-item">'
            f'<a href="/r/{card.id}" class="report-card-link">'
            f'<div class="report-card-surface">{surface_label}</div>'
            f'<div class="report-card-name">{esc(card.name)}{error_badge}</div>'
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
        f' data-rid="{esc(rid)}">'
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
    has_markdown: bool = False,
    has_csv: bool = False,
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
                      When provided, filter params are appended to CSV/JSON
                      download links so the downloaded data reflects the
                      currently filtered set.
        has_markdown: Whether to include a Markdown download link.
        has_csv:      Whether to include a CSV download link.
        oob:          When ``True``, add ``hx-swap-oob="true"`` so HTMX
                      replaces the sidebar in-place without it being inside
                      the primary swap target.

    Returns:
        An HTML ``<section>`` fragment.
    """
    safe_rid = esc(rid)

    # Build the query-string from selections (multi-value) using urlencode so
    # that & separators are NOT HTML-escaped.  Only the rid path segment is
    # escaped via esc().
    if selections:
        # Flatten dict[str, list[str]] → list of (key, val) pairs for urlencode.
        pairs: list[tuple[str, str]] = [
            (k, v) for k, vals in selections.items() for v in vals
        ]
        qs = f'?{urlencode(pairs)}' if pairs else ''
    else:
        qs = ''

    oob_attr = ' hx-swap-oob="true"' if oob else ''

    links: list[str] = [
        f'<a class="download-link" href="/r/{safe_rid}/export.html">HTML</a>',
    ]
    if has_markdown:
        links.append(f'<a class="download-link" href="/r/{safe_rid}/export.md">Markdown</a>')
    if has_csv:
        links.append(f'<a class="download-link" href="/r/{safe_rid}/export.csv{qs}">CSV</a>')
    links.append(f'<a class="download-link" href="/r/{safe_rid}/export.json{qs}">JSON</a>')

    inner = '\n'.join(links)
    return (
        f'<section id="{_DOWNLOAD_SIDEBAR_ID}" class="download-sidebar"{oob_attr}>'
        f'<h3 class="download-title">Downloads</h3>{inner}</section>'
    )


def render_message_list(
    messages: list[Any],
    *,
    role_labels: dict[str, str],
    class_prefix: str,
) -> str:
    """Render a role-labeled message list as a series of ``<div>`` elements.

    Each message produces:

    .. code-block:: html

        <div class="{class_prefix}-msg {class_prefix}-msg-{css_role}">
          <span class="{class_prefix}-msg-role">{label}</span>
          <pre class="{class_prefix}-msg-content">{esc(content)}</pre>
        </div>

    Where ``{css_role}`` is the raw role value when it is one of
    ``user``, ``assistant``, ``system``, ``tool``; otherwise ``unknown``.

    Every content string is passed through ``esc()`` (HTML-escaping) to
    prevent stored-XSS vectors.

    Args:
        messages:     Sequence of message dicts (``role`` / ``content`` keys)
                      or objects with ``.role`` / ``.content`` attributes.
        role_labels:  Mapping from role name → display label.  Falls back to
                      the raw role string when a role is not in the map.
        class_prefix: CSS class namespace (e.g. ``"sim"`` → ``sim-msg``).

    Returns:
        Concatenated HTML string (empty string when *messages* is empty).
    """
    known_roles = frozenset({"user", "assistant", "system", "tool"})

    parts: list[str] = []
    for msg in messages:
        # Support both dict-style and attribute-style message objects.
        if isinstance(msg, dict):
            role = str(msg.get("role", "unknown"))
            raw_content = msg.get("content", "")
        else:
            role = str(getattr(msg, "role", "unknown"))
            raw_content = getattr(msg, "content", "")

        label = role_labels.get(role, role)
        content_text: str = raw_content if isinstance(raw_content, str) else str(raw_content or "")
        safe_content = esc(content_text)
        css_role = role if role in known_roles else "unknown"

        parts.append(
            f'<div class="{class_prefix}-msg {class_prefix}-msg-{esc(css_role)}">'
            f'<span class="{class_prefix}-msg-role">{esc(label)}</span>'
            f'<pre class="{class_prefix}-msg-content">{safe_content}</pre>'
            f'</div>'
        )

    return "".join(parts)


def _sim_rowlist_wrapper(rid: str, inner: str) -> str:
    """Wrap *inner* HTML in the HTMX container div for the sim row-list.

    The div re-fetches itself when the ``orq:filter-changed`` event fires,
    carrying the current filter form values via ``hx-include``.  Both the
    initial page render (``sim_interactive_panels``) and the HTMX fragment
    route (``GET /r/{rid}/sim/row-list``) must return this identical wrapper
    so that ``hx-swap="outerHTML"`` replaces the correct element.

    Args:
        rid:   Report ID (URL-safe).
        inner: Already-rendered row-list HTML to embed inside the div.

    Returns:
        An HTML ``<div hx-get=...>`` string.
    """
    safe_rid = esc(rid)
    return (
        f'<div'
        f' hx-get="/r/{safe_rid}/sim/row-list"'
        f' hx-trigger="orq:filter-changed from:body"'
        f' hx-include="#filter-form"'
        f' hx-target="this"'
        f' hx-swap="outerHTML">'
        f'{inner}'
        f'</div>'
    )


def sim_interactive_panels(rid: str, entries: list[Any]) -> str:
    """Render the interactive sim panels section (conversation list + transcript).

    Embeds the sim row list table with HTMX-wired transcript drill-down panel.
    Parity: Streamlit ``_render_transcripts`` (dashboard.py:316-390).

    The outer section carries ``hx-include="#filter-form"`` and
    ``hx-trigger="load, orq:filter-changed from:body"`` so that every row's
    ``hx-get`` request for the transcript detail automatically includes the
    active filter selections, and the section reloads when the filter changes.

    Args:
        rid:     Report ID (URL-safe).
        entries: Individual-results entries from the section layer.

    Returns:
        An HTML ``<section class="sim-interactive-panels">`` fragment.
    """
    from evaluatorq.dashboard.sim_views import render_sim_row_list

    row_list = render_sim_row_list(rid, entries)
    return (
        f'<section class="sim-interactive-panels">'
        f'<h1 class="sim-panels-title">Conversations</h1>'
        f'{_sim_rowlist_wrapper(rid, row_list)}'
        f'</section>'
    )


def redteam_interactive_panels(rid: str) -> str:
    """Render the interactive dashboard panels section for a redteam report.

    Returns an HTML ``<section>`` containing four HTMX-wired panels that load
    their content via ``GET /r/{rid}/view/*`` routes:

    - Interactive breakdown (group_by x stack_by bar chart)
    - Agent heatmap (dimension selector — multi-agent reports only)
    - Conversation viewer (per-row transcript drill-down)
    - Disagreement viewer (agent-pair side-by-side — multi-agent only)

    Each panel placeholder ``<div>`` carries:

    - ``hx-trigger="load, orq:filter-changed from:body"`` so it fetches on
      initial page load AND refetches whenever the filter form fires the
      ``orq:filter-changed`` custom event (emitted by the POST /filter handler
      via the ``HX-Trigger`` response header).
    - ``hx-include="#filter-form"`` so each ``hx-get`` carries the current
      filter selections as query params, giving the view routes the same filter
      state the static body already uses.

    The panel-own params (group_by, stack_by, dim, a, b, page, idx) live in the
    ``hx-get`` URL and are preserved by ``hx-include`` being additive (it only
    appends form fields; it does not replace URL params).  The filter dimension
    names (result, agent, category, severity, technique, delivery_method,
    vulnerability) do not collide with any panel-own param names.

    Task 6's ``dashboard.js`` re-embeds ``render_embed`` Vega charts after
    each HTMX swap.
    """
    safe_rid = esc(rid)

    breakdown = (
        f'<div class="rt-panel" id="panel-breakdown">'
        f'<h2 class="rt-panel-title">Interactive Breakdown</h2>'
        f'<div'
        f' hx-get="/r/{safe_rid}/view/breakdown?group_by=vulnerability&amp;stack_by=none"'
        f' hx-trigger="load, orq:filter-changed from:body"'
        f' hx-include="#filter-form"'
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
        f' hx-trigger="load, orq:filter-changed from:body"'
        f' hx-include="#filter-form"'
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
        f' hx-trigger="load, orq:filter-changed from:body"'
        f' hx-include="#filter-form"'
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
        f' hx-trigger="load, orq:filter-changed from:body"'
        f' hx-include="#filter-form"'
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
