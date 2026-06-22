"""View helpers: HTML fragments and FastHTML header assets for the dashboard.

``head_assets()`` returns a tuple of FastHTML header elements (Script/Link
tags) for vendored JS (htmx, vega trio, dashboard.js).  The referenced paths
under ``/static/`` do not yet exist on disk — they will be vendored in a
later task.  The tags can be emitted now so routes are wired correctly.

``index_body(cards)`` and ``report_not_found()`` render pure-HTML fragments
that are injected into the shell via ``shell.page()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fasthtml.common import Script

if TYPE_CHECKING:
    from evaluatorq.dashboard.library import ReportCard


def head_assets() -> tuple[Script, ...]:
    """Return FastHTML header elements for vendored JS assets.

    The ``/static/`` paths will be populated in a later task (Task 6/10).
    Returning Script tags now ensures routes that call ``head_assets()`` emit
    the correct ``<script src>`` markup without requiring the files to exist.
    """
    return (
        Script(src="/static/htmx.min.js"),
        Script(src="/static/vega.min.js"),
        Script(src="/static/vega-lite.min.js"),
        Script(src="/static/vega-embed.min.js"),
        Script(src="/static/dashboard.js"),
    )


def index_body(cards: list[ReportCard]) -> str:
    """Render the report-listing page body as an HTML fragment.

    Returns a ``<section>`` element containing either a grid of report cards
    or a friendly "no reports" message when *cards* is empty.
    """
    if not cards:
        return (
            '<section class="report-index">'
            "<h1>Reports</h1>"
            '<p class="empty-state">No reports found. Run a red team or simulation job to generate reports.</p>'
            "</section>"
        )

    items: list[str] = []
    for card in cards:
        error_badge = (
            f'<span class="card-error" title="{card.error}">error</span>' if card.error else ""
        )
        created = card.created_at.strftime("%Y-%m-%d %H:%M") if card.created_at else ""
        surface_label = card.surface.replace("redteam", "Red Team").replace("sim", "Simulation")
        items.append(
            f'<article class="report-card-item">'
            f'<a href="/r/{card.id}" class="report-card-link">'
            f'<div class="report-card-surface">{surface_label}</div>'
            f'<div class="report-card-name">{card.name}{error_badge}</div>'
            f'<div class="report-card-meta">{created}'
            f"{' · ' + card.headline if card.headline else ''}"
            f"</div>"
            f"</a>"
            f'<a href="/r/{card.id}/export" class="report-card-export" title="Download standalone HTML">export</a>'
            f"</article>"
        )

    grid = f'<div class="report-grid">{"".join(items)}</div>'
    return f'<section class="report-index"><h1>Reports</h1>{grid}</section>'


def report_not_found(rid: str) -> str:
    """Render a 404-style body fragment for an unknown report ID."""
    return (
        '<section class="report-not-found">'
        "<h1>Report not found</h1>"
        f'<p>No report with id <code>{rid}</code> could be located.</p>'
        '<p><a href="/">Back to reports</a></p>'
        "</section>"
    )
