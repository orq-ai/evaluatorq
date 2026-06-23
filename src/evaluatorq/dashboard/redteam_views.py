"""Thin route-wiring module for the four redteam interactive views.

Routes (all return HTML fragments, no full page shell):

    GET /r/{rid}/view/breakdown?group_by=&stack_by=
    GET /r/{rid}/view/agent-heatmap?dim=
    GET /r/{rid}/view/conversation?idx=
    GET /r/{rid}/view/disagreement?a=&b=&page=

Each route loads the RedTeamReport via library.resolve / surfaces.ADAPTERS['redteam'],
applies the active filter selections, and delegates to the appropriate render
function from ``redteam_charts`` or ``redteam_transcripts``.

Public entry point:
    register_redteam_view_routes(app, roots)

Imported by ``evaluatorq.dashboard.app.build_app``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from starlette.requests import Request  # noqa: TC002 — FastHTML inspects this annotation at runtime
from starlette.responses import Response

from evaluatorq.common.reports import esc
from evaluatorq.dashboard.redteam_charts import render_agent_heatmap, render_breakdown
from evaluatorq.dashboard.redteam_transcripts import render_conversation, render_disagreement

if TYPE_CHECKING:
    from pathlib import Path

    from evaluatorq.redteam.contracts import RedTeamReport


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_report(rid: str, roots: list[Path] | None) -> RedTeamReport | None:
    """Load a RedTeamReport by report ID, returning None on miss or error."""
    from evaluatorq.dashboard import library
    from evaluatorq.dashboard.surfaces import ADAPTERS

    path = library.resolve(rid, roots)
    if path is None:
        logger.debug("redteam_views: report id not found: {}", rid)
        return None
    surface, _raw = library.load_surface(path)
    if surface != "redteam":
        logger.debug("redteam_views: surface mismatch for {}: {}", rid, surface)
        return None
    adapter = ADAPTERS.get("redteam")
    if adapter is None:
        return None
    try:
        report: RedTeamReport = adapter.load(path)
    except Exception:
        logger.opt(exception=True).warning("redteam_views: failed to load {}", path)
        return None
    return report


def _404(message: str) -> str:
    return f'<div class="rt-view-error"><p>{esc(message)}</p></div>'


def _parse_redteam_filter(req: Request) -> dict[str, list[str]]:
    """Parse redteam filter selections from the request query-string.

    Reads the same dimension names that ``FILTERS['redteam']`` uses so that
    ``hx-include="#filter-form"`` on each panel container automatically
    carries the current filter state into every panel ``hx-get`` request.

    Returns an empty dict when no filter params are present (≡ "show all").
    """
    from evaluatorq.dashboard.filters import FILTERS

    filter_def = FILTERS.get("redteam")
    if filter_def is None:
        return {}
    selections: dict[str, list[str]] = {}
    for dim in filter_def.dimensions:
        vals = req.query_params.getlist(dim)
        if vals:
            selections[dim] = vals
    return selections


def _apply_redteam_filter(report: RedTeamReport, selections: dict[str, list[str]]) -> list[Any]:
    """Return the filtered result list; empty selections ≡ all results."""
    from evaluatorq.dashboard.filters import FILTERS

    filter_def = FILTERS.get("redteam")
    if filter_def is None or not selections:
        return list(report.results)
    return filter_def.apply(report, selections)


# ---------------------------------------------------------------------------
# Route factory: register all four views on a FastHTML app
# ---------------------------------------------------------------------------


def register_redteam_view_routes(app: Any, roots: list[Any] | None = None) -> None:
    """Register the four /r/{rid}/view/* HTMX routes on *app*.

    Called from ``evaluatorq.dashboard.app.build_app`` after the main routes.

    Each route reads the same filter dimension query params that the filter
    form POSTs (carried via ``hx-include="#filter-form"`` on each panel
    container), applies them to the loaded report, and renders the panel from
    the filtered result set.  This gives filter parity with the static report
    body that ``POST /r/{rid}/filter`` already handles correctly.
    """
    @app.get("/r/{rid}/view/breakdown")
    def view_breakdown(rid: str, req: Request) -> Response:
        group_by = req.query_params.get("group_by", "vulnerability")
        stack_by_raw = req.query_params.get("stack_by", "none")
        stack_by = None if stack_by_raw in ("none", "") else stack_by_raw

        report = _load_report(rid, roots)
        if report is None:
            return Response(_404(f"Report {rid} not found"), status_code=404, media_type="text/html")

        selections = _parse_redteam_filter(req)
        filtered_results = _apply_redteam_filter(report, selections)
        filtered_report = report.model_copy(update={"results": filtered_results})

        html = render_breakdown(report=filtered_report, group_by=group_by, stack_by=stack_by, rid=rid)
        return Response(html, media_type="text/html")

    @app.get("/r/{rid}/view/agent-heatmap")
    def view_agent_heatmap(rid: str, req: Request) -> Response:
        dim = req.query_params.get("dim", "vulnerability")

        report = _load_report(rid, roots)
        if report is None:
            return Response(_404(f"Report {rid} not found"), status_code=404, media_type="text/html")

        selections = _parse_redteam_filter(req)
        filtered_results = _apply_redteam_filter(report, selections)
        filtered_report = report.model_copy(update={"results": filtered_results})

        html = render_agent_heatmap(report=filtered_report, dim=dim, rid=rid)
        return Response(html, media_type="text/html")

    @app.get("/r/{rid}/view/conversation")
    def view_conversation(rid: str, req: Request) -> Response:
        try:
            idx = int(req.query_params.get("idx", "0"))
        except (ValueError, TypeError):
            idx = 0

        report = _load_report(rid, roots)
        if report is None:
            return Response(_404(f"Report {rid} not found"), status_code=404, media_type="text/html")

        selections = _parse_redteam_filter(req)
        filtered_results = _apply_redteam_filter(report, selections)
        filtered_report = report.model_copy(update={"results": filtered_results})

        if filtered_results and idx >= len(filtered_results):
            idx = 0

        html = render_conversation(report=filtered_report, idx=idx, rid=rid)
        return Response(html, media_type="text/html")

    @app.get("/r/{rid}/view/disagreement")
    def view_disagreement(rid: str, req: Request) -> Response:
        agent_a = req.query_params.get("a", "")
        agent_b = req.query_params.get("b", "")
        try:
            page = int(req.query_params.get("page", "1"))
        except (ValueError, TypeError):
            page = 1

        report = _load_report(rid, roots)
        if report is None:
            return Response(_404(f"Report {rid} not found"), status_code=404, media_type="text/html")

        selections = _parse_redteam_filter(req)
        filtered_results = _apply_redteam_filter(report, selections)
        filtered_report = report.model_copy(update={"results": filtered_results})

        html = render_disagreement(
            report=filtered_report, agent_a=agent_a, agent_b=agent_b, page=page, rid=rid
        )
        return Response(html, media_type="text/html")
