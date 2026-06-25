"""HTMX fragment routes for simulation-dashboard interactive views.

Routes (all return HTML fragments, no full page shell):

    GET /r/{rid}/sim/transcript?idx=   → conversation detail: header, metrics,
                                          judge reason, criteria, full transcript

The row list is rendered inline in the report page body (not via HTMX) but
each row carries an ``hx-get`` link to this transcript endpoint.

Parity source: src/evaluatorq/simulation/ui/dashboard.py lines 316-390
(``_render_transcripts``  / row-click drill-down).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger
from starlette.requests import Request  # noqa: TC002 — FastHTML inspects at runtime
from starlette.responses import Response

from evaluatorq.common.messages import coerce_content_text
from evaluatorq.common.reports import esc
from evaluatorq.dashboard.filter_request import parse_selections
from evaluatorq.dashboard.filters import apply_or_all
from evaluatorq.dashboard.view import _sim_rowlist_wrapper, render_message_list
from evaluatorq.simulation.types import SimulationEntry

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Role label mapping (parity: dashboard.py:385)
# ---------------------------------------------------------------------------

_ROLE_LABELS: dict[str, str] = {
    'user': 'User (sim)',
    'assistant': 'Target',
    'system': 'System',
    'tool': 'Tool',
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_run(rid: str, roots: list[Path] | None) -> Any | None:
    """Load a SimulationRun by report id; returns None on miss/error."""
    from evaluatorq.dashboard import library
    from evaluatorq.dashboard.surfaces import ADAPTERS

    path = library.resolve(rid, roots)
    if path is None:
        logger.debug('sim_views: report id not found: {}', rid)
        return None
    surface, _raw = library.load_surface(path)
    if surface != 'sim':
        logger.debug('sim_views: not a sim report: {}', rid)
        return None
    adapter = ADAPTERS.get('sim')
    if adapter is None:
        return None
    try:
        return adapter.load(path)
    except Exception as exc:
        logger.warning('sim_views: failed to load {}: {}', path.name, exc)
        return None


def _entries_from_run(run: Any) -> list[SimulationEntry]:
    """Build the typed individual-results entry list from a SimulationRun."""
    from evaluatorq.simulation.reports.sections import individual_entries

    return individual_entries(run.results)


# ---------------------------------------------------------------------------
# Row list (embedded in the sim report page, not a separate HTMX route)
# ---------------------------------------------------------------------------


def render_sim_row_list(rid: str, entries: list[SimulationEntry]) -> str:
    """Render the conversation list panel for a sim report.

    Each row is a clickable element with ``hx-get`` pointing to the
    transcript endpoint.  Mirrors the ``table`` dict in dashboard.py:322-334
    (persona, scenario, goal, score, turns, terminated).

    Args:
        rid:     Report ID (URL-safe).
        entries: Typed entry list from ``_entries_from_run`` / ``individual_entries``.

    Returns:
        HTML fragment containing a ``<section class="sim-row-list">``.
    """
    if not entries:
        return '<section class="sim-row-list"><p class="sim-empty">No conversations found.</p></section>'

    safe_rid = esc(rid)
    rows_html: list[str] = []
    for e in entries:
        idx = e.index
        persona = esc(e.persona)
        scenario = esc(e.scenario)
        goal = 'yes' if e.goal_achieved else 'no'
        score = f'{e.goal_completion_score:.2f}'
        turns = str(e.turn_count)
        terminated = esc(e.terminated_by)

        rows_html.append(
            f'<tr class="sim-row-item"'
            f' hx-get="/r/{safe_rid}/sim/transcript?idx={idx}"'
            f' hx-target="#sim-transcript-panel"'
            f' hx-swap="innerHTML"'
            f' style="cursor:pointer">'
            f'<td>{idx + 1}</td>'
            f'<td>{persona}</td>'
            f'<td>{scenario}</td>'
            f'<td>{goal}</td>'
            f'<td>{score}</td>'
            f'<td>{turns}</td>'
            f'<td>{terminated}</td>'
            f'</tr>'
        )

    table_html = (
        '<table class="sim-row-table">'
        '<thead><tr>'
        '<th>#</th><th>Persona</th><th>Scenario</th>'
        '<th>Goal</th><th>Score</th><th>Turns</th><th>Terminated</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        '</table>'
    )

    panel = '<div id="sim-transcript-panel" class="sim-transcript-panel"><p class="sim-select-prompt">Select a row above to view its transcript.</p></div>'

    return f'<section class="sim-row-list"><h2>Conversations</h2>{table_html}{panel}</section>'


# ---------------------------------------------------------------------------
# Transcript fragment
# ---------------------------------------------------------------------------


def render_transcript_fragment(entry: SimulationEntry) -> str:
    """Render the drill-down transcript fragment for a single sim result entry.

    Parity: dashboard.py:356-390.

    All user-supplied content (persona, scenario, judge_reason, message
    content) goes through ``esc()`` — stored-XSS vector.

    Args:
        entry: A typed ``SimulationEntry`` from ``individual_entries``.

    Returns:
        An HTML fragment (no full-page shell).
    """
    idx: int = entry.index
    persona = esc(entry.persona)
    scenario = esc(entry.scenario)
    goal_achieved: bool = entry.goal_achieved
    score: float = entry.goal_completion_score
    turns: int = entry.turn_count
    terminated_by = esc(entry.terminated_by)
    judge_reason = esc(entry.judge_reason or '')
    error = entry.error

    # Header (parity: st.subheader f"#{idx+1} · {persona} · {scenario}")
    header = f'<h3 class="sim-transcript-header">#{idx + 1} &middot; {persona} &middot; {scenario}</h3>'

    # Metrics (parity: 4 columns: goal achieved, score, turns, terminated by)
    metrics = (
        f'<div class="sim-transcript-metrics">'
        f'<div class="sim-metric"><span class="sim-metric-label">Goal achieved</span>'
        f'<span class="sim-metric-value">{"yes" if goal_achieved else "no"}</span></div>'
        f'<div class="sim-metric"><span class="sim-metric-label">Score</span>'
        f'<span class="sim-metric-value">{score:.2f}</span></div>'
        f'<div class="sim-metric"><span class="sim-metric-label">Turns</span>'
        f'<span class="sim-metric-value">{turns}</span></div>'
        f'<div class="sim-metric"><span class="sim-metric-label">Terminated by</span>'
        f'<span class="sim-metric-value">{terminated_by}</span></div>'
        f'</div>'
    )

    # Judge reason
    judge_html = ''
    if judge_reason:
        judge_html = f'<p class="sim-judge-reason"><strong>Judge:</strong> {judge_reason}</p>'

    # Error (parity: st.error)
    error_html = ''
    if error:
        error_html = f'<p class="sim-transcript-error"><strong>Error:</strong> {esc(str(error))}</p>'

    # Criteria (parity: dashboard.py:371-377)
    criteria = entry.criteria or []
    criteria_html = ''
    if criteria:
        rows_parts: list[str] = []
        for c in criteria:
            if c.passed:
                icon = '&#x2705;'  # ✅
            elif c.safety:
                icon = '&#x26D4;'  # ⛔
            else:
                icon = '&#x274C;'  # ❌
            ctype = c.type or ''
            ctype_html = f' <em class="sim-ctype">{esc(ctype)}</em>' if ctype else ''
            desc = esc(c.description)
            rows_parts.append(f'<li class="sim-criterion">{icon} {desc}{ctype_html}</li>')
        criteria_html = (
            f'<div class="sim-criteria">'
            f'<strong>Criteria</strong>'
            f'<ul class="sim-criteria-list">{"".join(rows_parts)}</ul>'
            f'</div>'
        )

    # Transcript (parity: dashboard.py:384-390)
    transcript = entry.transcript or []

    # Normalise content via coerce_content_text (handles OpenAI content blocks)
    # before handing off to the shared renderer.  The '(empty)' fallback is
    # sim-specific so we apply it here rather than inside render_message_list.
    normalised_msgs: list[dict[str, Any]] = []
    for msg in transcript:
        raw_content = msg.content
        content_text = coerce_content_text(raw_content) or '(empty)'
        normalised_msgs.append({'role': msg.role, 'content': content_text})

    transcript_html = (
        (
            f'<div class="sim-transcript">'
            f'<strong>Transcript</strong>'
            f'<div class="sim-transcript-messages">'
            f'{render_message_list(normalised_msgs, role_labels=_ROLE_LABELS, class_prefix="sim")}'
            f'</div>'
            f'</div>'
        )
        if transcript
        else ''
    )

    return (
        f'<div class="sim-transcript-detail">'
        f'{header}'
        f'{metrics}'
        f'{judge_html}'
        f'{error_html}'
        f'{criteria_html}'
        f'{transcript_html}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Route factory
# ---------------------------------------------------------------------------


def register_sim_view_routes(app: Any, roots: list[Any] | None = None) -> None:
    """Register simulation view routes on *app*.

    Called from ``evaluatorq.dashboard.app.build_app``.

    Routes registered here:

    - ``GET /r/{rid}/sim/transcript?idx=`` — transcript drill-down fragment.
      Reads filter dimension query params (via ``hx-include="#filter-form"``)
      and indexes into the FILTERED entry list so the transcript panel stays
      consistent with the row-list.

    - ``GET /r/{rid}/sim/row-list`` — filtered row list fragment.  Used by the
      sim interactive panel container to refetch the conversation table when the
      filter changes (``hx-trigger="orq:filter-changed from:body"``).
    """

    @app.get('/r/{rid}/sim/row-list')
    def sim_row_list(rid: str, req: Request) -> Response:
        """Return the filtered sim conversation row-list fragment.

        Called when the ``orq:filter-changed`` event fires so the row-list
        table reflects the same filter state as the static report body.
        """
        run = _load_run(rid, roots)
        if run is None:
            return Response('Report not found.', status_code=404, media_type='text/html')

        selections = parse_selections(req, 'sim')
        filtered_results = apply_or_all(run, 'sim', selections)
        from evaluatorq.simulation.reports.sections import individual_entries

        entries = individual_entries(filtered_results)

        html = render_sim_row_list(rid, entries)
        # Return wrapped in the same container div that sim_interactive_panels
        # renders so the outerHTML swap replaces the correct element.
        return Response(_sim_rowlist_wrapper(rid, html), media_type='text/html')

    @app.get('/r/{rid}/sim/transcript')
    def sim_transcript(rid: str, req: Request) -> Response:
        """Return the transcript drill-down fragment for a sim result row.

        Query param ``idx`` selects which entry in the filtered result list to
        render (0-based).  Filter dimensions from the active filter form are
        read from the query-string via ``hx-include="#filter-form"`` and
        applied before indexing, so the transcript panel stays consistent with
        the row-list.  Missing or out-of-range ``idx`` returns a graceful
        empty message rather than a 500.
        """
        try:
            idx = int(req.query_params.get('idx', '0'))
        except (ValueError, TypeError):
            idx = 0

        run = _load_run(rid, roots)
        if run is None:
            return Response(
                'Report not found.',
                status_code=404,
                media_type='text/html',
            )

        # Apply any active filter before building the entry list so the idx
        # refers to a position in the same filtered ordering as the row-list.
        selections = parse_selections(req, 'sim')
        filtered_results = apply_or_all(run, 'sim', selections)
        from evaluatorq.simulation.reports.sections import individual_entries

        entries = individual_entries(filtered_results)

        if not entries or idx < 0 or idx >= len(entries):
            return Response(
                '<p class="sim-empty">No conversation at that index.</p>',
                status_code=200,
                media_type='text/html',
            )

        fragment = render_transcript_fragment(entries[idx])
        return Response(fragment, media_type='text/html')
