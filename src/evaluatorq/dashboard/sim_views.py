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


def _entries_from_run(run: Any) -> list[dict[str, Any]]:
    """Build the individual-results entry list from a SimulationRun.

    Uses the same section builder that the Streamlit dashboard does so the
    fields (transcript, criteria, evaluator_scores, etc.) are identical.
    """
    from evaluatorq.simulation.reports.sections import build_report_sections

    sections = build_report_sections(run.results)
    for s in sections:
        if s.kind == 'individual_results':
            return s.data.get('entries', [])
    return []


def _meta(run: Any, key: str) -> str:
    """Return metadata string from the run-level meta if available.

    The per-result ``metadata`` dict is accessed inside entries; this helper
    is for the filters._meta parity — unused here but kept for symmetry.
    """
    return str(getattr(run, key, 'unknown'))


# ---------------------------------------------------------------------------
# Row list (embedded in the sim report page, not a separate HTMX route)
# ---------------------------------------------------------------------------


def render_sim_row_list(rid: str, entries: list[dict[str, Any]]) -> str:
    """Render the conversation list panel for a sim report.

    Each row is a clickable element with ``hx-get`` pointing to the
    transcript endpoint.  Mirrors the ``table`` dict in dashboard.py:322-334
    (persona, scenario, goal, score, turns, terminated).

    Args:
        rid:     Report ID (URL-safe).
        entries: Entry list from ``_entries_from_run``.

    Returns:
        HTML fragment containing a ``<section class="sim-row-list">``.
    """
    if not entries:
        return '<section class="sim-row-list"><p class="sim-empty">No conversations found.</p></section>'

    safe_rid = esc(rid)
    rows_html: list[str] = []
    for e in entries:
        idx = e['index']
        persona = esc(str(e.get('persona', 'unknown')))
        scenario = esc(str(e.get('scenario', 'unknown')))
        goal = 'yes' if e.get('goal_achieved') else 'no'
        score = f'{e.get("goal_completion_score", 0.0):.2f}'
        turns = str(e.get('turn_count', 0))
        terminated = esc(str(e.get('terminated_by', '')))

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


def render_transcript_fragment(entry: dict[str, Any]) -> str:
    """Render the drill-down transcript fragment for a single sim result entry.

    Parity: dashboard.py:356-390.

    All user-supplied content (persona, scenario, judge_reason, message
    content) goes through ``esc()`` — stored-XSS vector.

    Args:
        entry: One element from ``_entries_from_run`` (the individual_results
               section data).

    Returns:
        An HTML fragment (no full-page shell).
    """
    idx: int = entry.get('index', 0)
    persona = esc(str(entry.get('persona', 'unknown')))
    scenario = esc(str(entry.get('scenario', 'unknown')))
    goal_achieved: bool = bool(entry.get('goal_achieved'))
    score: float = entry.get('goal_completion_score', 0.0)
    turns: int = entry.get('turn_count', 0)
    terminated_by = esc(str(entry.get('terminated_by', '')))
    judge_reason = esc(str(entry.get('judge_reason', '') or ''))
    error = entry.get('error')

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
    criteria: list[dict[str, Any]] = entry.get('criteria', []) or []
    criteria_html = ''
    if criteria:
        rows_parts: list[str] = []
        for c in criteria:
            passed = c.get('passed', True)
            is_safety = c.get('safety', False)
            if passed:
                icon = '&#x2705;'  # ✅
            elif is_safety:
                icon = '&#x26D4;'  # ⛔
            else:
                icon = '&#x274C;'  # ❌
            ctype = c.get('type') or ''
            ctype_html = f' <em class="sim-ctype">{esc(ctype)}</em>' if ctype else ''
            desc = esc(str(c.get('description', '')))
            rows_parts.append(f'<li class="sim-criterion">{icon} {desc}{ctype_html}</li>')
        criteria_html = (
            f'<div class="sim-criteria">'
            f'<strong>Criteria</strong>'
            f'<ul class="sim-criteria-list">{"".join(rows_parts)}</ul>'
            f'</div>'
        )

    # Transcript (parity: dashboard.py:384-390)
    transcript: list[dict[str, Any]] = entry.get('transcript', []) or []
    msg_parts: list[str] = []
    for msg in transcript:
        role = msg.get('role', 'unknown')
        label = _ROLE_LABELS.get(role, role)
        raw_content = msg.get('content', '')
        content_text = coerce_content_text(raw_content) or '(empty)'
        safe_content = esc(content_text)
        css_role = role if role in ('user', 'assistant', 'system', 'tool') else 'unknown'
        msg_parts.append(
            f'<div class="sim-msg sim-msg-{esc(css_role)}">'
            f'<span class="sim-msg-role">{esc(label)}</span>'
            f'<pre class="sim-msg-content">{safe_content}</pre>'
            f'</div>'
        )

    transcript_html = (
        (
            f'<div class="sim-transcript">'
            f'<strong>Transcript</strong>'
            f'<div class="sim-transcript-messages">{"".join(msg_parts)}</div>'
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
    """

    @app.get('/r/{rid}/sim/transcript')
    def sim_transcript(rid: str, req: Request) -> Response:
        """Return the transcript drill-down fragment for a sim result row.

        Query param ``idx`` selects which entry in the ordered result list to
        render (0-based, matching ``entry["index"]``).  Missing or out-of-range
        ``idx`` returns a graceful empty message rather than a 500.
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

        entries = _entries_from_run(run)
        if not entries or idx < 0 or idx >= len(entries):
            return Response(
                '<p class="sim-empty">No conversation at that index.</p>',
                status_code=200,
                media_type='text/html',
            )

        fragment = render_transcript_fragment(entries[idx])
        return Response(fragment, media_type='text/html')
