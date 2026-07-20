"""HTMX fragment routes for simulation-dashboard interactive views.

Routes (all return HTML fragments, no full page shell):

    GET /r/{rid}/sim/transcript?idx=   → conversation detail: header, judge
                                          callout, chat bubbles, criteria column

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
from evaluatorq.dashboard.trace_links import thread_trace_url, trace_link_button
from evaluatorq.dashboard.view import _sim_rowlist_wrapper, render_message_list

if TYPE_CHECKING:
    from pathlib import Path

    from evaluatorq.simulation.types import SimulationEntry

# ---------------------------------------------------------------------------
# Role label mapping (parity: dashboard.py:385)
# ---------------------------------------------------------------------------

_ROLE_LABELS: dict[str, str] = {
    'user': 'USR',
    'assistant': 'AGT',
    'system': 'SYS',
    'tool': 'TOOL',
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
    """Render flat, lazy drawer-trigger conversation rows for a sim report.

    Each row contains the summary metadata and a ``data-drawer-url`` for its
    transcript. The transcript body remains absent until the drawer opens.

    Args:
        rid:     Report ID (URL-safe).
        entries: Typed entry list from ``_entries_from_run`` / ``individual_entries``.

    Returns:
        HTML fragment containing a ``<section class="sim-row-list">``.
    """
    if not entries:
        return '<section class="sim-row-list"><p class="sim-empty">No conversations found.</p></section>'

    from evaluatorq.common.reports import status_badge
    from evaluatorq.dashboard.report_kit import tag

    safe_rid = esc(rid)
    rows_html: list[str] = []
    for e in entries:
        idx = e.index
        persona = esc(e.persona)
        scenario = esc(e.scenario)

        is_error = e.terminated_by == 'error'
        if is_error:
            tint = 'sim-tint-error'
            badge = status_badge('Error', 'warn')
        elif e.goal_achieved:
            tint = 'sim-tint-achieved'
            badge = status_badge('Goal met', 'pass')
        else:
            tint = 'sim-tint-missed'
            badge = status_badge('Goal missed', 'fail')

        # Trace deep-link on the row, next to the outcome badge. stopPropagation
        # keeps it from opening the conversation drawer.
        trace_btn = trace_link_button(
            thread_trace_url(e.thread_id),
            'View Traces',
            onclick='event.stopPropagation()',
            extra_attributes={'data-no-drawer': None},
        )
        right_cluster = (
            f'{tag(f"{e.turn_count} turns")}{tag(f"score {e.goal_completion_score:.2f}")}'
            f'{tag(e.terminated_by)}{badge}{trace_btn}'
        )
        row_attrs = (
            f'class="sim-conv-row {tint}" role="button" tabindex="0" '
            'data-sim-entity-trigger data-entity-kind="conversation" '
            f'data-drawer-url="/r/{safe_rid}/sim/transcript?idx={idx}"'
        )
        rows_html.append(
            f'<div {row_attrs}>'
            f'<span class="sim-conv-idx">#{idx + 1}</span>'
            f'<span class="sim-conv-persona">{persona}</span>'
            f'<span class="sim-conv-sep">&middot;</span>'
            f'<span class="sim-conv-scenario">{scenario}</span>'
            f'<span class="sim-conv-right">{right_cluster}</span>'
            '</div>'
        )

    return f'<section class="sim-row-list">{"".join(rows_html)}</section>'


# ---------------------------------------------------------------------------
# Transcript fragment
# ---------------------------------------------------------------------------


def _render_criteria_column(entry: SimulationEntry) -> str:
    """Render the CRITERIA column: two-state icon + description + type label.

    Deviation #4 (deliberate): the old three-state icon (✅ passed / ⛔ unsafe
    failure / ❌ other failure) is replaced by a plain two-state pass/fail
    icon. No information is lost — the safety distinction is preserved via
    the ``must_not_happen`` type label, rendered in red.
    """
    criteria = entry.criteria or []
    items: list[str] = []
    for c in criteria:
        state_class = 'sim-criterion-pass' if c.passed else 'sim-criterion-fail'
        icon = '&#x2713;' if c.passed else '&#x2717;'  # ✓ / ✗
        ctype = c.type or ''
        type_class = 'sim-ctype sim-ctype-unsafe' if ctype == 'must_not_happen' else 'sim-ctype'
        # Mockup shows the requirement as words ("MUST HAPPEN"), not the raw enum.
        type_html = f'<span class="{type_class}">{esc(ctype.replace("_", " "))}</span>' if ctype else ''
        desc = esc(c.description)
        items.append(
            f'<li class="sim-criterion {state_class}">'
            f'<span class="sim-criterion-icon">{icon}</span>'
            f'<span class="sim-criterion-desc">{desc}</span>'
            f'{type_html}'
            f'</li>'
        )

    return (
        f'<div class="sim-criteria">'
        f'<div class="sim-criteria-header">CRITERIA</div>'
        f'<ul class="sim-criteria-list">{"".join(items)}</ul>'
        f'</div>'
    )


def render_transcript_fragment(entry: SimulationEntry) -> str:
    """Render the drill-down transcript fragment for a single sim result entry.

    Design-aligned layout (spec §Transcripts): a Judge callout (sunken
    background, teal-600 left border, "Judge" mono-label + ``judge_reason``)
    followed by a two-column grid — chat bubbles (via ``render_message_list``)
    on the left, the CRITERIA column on the right. Turn-count / score /
    terminated-by metrics now live in the conversation card's ``<summary>``
    header (``render_sim_row_list``), so this fragment no longer duplicates
    them.

    Error entries substitute a red error message for the judge callout and
    the transcript bubbles (the criteria column is still shown — safety
    findings on a crashed run are still relevant).

    All user-supplied content (persona, scenario, judge_reason, criteria
    description, message content) goes through ``esc()`` — stored-XSS vector.

    Args:
        entry: A typed ``SimulationEntry`` from ``individual_entries``.

    Returns:
        An HTML fragment (no full-page shell).
    """
    judge_reason = esc(entry.judge_reason or '')
    error = entry.error

    # No title here: the collapsed conversation card already shows
    # "#N · persona · scenario" directly above this fragment, so repeating it
    # (the old sim-transcript-header) was a duplicate the mockup doesn't have.
    criteria_col = _render_criteria_column(entry)

    if error:
        error_html = f'<div class="sim-transcript-error"><strong>Error:</strong> {esc(str(error))}</div>'
        return f'<div class="sim-transcript-detail">{error_html}{criteria_col}</div>'

    judge_html = ''
    if judge_reason:
        judge_html = (
            f'<div class="sim-judge">'
            f'<span class="sim-judge-label">Judge</span>'
            f'<p class="sim-judge-reason">{judge_reason}</p>'
            f'</div>'
        )

    # Transcript (parity: dashboard.py:384-390). Normalise content via
    # coerce_content_text (handles OpenAI content blocks) before handing off
    # to the shared renderer. The '(empty)' fallback is sim-specific so we
    # apply it here rather than inside render_message_list.
    transcript = entry.transcript or []
    normalised_msgs: list[dict[str, Any]] = []
    for msg in transcript:
        content_text = coerce_content_text(msg.content) or '(empty)'
        normalised_msgs.append({'role': msg.role, 'content': content_text})

    bubbles_html = (
        f'<div class="sim-transcript-bubbles">'
        f'{render_message_list(normalised_msgs, role_labels=_ROLE_LABELS, class_prefix="sim")}'
        f'</div>'
    )
    grid_html = f'<div class="sim-transcript-grid">{bubbles_html}{criteria_col}</div>'

    return f'<div class="sim-transcript-detail">{judge_html}{grid_html}</div>'


# ---------------------------------------------------------------------------
# Route factory
# ---------------------------------------------------------------------------


def register_sim_view_routes(app: Any, roots: list[Any] | None = None) -> None:
    """Register simulation view routes on *app*.

    Called from ``evaluatorq.dashboard.app.build_app``.

    Routes registered here:

    - ``GET /r/{rid}/sim/transcript?idx=`` — transcript drill-down fragment,
      resolved against the full run by the row's stable index.

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

        # Keep each surviving conversation's ORIGINAL index (its position in the
        # full run) so drill-down links stay stable under filtering — filtering
        # hides rows, it does not renumber them.
        kept = {id(r) for r in filtered_results}
        entries = [e for e, r in zip(individual_entries(run.results), run.results, strict=True) if id(r) in kept]

        html = render_sim_row_list(rid, entries)
        # Return wrapped in the same container div that sim_interactive_panels
        # renders so the outerHTML swap replaces the correct element.
        return Response(_sim_rowlist_wrapper(rid, html), media_type='text/html')

    @app.get('/r/{rid}/sim/transcript')
    def sim_transcript(rid: str, req: Request) -> Response:
        """Return the transcript drill-down fragment for a sim result row.

        Query param ``idx`` selects the full run's 0-based result position.
        Missing or out-of-range ``idx`` returns a graceful empty message rather
        than a 500.
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

        # Resolve by stable index into the full, unfiltered run: the row-list
        # carries each conversation's original position (see sim_row_list), so a
        # visible row always maps to its transcript regardless of the active
        # filter sent via hx-include. Re-applying the filter here would re-index
        # the list and make an unfiltered row idx overflow the filtered list.
        from evaluatorq.simulation.reports.sections import individual_entries

        entries = individual_entries(run.results)

        if not entries or idx < 0 or idx >= len(entries):
            return Response(
                '<p class="sim-empty">No conversation at that index.</p>',
                status_code=200,
                media_type='text/html',
            )

        fragment = render_transcript_fragment(entries[idx])
        return Response(fragment, media_type='text/html')
