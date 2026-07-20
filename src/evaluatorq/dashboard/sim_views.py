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


# Conversation table columns. Non-sortable action columns (Status, Traces) are
# rendered separately below — this list is only the sortable ones, keyed by the
# ``sort`` query param.
#   (param, header label, sort-key callable)
_SIM_COLUMNS: list[tuple[str, str, Any]] = [
    ('index', '#', lambda e: e.index),
    ('persona', 'Persona', lambda e: e.persona.lower()),
    ('scenario', 'Scenario', lambda e: e.scenario.lower()),
    ('turn_count', 'Turns', lambda e: e.turn_count),
    ('goal_completion_score', 'Score', lambda e: e.goal_completion_score),
    ('terminated_by', 'Terminated by', lambda e: e.terminated_by),
]
_SORT_KEYS: dict[str, Any] = {param: key for param, _, key in _SIM_COLUMNS}
_DEFAULT_SORT = 'index'
_PAGE_SIZE = 25
_PAGE_SIZES = (5, 10, 25)  # selectable rows-per-page; last is the default


def _coerce_page_size(raw: str | int | None) -> int:
    """Clamp an incoming page-size to an allowed option; bad input → default."""
    try:
        n = int(raw)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return _PAGE_SIZE
    return n if n in _PAGE_SIZES else _PAGE_SIZE


def _hx_control(rid: str, sort: str, direction: str, page: int, size: int) -> str:
    """Shared HTMX attributes for a header/pager/size control.

    Each control re-fetches ``/sim/row-list`` with the new sort/page/size,
    pulling the current filter state in via ``hx-include`` so sorting and
    filtering compose, and swaps the whole ``#sim-row-list-{rid}`` wrapper.
    """
    safe_rid = esc(rid)
    return (
        f'hx-get="/r/{safe_rid}/sim/row-list?sort={esc(sort)}&dir={esc(direction)}&page={page}&size={size}" '
        'hx-include="#filter-form" '
        f'hx-target="#sim-row-list-{safe_rid}" hx-swap="outerHTML"'
    )


def _sim_header_row(rid: str, sort: str, direction: str, size: int) -> str:
    """Build the sortable ``<thead>`` row. Clicking a header toggles direction."""
    cells: list[str] = []
    for param, label, _key in _SIM_COLUMNS:
        active = param == sort
        # Toggle direction on the active column; a fresh column starts ascending.
        next_dir = 'desc' if (active and direction == 'asc') else 'asc'
        aria = {'asc': 'ascending', 'desc': 'descending'}[direction] if active else 'none'
        # Active column shows its direction (▲/▼); inactive sortable columns show
        # a faint up-down glyph (⇅) so the "click to sort" affordance is always
        # visible, not only after the first click.
        if active:
            caret = '&#x25B2;' if direction == 'asc' else '&#x25BC;'
            caret_cls = 'sim-th-caret sim-th-caret-active'
        else:
            caret = '&#x21C5;'
            caret_cls = 'sim-th-caret'
        cells.append(
            f'<th scope="col" aria-sort="{aria}">'
            f'<button type="button" class="sim-th-sort" {_hx_control(rid, param, next_dir, 1, size)}>'
            f'{esc(label)}<span class="{caret_cls}">{caret}</span>'
            f'</button></th>'
        )
    # Non-sortable action columns.
    cells.extend(('<th scope="col">Status</th>', '<th scope="col">Traces</th>'))
    return f'<thead><tr>{"".join(cells)}</tr></thead>'


def _sim_size_selector(rid: str, sort: str, direction: str, size: int) -> str:
    """Rows-per-page selector (5/10/25). Changing size resets to page 1."""
    opts = ''.join(
        (
            f'<button type="button" class="sim-size-btn" disabled aria-current="true">{n}</button>'
            if n == size
            else f'<button type="button" class="sim-size-btn" {_hx_control(rid, sort, direction, 1, n)}>{n}</button>'
        )
        for n in _PAGE_SIZES
    )
    return f'<div class="sim-size" role="group" aria-label="Rows per page"><span class="sim-size-label">Show</span>{opts}</div>'


def _sim_pager(rid: str, sort: str, direction: str, page: int, pages: int, total: int, size: int) -> str:
    """Render the pager nav + rows-per-page selector; buttons carry sort + filter state."""
    selector = _sim_size_selector(rid, sort, direction, size)
    # 3-column grid centres the pager group (col 2) while the selector stays
    # pinned right (col 3); col 1 is an empty spacer that balances the selector.
    if pages <= 1:
        nav = f'<div class="sim-pager-nav"><span class="sim-pager-info">{total} conversations</span></div>'
        return f'<nav class="sim-pager">{nav}{selector}</nav>'

    def btn(label: str, target_page: int, *, disabled: bool) -> str:
        if disabled:
            return f'<button type="button" class="sim-pager-btn" disabled>{label}</button>'
        return f'<button type="button" class="sim-pager-btn" {_hx_control(rid, sort, direction, target_page, size)}>{label}</button>'

    nav = (
        '<div class="sim-pager-nav">'
        f'{btn("&#x2039; Prev", page - 1, disabled=page <= 1)}'
        f'<span class="sim-pager-info">Page {page} of {pages} &middot; {total} conversations</span>'
        f'{btn("Next &#x203A;", page + 1, disabled=page >= pages)}'
        '</div>'
    )
    return f'<nav class="sim-pager">{nav}{selector}</nav>'


def render_sim_row_list(
    rid: str,
    entries: list[SimulationEntry],
    *,
    sort: str = _DEFAULT_SORT,
    direction: str = 'asc',
    page: int = 1,
    page_size: int = _PAGE_SIZE,
) -> str:
    """Render a sortable, paginated conversation table for a sim report.

    Rows are lazy drawer-triggers: each ``<tr>`` carries a ``data-drawer-url``
    for its transcript, which stays absent until the drawer opens. Sorting and
    pagination are server-side — clicking a header or pager button re-fetches
    ``/sim/row-list`` (see ``_hx_control``) with the current filter state.

    Args:
        rid:       Report ID (URL-safe).
        entries:   Typed entry list (already filtered) from ``individual_entries``.
        sort:      Active sort column (``_SIM_COLUMNS`` param); invalid → ``index``.
        direction: ``'asc'`` or ``'desc'``; anything else → ``'asc'``.
        page:      1-based page number; clamped to ``[1, pages]``.
        page_size: Rows per page.

    Returns:
        HTML fragment containing a ``<section class="sim-row-list">``.
    """
    if not entries:
        return '<section class="sim-row-list"><p class="sim-empty">No conversations found.</p></section>'

    from evaluatorq.common.reports import status_badge

    if sort not in _SORT_KEYS:
        sort = _DEFAULT_SORT
    if direction not in ('asc', 'desc'):
        direction = 'asc'
    if page_size not in _PAGE_SIZES:
        page_size = _PAGE_SIZE

    ordered = sorted(entries, key=_SORT_KEYS[sort], reverse=(direction == 'desc'))
    total = len(ordered)
    pages = max(1, -(-total // page_size))  # ceil division
    page = max(1, min(page, pages))
    start = (page - 1) * page_size
    visible = ordered[start : start + page_size]

    safe_rid = esc(rid)
    rows_html: list[str] = []
    for e in visible:
        idx = e.index
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

        # ``data-no-drawer`` keeps a trace-link click from also opening the drawer
        # (the whole row is the drawer trigger).
        trace_btn = trace_link_button(
            thread_trace_url(e.thread_id),
            'View Traces',
            extra_attributes={'data-no-drawer': None},
        )
        rows_html.append(
            f'<tr class="sim-conv-row {tint}" role="button" tabindex="0" '
            'data-sim-entity-trigger data-entity-kind="conversation" '
            f'data-drawer-url="/r/{safe_rid}/sim/transcript?idx={idx}">'
            f'<td class="sim-conv-idx">#{idx + 1}</td>'
            f'<td class="sim-conv-persona">{esc(e.persona)}</td>'
            f'<td class="sim-conv-scenario">{esc(e.scenario)}</td>'
            f'<td class="sim-conv-turns">{e.turn_count}</td>'
            f'<td class="sim-conv-score">{e.goal_completion_score:.2f}</td>'
            f'<td class="sim-conv-term">{esc(e.terminated_by)}</td>'
            f'<td class="sim-conv-status">{badge}</td>'
            f'<td class="sim-conv-trace">{trace_btn}</td>'
            '</tr>'
        )

    header = _sim_header_row(rid, sort, direction, page_size)
    pager = _sim_pager(rid, sort, direction, page, pages, total, page_size)
    return (
        f'<section class="sim-row-list">'
        f'<table class="sim-conv-table">{header}<tbody>{"".join(rows_html)}</tbody></table>'
        f'{pager}'
        f'</section>'
    )


# ---------------------------------------------------------------------------
# Transcript fragment
# ---------------------------------------------------------------------------


def _render_criteria_column(entry: SimulationEntry) -> str:
    """Render the CRITERIA section: verdict header + polarity-explicit rows.

    Outcome polarity is sacred here (DESIGN.md): a green check on a
    ``must_not_happen`` criterion reads backwards unless the requirement is
    stated *next to* the check. So each row leads with the icon (result:
    pass/fail) immediately followed by a ``Required`` / ``Prohibited`` chip
    (the requirement that gives the check its meaning) — no far-right orphan
    label to hunt for. A verdict line above the list answers the reader's
    first question ("did it pass, how many met?") before they scan the rows.
    """
    criteria = entry.criteria or []
    items: list[str] = []
    for c in criteria:
        state_class = 'sim-criterion-pass' if c.passed else 'sim-criterion-fail'
        icon = '&#x2713;' if c.passed else '&#x2717;'  # ✓ / ✗
        prohibited = c.type == 'must_not_happen'
        # The chip carries the requirement's polarity right beside the result
        # icon; keep the unsafe hook so `must_not_happen` reads red.
        chip_html = ''
        if c.type:
            chip_label = 'Prohibited' if prohibited else 'Required'
            chip_class = 'sim-ctype sim-ctype-unsafe' if prohibited else 'sim-ctype'
            chip_html = f'<span class="{chip_class}">{chip_label}</span>'
        desc = esc(c.description)
        items.append(
            f'<li class="sim-criterion {state_class}">'
            f'<span class="sim-criterion-icon">{icon}</span>'
            f'{chip_html}'
            f'<span class="sim-criterion-desc">{desc}</span>'
            f'</li>'
        )

    # Goal outcome and criteria-met count are two different facts: an agent can
    # miss the goal while still satisfying every criterion (and vice versa), so
    # "FAIL · 4/4 met" reads as a contradiction. Show them as separate spans —
    # the coloured goal verdict, then a neutral criteria tally.
    verdict_html = ''
    if criteria:
        met = sum(1 for c in criteria if c.passed)
        passed = entry.goal_achieved
        verdict_class = 'sim-criteria-verdict--pass' if passed else 'sim-criteria-verdict--fail'
        verdict_word = 'Goal met' if passed else 'Goal missed'
        verdict_html = (
            f'<span class="sim-criteria-verdict {verdict_class}">{verdict_word}</span>'
            f'<span class="sim-criteria-count">{met}/{len(criteria)} criteria met</span>'
        )

    # The judge rationale is folded into the criteria block (it explains the
    # verdict above it), not a separate callout — one "outcome" section.
    judge_reason = esc(entry.judge_reason or '')
    judge_html = ''
    if judge_reason:
        judge_html = (
            f'<div class="sim-judge">'
            f'<span class="sim-judge-label">Judge</span>'
            f'<p class="sim-judge-reason">{judge_reason}</p>'
            f'</div>'
        )

    return (
        f'<div class="sim-criteria">'
        f'<div class="sim-criteria-head">'
        f'<span class="sim-criteria-header">CRITERIA</span>{verdict_html}'
        f'</div>'
        f'<ul class="sim-criteria-list">{"".join(items)}</ul>'
        f'{judge_html}'
        f'</div>'
    )


def _render_conversation_summary(entry: SimulationEntry) -> str:
    """Persona + scenario recap and a turn-count chip for the drawer header.

    The collapsed row shows only "#N · persona · scenario" truncated; here we
    give the full text so the reader has the setup in view while reading the
    transcript. All fields are user-supplied — esc() guards the stored-XSS
    vector.
    """
    turns = entry.turn_count
    turn_word = 'turn' if turns == 1 else 'turns'
    return (
        f'<div class="sim-conv-summary">'
        f'<span class="sim-conv-index">#{entry.index}</span>'
        f'<div class="sim-conv-meta">'
        f'<div class="sim-conv-field"><span class="sim-conv-label">Persona</span>'
        f'<span class="sim-conv-value">{esc(entry.persona)}</span></div>'
        f'<div class="sim-conv-field"><span class="sim-conv-label">Scenario</span>'
        f'<span class="sim-conv-value">{esc(entry.scenario)}</span></div>'
        f'</div>'
        f'<span class="sim-conv-turns-pill">{turns} {turn_word}</span>'
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
    error = entry.error

    criteria_col = _render_criteria_column(entry)
    summary_html = _render_conversation_summary(entry)

    if error:
        error_html = f'<div class="sim-transcript-error"><strong>Error:</strong> {esc(str(error))}</div>'
        return f'<div class="sim-transcript-detail">{summary_html}{criteria_col}{error_html}</div>'

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
    # Criteria above the conversation: the outcome verdict frames the
    # transcript the reader is about to scroll through, not a footnote after it.
    grid_html = f'<div class="sim-transcript-grid">{criteria_col}{bubbles_html}</div>'

    return f'<div class="sim-transcript-detail">{summary_html}{grid_html}</div>'


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

        sort = req.query_params.get('sort', _DEFAULT_SORT)
        direction = req.query_params.get('dir', 'asc')
        try:
            page = int(req.query_params.get('page', '1'))
        except (ValueError, TypeError):
            page = 1
        page_size = _coerce_page_size(req.query_params.get('size'))

        html = render_sim_row_list(rid, entries, sort=sort, direction=direction, page=page, page_size=page_size)
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
