"""Apply-recommendations UI for the dashboard (RES-1143).

Serves both surfaces on top of the shared engine in
`evaluatorq.common.apply`: red team surfaces the report's
``focus_area_recommendations`` in the Focus areas tab, agent simulation
surfaces the run's ``recommendations`` in its Recommendations tab. Same bar,
drawer, and confirm flow; only the routes, the write-back field
(``applied_recommendations`` vs ``applied_suggestions``), and the breakdown
block differ. The red-team flow:

1. The apply panel (rendered into the Focus areas tab) shows how many
   recommendations are pending vs already applied, an agent picker when the
   run tested several agents, and a "Preview changes" button.
2. Preview POSTs to ``/r/{rid}/redteam/apply/preview``, which runs
   ``apply_recommendations(apply=False)`` — the agent is only READ, an LLM
   folds the pending recommendations into its instructions, and the result
   (recommendation list + unified diff) renders into a right-hand drawer.
3. The drawer's "Apply to agent" button POSTs the previewed instructions to
   ``/r/{rid}/redteam/apply/confirm``, which writes them back as a new minor
   agent version and records the applied recommendations on the report JSON
   (``applied_recommendations``), so a later preview skips them.

Nothing is written to the platform until the user clicks Apply in the drawer.
The confirm step reuses the previewed instructions verbatim — no second LLM
call, what you saw is what is written.

Public entry points:
    render_apply_panel(rid, report)      — red-team panel + drawer mount
    render_sim_apply_panel(rid, run)     — simulation panel + drawer mount
    register_apply_routes(app, roots)
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import threading
from typing import TYPE_CHECKING, Any

from loguru import logger
from starlette.requests import Request  # noqa: TC002 — FastHTML inspects this annotation at runtime
from starlette.responses import Response

from evaluatorq.common.reports import esc

if TYPE_CHECKING:
    from pathlib import Path

    from evaluatorq.redteam.contracts import FocusAreaRecommendation, RedTeamReport
    from evaluatorq.redteam.reports.apply import ApplyRecommendationsResult
    from evaluatorq.simulation.types import SimulationRecommendation, SimulationRun

DRAWER_ID = 'rt-apply-drawer'
AGENT_FIELD_ID = 'rt-apply-agent-field'

# One process-wide CSRF token, minted into every apply form. A cross-origin page
# cannot read this dashboard's HTML, so it cannot supply the token - which closes
# the form-POST CSRF hole on the dashboard's state-changing routes (review).
CSRF_FIELD = 'csrf'
_CSRF_TOKEN = secrets.token_urlsafe(32)

# Server-side previews keyed by a SINGLE-USE token. Confirm accepts only the
# token, so the write is exactly what this server previewed - "what you saw is
# what is written" becomes an invariant instead of a UI convention (review).
_PREVIEWS: dict[str, dict[str, Any]] = {}
_PREVIEWS_LOCK = threading.Lock()
_PREVIEWS_CAP = 32  # FIFO evict: a dashboard session never has this many live drawers


def _store_preview(entry: dict[str, Any]) -> str:
    token = secrets.token_urlsafe(16)
    with _PREVIEWS_LOCK:
        while len(_PREVIEWS) >= _PREVIEWS_CAP:
            _PREVIEWS.pop(next(iter(_PREVIEWS)))
        _PREVIEWS[token] = entry
    return token


def _pop_preview(token: str) -> dict[str, Any] | None:
    with _PREVIEWS_LOCK:
        return _PREVIEWS.pop(token, None)


def _csrf_field() -> str:
    return f'<input type="hidden" name="{CSRF_FIELD}" value="{_CSRF_TOKEN}">'


def _request_rejected(req: Request, form: Any) -> str | None:
    """CSRF/origin gate for the apply routes. Returns an error message or None.

    Two independent checks: the form must echo the per-process token (unreadable
    cross-origin), and when the browser sends Sec-Fetch-Site it must be a
    same-origin (or direct) request. Absent headers pass - test clients and
    older browsers do not send them; the token check still holds then."""
    sec_fetch = req.headers.get('sec-fetch-site', '')
    if sec_fetch and sec_fetch not in ('same-origin', 'none'):
        return 'Cross-origin request rejected.'
    if str(form.get(CSRF_FIELD) or '') != _CSRF_TOKEN:
        return 'Stale or missing form token; reload the page and try again.'
    return None


# Model for the instruction-merge call. A dashboard config setting (shown on
# the Settings page) rather than the red-team pipeline's evaluator default:
# the merge rewrites production agent instructions, so it warrants a stronger
# model than the scoring pipeline needs, independently configurable.
APPLY_MODEL_ENV = 'EVALUATORQ_APPLY_MODEL'
DEFAULT_APPLY_MODEL = 'gpt-5.6-luna'


def apply_model() -> str:
    """The model used to merge recommendations into agent instructions."""
    return os.environ.get(APPLY_MODEL_ENV, '').strip() or DEFAULT_APPLY_MODEL


# ---------------------------------------------------------------------------
# Pending-recommendation accounting
# ---------------------------------------------------------------------------


def pending_recommendations(report: RedTeamReport) -> list[str]:
    """Recommendation strings not yet recorded as applied, order-preserving."""
    from evaluatorq.common.apply import collect_recommendations

    return collect_recommendations(
        report.focus_area_recommendations or [],
        max_recommendations=10_000,
        already_applied=report.applied_recommendations,
    )


# ---------------------------------------------------------------------------
# Panel (rendered inside the Focus areas tab)
# ---------------------------------------------------------------------------


def is_apply_enabled(report: RedTeamReport) -> bool:
    """Apply is a single-agent feature (review decision): multi-agent runs are
    aimed at comparison, so the whole flow is omitted there rather than asking
    the user to pick a write target mid-report."""
    return bool(report.focus_area_recommendations) and len(report.tested_agents or []) == 1


def _panel_html(rid: str, surface: str, agent_key: str, pending: list[str], applied_n: int) -> str:
    """The apply bar + drawer mount, shared by both surfaces; *surface* is the
    route segment ('redteam' or 'sim')."""
    safe_rid = esc(rid)

    if not pending:
        status = (
            f'<span class="rt-apply-count rt-apply-count--done">✓ all {applied_n} recommendation(s) applied</span>'
            if applied_n
            else '<span class="rt-apply-count">no recommendations to apply</span>'
        )
        return f'<div class="rt-apply-bar">{status}</div><div id="{DRAWER_ID}"></div>'

    # The run's single agent rides along invisibly: showing it as a tag read
    # like a mystery button in review, and with one agent there is nothing to
    # choose. The stable id lets the per-recommendation buttons hx-include it.
    agent_field = f'<input type="hidden" id="{AGENT_FIELD_ID}" name="agent_key" value="{esc(agent_key)}">'

    applied_note = f' · {applied_n} already applied' if applied_n else ''
    return (
        '<div class="rt-apply-bar">'
        '<div class="rt-apply-bar-text">'
        f'<span class="rt-apply-count">{len(pending)} recommendation(s) ready to apply{esc(applied_note)}</span>'
        '<span class="rt-apply-hint">Apply one from its card below, or all at once here. Preview folds '
        'them into the agent instructions; nothing is written until you approve the diff.</span>'
        '</div>'
        f'<form class="rt-apply-form" hx-post="/r/{safe_rid}/{surface}/apply/preview" '
        f'hx-target="#{DRAWER_ID}" hx-swap="innerHTML">'
        f'{agent_field}{_csrf_field()}'
        f'<button type="submit" class="rt-apply-btn">Preview &amp; apply all '
        f'{len(pending)} recommendation{"s" if len(pending) != 1 else ""}</button>'
        '</form>'
        '</div>'
        f'<div id="{DRAWER_ID}"></div>'
    )


def render_apply_panel(rid: str, report: RedTeamReport) -> str:
    """Red-team apply bar + drawer mount. Empty string when there is nothing
    to show (no recommendations, or a multi-agent comparison run)."""
    if not is_apply_enabled(report):
        return ''
    pending = pending_recommendations(report)
    return _panel_html(rid, 'redteam', report.tested_agents[0], pending, len(report.applied_recommendations))


# ---------------------------------------------------------------------------
# Simulation surface
# ---------------------------------------------------------------------------


def is_sim_apply_enabled(run: SimulationRun) -> bool:
    """Sim apply targets the run's single orq agent; other target kinds (models,
    callbacks, deployments) have no agent instructions to write back to."""
    return bool(run.recommendations) and run.target_kind == 'orq_agent' and bool(run.target)


def pending_sim_suggestions(run: SimulationRun) -> list[str]:
    """Suggestion strings not yet recorded as applied, order-preserving."""
    from evaluatorq.common.apply import collect_recommendations

    return collect_recommendations(
        run.recommendations, max_recommendations=10_000, already_applied=run.applied_suggestions
    )


def render_sim_apply_panel(rid: str, run: SimulationRun) -> str:
    """Simulation apply bar + drawer mount. Empty string when the run has no
    recommendations or does not target an orq agent."""
    if not is_sim_apply_enabled(run):
        return ''
    pending = pending_sim_suggestions(run)
    return _panel_html(rid, 'sim', run.target or '', pending, len(run.applied_suggestions))


def _rec_apply_button(rid: str, surface: str, narrow_field: str, narrow_value: str, rec: str) -> str:
    """Per-recommendation Apply button, shared by both surfaces.

    Posts the single bullet plus one narrowing field (red team: ``category``;
    simulation: ``result_index``) to the surface's preview route so the drawer
    can show that finding's breakdown; the agent rides along from the apply
    bar's hidden field via ``hx-include``.
    """
    return (
        f'<form class="rt-focus-rec-apply" hx-post="/r/{esc(rid)}/{surface}/apply/preview" '
        f'hx-target="#{DRAWER_ID}" hx-swap="innerHTML" hx-include="#{AGENT_FIELD_ID}">'
        f'<input type="hidden" name="rec" value="{esc(rec)}">{_csrf_field()}'
        f'<input type="hidden" name="{narrow_field}" value="{esc(narrow_value)}">'
        '<button type="submit" class="rt-apply-btn rt-apply-btn--sm">Apply</button>'
        '</form>'
    )


def render_sim_rec_apply_button(rid: str, result_index: int, rec: str) -> str:
    """Per-suggestion Apply button for a simulation recommendation card."""
    return _rec_apply_button(rid, 'sim', 'result_index', str(result_index), rec)


def render_rec_apply_button(rid: str, category: str, rec: str) -> str:
    """Per-recommendation Apply button for a red-team focus-card bullet."""
    return _rec_apply_button(rid, 'redteam', 'category', category, rec)


# ---------------------------------------------------------------------------
# Drawer fragments
# ---------------------------------------------------------------------------


def _drawer(title: str, body: str, footer: str = '') -> str:
    """Right-hand drawer shell: overlay + panel; the close button empties the mount."""
    close = (
        f'<button class="rt-drawer-close" aria-label="Close" '
        f'hx-get="/apply/dismiss" hx-target="#{DRAWER_ID}" hx-swap="innerHTML">&times;</button>'
    )
    overlay = (
        f'<div class="rt-drawer-overlay" hx-get="/apply/dismiss" hx-target="#{DRAWER_ID}" hx-swap="innerHTML"></div>'
    )
    footer_html = f'<div class="rt-drawer-footer">{footer}</div>' if footer else ''
    return (
        f'{overlay}'
        '<aside class="rt-drawer" role="dialog" aria-modal="true">'
        f'<div class="rt-drawer-head"><h3 class="rt-drawer-title">{title}</h3>{close}</div>'
        f'<div class="rt-drawer-body">{body}</div>'
        f'{footer_html}'
        '</aside>'
    )


def render_error_drawer(message: str) -> str:
    return _drawer('Apply recommendations', f'<p class="rt-drawer-error">{esc(message)}</p>')


def _diff_html(diff: str) -> str:
    """Colorized unified diff, line by line."""
    out: list[str] = []
    for line in diff.splitlines():
        cls = 'ctx'
        if line.startswith('+') and not line.startswith('+++'):
            cls = 'add'
        elif line.startswith('-') and not line.startswith('---'):
            cls = 'del'
        elif line.startswith('@@'):
            cls = 'hunk'
        elif line.startswith(('+++', '---')):
            cls = 'file'
        out.append(f'<span class="rt-diff-line rt-diff-{cls}">{esc(line)}</span>')
    return f'<pre class="rt-diff">{"".join(out)}</pre>'


def _area_breakdown_html(area: FocusAreaRecommendation) -> str:
    """Focus-area context block for a single-recommendation preview: where the
    finding came from and what the analysis saw."""
    from evaluatorq.dashboard.report_kit import focus_tier

    code, label, color = focus_tier(area.risk_score)
    tier = f'{code} · {label}'
    patterns = f'<div class="rt-drawer-patterns">{esc(area.patterns_observed)}</div>' if area.patterns_observed else ''
    return (
        '<div class="rt-drawer-section-label">Focus area</div>'
        '<div class="rt-drawer-area">'
        f'<span class="rt-drawer-area-tier" style="color:{color}">{esc(tier)}</span>'
        f'<span class="rt-drawer-area-name">{esc(area.category_name)} '
        f'<span class="rt-drawer-area-code">{esc(area.category)}</span></span>'
        f'<span class="rt-drawer-area-meta">risk {area.risk_score:.1f} · '
        f'{area.traces_analyzed} failed trace(s) analyzed</span>'
        f'{patterns}'
        '</div>'
    )


def sim_breakdown_html(rec: SimulationRecommendation) -> str:
    """Simulation context block for a single-suggestion preview: the persona,
    scenario, and triggers the suggestion came from."""
    triggers = f'<div class="rt-drawer-patterns">{esc("; ".join(rec.triggers))}</div>' if rec.triggers else ''
    return (
        '<div class="rt-drawer-section-label">Simulation finding</div>'
        '<div class="rt-drawer-area">'
        f'<span class="rt-drawer-area-name">{esc(rec.persona)}</span>'
        f'<span class="rt-drawer-area-meta">{esc(rec.scenario)}</span>'
        f'{triggers}'
        '</div>'
    )


# The merge is an LLM rewrite of a live prompt, so it can reword or drop lines
# the recommendations never mentioned. The diff is the only review gate before
# the agent changes - say that where the reader cannot miss it.
_REVIEW_CALLOUT = (
    '<div class="rt-drawer-review">'
    '<span class="rt-drawer-review-icon" aria-hidden="true">&#9888;</span>'
    '<div class="rt-drawer-review-text">'
    '<b>Read this diff before applying.</b> An LLM rewrote the instructions, so it may also '
    'reword or remove lines the recommendations never mentioned. Applying replaces the whole '
    'instructions field with the right-hand side.'
    '</div></div>'
)


def render_preview_drawer(
    rid: str,
    result: ApplyRecommendationsResult,
    area: FocusAreaRecommendation | None = None,
    *,
    surface: str = 'redteam',
    breakdown: str = '',
    confirm_token: str | None = None,
) -> str:
    """Breakdown drawer for a preview: where the recommendation(s) came from
    (single-rec applies carry their focus area or, for sim, a pre-rendered
    *breakdown* block), what will be merged, and the exact instructions diff.
    The confirm form posts only *confirm_token* - the previewed content lives
    server-side, keyed by that single-use token."""
    recs = ''.join(f'<li>{esc(r)}</li>' for r in result.recommendations)
    unchanged = result.new_instructions.strip() == result.original_instructions.strip()
    rec_label = (
        'Recommendation to fold in'
        if len(result.recommendations) == 1
        else f'{len(result.recommendations)} recommendations to fold in'
    )
    body = (
        f'<div class="rt-drawer-section-label">Agent</div>'
        f'<div class="rt-drawer-agent">{esc(result.agent_key)}</div>'
        + (_area_breakdown_html(area) if area is not None else breakdown)
        + f'<div class="rt-drawer-section-label">{esc(rec_label)}</div>'
        f'<ul class="rt-drawer-recs">{recs}</ul>'
        '<div class="rt-drawer-section-label">Instructions diff</div>'
        + (
            '<p class="rt-drawer-note">The merge produced no change to the instructions.</p>'
            if unchanged
            else _REVIEW_CALLOUT + _diff_html(result.diff)
        )
    )
    if unchanged:
        return _drawer('Preview: no change', body)

    safe_rid = esc(rid)
    footer = (
        f'<form hx-post="/r/{safe_rid}/{surface}/apply/confirm" hx-target="#{DRAWER_ID}" hx-swap="innerHTML">'
        f'<input type="hidden" name="confirm_token" value="{esc(confirm_token or "")}">{_csrf_field()}'
        '<button type="submit" class="rt-apply-btn rt-apply-btn--confirm">Apply to agent</button>'
        '</form>'
        f'<button class="rt-apply-btn rt-apply-btn--ghost" hx-get="/apply/dismiss" '
        f'hx-target="#{DRAWER_ID}" hx-swap="innerHTML">Cancel</button>'
        '<span class="rt-drawer-footnote">Applies the diff above, as a new minor version of the agent.</span>'
    )
    return _drawer('Preview changes', body, footer)


def render_applied_drawer(agent_key: str, applied_count: int, new_version: str | None) -> str:
    """Celebration screen: centered, animated green check, the what and where,
    and the follow-up note. The version chip renders only when the platform
    returned one."""
    version_chip = f'<span class="rt-applied-version">v{esc(new_version)}</span>' if new_version else ''
    # Inline SVG: circle + check drawn via stroke-dashoffset keyframes.
    check_svg = (
        '<svg class="rt-applied-check" viewBox="0 0 72 72" aria-hidden="true">'
        '<circle class="rt-applied-check-ring" cx="36" cy="36" r="32" fill="none" stroke-width="4"/>'
        '<path class="rt-applied-check-mark" fill="none" stroke-width="5" stroke-linecap="round" '
        'stroke-linejoin="round" d="M22 37 L32 47 L51 27"/>'
        '</svg>'
    )
    body = (
        '<div class="rt-drawer-body--applied">'
        f'{check_svg}'
        '<p class="rt-applied-headline">'
        f'Applied {applied_count} recommendation(s)</p>'
        f'<p class="rt-applied-target">to <b>{esc(agent_key)}</b>{version_chip}</p>'
        '<p class="rt-drawer-note">The report now records them as applied; reload the page to '
        'see the updated state. Review the new version in the Orq UI before routing traffic to it.</p>'
        '</div>'
    )
    return _drawer('Applied', body)


# ---------------------------------------------------------------------------
# Report write-back
# ---------------------------------------------------------------------------


# ponytail: process-global lock. The read-modify-write below is not atomic, so
# two concurrent confirms could each read the old list and drop one another's
# bookkeeping. One lock serializes all report writes in this process, which is
# all a single dashboard needs; move to per-path locks only if it contends.
_RECORD_LOCK = threading.Lock()


def record_applied_on_report(path: Path, recommendations: list[str], field: str = 'applied_recommendations') -> None:
    """Append newly applied recommendations to the report JSON, atomically.

    *field* is the surface's tracking field (``applied_recommendations`` for
    red team, ``applied_suggestions`` for simulation). Deduplicates against
    what is already recorded so a double-click cannot inflate the list. Blocking
    file IO guarded by a process lock; call it off the event loop (``to_thread``).
    """
    with _RECORD_LOCK:
        raw = json.loads(path.read_text(encoding='utf-8'))
        existing = raw.get(field) or []
        seen = {str(r).strip() for r in existing}
        for rec in recommendations:
            if rec.strip() not in seen:
                existing.append(rec)
                seen.add(rec.strip())
        raw[field] = existing
        # Unique temp name so concurrent writers never share (and clobber) one
        # scratch file; os.replace is atomic on the same filesystem.
        tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}.{threading.get_ident()}')
        tmp.write_text(json.dumps(raw, indent=2, default=str), encoding='utf-8')
        tmp.replace(path)


# ---------------------------------------------------------------------------
# Clients (lazy; the dashboard must keep working without ORQ_API_KEY)
# ---------------------------------------------------------------------------


def _build_clients() -> tuple[Any, Any, str]:
    """(orq_client, llm_client, model) for the apply flow, or raise ValueError.

    The call config (temperature, retries) follows the red-team pipeline's
    evaluator role; the MODEL is the dashboard's apply-model setting
    (``EVALUATORQ_APPLY_MODEL``, default ``gpt-5.6-luna``), shown on the
    Settings page.
    """
    api_key = os.environ.get('ORQ_API_KEY', '')
    if not api_key:
        raise ValueError('ORQ_API_KEY is not set; the dashboard cannot reach the Orq API to apply recommendations.')
    try:
        from orq_ai_sdk import Orq
    except ModuleNotFoundError as e:  # pragma: no cover - extra not installed
        raise ValueError("The 'orq-ai-sdk' package is required to apply recommendations (install extra 'orq').") from e

    from evaluatorq.redteam.backends.registry import create_async_llm_client
    from evaluatorq.redteam.contracts import PIPELINE_CONFIG

    orq_client = Orq(api_key=api_key, server_url=os.environ.get('ORQ_BASE_URL', 'https://my.orq.ai'))
    llm_client = create_async_llm_client(
        role_config=PIPELINE_CONFIG.evaluator.as_call_config(), max_retries=PIPELINE_CONFIG.retry_count
    )
    return orq_client, llm_client, apply_model()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


async def _confirm_response(
    rid: str,
    req: Request,
    roots: list[Any] | None,
    *,
    surface: str,
    field: str,
    version_note: str,
    expected_agent: str | None,
    already_applied: list[str],
) -> Response:
    """Shared confirm handler: write a SERVER-STORED preview to the agent and
    record the applied recommendations on the report JSON under *field*.

    The request carries only a single-use token (plus the CSRF field); the
    instructions and recommendations come from this server's preview store, so
    a hand-crafted POST cannot choose what gets written. The stored preview is
    validated against the reloaded report (*expected_agent*, *already_applied*)
    and against the agent's CURRENT instructions - if they changed since the
    preview, the write is refused instead of silently clobbering the change.
    """
    from evaluatorq.common.apply import read_instructions, write_instructions
    from evaluatorq.dashboard import library

    form = await req.form()
    rejected = _request_rejected(req, form)
    if rejected:
        return Response(render_error_drawer(rejected), media_type='text/html')

    token = str(form.get('confirm_token') or '')
    entry = _pop_preview(token) if token else None
    if entry is None:
        return Response(
            render_error_drawer('This preview has expired or was already applied; run the preview again.'),
            media_type='text/html',
        )
    if entry['rid'] != rid or entry['surface'] != surface:
        return Response(
            render_error_drawer('This preview belongs to a different report; run the preview again.'),
            media_type='text/html',
        )
    agent_key = str(entry['agent_key'])
    new_instructions = str(entry['new_instructions'])
    recommendations = [str(r) for r in entry['recommendations']]

    # Validate against the reloaded report, not the drawer: a preview built when
    # the report said something else must not write now.
    if expected_agent and agent_key != expected_agent:
        return Response(
            render_error_drawer('This preview was for a different agent; reload the page and preview again.'),
            media_type='text/html',
        )
    seen = {r.strip() for r in already_applied}
    recommendations = [r for r in recommendations if r.strip() not in seen]
    if not recommendations:
        return Response(
            render_error_drawer('These recommendations were already applied; reload the page.'),
            media_type='text/html',
        )

    try:
        orq_client, _llm_client, _model = _build_clients()
    except ValueError as e:
        return Response(render_error_drawer(str(e)), media_type='text/html')

    # Lost-update guard: the preview merged into the instructions as they were
    # at preview time. If the agent changed since, refuse rather than clobber.
    try:
        current = await read_instructions(orq_client, agent_key)
    except Exception as e:
        logger.opt(exception=True).warning('apply_ui: pre-write read failed for {}', agent_key)
        return Response(render_error_drawer(f'Could not re-read the agent before writing: {e}'), media_type='text/html')
    if current.strip() != str(entry['original_instructions']).strip():
        return Response(
            render_error_drawer(
                'The agent instructions changed after this preview was made; run the preview again '
                'so the diff reflects the current instructions.'
            ),
            media_type='text/html',
        )

    try:
        new_version = await write_instructions(
            orq_client,
            agent_key,
            new_instructions,
            version_description=f'Applied {len(recommendations)} {version_note}',
        )
    except Exception as e:
        logger.opt(exception=True).warning('apply_ui: agent update failed for {}', agent_key)
        return Response(render_error_drawer(f'Agent update failed: {e}'), media_type='text/html')

    # Record on the report so a later preview skips these. A write-back
    # failure must not hide that the agent WAS updated - report it as a
    # warning inside the success drawer instead of failing the request. The
    # note names the actual consequence (review): unrecorded bullets come back
    # as pending, and applying again folds them in a second time.
    unrecorded = ' They will show as pending again, and applying them again would merge them into the agent twice.'
    path = library.resolve(rid, roots)
    record_note = ''
    if path is not None:
        try:
            await asyncio.to_thread(record_applied_on_report, path, recommendations, field)
        except Exception:
            logger.opt(exception=True).warning('apply_ui: could not record applied recs on {}', path)
            record_note = f' The agent was updated, but the report file could not be written.{unrecorded}'
    else:
        logger.warning('apply_ui: could not resolve report path for {}; applied recs not recorded', rid)
        record_note = f' The agent was updated, but the report file could not be located.{unrecorded}'

    html = render_applied_drawer(agent_key, len(recommendations), new_version)
    if record_note:
        html = html.replace('</p>', esc(record_note) + '</p>', 1)
    return Response(html, media_type='text/html')


def _rt_narrow(form: Any, report: RedTeamReport) -> tuple[list[Any], FocusAreaRecommendation | None, str, str | None]:
    """Narrow the red-team preview to a single bullet when `rec` is posted.

    Returns ``(items, area, breakdown, error)``: the focus areas to merge, the
    single area for the drawer breakdown (None for apply-all), an empty
    breakdown (red team uses *area* instead), and an error message or None.
    """
    areas = list(report.focus_area_recommendations or [])
    single_rec = str(form.get('rec') or '').strip()
    if not single_rec:
        return areas, None, '', None
    category = str(form.get('category') or '').strip()
    area = next(
        (a for a in areas if a.category == category and single_rec in [r.strip() for r in a.recommendations]),
        None,
    )
    if area is None:
        return [], None, '', 'That recommendation is no longer on the report; reload the page.'
    if single_rec in {r.strip() for r in report.applied_recommendations}:
        return [], None, '', 'That recommendation is already applied to the agent.'
    return [area.model_copy(update={'recommendations': [single_rec]})], area, '', None


def _sim_narrow(form: Any, run: SimulationRun) -> tuple[list[Any], None, str, str | None]:
    """Narrow the simulation preview to a single suggestion when `rec` is posted.

    Returns ``(items, None, breakdown, error)``: the recommendations to merge,
    no focus area (sim renders its own *breakdown* block), and an error or None.
    """
    recs = list(run.recommendations)
    single_rec = str(form.get('rec') or '').strip()
    if not single_rec:
        return recs, None, '', None
    try:
        idx = int(str(form.get('result_index') or ''))
    except ValueError:
        idx = -1
    card = next(
        (r for r in recs if r.result_index == idx and single_rec in [s.strip() for s in r.suggestions]),
        None,
    )
    if card is None:
        return [], None, '', 'That suggestion is no longer on the report; reload the page.'
    if single_rec in {s.strip() for s in run.applied_suggestions}:
        return [], None, '', 'That suggestion is already applied to the agent.'
    return [card.model_copy(update={'suggestions': [single_rec]})], None, sim_breakdown_html(card), None


async def _preview_response(
    rid: str,
    req: Request,
    *,
    surface: str,
    obj: Any,
    not_found_html: str,
    enable_error: str | None,
    agent_default: str,
    narrow: Any,
    apply_fn: Any,
    already_applied: list[str],
    empty_msg: str,
) -> Response:
    """Shared preview handler: resolve the agent, narrow to the requested
    bullet(s), run ``apply(apply=False)``, and render the drawer. Only the
    loader, enable gate, narrowing, and apply wrapper differ between surfaces.
    """
    if obj is None:
        return Response(not_found_html, status_code=404, media_type='text/html')
    if enable_error:
        return Response(render_error_drawer(enable_error), media_type='text/html')

    form = await req.form()
    rejected = _request_rejected(req, form)
    if rejected:
        return Response(render_error_drawer(rejected), media_type='text/html')
    agent_key = str(form.get('agent_key') or '').strip() or agent_default
    if not agent_key:
        return Response(
            render_error_drawer('This run does not record which agent it tested, so there is nothing to apply to.'),
            media_type='text/html',
        )

    items, area, breakdown, narrow_error = narrow(form, obj)
    if narrow_error:
        return Response(render_error_drawer(narrow_error), media_type='text/html')

    try:
        orq_client, llm_client, model = _build_clients()
    except ValueError as e:
        return Response(render_error_drawer(str(e)), media_type='text/html')

    try:
        result = await apply_fn(items, agent_key, orq_client, llm_client, model, already_applied)
    except Exception as e:
        logger.opt(exception=True).warning('apply_ui: preview failed for {}', agent_key)
        return Response(render_error_drawer(f'Preview failed: {e}'), media_type='text/html')

    if result.merge_failed:
        return Response(
            render_error_drawer('The merge did not produce usable instructions; please try the preview again.'),
            media_type='text/html',
        )
    if not result.recommendations:
        return Response(render_error_drawer(empty_msg), media_type='text/html')
    token = _store_preview({
        'rid': rid,
        'surface': surface,
        'agent_key': result.agent_key,
        'original_instructions': result.original_instructions,
        'new_instructions': result.new_instructions,
        'recommendations': list(result.recommendations),
    })
    return Response(
        render_preview_drawer(rid, result, area, surface=surface, breakdown=breakdown, confirm_token=token),
        media_type='text/html',
    )


def register_apply_routes(app: Any, roots: list[Any] | None = None) -> None:
    """Register the apply preview/confirm/dismiss routes for both surfaces."""
    from evaluatorq.dashboard.redteam_views import _404, _load_report
    from evaluatorq.dashboard.sim_views import _load_run

    @app.get('/apply/dismiss')
    def apply_dismiss() -> Response:
        return Response('', media_type='text/html')

    @app.post('/r/{rid}/redteam/apply/preview')
    async def apply_preview(rid: str, req: Request) -> Response:
        report = _load_report(rid, roots)
        multi = report is not None and len(report.tested_agents or []) > 1

        async def _apply(items: Any, ak: str, orq: Any, llm: Any, model: str, applied: list[str]) -> Any:
            from evaluatorq.redteam.reports.apply import apply_recommendations

            return await apply_recommendations(items, ak, orq, llm, model, apply=False, already_applied=applied)

        return await _preview_response(
            rid,
            req,
            surface='redteam',
            obj=report,
            not_found_html=_404(f'Report {rid} not found'),
            enable_error='Applying recommendations is available for single-agent runs; this run compared several agents.'
            if multi
            else None,
            agent_default='',
            narrow=_rt_narrow,
            apply_fn=_apply,
            already_applied=list(report.applied_recommendations) if report else [],
            empty_msg='No new recommendations to apply — everything is already applied.',
        )

    @app.post('/r/{rid}/redteam/apply/confirm')
    async def apply_confirm(rid: str, req: Request) -> Response:
        report = _load_report(rid, roots)
        if report is None:
            return Response(
                render_error_drawer('That report is no longer available; reload the page.'), media_type='text/html'
            )
        expected = report.tested_agents[0] if len(report.tested_agents or []) == 1 else None
        return await _confirm_response(
            rid,
            req,
            roots,
            surface='redteam',
            field='applied_recommendations',
            version_note='red-team remediation recommendation(s)',
            expected_agent=expected,
            already_applied=list(report.applied_recommendations),
        )

    @app.post('/r/{rid}/sim/apply/preview')
    async def sim_apply_preview(rid: str, req: Request) -> Response:
        run = _load_run(rid, roots)
        enabled = run is not None and is_sim_apply_enabled(run)

        async def _apply(items: Any, ak: str, orq: Any, llm: Any, model: str, applied: list[str]) -> Any:
            from evaluatorq.simulation.reports.apply import apply_suggestions

            return await apply_suggestions(items, ak, orq, llm, model, apply=False, already_applied=applied)

        return await _preview_response(
            rid,
            req,
            surface='sim',
            obj=run,
            not_found_html=_404(f'Report {rid} not found'),
            enable_error=None if enabled else 'Applying suggestions needs a run that targeted an orq agent.',
            agent_default=str(run.target or '') if run else '',
            narrow=_sim_narrow,
            apply_fn=_apply,
            already_applied=list(run.applied_suggestions) if run else [],
            empty_msg='No new suggestions to apply — everything is already applied.',
        )

    @app.post('/r/{rid}/sim/apply/confirm')
    async def sim_apply_confirm(rid: str, req: Request) -> Response:
        run = _load_run(rid, roots)
        if run is None:
            return Response(
                render_error_drawer('That report is no longer available; reload the page.'), media_type='text/html'
            )
        expected = run.target if is_sim_apply_enabled(run) else None
        return await _confirm_response(
            rid,
            req,
            roots,
            surface='sim',
            field='applied_suggestions',
            version_note='simulation remediation suggestion(s)',
            expected_agent=expected,
            already_applied=list(run.applied_suggestions),
        )
