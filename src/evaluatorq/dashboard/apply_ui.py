"""Apply-recommendations UI for the dashboard (RES-1143).

Serves both surfaces on top of the shared engine in
:mod:`evaluatorq.common.apply`: red team surfaces the report's
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

import json
import os
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
    from evaluatorq.redteam.reports.apply import _collect_recommendations

    return _collect_recommendations(
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
        f'{agent_field}'
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


def render_sim_rec_apply_button(rid: str, result_index: int, rec: str) -> str:
    """Per-suggestion Apply button for a simulation recommendation card.

    Posts the single suggestion (plus its card's ``result_index``, so the
    drawer can show that card's persona/scenario breakdown) to the sim preview
    route; the agent comes from the apply bar's field via ``hx-include``.
    """
    return (
        f'<form class="rt-focus-rec-apply" hx-post="/r/{esc(rid)}/sim/apply/preview" '
        f'hx-target="#{DRAWER_ID}" hx-swap="innerHTML" hx-include="#{AGENT_FIELD_ID}">'
        f'<input type="hidden" name="rec" value="{esc(rec)}">'
        f'<input type="hidden" name="result_index" value="{result_index}">'
        '<button type="submit" class="rt-apply-btn rt-apply-btn--sm">Apply</button>'
        '</form>'
    )


def render_rec_apply_button(rid: str, category: str, rec: str) -> str:
    """Per-recommendation Apply button for a focus-card bullet.

    Posts the single recommendation (plus its area category, so the drawer can
    show that area's breakdown) to the preview route; the agent comes from the
    apply bar's field via ``hx-include``.
    """
    return (
        f'<form class="rt-focus-rec-apply" hx-post="/r/{esc(rid)}/redteam/apply/preview" '
        f'hx-target="#{DRAWER_ID}" hx-swap="innerHTML" hx-include="#{AGENT_FIELD_ID}">'
        f'<input type="hidden" name="rec" value="{esc(rec)}">'
        f'<input type="hidden" name="category" value="{esc(category)}">'
        '<button type="submit" class="rt-apply-btn rt-apply-btn--sm">Apply</button>'
        '</form>'
    )


# ---------------------------------------------------------------------------
# Drawer fragments
# ---------------------------------------------------------------------------


def _drawer(title: str, body: str, footer: str = '') -> str:
    """Right-hand drawer shell: overlay + panel; the close button empties the mount."""
    close = (
        f'<button class="rt-drawer-close" aria-label="Close" '
        f'hx-get="/redteam/apply/dismiss" hx-target="#{DRAWER_ID}" hx-swap="innerHTML">&times;</button>'
    )
    overlay = (
        f'<div class="rt-drawer-overlay" hx-get="/redteam/apply/dismiss" '
        f'hx-target="#{DRAWER_ID}" hx-swap="innerHTML"></div>'
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
    # Same tier thresholds as the focus cards (report_tabs._rt_focus_tier);
    # duplicated here because report_tabs imports this module.
    if area.risk_score >= 2:
        tier, color = 'P1 · Critical priority', 'var(--red-600)'
    elif area.risk_score >= 1:
        tier, color = 'P2 · High priority', 'var(--orange-600)'
    else:
        tier, color = 'P3 · Medium priority', 'var(--amber-600)'
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


def render_preview_drawer(
    rid: str,
    result: ApplyRecommendationsResult,
    area: FocusAreaRecommendation | None = None,
    *,
    surface: str = 'redteam',
    breakdown: str = '',
) -> str:
    """Breakdown drawer for a preview: where the recommendation(s) came from
    (single-rec applies carry their focus area or, for sim, a pre-rendered
    *breakdown* block), what will be merged, and the exact instructions diff."""
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
            else _diff_html(result.diff)
        )
    )
    if unchanged:
        return _drawer('Preview: no change', body)

    payload = {
        'agent_key': result.agent_key,
        'new_instructions': result.new_instructions,
        'recommendations': result.recommendations,
    }
    safe_rid = esc(rid)
    footer = (
        f'<form hx-post="/r/{safe_rid}/{surface}/apply/confirm" hx-target="#{DRAWER_ID}" hx-swap="innerHTML">'
        f'<input type="hidden" name="payload" value="{esc(json.dumps(payload))}">'
        '<button type="submit" class="rt-apply-btn rt-apply-btn--confirm">Apply to agent</button>'
        '</form>'
        f'<button class="rt-apply-btn rt-apply-btn--ghost" hx-get="/redteam/apply/dismiss" '
        f'hx-target="#{DRAWER_ID}" hx-swap="innerHTML">Cancel</button>'
        '<span class="rt-drawer-footnote">Writes a new minor version of the agent.</span>'
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


def record_applied_on_report(path: Path, recommendations: list[str], field: str = 'applied_recommendations') -> None:
    """Append newly applied recommendations to the report JSON, atomically.

    *field* is the surface's tracking field (``applied_recommendations`` for
    red team, ``applied_suggestions`` for simulation). Deduplicates against
    what is already recorded so a double-click cannot inflate the list.
    """
    raw = json.loads(path.read_text(encoding='utf-8'))
    existing = raw.get(field) or []
    seen = {str(r).strip() for r in existing}
    for rec in recommendations:
        if rec.strip() not in seen:
            existing.append(rec)
            seen.add(rec.strip())
    raw[field] = existing
    tmp = path.with_suffix(path.suffix + '.tmp')
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
    rid: str, req: Request, roots: list[Any] | None, *, field: str, version_note: str
) -> Response:
    """Shared confirm handler: write the previewed instructions to the agent,
    record the applied recommendations on the report JSON under *field*."""
    from evaluatorq.dashboard import library

    form = await req.form()
    try:
        payload = json.loads(str(form.get('payload') or '{}'))
        agent_key = str(payload['agent_key'])
        new_instructions = str(payload['new_instructions'])
        recommendations = [str(r) for r in payload['recommendations']]
    except (KeyError, ValueError, TypeError):
        return Response(render_error_drawer('Malformed apply payload; re-run the preview.'), media_type='text/html')
    if not agent_key or not new_instructions or not recommendations:
        return Response(render_error_drawer('Nothing to apply; re-run the preview.'), media_type='text/html')

    try:
        orq_client, _llm_client, _model = _build_clients()
    except ValueError as e:
        return Response(render_error_drawer(str(e)), media_type='text/html')

    import asyncio
    import functools

    try:
        updated = await asyncio.to_thread(
            functools.partial(
                orq_client.agents.update,
                agent_key=agent_key,
                instructions=new_instructions,
                version_increment='minor',
                version_description=f'Applied {len(recommendations)} {version_note}',
            )
        )
    except Exception as e:
        logger.opt(exception=True).warning('apply_ui: agent update failed for {}', agent_key)
        return Response(render_error_drawer(f'Agent update failed: {e}'), media_type='text/html')

    # Record on the report so a later preview skips these. A write-back
    # failure must not hide that the agent WAS updated - report it as a
    # warning inside the success drawer instead of failing the request.
    path = library.resolve(rid, roots)
    record_note = ''
    if path is not None:
        try:
            record_applied_on_report(path, recommendations, field)
        except Exception:
            logger.opt(exception=True).warning('apply_ui: could not record applied recs on {}', path)
            record_note = ' (warning: the report file could not be updated, so these may show as pending again)'

    version = getattr(updated, 'version', None)
    html = render_applied_drawer(agent_key, len(recommendations), str(version) if version is not None else None)
    if record_note:
        html = html.replace('</p>', esc(record_note) + '</p>', 1)
    return Response(html, media_type='text/html')


def register_apply_routes(app: Any, roots: list[Any] | None = None) -> None:
    """Register the apply preview/confirm/dismiss routes for both surfaces."""
    from evaluatorq.dashboard.redteam_views import _404, _load_report
    from evaluatorq.dashboard.sim_views import _load_run

    @app.get('/redteam/apply/dismiss')
    def apply_dismiss() -> Response:
        return Response('', media_type='text/html')

    @app.post('/r/{rid}/redteam/apply/preview')
    async def apply_preview(rid: str, req: Request) -> Response:
        report = _load_report(rid, roots)
        if report is None:
            return Response(_404(f'Report {rid} not found'), status_code=404, media_type='text/html')

        if len(report.tested_agents or []) > 1:
            return Response(
                render_error_drawer(
                    'Applying recommendations is available for single-agent runs; this run compared several agents.'
                ),
                media_type='text/html',
            )
        form = await req.form()
        agent_key = str(form.get('agent_key') or '').strip()
        if not agent_key:
            return Response(
                render_error_drawer('This run does not record which agent it tested, so there is nothing to apply to.'),
                media_type='text/html',
            )

        # Per-recommendation apply: `rec` (+ its area `category`) narrows the
        # merge to that single bullet, and the drawer shows the area breakdown.
        # Without `rec` the whole pending set is merged (the apply-all bar).
        areas = list(report.focus_area_recommendations or [])
        single_rec = str(form.get('rec') or '').strip()
        area: FocusAreaRecommendation | None = None
        if single_rec:
            category = str(form.get('category') or '').strip()
            area = next(
                (a for a in areas if a.category == category and single_rec in [r.strip() for r in a.recommendations]),
                None,
            )
            if area is None:
                return Response(
                    render_error_drawer('That recommendation is no longer on the report; reload the page.'),
                    media_type='text/html',
                )
            if single_rec in {r.strip() for r in report.applied_recommendations}:
                return Response(
                    render_error_drawer('That recommendation is already applied to the agent.'),
                    media_type='text/html',
                )
            areas = [area.model_copy(update={'recommendations': [single_rec]})]

        try:
            orq_client, llm_client, model = _build_clients()
        except ValueError as e:
            return Response(render_error_drawer(str(e)), media_type='text/html')

        from evaluatorq.redteam.reports.apply import apply_recommendations

        try:
            result = await apply_recommendations(
                areas,
                agent_key,
                orq_client,
                llm_client,
                model,
                apply=False,
                already_applied=report.applied_recommendations,
            )
        except Exception as e:
            logger.opt(exception=True).warning('apply_ui: preview failed for {}', agent_key)
            return Response(render_error_drawer(f'Preview failed: {e}'), media_type='text/html')

        if not result.recommendations:
            return Response(
                render_error_drawer('No new recommendations to apply — everything is already applied.'),
                media_type='text/html',
            )
        return Response(render_preview_drawer(rid, result, area), media_type='text/html')

    @app.post('/r/{rid}/redteam/apply/confirm')
    async def apply_confirm(rid: str, req: Request) -> Response:
        return await _confirm_response(
            rid, req, roots, field='applied_recommendations', version_note='red-team remediation recommendation(s)'
        )

    @app.post('/r/{rid}/sim/apply/preview')
    async def sim_apply_preview(rid: str, req: Request) -> Response:
        run = _load_run(rid, roots)
        if run is None:
            return Response(_404(f'Report {rid} not found'), status_code=404, media_type='text/html')
        if not is_sim_apply_enabled(run):
            return Response(
                render_error_drawer('Applying suggestions needs a run that targeted an orq agent.'),
                media_type='text/html',
            )
        form = await req.form()
        agent_key = str(form.get('agent_key') or '').strip() or str(run.target or '')
        if not agent_key:
            return Response(
                render_error_drawer('This run does not record which agent it tested, so there is nothing to apply to.'),
                media_type='text/html',
            )

        # Per-suggestion apply: `rec` (+ its card's `result_index`) narrows the
        # merge to that single bullet, and the drawer shows the card breakdown.
        recs = list(run.recommendations)
        single_rec = str(form.get('rec') or '').strip()
        breakdown = ''
        if single_rec:
            try:
                idx = int(str(form.get('result_index') or ''))
            except ValueError:
                idx = -1
            card = next(
                (r for r in recs if r.result_index == idx and single_rec in [s.strip() for s in r.suggestions]),
                None,
            )
            if card is None:
                return Response(
                    render_error_drawer('That suggestion is no longer on the report; reload the page.'),
                    media_type='text/html',
                )
            if single_rec in {s.strip() for s in run.applied_suggestions}:
                return Response(
                    render_error_drawer('That suggestion is already applied to the agent.'),
                    media_type='text/html',
                )
            breakdown = sim_breakdown_html(card)
            recs = [card.model_copy(update={'suggestions': [single_rec]})]

        try:
            orq_client, llm_client, model = _build_clients()
        except ValueError as e:
            return Response(render_error_drawer(str(e)), media_type='text/html')

        from evaluatorq.simulation.reports.apply import apply_suggestions

        try:
            result = await apply_suggestions(
                recs,
                agent_key,
                orq_client,
                llm_client,
                model,
                apply=False,
                already_applied=run.applied_suggestions,
            )
        except Exception as e:
            logger.opt(exception=True).warning('apply_ui: sim preview failed for {}', agent_key)
            return Response(render_error_drawer(f'Preview failed: {e}'), media_type='text/html')

        if not result.recommendations:
            return Response(
                render_error_drawer('No new suggestions to apply — everything is already applied.'),
                media_type='text/html',
            )
        return Response(render_preview_drawer(rid, result, surface='sim', breakdown=breakdown), media_type='text/html')

    @app.post('/r/{rid}/sim/apply/confirm')
    async def sim_apply_confirm(rid: str, req: Request) -> Response:
        return await _confirm_response(
            rid, req, roots, field='applied_suggestions', version_note='simulation remediation suggestion(s)'
        )
