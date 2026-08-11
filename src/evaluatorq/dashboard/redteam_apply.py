"""Apply-recommendations UI for the red-team dashboard (RES-1143).

Surfaces the report's ``focus_area_recommendations`` in the Focus areas tab
with an "Apply to agent" flow backed by ``redteam.reports.apply``:

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
    render_apply_panel(rid, report)   — panel + drawer mount for the tab body
    register_redteam_apply_routes(app, roots)
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

    from evaluatorq.redteam.contracts import RedTeamReport
    from evaluatorq.redteam.reports.apply import ApplyRecommendationsResult

DRAWER_ID = 'rt-apply-drawer'


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


def render_apply_panel(rid: str, report: RedTeamReport) -> str:
    """The apply bar + drawer mount. Empty string when there is nothing to show."""
    if not report.focus_area_recommendations:
        return ''
    pending = pending_recommendations(report)
    applied_n = len(report.applied_recommendations)
    safe_rid = esc(rid)

    if not pending:
        status = (
            f'<span class="rt-apply-count rt-apply-count--done">✓ all {applied_n} recommendation(s) applied</span>'
            if applied_n
            else '<span class="rt-apply-count">no recommendations to apply</span>'
        )
        return f'<div class="rt-apply-bar">{status}</div><div id="{DRAWER_ID}"></div>'

    agents = report.tested_agents or []
    if len(agents) > 1:
        options = ''.join(f'<option value="{esc(a)}">{esc(a)}</option>' for a in agents)
        agent_field = f'<select class="rt-apply-agent" name="agent_key" aria-label="Agent">{options}</select>'
    else:
        # Single (or unknown) agent: fixed value, no chrome.
        only = agents[0] if agents else ''
        agent_field = f'<input type="hidden" name="agent_key" value="{esc(only)}">'
        if only:
            agent_field += f'<span class="rt-apply-agent-name">{esc(only)}</span>'

    applied_note = f' · {applied_n} already applied' if applied_n else ''
    return (
        '<div class="rt-apply-bar">'
        '<div class="rt-apply-bar-text">'
        f'<span class="rt-apply-count">{len(pending)} recommendation(s) ready to apply{esc(applied_note)}</span>'
        '<span class="rt-apply-hint">Preview folds them into the agent instructions; nothing is '
        'written until you approve the diff.</span>'
        '</div>'
        f'<form class="rt-apply-form" hx-post="/r/{safe_rid}/redteam/apply/preview" '
        f'hx-target="#{DRAWER_ID}" hx-swap="innerHTML">'
        f'{agent_field}'
        '<button type="submit" class="rt-apply-btn">Preview &amp; apply…</button>'
        '</form>'
        '</div>'
        f'<div id="{DRAWER_ID}"></div>'
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


def render_preview_drawer(rid: str, result: ApplyRecommendationsResult) -> str:
    """Breakdown drawer for a preview: what will be merged, and the exact diff."""
    recs = ''.join(f'<li>{esc(r)}</li>' for r in result.recommendations)
    unchanged = result.new_instructions.strip() == result.original_instructions.strip()
    body = (
        f'<div class="rt-drawer-section-label">Agent</div>'
        f'<div class="rt-drawer-agent">{esc(result.agent_key)}</div>'
        f'<div class="rt-drawer-section-label">{len(result.recommendations)} recommendation(s) to fold in</div>'
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
        f'<form hx-post="/r/{safe_rid}/redteam/apply/confirm" hx-target="#{DRAWER_ID}" hx-swap="innerHTML">'
        f'<input type="hidden" name="payload" value="{esc(json.dumps(payload))}">'
        '<button type="submit" class="rt-apply-btn rt-apply-btn--confirm">Apply to agent</button>'
        '</form>'
        f'<button class="rt-apply-btn rt-apply-btn--ghost" hx-get="/redteam/apply/dismiss" '
        f'hx-target="#{DRAWER_ID}" hx-swap="innerHTML">Cancel</button>'
        '<span class="rt-drawer-footnote">Writes a new minor version of the agent.</span>'
    )
    return _drawer('Preview changes', body, footer)


def render_applied_drawer(agent_key: str, applied_count: int, new_version: str | None) -> str:
    version_note = f' as version <b>{esc(new_version)}</b>' if new_version else ''
    body = (
        f'<p class="rt-drawer-success">✓ Applied {applied_count} recommendation(s) to '
        f'<b>{esc(agent_key)}</b>{version_note}.</p>'
        '<p class="rt-drawer-note">The report now records them as applied; reload the page to '
        'see the updated state. Review the new version in the Orq UI before routing traffic to it.</p>'
    )
    return _drawer('Applied', body)


# ---------------------------------------------------------------------------
# Report write-back
# ---------------------------------------------------------------------------


def record_applied_on_report(path: Path, recommendations: list[str]) -> None:
    """Append newly applied recommendations to the report JSON, atomically.

    Deduplicates against what is already recorded so a double-click cannot
    inflate the list.
    """
    raw = json.loads(path.read_text(encoding='utf-8'))
    existing = raw.get('applied_recommendations') or []
    seen = {str(r).strip() for r in existing}
    for rec in recommendations:
        if rec.strip() not in seen:
            existing.append(rec)
            seen.add(rec.strip())
    raw['applied_recommendations'] = existing
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(raw, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Clients (lazy; the dashboard must keep working without ORQ_API_KEY)
# ---------------------------------------------------------------------------


def _build_clients() -> tuple[Any, Any, str]:
    """(orq_client, llm_client, model) for the apply flow, or raise ValueError.

    Uses the same evaluator-role config the red-team pipeline used to GENERATE
    the recommendations, so the merge model matches the analysis model.
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
    model = PIPELINE_CONFIG.evaluator.model or PIPELINE_CONFIG.evaluator.judges[0]
    return orq_client, llm_client, model


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def register_redteam_apply_routes(app: Any, roots: list[Any] | None = None) -> None:
    """Register the apply preview/confirm/dismiss routes on *app*."""
    from evaluatorq.dashboard.redteam_views import _404, _load_report

    @app.get('/redteam/apply/dismiss')
    def apply_dismiss() -> Response:
        return Response('', media_type='text/html')

    @app.post('/r/{rid}/redteam/apply/preview')
    async def apply_preview(rid: str, req: Request) -> Response:
        report = _load_report(rid, roots)
        if report is None:
            return Response(_404(f'Report {rid} not found'), status_code=404, media_type='text/html')

        form = await req.form()
        agent_key = str(form.get('agent_key') or '').strip()
        if not agent_key:
            return Response(
                render_error_drawer('This run does not record which agent it tested, so there is nothing to apply to.'),
                media_type='text/html',
            )
        try:
            orq_client, llm_client, model = _build_clients()
        except ValueError as e:
            return Response(render_error_drawer(str(e)), media_type='text/html')

        from evaluatorq.redteam.reports.apply import apply_recommendations

        try:
            result = await apply_recommendations(
                report.focus_area_recommendations or [],
                agent_key,
                orq_client,
                llm_client,
                model,
                apply=False,
                already_applied=report.applied_recommendations,
            )
        except Exception as e:
            logger.opt(exception=True).warning('redteam_apply: preview failed for {}', agent_key)
            return Response(render_error_drawer(f'Preview failed: {e}'), media_type='text/html')

        if not result.recommendations:
            return Response(
                render_error_drawer('No new recommendations to apply — everything is already applied.'),
                media_type='text/html',
            )
        return Response(render_preview_drawer(rid, result), media_type='text/html')

    @app.post('/r/{rid}/redteam/apply/confirm')
    async def apply_confirm(rid: str, req: Request) -> Response:
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
                    version_description=f'Applied {len(recommendations)} red-team remediation recommendation(s)',
                )
            )
        except Exception as e:
            logger.opt(exception=True).warning('redteam_apply: agent update failed for {}', agent_key)
            return Response(render_error_drawer(f'Agent update failed: {e}'), media_type='text/html')

        # Record on the report so a later preview skips these. A write-back
        # failure must not hide that the agent WAS updated - report it as a
        # warning inside the success drawer instead of failing the request.
        path = library.resolve(rid, roots)
        record_note = ''
        if path is not None:
            try:
                record_applied_on_report(path, recommendations)
            except Exception:
                logger.opt(exception=True).warning('redteam_apply: could not record applied recs on {}', path)
                record_note = ' (warning: the report file could not be updated, so these may show as pending again)'

        version = getattr(updated, 'version', None)
        html = render_applied_drawer(agent_key, len(recommendations), str(version) if version is not None else None)
        if record_note:
            html = html.replace('</p>', esc(record_note) + '</p>', 1)
        return Response(html, media_type='text/html')
