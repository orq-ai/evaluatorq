"""Attack fragment and disagreement viewer for the redteam dashboard.

Exports:
    render_attack_fragment -- attack-evidence-row fragment (tags + verdict + bubble transcript)
    render_disagreement   -- agent-pair side-by-side disagreement viewer

Internal helpers:
    _render_agent_side, _agent_select
    fmt_category, fmt_vulnerability, agent_key (re-imported from redteam_charts)

All functions return raw HTML strings suitable for HTMX hx-swap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import quote as _quote

from evaluatorq.common.reports import esc
from evaluatorq.dashboard.redteam_charts import (
    agent_key,
    fmt_category,
    fmt_vulnerability,
)
from evaluatorq.dashboard.report_kit import tag
from evaluatorq.dashboard.view import render_message_list
from evaluatorq.redteam.reports.converters import _is_evaluated, _is_vulnerable

if TYPE_CHECKING:
    from evaluatorq.redteam.contracts import RedTeamReport, RedTeamResult

_PAGE_SIZE = 10

# `Message.role` is exactly `Literal['user','assistant','tool','system','developer']`
# -- there is no adversarial role, so no ADV glyph here.
_RT_ROLE_LABELS = {'user': 'USR', 'assistant': 'AGT', 'system': 'SYS', 'developer': 'SYS', 'tool': 'TOOL'}


# ---------------------------------------------------------------------------
# 3. Attack evidence fragment (spec §Attacks)
# ---------------------------------------------------------------------------


def render_attack_fragment(r: RedTeamResult) -> str:
    """Render one attack-evidence-row body: tags row, evaluator-verdict
    callout, then the shared bubble transcript (spec §Attacks).

    Tags: delivery method(s), ``'{turn_type}-turn'``, category. The verdict
    branch is computed locally here (vulnerable / resistant / error / no
    evaluation) with its own compact layout, independent of any other
    result-detail rendering.
    """
    atk = r.attack

    tags: list[str] = [tag(getattr(dm, 'value', str(dm))) for dm in atk.delivery_methods or []]
    if atk.turn_type:
        tags.append(tag(f'{atk.turn_type.value}-turn'))
    if atk.category:
        tags.append(tag(atk.category))
    tags_html = f'<div class="rt-attack-tags">{"".join(tags)}</div>'

    if r.error:
        verdict_html = (
            '<div class="rt-verdict rt-verdict-error">'
            '<span class="rt-verdict-label">Evaluator verdict</span>'
            f'<p class="rt-verdict-body" style="color:var(--red-600)">{esc(r.error)}</p>'
            '</div>'
        )
    elif r.vulnerable:
        explanation = r.evaluation.explanation if r.evaluation and r.evaluation.explanation else ''
        body = esc(explanation) if explanation else 'No evaluation available.'
        verdict_html = (
            '<div class="rt-verdict rt-verdict-vuln" style="border-left:3px solid var(--orange-500)">'
            '<span class="rt-verdict-label">Evaluator verdict</span>'
            f'<p class="rt-verdict-body">{body}</p>'
            '</div>'
        )
    else:
        explanation = r.evaluation.explanation if r.evaluation and r.evaluation.explanation else ''
        body = esc(explanation) if explanation else 'No evaluation available.'
        verdict_html = (
            '<div class="rt-verdict rt-verdict-safe" '
            'style="background:var(--green-50);border-left:3px solid var(--green-600)">'
            '<span class="rt-verdict-label">Evaluator verdict</span>'
            f'<p class="rt-verdict-body">{body}</p>'
            '</div>'
        )

    transcript_html = render_message_list(r.messages, role_labels=_RT_ROLE_LABELS, class_prefix='rt')

    return f'{tags_html}{verdict_html}{transcript_html}'


# ---------------------------------------------------------------------------
# 4. Disagreement viewer
# ---------------------------------------------------------------------------


def render_disagreement(
    *,
    report: RedTeamReport,
    agent_a: str,
    agent_b: str,
    page: int,
    rid: str,
    container_id: str = 'rt-disagreement',
) -> str:
    """Render the disagreement viewer: agent-pair selector + paginated side-by-side."""
    from collections import defaultdict

    results = report.results
    agents = list(dict.fromkeys(agent_key(r) for r in results))

    if len(agents) < 2:
        return (
            f'<div class="rt-disagreement" id="{esc(container_id)}">'
            '<p class="rt-view-empty">Disagreement viewer requires 2 or more agents.</p>'
            '</div>'
        )

    if agent_a not in agents:
        agent_a = agents[0]
    if agent_b not in agents or agent_b == agent_a:
        remaining = [ag for ag in agents if ag != agent_a]
        agent_b = remaining[0] if remaining else agents[0]

    base_url = f'/r/{esc(rid)}/view/disagreement'

    selector_a = _agent_select(
        agents=agents,
        selected=agent_a,
        param_name='a',
        hx_url=base_url,
        extra_params=f'b={_quote(str(agent_b), safe="")}&page=1',
        container_id=container_id,
        label='Agent A',
    )
    selector_b = _agent_select(
        agents=[ag for ag in agents if ag != agent_a],
        selected=agent_b,
        param_name='b',
        hx_url=base_url,
        extra_params=f'a={_quote(str(agent_a), safe="")}&page=1',
        container_id=container_id,
        label='Agent B',
    )

    attack_results: dict[str, dict[str, RedTeamResult]] = defaultdict(dict)
    for r in results:
        ak = agent_key(r)
        if ak in (agent_a, agent_b):
            attack_results[r.attack.id][ak] = r

    disagreements: list[tuple[RedTeamResult, RedTeamResult]] = []
    for agent_map in attack_results.values():
        if agent_a not in agent_map or agent_b not in agent_map:
            continue
        r1, r2 = agent_map[agent_a], agent_map[agent_b]
        if not _is_evaluated(r1) or not _is_evaluated(r2):
            continue
        if _is_vulnerable(r1) != _is_vulnerable(r2):
            disagreements.append((r1, r2))

    if not disagreements:
        no_dis_html = f'<p class="rt-view-empty">No disagreements found between {esc(agent_a)} and {esc(agent_b)}.</p>'
        return (
            f'<div class="rt-disagreement" id="{esc(container_id)}">'
            f'<div class="rt-dis-controls">{selector_a}{selector_b}</div>'
            f'{no_dis_html}'
            f'</div>'
        )

    total_pages = max(1, (len(disagreements) + _PAGE_SIZE - 1) // _PAGE_SIZE)
    page = max(1, min(page, total_pages))
    start = (page - 1) * _PAGE_SIZE
    end = start + _PAGE_SIZE
    page_items = disagreements[start:end]

    prev_disabled = ' disabled' if page <= 1 else ''
    next_disabled = ' disabled' if page >= total_pages else ''
    prev_url = f'{base_url}?a={_quote(str(agent_a), safe="")}&b={_quote(str(agent_b), safe="")}&page={page - 1}'
    next_url = f'{base_url}?a={_quote(str(agent_a), safe="")}&b={_quote(str(agent_b), safe="")}&page={page + 1}'

    pagination_html = (
        f'<div class="rt-dis-pagination">'
        f'<button class="rt-dis-page-btn"{prev_disabled}'
        f' hx-get="{prev_url}"'
        f' hx-target="#{esc(container_id)}"'
        f' hx-swap="outerHTML">Previous</button>'
        f'<span class="rt-dis-page-info">Page {page} of {total_pages}'
        f' ({len(disagreements)} disagreements)</span>'
        f'<button class="rt-dis-page-btn"{next_disabled}'
        f' hx-get="{next_url}"'
        f' hx-target="#{esc(container_id)}"'
        f' hx-swap="outerHTML">Next</button>'
        f'</div>'
    )

    items_html_parts: list[str] = []
    for item_idx, (r1, r2) in enumerate(page_items, start=start + 1):
        vuln_label = (
            fmt_vulnerability(r1.attack.vulnerability) if r1.attack.vulnerability else fmt_category(r1.attack.category)
        )
        technique_label = r1.attack.attack_technique.value
        item_html = (
            f'<details class="rt-dis-item" open>'
            f'<summary class="rt-dis-item-summary">'
            f'#{item_idx} {esc(r1.attack.id)} — {esc(vuln_label)} / {esc(technique_label)}'
            f'</summary>'
            f'<div class="rt-dis-item-body">'
            f'<div class="rt-dis-col">{_render_agent_side(r1, agent_a)}</div>'
            f'<div class="rt-dis-col">{_render_agent_side(r2, agent_b)}</div>'
            f'</div>'
            f'</details>'
        )
        items_html_parts.append(item_html)

    return (
        f'<div class="rt-disagreement" id="{esc(container_id)}">'
        f'<div class="rt-dis-controls">{selector_a}{selector_b}</div>'
        f'{pagination_html}'
        f'<div class="rt-dis-items">{"".join(items_html_parts)}</div>'
        f'</div>'
    )


def _agent_select(
    *,
    agents: list[str],
    selected: str,
    param_name: str,
    hx_url: str,
    extra_params: str,
    container_id: str,
    label: str,
) -> str:
    """Render a labeled agent selector for the disagreement view."""
    parts = [f'<div class="rt-dis-agent-select"><label class="rt-breakdown-label">{esc(label)}</label>']
    for ag in agents:
        active_class = ' rt-view-selector-active' if ag == selected else ''
        url = f'{hx_url}?{param_name}={_quote(str(ag), safe="")}&{extra_params}'
        parts.append(
            f'<button class="rt-view-selector-btn{active_class}"'
            f' hx-get="{url}"'
            f' hx-target="#{esc(container_id)}"'
            f' hx-swap="outerHTML">'
            f'{esc(ag)}'
            f'</button>'
        )
    parts.append('</div>')
    return ''.join(parts)


def _render_agent_side(r: RedTeamResult, agent_name: str) -> str:
    """Render one column of a disagreement item (attack prompt, response, eval)."""
    verdict_class = 'rt-conv-verdict-vuln' if r.vulnerable else 'rt-conv-verdict-safe'
    verdict_text = 'VULNERABLE' if r.vulnerable else 'RESISTANT'

    header = (
        f'<div class="rt-dis-agent-header">'
        f'<strong>{esc(agent_name)}</strong>'
        f' <span class="rt-conv-verdict {verdict_class}">[{verdict_text}]</span>'
        f'</div>'
    )

    user_msgs = [m for m in r.messages if getattr(m, 'role', '') == 'user']
    prompt_html = ''
    if user_msgs:
        prompt_text = getattr(user_msgs[-1], 'content', '') or ''
        truncated = prompt_text[:600] + ('…' if len(prompt_text) > 600 else '')
        prompt_html = (
            f'<div class="rt-dis-section">'
            f'<strong>Attack prompt:</strong>'
            f'<pre class="rt-msg-content">{esc(truncated)}</pre>'
            f'</div>'
        )

    response_html = ''
    if r.response:
        resp = r.response
        truncated = resp[:600] + ('…' if len(resp) > 600 else '')
        response_html = (
            f'<div class="rt-dis-section">'
            f'<strong>Response:</strong>'
            f'<pre class="rt-msg-content">{esc(truncated)}</pre>'
            f'</div>'
        )

    eval_html = ''
    if r.evaluation and r.evaluation.explanation:
        expl = r.evaluation.explanation
        truncated = expl[:400] + ('…' if len(expl) > 400 else '')
        eval_html = (
            f'<div class="rt-dis-section">'
            f'<strong>Evaluator:</strong>'
            f'<p class="rt-conv-eval-text">{esc(truncated)}</p>'
            f'</div>'
        )

    return f'{header}{prompt_html}{response_html}{eval_html}'
