"""Conversation viewer and disagreement viewer for the redteam dashboard.

Exports:
    render_conversation   -- per-result transcript drill-down fragment
    render_disagreement   -- agent-pair side-by-side disagreement viewer

Internal helpers:
    _render_result_detail, _render_messages, _render_agent_side, _agent_select
    _fmt_category, _fmt_vulnerability, _agent_key (re-imported from redteam_charts)

All functions return raw HTML strings suitable for HTMX hx-swap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote as _quote

from evaluatorq.common.reports import esc
from evaluatorq.dashboard.redteam_charts import (
    _agent_key,
    _fmt_category,
    _fmt_vulnerability,
)
from evaluatorq.redteam.reports.converters import _is_evaluated, _is_vulnerable

if TYPE_CHECKING:
    from evaluatorq.redteam.contracts import RedTeamReport, RedTeamResult

_PAGE_SIZE = 10


# ---------------------------------------------------------------------------
# 3. Conversation viewer
# ---------------------------------------------------------------------------


def render_conversation(
    *,
    report: RedTeamReport,
    idx: int,
    rid: str,
    container_id: str = "rt-conversation",
) -> str:
    """Render the conversation viewer fragment: row list + transcript detail."""
    results = report.results

    list_items: list[str] = []
    base_url = f"/r/{esc(rid)}/view/conversation"
    for i, r in enumerate(results):
        status = "VULN" if r.vulnerable else "SAFE"
        label = (
            f"[{status}] {esc(r.attack.id)} / "
            f"{esc(r.attack.category)} / "
            f"{esc(r.attack.attack_technique.value)}"
        )
        active_class = " rt-conv-row-active" if i == idx else ""
        list_items.append(
            f'<li class="rt-conv-row{active_class}">'
            f'<button class="rt-conv-row-btn"'
            f' hx-get="{base_url}?idx={i}"'
            f' hx-target="#{esc(container_id)}"'
            f' hx-swap="outerHTML">'
            f'{label}'
            f'</button>'
            f'</li>'
        )

    list_html = (
        f'<ul class="rt-conv-list">{"".join(list_items)}</ul>'
        if list_items
        else '<p class="rt-view-empty">No results.</p>'
    )

    detail_html = ""
    if 0 <= idx < len(results):
        detail_html = _render_result_detail(results[idx])

    return (
        f'<div class="rt-conversation" id="{esc(container_id)}">'
        f'<div class="rt-conv-layout">'
        f'<div class="rt-conv-sidebar">{list_html}</div>'
        f'<div class="rt-conv-detail">{detail_html}</div>'
        f'</div>'
        f'</div>'
    )


def _render_result_detail(r: RedTeamResult) -> str:
    """Render the detail pane for a single result: metadata + transcript."""
    atk = r.attack

    vuln_name = _fmt_vulnerability(atk.vulnerability) if atk.vulnerability else "-"
    category = _fmt_category(atk.category)
    technique = atk.attack_technique.value
    severity = atk.severity.value
    turn_type = atk.turn_type.value if atk.turn_type else "-"

    delivery_str = ""
    if atk.delivery_methods:
        dms = [getattr(dm, "value", str(dm)) for dm in atk.delivery_methods]
        delivery_str = (
            f'<div class="rt-conv-meta-item">'
            f'<span class="rt-conv-meta-label">Delivery Methods</span>'
            f'<span class="rt-conv-meta-value">{esc(", ".join(dms))}</span>'
            f'</div>'
        )

    verdict_class = "rt-conv-verdict-vuln" if r.vulnerable else "rt-conv-verdict-safe"
    verdict_text = "VULNERABLE" if r.vulnerable else "RESISTANT"

    meta_html = (
        f'<div class="rt-conv-meta">'
        f'<div class="rt-conv-meta-item">'
        f'<span class="rt-conv-meta-label">Vulnerability</span>'
        f'<span class="rt-conv-meta-value">{esc(vuln_name)}</span>'
        f'</div>'
        f'<div class="rt-conv-meta-item">'
        f'<span class="rt-conv-meta-label">Category</span>'
        f'<span class="rt-conv-meta-value">{esc(category)}</span>'
        f'</div>'
        f'<div class="rt-conv-meta-item">'
        f'<span class="rt-conv-meta-label">Technique</span>'
        f'<span class="rt-conv-meta-value">{esc(technique)}</span>'
        f'</div>'
        f'<div class="rt-conv-meta-item">'
        f'<span class="rt-conv-meta-label">Severity</span>'
        f'<span class="rt-conv-meta-value">{esc(severity)}</span>'
        f'</div>'
        f'<div class="rt-conv-meta-item">'
        f'<span class="rt-conv-meta-label">Turn Type</span>'
        f'<span class="rt-conv-meta-value">{esc(turn_type)}</span>'
        f'</div>'
        f'{delivery_str}'
        f'<div class="rt-conv-meta-item">'
        f'<span class="rt-conv-meta-label">Result</span>'
        f'<span class="rt-conv-verdict {verdict_class}">{verdict_text}</span>'
        f'</div>'
        f'</div>'
    )

    eval_html = ""
    if r.evaluation and r.evaluation.explanation:
        expl = r.evaluation.explanation
        eval_html = (
            f'<div class="rt-conv-eval">'
            f'<span class="rt-conv-eval-label">Evaluator explanation:</span>'
            f'<p class="rt-conv-eval-text">{esc(expl)}</p>'
            f'</div>'
        )

    msgs_html = _render_messages(r.messages)

    return (
        f'<div class="rt-conv-detail-inner">'
        f'{meta_html}'
        f'{eval_html}'
        f'<div class="rt-conv-transcript">'
        f'<h4 class="rt-conv-transcript-title">Conversation</h4>'
        f'{msgs_html}'
        f'</div>'
        f'</div>'
    )


def _render_messages(messages: list[Any]) -> str:
    """Render a list of Message objects as HTML.

    Handles the redteam-specific extras: tool_calls on assistant messages
    and collapsible system messages / tool responses.
    """
    if not messages:
        return '<p class="rt-view-empty">No messages recorded.</p>'

    parts: list[str] = []
    for msg in messages:
        role: str = getattr(msg, "role", "unknown")
        content: str = getattr(msg, "content", "") or ""
        tool_calls = getattr(msg, "tool_calls", None)
        name: str = getattr(msg, "name", "") or ""

        if role == "system":
            parts.append(
                f'<details class="rt-msg rt-msg-system">'
                f'<summary class="rt-msg-role">System prompt</summary>'
                f'<pre class="rt-msg-content">{esc(content)}</pre>'
                f'</details>'
            )
        elif role == "user":
            parts.append(
                f'<div class="rt-msg rt-msg-user">'
                f'<span class="rt-msg-role">User</span>'
                f'<pre class="rt-msg-content">{esc(content)}</pre>'
                f'</div>'
            )
        elif role == "assistant":
            inner_parts: list[str] = []
            if content:
                inner_parts.append(f'<pre class="rt-msg-content">{esc(content)}</pre>')
            if tool_calls:
                for tc in tool_calls:
                    fn = getattr(tc, "function", None)
                    fn_name = getattr(fn, "name", "?") if fn else "?"
                    fn_args = getattr(fn, "arguments", "") if fn else ""
                    inner_parts.append(
                        f'<pre class="rt-msg-tool-call">'
                        f'Tool call: {esc(fn_name)}({esc(fn_args)})'
                        f'</pre>'
                    )
            parts.append(
                f'<div class="rt-msg rt-msg-assistant">'
                f'<span class="rt-msg-role">Assistant</span>'
                f'{"".join(inner_parts)}'
                f'</div>'
            )
        elif role == "tool":
            tool_name = name or "tool"
            parts.append(
                f'<details class="rt-msg rt-msg-tool">'
                f'<summary class="rt-msg-role">Tool response: {esc(tool_name)}</summary>'
                f'<pre class="rt-msg-content">{esc(content)}</pre>'
                f'</details>'
            )
        else:
            parts.append(
                f'<div class="rt-msg rt-msg-unknown">'
                f'<span class="rt-msg-role">{esc(role)}</span>'
                f'<pre class="rt-msg-content">{esc(content)}</pre>'
                f'</div>'
            )

    return "".join(parts)


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
    container_id: str = "rt-disagreement",
) -> str:
    """Render the disagreement viewer: agent-pair selector + paginated side-by-side."""
    from collections import defaultdict

    results = report.results
    agents = list(dict.fromkeys(_agent_key(r) for r in results))

    if len(agents) < 2:
        return (
            f'<div class="rt-disagreement" id="{esc(container_id)}">'
            '<p class="rt-view-empty">Disagreement viewer requires 2 or more agents.</p>'
            "</div>"
        )

    if agent_a not in agents:
        agent_a = agents[0]
    if agent_b not in agents or agent_b == agent_a:
        remaining = [ag for ag in agents if ag != agent_a]
        agent_b = remaining[0] if remaining else agents[0]

    base_url = f"/r/{esc(rid)}/view/disagreement"

    selector_a = _agent_select(
        agents=agents,
        selected=agent_a,
        param_name="a",
        hx_url=base_url,
        extra_params=f"b={_quote(str(agent_b), safe='')}&page=1",
        container_id=container_id,
        label="Agent A",
    )
    selector_b = _agent_select(
        agents=[ag for ag in agents if ag != agent_a],
        selected=agent_b,
        param_name="b",
        hx_url=base_url,
        extra_params=f"a={_quote(str(agent_a), safe='')}&page=1",
        container_id=container_id,
        label="Agent B",
    )

    attack_results: dict[str, dict[str, RedTeamResult]] = defaultdict(dict)
    for r in results:
        ak = _agent_key(r)
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
        no_dis_html = (
            f'<p class="rt-view-empty">'
            f'No disagreements found between {esc(agent_a)} and {esc(agent_b)}.'
            f'</p>'
        )
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
    prev_url = f"{base_url}?a={_quote(str(agent_a), safe='')}&b={_quote(str(agent_b), safe='')}&page={page - 1}"
    next_url = f"{base_url}?a={_quote(str(agent_a), safe='')}&b={_quote(str(agent_b), safe='')}&page={page + 1}"

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
            _fmt_vulnerability(r1.attack.vulnerability) if r1.attack.vulnerability
            else _fmt_category(r1.attack.category)
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
        active_class = " rt-view-selector-active" if ag == selected else ""
        url = f"{hx_url}?{param_name}={_quote(str(ag), safe='')}&{extra_params}"
        parts.append(
            f'<button class="rt-view-selector-btn{active_class}"'
            f' hx-get="{url}"'
            f' hx-target="#{esc(container_id)}"'
            f' hx-swap="outerHTML">'
            f'{esc(ag)}'
            f'</button>'
        )
    parts.append("</div>")
    return "".join(parts)


def _render_agent_side(r: RedTeamResult, agent_name: str) -> str:
    """Render one column of a disagreement item (attack prompt, response, eval)."""
    verdict_class = "rt-conv-verdict-vuln" if r.vulnerable else "rt-conv-verdict-safe"
    verdict_text = "VULNERABLE" if r.vulnerable else "RESISTANT"

    header = (
        f'<div class="rt-dis-agent-header">'
        f'<strong>{esc(agent_name)}</strong>'
        f' <span class="rt-conv-verdict {verdict_class}">[{verdict_text}]</span>'
        f'</div>'
    )

    user_msgs = [m for m in r.messages if getattr(m, "role", "") == "user"]
    prompt_html = ""
    if user_msgs:
        prompt_text = getattr(user_msgs[-1], "content", "") or ""
        truncated = prompt_text[:600] + ("…" if len(prompt_text) > 600 else "")
        prompt_html = (
            f'<div class="rt-dis-section">'
            f'<strong>Attack prompt:</strong>'
            f'<pre class="rt-msg-content">{esc(truncated)}</pre>'
            f'</div>'
        )

    response_html = ""
    if r.response:
        resp = r.response
        truncated = resp[:600] + ("…" if len(resp) > 600 else "")
        response_html = (
            f'<div class="rt-dis-section">'
            f'<strong>Response:</strong>'
            f'<pre class="rt-msg-content">{esc(truncated)}</pre>'
            f'</div>'
        )

    eval_html = ""
    if r.evaluation and r.evaluation.explanation:
        expl = r.evaluation.explanation
        truncated = expl[:400] + ("…" if len(expl) > 400 else "")
        eval_html = (
            f'<div class="rt-dis-section">'
            f'<strong>Evaluator:</strong>'
            f'<p class="rt-conv-eval-text">{esc(truncated)}</p>'
            f'</div>'
        )

    return f'{header}{prompt_html}{response_html}{eval_html}'
