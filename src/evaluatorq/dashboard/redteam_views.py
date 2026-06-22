"""HTMX fragment routes for the four redteam-dashboard-only interactive views.


Routes (all return HTML fragments, no full page shell):

    GET /r/{rid}/view/breakdown?group_by=&stack_by=
    GET /r/{rid}/view/agent-heatmap?dim=
    GET /r/{rid}/view/conversation?idx=
    GET /r/{rid}/view/disagreement?a=&b=&page=

Each route loads the RedTeamReport via library.resolve / surfaces.ADAPTERS['redteam'],
recomputes the view from the loaded data, and returns a fragment suitable for
HTMX hx-swap="outerHTML" or hx-swap="innerHTML".

Parity source: src/evaluatorq/redteam/ui/dashboard.py (Streamlit reference).
"""

from __future__ import annotations

import operator
from collections import defaultdict
from typing import TYPE_CHECKING, Any
from urllib.parse import quote as _quote

from loguru import logger
from starlette.requests import Request  # noqa: TC002 — FastHTML inspects this annotation at runtime
from starlette.responses import Response

from evaluatorq.common.reports import (
    COLORS,
    ORQ_SCALE_HEAT,
    QUALITATIVE,
    SEVERITY_ORDER,
    esc,
    render_embed,
    scale_color,
)
from evaluatorq.common.reports.vega import vl_bar_h, vl_heatmap, vl_stacked_bar
from evaluatorq.redteam.contracts import OWASP_CATEGORY_NAMES, RedTeamReport, RedTeamResult

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Internal helpers (ported from Streamlit dashboard)
# ---------------------------------------------------------------------------

_DIM_LABELS: dict[str, str] = {
    "vulnerability": "Vulnerability",
    "category": "Framework Category",
    "severity": "Severity",
    "attack_technique": "Attack Technique",
    "delivery_method": "Delivery Method",
    "turn_type": "Turn Type",
    "source": "Source",
}

_HEATMAP_DIM_LABELS: dict[str, str] = {
    "vulnerability": "Vulnerability",
    "category": "Category",
    "technique": "Technique",
    "severity": "Severity",
}

_PAGE_SIZE = 10


def _fmt_category(code: str) -> str:
    name = OWASP_CATEGORY_NAMES.get(code)
    return f"{code} - {name}" if name else code


def _fmt_vulnerability(vuln_id: str) -> str:
    """Format a vulnerability ID into a human-readable name."""
    from evaluatorq.redteam.vulnerability_registry import VULNERABILITY_DEFS, Vulnerability

    try:
        vuln_enum = Vulnerability(vuln_id)
        vdef = VULNERABILITY_DEFS.get(vuln_enum)
        if vdef:
            return vdef.name
    except ValueError:
        pass
    return vuln_id.replace("_", " ").title()


def _dim_value(r: RedTeamResult, dim: str) -> str:
    """Extract a string dimension value from a RedTeamResult."""
    if dim == "category":
        return _fmt_category(r.attack.category)
    if dim == "vulnerability":
        return _fmt_vulnerability(r.attack.vulnerability) if r.attack.vulnerability else "unknown"
    if dim == "severity":
        return r.attack.severity.value
    if dim == "attack_technique":
        return r.attack.attack_technique.value
    if dim == "delivery_method":
        if r.attack.delivery_methods:
            dm = r.attack.delivery_methods[0]
            return getattr(dm, "value", str(dm))
        return "unknown"
    if dim == "turn_type":
        return r.attack.turn_type.value if r.attack.turn_type else "unknown"
    if dim == "source":
        return r.attack.source
    return "unknown"


def _heatmap_dim_value(r: RedTeamResult, dim: str) -> str:
    """Extract a dimension value for the agent heatmap view."""
    if dim == "vulnerability":
        return _fmt_vulnerability(r.attack.vulnerability) if r.attack.vulnerability else "unknown"
    if dim == "category":
        return _fmt_category(r.attack.category)
    if dim == "technique":
        return r.attack.attack_technique.value
    if dim == "severity":
        return r.attack.severity.value
    return "unknown"


def _agent_key(r: RedTeamResult) -> str:
    return r.agent.key or r.agent.display_name or "unknown"


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


# ---------------------------------------------------------------------------
# Selector widget helpers
# ---------------------------------------------------------------------------


def _select_buttons(
    *,
    options: list[str],
    labels: dict[str, str],
    selected: str,
    hx_url: str,
    param_name: str,
    extra_params: str = "",
    container_id: str,
) -> str:
    """Render a row of hx-get buttons that swap ``container_id``."""
    parts: list[str] = [f'<div class="rt-view-selector" id="{esc(container_id)}">']
    for opt in options:
        active_class = " rt-view-selector-active" if opt == selected else ""
        # Build the URL with this option selected, preserving extra params.
        # Use urllib.parse.quote for query-param values so spaces/ampersands
        # in option strings produce valid URLs.  esc() (html.escape) is kept
        # only for HTML-attribute / text contexts.
        sep = "&" if extra_params else ""
        url = f"{hx_url}?{param_name}={_quote(str(opt), safe='')}{sep}{extra_params}"
        lbl = labels.get(opt, opt)
        parts.append(
            f'<button class="rt-view-selector-btn{active_class}"'
            f' hx-get="{url}"'
            f' hx-target="#{esc(container_id)}"'
            f' hx-swap="outerHTML">'
            f'{esc(lbl)}</button>'
        )
    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# 1. Interactive breakdown
# ---------------------------------------------------------------------------


def render_breakdown(
    *,
    report: RedTeamReport,
    group_by: str,
    stack_by: str | None,
    rid: str,
    container_id: str = "rt-breakdown",
) -> str:
    """Render the breakdown fragment: selector controls + Vega-Embed chart."""
    results = report.results
    dimensions = list(_DIM_LABELS.keys())

    # Clamp invalid inputs
    if group_by not in dimensions:
        group_by = "vulnerability"
    if stack_by is not None and (stack_by not in dimensions or stack_by == group_by):
        stack_by = None

    base_url = f"/r/{esc(rid)}/view/breakdown"
    stack_options = ["none"] + [d for d in dimensions if d != group_by]
    stack_labels: dict[str, str] = {"none": "None", **_DIM_LABELS}
    cur_stack = stack_by or "none"

    # Selector: group_by buttons; preserve stack_by as extra param.
    # URL-encode the preserved values so dimension names with special chars are safe.
    extra_stack = f"stack_by={_quote(str(cur_stack), safe='')}" if cur_stack != "none" else ""
    group_selector = _select_buttons(
        options=dimensions,
        labels=_DIM_LABELS,
        selected=group_by,
        hx_url=base_url,
        param_name="group_by",
        extra_params=extra_stack,
        container_id=container_id,
    )
    extra_group = f"group_by={_quote(str(group_by), safe='')}"
    stack_selector = _select_buttons(
        options=stack_options,
        labels=stack_labels,
        selected=cur_stack,
        hx_url=base_url,
        param_name="stack_by",
        extra_params=extra_group,
        container_id=container_id,
    )

    if not results:
        return (
            f'<div class="rt-breakdown" id="{esc(container_id)}">'
            f'{group_selector}{stack_selector}'
            '<p class="rt-view-empty">No results to display.</p>'
            "</div>"
        )

    chart_html = _build_breakdown_chart(results, group_by, stack_by, container_id)

    return (
        f'<div class="rt-breakdown" id="{esc(container_id)}">'
        f'<div class="rt-breakdown-controls">'
        f'<div class="rt-breakdown-control-group">'
        f'<label class="rt-breakdown-label">Group by (Y-Axis)</label>'
        f'{group_selector}'
        f'</div>'
        f'<div class="rt-breakdown-control-group">'
        f'<label class="rt-breakdown-label">Stack / Color by</label>'
        f'{stack_selector}'
        f'</div>'
        f'</div>'
        f'{chart_html}'
        f"</div>"
    )


def _build_breakdown_chart(
    results: list[RedTeamResult],
    group_by: str,
    stack_by: str | None,
    container_id: str,
) -> str:
    """Build and embed the breakdown bar chart."""
    chart_id = f"{container_id}-chart"

    if stack_by is None:
        # Simple horizontal bar: group_by dimension → ASR%
        groups: dict[str, dict[str, int]] = defaultdict(lambda: {"vuln": 0, "total": 0})
        for r in results:
            key = _dim_value(r, group_by)
            groups[key]["total"] += 1
            if r.vulnerable:
                groups[key]["vuln"] += 1

        chart_rows: list[dict[str, Any]] = []
        for name, counts in groups.items():
            asr = (counts["vuln"] / counts["total"] * 100) if counts["total"] else 0.0
            chart_rows.append({"dimension": name, "asr": round(asr, 1), "n": counts["total"]})

        chart_rows.sort(key=operator.itemgetter("asr"), reverse=True)

        labels = [row["dimension"] for row in chart_rows]
        values = [row["asr"] for row in chart_rows]
        value_labels = [f'{row["asr"]:.1f}% (n={row["n"]})' for row in chart_rows]

        # Per-bar colors from QUALITATIVE
        colors = [QUALITATIVE[i % len(QUALITATIVE)] for i in range(len(labels))]

        spec = vl_bar_h(
            labels=labels,
            values=values,
            color=QUALITATIVE[0],
            x_title="ASR (%)",
            value_labels=value_labels,
            colors=colors,
        )
    else:
        # Stacked bar: group_by x stack_by -> ASR%
        groups_stacked: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"vuln": 0, "total": 0}
        )
        for r in results:
            g = _dim_value(r, group_by)
            s = _dim_value(r, stack_by)
            groups_stacked[g, s]["total"] += 1
            if r.vulnerable:
                groups_stacked[g, s]["vuln"] += 1

        stacked_rows: list[dict[str, Any]] = []
        for (g, s), counts in groups_stacked.items():
            asr = (counts["vuln"] / counts["total"] * 100) if counts["total"] else 0.0
            stacked_rows.append({"dimension": g, "stack": s, "asr": round(asr, 1), "n": counts["total"]})

        # Sort dimensions by average ASR descending
        dim_asr: dict[str, list[float]] = defaultdict(list)
        for row in stacked_rows:
            dim_asr[row["dimension"]].append(row["asr"])
        dim_order = sorted(dim_asr.keys(), key=lambda d: sum(dim_asr[d]) / max(len(dim_asr[d]), 1), reverse=True)

        stack_vals = sorted({row["stack"] for row in stacked_rows})

        # Build series: (stack_value, [asr_per_dim_in_dim_order])
        row_by_gs: dict[tuple[str, str], dict[str, Any]] = {(row["dimension"], row["stack"]): row for row in stacked_rows}
        series: list[tuple[str, list[float]]] = []
        # Build per-series/per-label n= text labels to match Streamlit parity
        # (ref: redteam/ui/dashboard.py:1413 labels each stacked segment "n=<count>").
        stacked_value_labels: list[list[str]] = []
        for sv in stack_vals:
            sv_vals: list[float] = []
            sv_texts: list[str] = []
            for d in dim_order:
                row = row_by_gs.get((d, sv), {"asr": 0.0, "n": 0})
                sv_vals.append(row["asr"])
                n = row["n"]
                sv_texts.append(f"n={n}" if n else "")
            series.append((sv, sv_vals))
            stacked_value_labels.append(sv_texts)

        spec = vl_stacked_bar(
            labels=dim_order,
            series=series,
            x_title="ASR (%)",
            value_labels=stacked_value_labels,
        )

    return render_embed(spec, chart_id)


# ---------------------------------------------------------------------------
# 2. Agent heatmap dimension selector
# ---------------------------------------------------------------------------


def render_agent_heatmap(
    *,
    report: RedTeamReport,
    dim: str,
    rid: str,
    container_id: str = "rt-agent-heatmap",
) -> str:
    """Render the agent heatmap fragment: dim selector + heatmap chart."""
    results = report.results
    dim_options = list(_HEATMAP_DIM_LABELS.keys())

    # Collect distinct agents
    agents = list(dict.fromkeys(_agent_key(r) for r in results))

    if len(agents) < 2:
        return (
            f'<div class="rt-agent-heatmap" id="{esc(container_id)}">'
            '<p class="rt-view-empty">Agent heatmap requires 2 or more agents.</p>'
            "</div>"
        )

    if dim not in dim_options:
        dim = "vulnerability"

    base_url = f"/r/{esc(rid)}/view/agent-heatmap"
    dim_selector = _select_buttons(
        options=dim_options,
        labels=_HEATMAP_DIM_LABELS,
        selected=dim,
        hx_url=base_url,
        param_name="dim",
        container_id=container_id,
    )

    if not results:
        return (
            f'<div class="rt-agent-heatmap" id="{esc(container_id)}">'
            f'{dim_selector}'
            '<p class="rt-view-empty">No results to display.</p>'
            "</div>"
        )

    chart_html = _build_heatmap_chart(results, agents, dim, container_id)

    return (
        f'<div class="rt-agent-heatmap" id="{esc(container_id)}">'
        f'<div class="rt-heatmap-controls">'
        f'<label class="rt-breakdown-label">Group by</label>'
        f'{dim_selector}'
        f'</div>'
        f'{chart_html}'
        f"</div>"
    )


def _build_heatmap_chart(
    results: list[RedTeamResult],
    agents: list[str],
    dim: str,
    container_id: str,
) -> str:
    """Build and embed the agent heatmap."""
    chart_id = f"{container_id}-chart"

    # Build pivot: dim_value -> agent_key -> {total, vuln}
    pivot: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "vuln": 0})
    )
    for r in results:
        dv = _heatmap_dim_value(r, dim)
        ak = _agent_key(r)
        pivot[dv][ak]["total"] += 1
        if r.vulnerable:
            pivot[dv][ak]["vuln"] += 1

    # Sort rows
    if dim == "severity":
        all_dim_vals = [s for s in SEVERITY_ORDER if s in pivot]
        other_vals = sorted(v for v in pivot if v not in SEVERITY_ORDER)
        all_dim_vals = all_dim_vals + other_vals
    else:
        all_dim_vals = sorted(pivot.keys())

    if not all_dim_vals:
        return '<p class="rt-view-empty">No data for this dimension.</p>'

    # Build cell data: rows=dim_vals (y), cols=agents (x)
    cell_colors: list[list[str]] = []
    cell_texts: list[list[str]] = []
    grey = COLORS.get("sand_400", "#e4e2df")

    for dv in all_dim_vals:
        row_colors: list[str] = []
        row_texts: list[str] = []
        for ak in agents:
            counts = pivot[dv].get(ak, {"total": 0, "vuln": 0})
            total = counts["total"]
            vuln = counts["vuln"]
            if total == 0:
                row_colors.append(grey)
                row_texts.append("—")
            else:
                asr = vuln / total  # 0-1 for scale_color
                color = scale_color(asr, ORQ_SCALE_HEAT)
                row_colors.append(color)
                row_texts.append(f"{asr * 100:.0f}%\nn={total}")
        cell_colors.append(row_colors)
        cell_texts.append(row_texts)

    spec = vl_heatmap(
        x_labels=agents,
        y_labels=all_dim_vals,
        cell_colors=cell_colors,
        cell_texts=cell_texts,
    )
    return render_embed(spec, chart_id)


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

    # Row list (selectable)
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

    # Detail pane
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

    # Evaluator explanation
    eval_html = ""
    if r.evaluation and r.evaluation.explanation:
        expl = r.evaluation.explanation
        eval_html = (
            f'<div class="rt-conv-eval">'
            f'<span class="rt-conv-eval-label">Evaluator explanation:</span>'
            f'<p class="rt-conv-eval-text">{esc(expl)}</p>'
            f'</div>'
        )

    # Transcript
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
    """Render a list of Message objects as HTML."""
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
    results = report.results
    agents = list(dict.fromkeys(_agent_key(r) for r in results))

    if len(agents) < 2:
        return (
            f'<div class="rt-disagreement" id="{esc(container_id)}">'
            '<p class="rt-view-empty">Disagreement viewer requires 2 or more agents.</p>'
            "</div>"
        )

    # Default agent pair if invalid
    if agent_a not in agents:
        agent_a = agents[0]
    if agent_b not in agents or agent_b == agent_a:
        remaining = [ag for ag in agents if ag != agent_a]
        agent_b = remaining[0] if remaining else agents[0]

    base_url = f"/r/{esc(rid)}/view/disagreement"

    # Agent-pair selectors — URL-encode agent keys inside extra_params too.
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

    # Find disagreements between the selected pair
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
        if r1.vulnerable != r2.vulnerable:
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

    # Pagination controls
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

    # Build each disagreement item
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
        # URL-encode the agent key so keys containing spaces or special chars
        # produce valid query strings; Starlette auto-decodes on the way in.
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

    # Last user message as attack prompt
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

    # Agent response
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

    # Evaluator explanation
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


# ---------------------------------------------------------------------------
# Route factory: register all four views on a FastHTML app
# ---------------------------------------------------------------------------


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


def _apply_redteam_filter(report: RedTeamReport, selections: dict[str, list[str]]) -> list[RedTeamResult]:
    """Return the filtered result list; empty selections ≡ all results."""
    from evaluatorq.dashboard.filters import FILTERS

    filter_def = FILTERS.get("redteam")
    if filter_def is None or not selections:
        return list(report.results)
    return filter_def.apply(report, selections)


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
        # Build a view of the report scoped to the filtered results.  RedTeamReport
        # is a Pydantic model so we use model_copy(update=...) rather than
        # dataclasses.replace.
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

        # Clamp idx to the filtered set so a stale idx after a filter change
        # shows index 0 rather than an empty detail pane.
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
