"""Interactive breakdown chart and agent heatmap for the redteam dashboard.

Exports:
    render_breakdown        -- interactive breakdown (group_by x stack_by) fragment
    render_agent_heatmap    -- agent heatmap fragment
    agent_key               -- extract agent key string from a RedTeamResult
    fmt_category            -- format a framework category code for display
    fmt_vulnerability       -- format a vulnerability ID for display

Internal helpers:
    _build_breakdown_chart, _build_heatmap_chart
    _select_buttons, _dim_value, _heatmap_dim_value
    _DIM_LABELS, _HEATMAP_DIM_LABELS

All functions return raw HTML strings suitable for HTMX hx-swap.
"""

from __future__ import annotations

import operator
from collections import defaultdict
from typing import Any
from urllib.parse import quote as _quote

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
from evaluatorq.redteam.reports.converters import _is_evaluated, _is_vulnerable

# ---------------------------------------------------------------------------
# Dimension label maps
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


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt_category(code: str) -> str:
    name = OWASP_CATEGORY_NAMES.get(code)
    return f"{code} - {name}" if name else code


def fmt_vulnerability(vuln_id: str) -> str:
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
        return fmt_category(r.attack.category)
    if dim == "vulnerability":
        return fmt_vulnerability(r.attack.vulnerability) if r.attack.vulnerability else "unknown"
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
        return fmt_vulnerability(r.attack.vulnerability) if r.attack.vulnerability else "unknown"
    if dim == "category":
        return fmt_category(r.attack.category)
    if dim == "technique":
        return r.attack.attack_technique.value
    if dim == "severity":
        return r.attack.severity.value
    return "unknown"


def agent_key(r: RedTeamResult) -> str:
    return r.agent.key or r.agent.display_name or "unknown"


# ---------------------------------------------------------------------------
# Selector widget helper
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
        groups: dict[str, dict[str, int]] = defaultdict(lambda: {"vuln": 0, "evaluated": 0})
        for r in results:
            key = _dim_value(r, group_by)
            if _is_evaluated(r):
                groups[key]["evaluated"] += 1
                if _is_vulnerable(r):
                    groups[key]["vuln"] += 1

        chart_rows: list[dict[str, Any]] = []
        for name, counts in groups.items():
            asr = (counts["vuln"] / counts["evaluated"] * 100) if counts["evaluated"] else 0.0
            chart_rows.append({"dimension": name, "asr": round(asr, 1), "n": counts["evaluated"]})

        chart_rows.sort(key=operator.itemgetter("asr"), reverse=True)

        labels = [row["dimension"] for row in chart_rows]
        values = [row["asr"] for row in chart_rows]
        value_labels = [f'{row["asr"]:.1f}% (n={row["n"]})' for row in chart_rows]

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
        groups_stacked: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"vuln": 0, "evaluated": 0}
        )
        for r in results:
            if _is_evaluated(r):
                g = _dim_value(r, group_by)
                s = _dim_value(r, stack_by)
                groups_stacked[g, s]["evaluated"] += 1
                if _is_vulnerable(r):
                    groups_stacked[g, s]["vuln"] += 1

        stacked_rows: list[dict[str, Any]] = []
        for (g, s), counts in groups_stacked.items():
            asr = (counts["vuln"] / counts["evaluated"] * 100) if counts["evaluated"] else 0.0
            stacked_rows.append({"dimension": g, "stack": s, "asr": round(asr, 1), "n": counts["evaluated"]})

        dim_asr: dict[str, list[float]] = defaultdict(list)
        for row in stacked_rows:
            dim_asr[row["dimension"]].append(row["asr"])
        dim_order = sorted(dim_asr.keys(), key=lambda d: sum(dim_asr[d]) / max(len(dim_asr[d]), 1), reverse=True)

        stack_vals = sorted({row["stack"] for row in stacked_rows})

        row_by_gs: dict[tuple[str, str], dict[str, Any]] = {(row["dimension"], row["stack"]): row for row in stacked_rows}
        series: list[tuple[str, list[float]]] = []
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

    agents = list(dict.fromkeys(agent_key(r) for r in results))

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

    pivot: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"evaluated": 0, "vuln": 0})
    )
    for r in results:
        if _is_evaluated(r):
            dv = _heatmap_dim_value(r, dim)
            ak = agent_key(r)
            pivot[dv][ak]["evaluated"] += 1
            if _is_vulnerable(r):
                pivot[dv][ak]["vuln"] += 1

    if dim == "severity":
        all_dim_vals = [s for s in SEVERITY_ORDER if s in pivot]
        other_vals = sorted(v for v in pivot if v not in SEVERITY_ORDER)
        all_dim_vals = all_dim_vals + other_vals
    else:
        all_dim_vals = sorted(pivot.keys())

    if not all_dim_vals:
        return '<p class="rt-view-empty">No data for this dimension.</p>'

    cell_colors: list[list[str]] = []
    cell_texts: list[list[str]] = []
    grey = COLORS.get("sand_400", "#e4e2df")

    for dv in all_dim_vals:
        row_colors: list[str] = []
        row_texts: list[str] = []
        for ak in agents:
            counts = pivot[dv].get(ak, {"evaluated": 0, "vuln": 0})
            evaluated = counts["evaluated"]
            vuln = counts["vuln"]
            if evaluated == 0:
                row_colors.append(grey)
                row_texts.append("—")
            else:
                asr = vuln / evaluated
                color = scale_color(asr, ORQ_SCALE_HEAT)
                row_colors.append(color)
                row_texts.append(f"{asr * 100:.0f}%\nn={evaluated}")
        cell_colors.append(row_colors)
        cell_texts.append(row_texts)

    spec = vl_heatmap(
        x_labels=agents,
        y_labels=all_dim_vals,
        cell_colors=cell_colors,
        cell_texts=cell_texts,
    )
    return render_embed(spec, chart_id)
