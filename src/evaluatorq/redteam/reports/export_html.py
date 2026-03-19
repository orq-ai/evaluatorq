"""HTML renderer for red team reports.

``export_html(report)`` converts a ``RedTeamReport`` to a self-contained HTML
string with inline SVG charts (via Plotly + kaleido), styled tables, and
print-friendly CSS.  No JavaScript is required — charts are static SVGs.

If ``plotly`` or ``kaleido`` are not installed the report degrades gracefully
to a tables-only layout.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from evaluatorq.redteam.contracts import RedTeamReport
from evaluatorq.redteam.reports.sections import ReportSection, build_report_sections

# ---------------------------------------------------------------------------
# Brand colors (inline to avoid hard dependency on ui package at import time)
# ---------------------------------------------------------------------------

_COLORS = {
    "orange_300": "#ff8f34",
    "teal_400": "#025558",
    "teal_500": "#01483d",
    "ink_700": "#25232e",
    "ink_800": "#1a1921",
    "sand_100": "#f9f8f6",
    "sand_400": "#e4e2df",
    "success_400": "#2ebd85",
    "yellow_400": "#f2b600",
    "red_400": "#d92d20",
    "blue_400": "#4fd2ff",
}

_SEVERITY_COLORS = {
    "critical": _COLORS["red_400"],
    "high": _COLORS["orange_300"],
    "medium": _COLORS["yellow_400"],
    "low": _COLORS["success_400"],
}

_STATUS_COLORS = {
    "vulnerable": _COLORS["red_400"],
    "resistant": _COLORS["success_400"],
    "error": _COLORS["yellow_400"],
}

# Canonical order for severity labels in charts
_SEVERITY_ORDER = ["critical", "high", "medium", "low"]

# ---------------------------------------------------------------------------
# CSS (loaded from report.css, with color placeholders interpolated)
# ---------------------------------------------------------------------------

_CSS_PATH = Path(__file__).with_name("report.css")
_CSS: str | None = None


def _load_css() -> str:
    global _CSS
    if _CSS is None:
        _CSS = _CSS_PATH.read_text(encoding="utf-8") % _COLORS  # pyright: ignore[reportConstantRedefinition]
    return _CSS


# ---------------------------------------------------------------------------
# Chart rendering helpers
# ---------------------------------------------------------------------------


def _try_render_svg_chart(fig: Any) -> str | None:
    """Attempt to render a Plotly figure as inline SVG."""
    try:
        svg_bytes = fig.to_image(format="svg", engine="kaleido")
        return svg_bytes.decode("utf-8") if isinstance(svg_bytes, bytes) else svg_bytes
    except Exception:
        return None


def _charts_available() -> bool:
    """Check if plotly and kaleido are importable."""
    try:
        import plotly  # noqa: F401
        import kaleido  # noqa: F401
        return True
    except ImportError:
        return False


def _render_donut_chart(summary_data: dict[str, Any]) -> str:
    """Render a donut chart: resistant vs vulnerable vs errors."""
    if not _charts_available():
        return ""
    import plotly.graph_objects as go

    vuln = summary_data.get("vulnerabilities_found", 0)
    total = summary_data.get("total_attacks", 0)
    errors = summary_data.get("total_errors", 0)
    resistant = max(0, total - vuln - errors)

    if total == 0:
        return ""

    labels = ["Resistant", "Vulnerable", "Error"]
    values = [resistant, vuln, errors]
    colors = [_STATUS_COLORS["resistant"], _STATUS_COLORS["vulnerable"], _STATUS_COLORS["error"]]

    # Remove zero-value segments
    filtered = [(l, v, c) for l, v, c in zip(labels, values, colors) if v > 0]
    if not filtered:
        return ""
    labels, values, colors = zip(*filtered)

    fig = go.Figure(data=[go.Pie(
        labels=list(labels),
        values=list(values),
        hole=0.5,
        marker=dict(colors=list(colors)),
        textinfo="label+percent",
        textfont=dict(size=12),
    )])
    fig.update_layout(
        width=400, height=300,
        margin=dict(t=30, b=30, l=30, r=30),
        showlegend=False,
        title=dict(text="Overall Results", font=dict(size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    svg = _try_render_svg_chart(fig)
    return f'<div class="chart-container">{svg}</div>' if svg else ""


def _render_severity_bar_chart(by_severity: dict[str, Any]) -> str:
    """Render a vertical bar chart of vulnerability counts by severity level.

    Args:
        by_severity: Mapping of severity level -> summary object (or dict) with
            a ``vulnerabilities_found`` attribute/key.  Corresponds to
            ``ReportSummary.by_severity``.
    """
    if not _charts_available():
        return ""
    import plotly.graph_objects as go

    if not by_severity:
        return ""

    # Build ordered lists, keeping only severities that have vulnerabilities.
    labels: list[str] = []
    values: list[int] = []
    colors: list[str] = []
    for sev in _SEVERITY_ORDER:
        entry = by_severity.get(sev)
        if entry is None:
            continue
        # Support both Pydantic model instances and plain dicts.
        found = (
            entry.get("vulnerabilities_found", 0)
            if isinstance(entry, dict)
            else getattr(entry, "vulnerabilities_found", 0)
        )
        if found > 0:
            labels.append(sev)
            values.append(found)
            colors.append(_SEVERITY_COLORS.get(sev, "#999"))

    if not labels:
        return ""

    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=values,
        textposition="outside",
    )])
    fig.update_layout(
        width=450, height=300,
        margin=dict(t=40, b=40, l=50, r=30),
        title=dict(text="Vulnerabilities by Severity", font=dict(size=14)),
        xaxis_title="Severity",
        yaxis_title="Count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    svg = _try_render_svg_chart(fig)
    return f'<div class="chart-container">{svg}</div>' if svg else ""


def _render_category_bar_chart(rows: list[dict[str, Any]]) -> str:
    """Render a horizontal bar chart of per-category vulnerability rate."""
    if not _charts_available():
        return ""
    import plotly.graph_objects as go

    if not rows:
        return ""

    # Take top 10 by vulnerability rate
    sorted_rows = sorted(rows, key=lambda r: r.get("vulnerability_rate", 0), reverse=True)[:10]
    labels = [r.get("category", "?") for r in sorted_rows]
    rates = [r.get("vulnerability_rate", 0) * 100 for r in sorted_rows]

    fig = go.Figure(data=[go.Bar(
        y=labels,
        x=rates,
        orientation="h",
        marker_color=_COLORS["orange_300"],
        text=[f"{r:.0f}%" for r in rates],
        textposition="outside",
    )])
    fig.update_layout(
        width=500, height=max(250, len(labels) * 35 + 80),
        margin=dict(t=40, b=40, l=80, r=50),
        title=dict(text="Vulnerability Rate by Category", font=dict(size=14)),
        xaxis_title="Vulnerability Rate (%)",
        xaxis=dict(range=[0, max(max(rates) * 1.2, 5) if rates else 100]),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    svg = _try_render_svg_chart(fig)
    return f'<div class="chart-container">{svg}</div>' if svg else ""


def _render_technique_bar_chart(rows: list[dict[str, Any]]) -> str:
    """Render a horizontal bar chart of ASR% by technique.

    Mirrors the "ASR by Technique" chart shown in the Streamlit dashboard.
    Techniques are sorted by vulnerability rate (highest first) and capped at
    the top 15 to keep the chart readable.
    """
    if not _charts_available():
        return ""
    import plotly.graph_objects as go

    if not rows:
        return ""

    sorted_rows = sorted(rows, key=lambda r: r.get("vulnerability_rate", 0), reverse=True)[:15]
    labels = [r.get("technique", "?") for r in sorted_rows]
    rates = [r.get("vulnerability_rate", 0) * 100 for r in sorted_rows]
    totals = [r.get("total_attacks", 0) for r in sorted_rows]

    hover_texts = [
        f"{label}<br>ASR: {rate:.1f}%<br>Attacks: {total}"
        for label, rate, total in zip(labels, rates, totals)
    ]

    fig = go.Figure(data=[go.Bar(
        y=labels,
        x=rates,
        orientation="h",
        marker_color=_COLORS["orange_300"],
        text=[f"{r:.0f}%" for r in rates],
        textposition="outside",
        hovertext=hover_texts,
        hoverinfo="text",
    )])
    fig.update_layout(
        width=500, height=max(250, len(labels) * 35 + 80),
        margin=dict(t=40, b=40, l=140, r=60),
        title=dict(text="ASR by Technique", font=dict(size=14)),
        xaxis_title="Attack Success Rate (%)",
        xaxis=dict(range=[0, max(max(rates) * 1.2, 5) if rates else 100]),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    svg = _try_render_svg_chart(fig)
    return f'<div class="chart-container">{svg}</div>' if svg else ""


def _render_vulnerability_bar_chart(rows: list[dict[str, Any]]) -> str:
    """Render a horizontal bar chart of ASR% by vulnerability.

    Mirrors the "ASR by Vulnerability" chart shown in the Streamlit dashboard.
    Vulnerabilities are sorted by vulnerability rate (highest first) and capped
    at the top 15 to keep the chart readable.
    """
    if not _charts_available():
        return ""
    import plotly.graph_objects as go

    if not rows:
        return ""

    sorted_rows = sorted(rows, key=lambda r: r.get("vulnerability_rate", 0), reverse=True)[:15]
    # Prefer the human-readable name; fall back to the ID.
    labels = [r.get("vulnerability_name") or r.get("vulnerability", "?") for r in sorted_rows]
    rates = [r.get("vulnerability_rate", 0) * 100 for r in sorted_rows]
    totals = [r.get("total_attacks", 0) for r in sorted_rows]

    hover_texts = [
        f"{label}<br>ASR: {rate:.1f}%<br>Attacks: {total}"
        for label, rate, total in zip(labels, rates, totals)
    ]

    fig = go.Figure(data=[go.Bar(
        y=labels,
        x=rates,
        orientation="h",
        marker_color=_COLORS["orange_300"],
        text=[f"{r:.0f}%" for r in rates],
        textposition="outside",
        hovertext=hover_texts,
        hoverinfo="text",
    )])
    fig.update_layout(
        width=500, height=max(250, len(labels) * 35 + 80),
        margin=dict(t=40, b=40, l=160, r=60),
        title=dict(text="ASR by Vulnerability", font=dict(size=14)),
        xaxis_title="Attack Success Rate (%)",
        xaxis=dict(range=[0, max(max(rates) * 1.2, 5) if rates else 100]),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    svg = _try_render_svg_chart(fig)
    return f'<div class="chart-container">{svg}</div>' if svg else ""


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _esc(text: str) -> str:
    """HTML-escape text."""
    return html.escape(str(text))


def _pct(rate: float) -> str:
    return f"{rate:.0%}"


def _truncate(text: str, max_chars: int = 800) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[truncated — full text in report JSON]"


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    parts = ["<table>", "<thead><tr>"]
    for h in headers:
        parts.append(f"<th>{_esc(h)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for cell in row:
            parts.append(f"<td>{cell}</td>")  # cell may contain HTML (badges)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_header_html(data: dict[str, Any]) -> str:
    target = _esc(data.get("target", "unknown"))
    pipeline = _esc(data.get("pipeline", "unknown"))
    created_at = data.get("created_at")
    vulnerability_rate = data.get("vulnerability_rate", 0.0)
    date_str = created_at.strftime("%Y-%m-%d %H:%M UTC") if created_at else "unknown"

    return (
        f"<h1>Red Team Security Report</h1>\n"
        f'<p class="meta">'
        f"<strong>Target:</strong> {target} &nbsp;|&nbsp; "
        f"<strong>Mode:</strong> {pipeline} &nbsp;|&nbsp; "
        f"<strong>Date:</strong> {date_str} &nbsp;|&nbsp; "
        f"<strong>ASR:</strong> {_pct(vulnerability_rate)}"
        f"</p>\n"
    )


def _render_kpi_cards(data: dict[str, Any]) -> str:
    """Render a row of KPI metric cards for the executive summary."""
    asr = data.get("vulnerability_rate", 0.0)
    critical_exposure = data.get("critical_exposure", 0)
    eval_coverage = data.get("evaluation_coverage", 0.0)
    total_errors = data.get("total_errors", 0)
    total_attacks = data.get("total_attacks", 0)

    asr_class = "kpi-alert" if asr >= 0.5 else ("kpi-warn" if asr >= 0.2 else "")
    critical_class = "kpi-alert" if critical_exposure > 0 else ""
    errors_class = "kpi-warn" if total_errors > 0 else ""

    cards = [
        f'<div class="kpi-card {asr_class}">'
        f'<div class="kpi-value">{_pct(asr)}</div>'
        f'<div class="kpi-label">ASR</div>'
        f'<div class="kpi-subtitle">Target: 0%</div>'
        f"</div>",

        f'<div class="kpi-card {critical_class}">'
        f'<div class="kpi-value">{_esc(str(critical_exposure))}</div>'
        f'<div class="kpi-label">Critical Exposure</div>'
        f'<div class="kpi-subtitle">{"Requires attention" if critical_exposure > 0 else "Clear"}</div>'
        f"</div>",

        f'<div class="kpi-card">'
        f'<div class="kpi-value">{_pct(eval_coverage)}</div>'
        f'<div class="kpi-label">Eval Coverage</div>'
        f"</div>",

        f'<div class="kpi-card {errors_class}">'
        f'<div class="kpi-value">{_esc(str(total_errors))}</div>'
        f'<div class="kpi-label">Errors</div>'
        f"</div>",

        f'<div class="kpi-card">'
        f'<div class="kpi-value">{_esc(str(total_attacks))}</div>'
        f'<div class="kpi-label">Total Attacks</div>'
        f"</div>",
    ]

    return '<div class="kpi-row">' + "".join(cards) + "</div>"


def _render_summary_html(section: ReportSection) -> str:
    data = section.data
    rows = [
        ["Total Attacks", _esc(str(data.get("total_attacks", 0)))],
        ["Evaluated", _esc(str(data.get("evaluated_attacks", 0)))],
        ["Successful Attacks", _esc(str(data.get("vulnerabilities_found", 0)))],
        ["ASR", _pct(data.get("vulnerability_rate", 0.0))],
        ["Evaluation Coverage", _pct(data.get("evaluation_coverage", 0.0))],
    ]
    if data.get("total_errors"):
        rows.append(["Errors", _esc(str(data["total_errors"]))])
    if data.get("duration_seconds") is not None:
        mins, secs = divmod(int(data["duration_seconds"]), 60)
        rows.append(["Duration", f"{mins}m {secs}s"])

    kpi_cards = _render_kpi_cards(data)
    table = _html_table(["Metric", "Value"], rows)
    donut_chart = _render_donut_chart(data)
    severity_chart = _render_severity_bar_chart(data.get("by_severity", {}))
    return f"<h2>{_esc(section.title)}</h2>\n{kpi_cards}\n{donut_chart}\n{severity_chart}\n{table}"


def _render_focus_areas_html(section: ReportSection) -> str:
    focus_areas = section.data.get("focus_areas", [])
    if not focus_areas:
        return f"<h2>{_esc(section.title)}</h2>\n<p>No significant risk areas identified.</p>"

    parts = [f"<h2>{_esc(section.title)}</h2>"]
    for i, area in enumerate(focus_areas, start=1):
        cat = _esc(area["category"])
        cat_name = _esc(area.get("category_name", cat))
        vuln = area.get("vulnerabilities_found", 0)
        vuln_rate = _pct(area.get("vulnerability_rate", 0.0))
        risk_score = area.get("risk_score", 0.0)
        remediation = area.get("remediation", "")

        card_lines = [
            '<div class="card">',
            f"<h3>{i}. {cat} — {cat_name}</h3>",
            f"<p><strong>Vulnerabilities:</strong> {vuln} ({vuln_rate} of attacks succeeded) "
            f"&nbsp;|&nbsp; <strong>Risk Score:</strong> {risk_score:.2f}</p>",
        ]
        if remediation:
            card_lines.append(f"<p><strong>Remediation guidance:</strong></p>")
            card_lines.append(f"<blockquote>{_esc(remediation)}</blockquote>")

        agent_remediation = area.get("agent_specific_remediation", "")
        if agent_remediation:
            card_lines.append(
                f'<div style="margin-top:8px;padding:8px 12px;background:{_COLORS["sand_100"]};border-left:3px solid {_COLORS["orange_300"]};border-radius:4px">'
                f'<strong>Agent-specific:</strong> {_esc(agent_remediation)}</div>'
            )

        llm_rec = area.get("llm_recommendations")
        if llm_rec:
            recs = llm_rec.get("recommendations", [])
            patterns = llm_rec.get("patterns_observed", "")
            traces_analyzed = llm_rec.get("traces_analyzed", 0)
            if recs:
                card_lines.append(
                    f'<p><strong>Actionable recommendations</strong> (based on {traces_analyzed} trace samples):</p>'
                )
                card_lines.append('<ul class="recommendations">')
                for rec_item in recs:
                    card_lines.append(f"<li>{_esc(rec_item)}</li>")
                card_lines.append("</ul>")
            if patterns:
                card_lines.append(f"<p><em>Patterns observed:</em> {_esc(patterns)}</p>")

        card_lines.append("</div>")
        parts.append("\n".join(card_lines))

    return "\n".join(parts)


def _render_vulnerability_breakdown_html(section: ReportSection) -> str:
    """Render the per-vulnerability breakdown as an HTML table with a bar chart."""
    rows = section.data.get("rows", [])
    if not rows:
        return f"<h2>{_esc(section.title)}</h2>\n<p>No vulnerability data available.</p>"

    chart = _render_vulnerability_bar_chart(rows)
    table_rows = [
        [
            _esc(r["vulnerability_name"] or r["vulnerability"]),
            _esc(r.get("domain", "")),
            _esc(str(r["total_attacks"])),
            _esc(str(r["vulnerabilities_found"])),
            _pct(r["vulnerability_rate"]),
        ]
        for r in rows
    ]
    table = _html_table(
        ["Vulnerability", "Domain", "Attacks", "Hits", "ASR"],
        table_rows,
    )
    return f"<h2>{_esc(section.title)}</h2>\n{chart}\n{table}"


def _render_category_breakdown_html(section: ReportSection) -> str:
    rows = section.data.get("rows", [])
    if not rows:
        return f"<h2>{_esc(section.title)}</h2>\n<p>No category data available.</p>"

    chart = _render_category_bar_chart(rows)
    table_rows = [
        [
            _esc(f"{r['category']} — {r['category_name']}"),
            _esc(str(r["total_attacks"])),
            _esc(str(r["vulnerabilities_found"])),
            _pct(r["vulnerability_rate"]),
        ]
        for r in rows
    ]
    table = _html_table(
        ["Category", "Attacks", "Hits", "ASR"],
        table_rows,
    )
    return f"<h2>{_esc(section.title)}</h2>\n{chart}\n{table}"


def _render_technique_breakdown_html(section: ReportSection) -> str:
    rows = section.data.get("rows", [])
    if not rows:
        return f"<h2>{_esc(section.title)}</h2>\n<p>No technique data available.</p>"

    chart = _render_technique_bar_chart(rows)
    table_rows = [
        [
            _esc(r["technique"]),
            _esc(str(r["total_attacks"])),
            _esc(str(r["vulnerabilities_found"])),
            _pct(r["vulnerability_rate"]),
        ]
        for r in rows
    ]
    table = _html_table(
        ["Technique", "Attacks", "Hits", "ASR"],
        table_rows,
    )
    return f"<h2>{_esc(section.title)}</h2>\n{chart}\n{table}"


def _render_delivery_bar_chart(rows: list[dict[str, Any]]) -> str:
    """Render a horizontal bar chart of ASR% by delivery method."""
    if not _charts_available():
        return ""
    import plotly.graph_objects as go

    if not rows:
        return ""

    sorted_rows = sorted(rows, key=lambda r: r.get("vulnerability_rate", 0), reverse=True)
    labels = [r.get("delivery_method", "?") for r in sorted_rows]
    rates = [r.get("vulnerability_rate", 0) * 100 for r in sorted_rows]
    totals = [r.get("total_attacks", 0) for r in sorted_rows]

    hover_texts = [
        f"{label}<br>ASR: {rate:.1f}%<br>Attacks: {total}"
        for label, rate, total in zip(labels, rates, totals)
    ]

    fig = go.Figure(data=[go.Bar(
        y=labels,
        x=rates,
        orientation="h",
        marker_color=_COLORS["blue_400"],
        text=[f"{r:.0f}%" for r in rates],
        textposition="outside",
        hovertext=hover_texts,
        hoverinfo="text",
    )])
    fig.update_layout(
        width=500, height=max(250, len(labels) * 35 + 80),
        margin=dict(t=40, b=40, l=140, r=60),
        title=dict(text="ASR by Delivery Method", font=dict(size=14)),
        xaxis_title="Attack Success Rate (%)",
        xaxis=dict(range=[0, max(max(rates) * 1.2, 5) if rates else 100]),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    svg = _try_render_svg_chart(fig)
    return f'<div class="chart-container">{svg}</div>' if svg else ""


def _render_delivery_breakdown_html(section: ReportSection) -> str:
    """Render the per-delivery-method ASR breakdown as an HTML table with a bar chart."""
    rows = section.data.get("rows", [])
    if not rows:
        return f"<h2>{_esc(section.title)}</h2>\n<p>No delivery method data available.</p>"

    chart = _render_delivery_bar_chart(rows)
    table_rows = [
        [
            _esc(r["delivery_method"]),
            _esc(str(r["total_attacks"])),
            _esc(str(r["vulnerabilities_found"])),
            _pct(r["vulnerability_rate"]),
        ]
        for r in rows
    ]
    table = _html_table(
        ["Delivery Method", "Attacks", "Hits", "ASR"],
        table_rows,
    )
    return f"<h2>{_esc(section.title)}</h2>\n{chart}\n{table}"


def _render_error_bar_chart(errors_by_type: dict[str, int]) -> str:
    """Render a simple inline bar chart of error counts by type (no Plotly needed)."""
    if not errors_by_type:
        return ""

    total = sum(errors_by_type.values())
    if total == 0:
        return ""

    parts = ['<div style="margin:1rem 0">']
    for etype, count in sorted(errors_by_type.items(), key=lambda x: x[1], reverse=True):
        pct_width = min(100, int(count / total * 100))
        parts.append(
            f'<div class="error-bar-container">'
            f'<span class="error-bar-label">{_esc(etype)}</span>'
            f'<div class="error-bar-track">'
            f'<div class="error-bar-fill" style="width:{pct_width}%"></div>'
            f"</div>"
            f'<span class="error-bar-count">{_esc(str(count))}</span>'
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _render_error_analysis_html(section: ReportSection) -> str:
    """Render the error analysis section with metric cards, bar chart, and detail table."""
    data = section.data
    total_errors = data.get("total_errors", 0)
    if total_errors == 0:
        return f"<h2>{_esc(section.title)}</h2>\n<p>No errors recorded.</p>"

    error_rate = data.get("error_rate", 0.0)
    errors_by_type = data.get("errors_by_type", {})
    error_types_count = data.get("error_types_count", len(errors_by_type))
    detail_rows = data.get("detail_rows", [])

    # Metric cards row
    cards = (
        f'<div class="kpi-row">'
        f'<div class="kpi-card kpi-warn">'
        f'<div class="kpi-value">{_esc(str(total_errors))}</div>'
        f'<div class="kpi-label">Total Errors</div>'
        f"</div>"
        f'<div class="kpi-card">'
        f'<div class="kpi-value">{_pct(error_rate)}</div>'
        f'<div class="kpi-label">Error Rate</div>'
        f"</div>"
        f'<div class="kpi-card">'
        f'<div class="kpi-value">{_esc(str(error_types_count))}</div>'
        f'<div class="kpi-label">Error Types</div>'
        f"</div>"
        f"</div>"
    )

    bar_chart = _render_error_bar_chart(errors_by_type)

    # Detail table (capped at 100 rows to keep HTML manageable)
    detail_html = ""
    if detail_rows:
        capped = detail_rows[:100]
        table_rows = [
            [
                _esc(r["id"]),
                _esc(r["category"]),
                _esc(r["technique"]),
                _esc(r["error_type"]),
                _esc(r["stage"]),
                _esc(_truncate(r["error"], 200)),
            ]
            for r in capped
        ]
        detail_html = _html_table(
            ["ID", "Category", "Technique", "Error Type", "Stage", "Error"],
            table_rows,
        )
        if len(detail_rows) > 100:
            detail_html += f"<p><em>{len(detail_rows) - 100} more rows omitted.</em></p>"

    return (
        f"<h2>{_esc(section.title)}</h2>\n"
        f"{cards}\n"
        f"{bar_chart}\n"
        f"{detail_html}"
    )


def _render_attack_heatmap_html(section: ReportSection) -> str:
    """Render vulnerability × technique attack success rate as a heatmap table."""
    data = section.data
    vulnerabilities: list[str] = data.get("vulnerabilities", [])
    techniques: list[str] = data.get("techniques", [])
    cells: list[dict[str, Any]] = data.get("cells", [])

    if not vulnerabilities or not techniques or not cells:
        return f"<h2>{_esc(section.title)}</h2>\n<p>Insufficient data for heatmap.</p>"

    # Build lookup: (vuln, technique) -> cell
    cell_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for c in cells:
        key = (c["vulnerability"], c["technique"])
        cell_lookup[key] = c

    def _heatmap_color(asr: float, total: int) -> str:
        """Interpolate from success_400 (0%) to red_400 (100%)."""
        if total == 0:
            return "#e0e0e0"
        # success_400: #2ebd85, red_400: #d92d20 via yellow_400 #f2b600 at 50%
        if asr <= 0.5:
            t = asr / 0.5
            r = int(46 + (242 - 46) * t)
            g = int(189 + (182 - 189) * t)
            b = int(133 + (0 - 133) * t)
        else:
            t = (asr - 0.5) / 0.5
            r = int(242 + (217 - 242) * t)
            g = int(182 + (45 - 182) * t)
            b = int(0 + (32 - 0) * t)
        return f"#{r:02X}{g:02X}{b:02X}"

    # Build table header row (techniques)
    header_cells = ["<th>Vulnerability</th>"]
    for tech in techniques:
        header_cells.append(f"<th>{_esc(tech)}</th>")

    body_rows: list[str] = []
    for vuln in vulnerabilities:
        row_parts = [f"<td><strong>{_esc(vuln)}</strong></td>"]
        for tech in techniques:
            c = cell_lookup.get((vuln, tech))
            if c and c["total_attacks"] > 0:
                asr = c["vulnerability_rate"]
                color = _heatmap_color(asr, c["total_attacks"])
                label = _pct(asr)
                row_parts.append(
                    f'<td><span class="heatmap-cell" style="background:{color}">{_esc(label)}</span></td>'
                )
            else:
                row_parts.append('<td><span class="heatmap-cell" style="background:#e0e0e0;color:#999">—</span></td>')
        body_rows.append("<tr>" + "".join(row_parts) + "</tr>")

    table_html = (
        '<div style="overflow-x:auto"><table class="heatmap-table">'
        "<thead><tr>" + "".join(header_cells) + "</tr></thead>"
        "<tbody>" + "".join(body_rows) + "</tbody>"
        "</table></div>"
    )

    legend = (
        '<p style="font-size:.8em;color:#888;margin-top:.5rem">'
        "Cell color: green = low ASR (resistant), yellow = medium, red = high ASR (vulnerable). "
        "Grey = no attacks for that combination."
        "</p>"
    )

    return f"<h2>{_esc(section.title)}</h2>\n{table_html}\n{legend}"


def _render_individual_results_html(section: ReportSection) -> str:
    """Render individual attack result entries as collapsible details elements."""
    entries = section.data.get("entries", [])
    if not entries:
        return f"<h2>{_esc(section.title)}</h2>\n<p>No individual results available.</p>"

    parts = [f"<h2>{_esc(section.title)}</h2>"]
    parts.append(f"<p>{_esc(str(len(entries)))} total results</p>")

    for entry in entries:
        is_vulnerable = entry.get("vulnerable", False)
        has_error = bool(entry.get("error"))

        if has_error:
            status_badge = '<span class="badge badge-error">ERROR</span>'
        elif is_vulnerable:
            status_badge = '<span class="badge badge-vulnerable">VULNERABLE</span>'
        else:
            status_badge = '<span class="badge badge-resistant">RESISTANT</span>'

        severity = entry.get("severity", "low")
        sev_html = f'<span class="severity-{_esc(severity)}">{_esc(severity.upper())}</span>'

        vuln_name = _esc(entry.get("vulnerability") or entry.get("category", "unknown"))
        technique = _esc(entry.get("technique", ""))
        entry_id = _esc(entry.get("id", ""))

        summary_line = (
            f"{status_badge} &nbsp; {sev_html} &nbsp; "
            f"<strong>{vuln_name}</strong> &nbsp; "
            f"<span class=\"meta\">{technique}</span>"
            f'<span class="meta" style="float:right;font-size:.75em">{entry_id}</span>'
        )

        inner_parts: list[str] = []

        prompt = entry.get("prompt", "")
        if prompt:
            inner_parts.append(
                f"<p><strong>Prompt:</strong></p>"
                f"<pre>{_esc(_truncate(prompt, 600))}</pre>"
            )

        response = entry.get("response", "")
        if response:
            inner_parts.append(
                f"<p><strong>Response:</strong></p>"
                f"<pre>{_esc(_truncate(response, 600))}</pre>"
            )

        explanation = entry.get("explanation", "")
        if explanation:
            inner_parts.append(
                f"<p><strong>Evaluation:</strong> {_esc(_truncate(explanation, 400))}</p>"
            )

        error = entry.get("error", "")
        if error:
            inner_parts.append(
                f"<p><strong>Error:</strong> {_esc(_truncate(error, 300))}</p>"
            )

        inner_html = "".join(inner_parts)

        parts.append(
            f"<details>"
            f"<summary>{summary_line}</summary>"
            f'<div style="padding:.5rem 0 .5rem 1rem">{inner_html}</div>'
            f"</details>"
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Phase 2 section renderers
# ---------------------------------------------------------------------------


def _render_agent_context_html(section: ReportSection) -> str:
    """Render agent context capability cards."""
    agents = section.data.get("agents", [])
    if not agents:
        return f"<h2>{_esc(section.title)}</h2>\n<p>No agent information available.</p>"

    parts = [f"<h2>{_esc(section.title)}</h2>", '<div class="agent-cards-grid">']
    for agent in agents:
        display_name = _esc(agent.get("display_name") or agent.get("key", "unknown"))
        model = _esc(agent.get("model", ""))
        description = _esc(agent.get("description", ""))
        tools: list[str] = agent.get("tools", [])
        memory_stores: list[str] = agent.get("memory_stores", [])
        knowledge_bases: list[str] = agent.get("knowledge_bases", [])

        card_lines = [
            '<div class="card">',
            f"<h3>{display_name}</h3>",
        ]
        if model:
            card_lines.append(f'<p class="meta"><strong>Model:</strong> {model}</p>')
        if description:
            card_lines.append(f"<p>{description}</p>")

        # Capability chips
        chip_groups: list[str] = []
        if tools:
            tool_chips = "".join(
                f'<span style="display:inline-block;background:#e8f4f8;border:1px solid #b3d9ea;'
                f'border-radius:12px;padding:.15rem .6rem;font-size:.8em;margin:.15rem .15rem .15rem 0">'
                f"{_esc(t)}</span>"
                for t in tools
            )
            chip_groups.append(f"<p><strong>Tools:</strong> {tool_chips}</p>")
        if memory_stores:
            mem_chips = "".join(
                f'<span style="display:inline-block;background:#f0f8e8;border:1px solid #c3e0a0;'
                f'border-radius:12px;padding:.15rem .6rem;font-size:.8em;margin:.15rem .15rem .15rem 0">'
                f"{_esc(m)}</span>"
                for m in memory_stores
            )
            chip_groups.append(f"<p><strong>Memory:</strong> {mem_chips}</p>")
        if knowledge_bases:
            kb_chips = "".join(
                f'<span style="display:inline-block;background:#fdf0e8;border:1px solid #f0c080;'
                f'border-radius:12px;padding:.15rem .6rem;font-size:.8em;margin:.15rem .15rem .15rem 0">'
                f"{_esc(k)}</span>"
                for k in knowledge_bases
            )
            chip_groups.append(f"<p><strong>Knowledge Bases:</strong> {kb_chips}</p>")

        card_lines.extend(chip_groups)
        card_lines.append("</div>")
        parts.append("\n".join(card_lines))

    parts.append("</div>")  # close agent-cards-grid
    return "\n".join(parts)


def _render_mini_donut_chart(label: str, data: dict[str, Any]) -> str:
    """Render a small donut chart for a single turn-type or domain group."""
    if not _charts_available():
        return ""
    import plotly.graph_objects as go

    total = data.get("total_attacks", 0)
    vuln = data.get("vulnerabilities_found", 0)
    if total == 0:
        return ""

    resistant = max(0, total - vuln)
    filtered = [
        (lbl, val, col)
        for lbl, val, col in [
            ("Resistant", resistant, _STATUS_COLORS["resistant"]),
            ("Vulnerable", vuln, _STATUS_COLORS["vulnerable"]),
        ]
        if val > 0
    ]
    if not filtered:
        return ""
    lbls, vals, cols = zip(*filtered)

    fig = go.Figure(data=[go.Pie(
        labels=list(lbls),
        values=list(vals),
        hole=0.55,
        marker=dict(colors=list(cols)),
        textinfo="percent",
        textfont=dict(size=11),
    )])
    asr_pct = f"{vuln / total:.0%}" if total > 0 else "0%"
    fig.update_layout(
        width=250, height=220,
        margin=dict(t=35, b=20, l=10, r=10),
        showlegend=True,
        legend=dict(orientation="h", y=-0.05, font=dict(size=10)),
        title=dict(text=f"{label}<br><sup>ASR: {asr_pct}</sup>", font=dict(size=12)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    svg = _try_render_svg_chart(fig)
    return f'<div class="chart-container" style="display:inline-block;vertical-align:top">{svg}</div>' if svg else ""


def _render_turn_scope_breakdown_html(section: ReportSection) -> str:
    """Render turn-type and domain breakdown as side-by-side mini donut charts and tables."""
    by_turn_type: dict[str, Any] = section.data.get("by_turn_type", {})
    by_domain: dict[str, Any] = section.data.get("by_domain", {})

    if not by_turn_type and not by_domain:
        return f"<h2>{_esc(section.title)}</h2>\n<p>No turn type or domain data available.</p>"

    parts = [f"<h2>{_esc(section.title)}</h2>"]

    charts_html = ""
    for group_label, group_data in by_turn_type.items():
        charts_html += _render_mini_donut_chart(group_label, group_data)
    for group_label, group_data in by_domain.items():
        charts_html += _render_mini_donut_chart(group_label, group_data)

    if charts_html:
        parts.append(f'<div style="display:flex;flex-wrap:wrap;gap:1rem;align-items:flex-start">{charts_html}</div>')

    # Turn type table
    if by_turn_type:
        tt_rows = [
            [
                _esc(turn_type),
                _esc(str(d.get("total_attacks", 0))),
                _esc(str(d.get("vulnerabilities_found", 0))),
                _pct(d.get("vulnerability_rate", 0.0)),
            ]
            for turn_type, d in by_turn_type.items()
        ]
        parts.append("<h3>By Turn Type</h3>")
        parts.append(_html_table(["Turn Type", "Attacks", "Hits", "ASR"], tt_rows))

    # Domain table
    if by_domain:
        domain_rows = [
            [
                _esc(domain),
                _esc(str(d.get("total_attacks", 0))),
                _esc(str(d.get("vulnerabilities_found", 0))),
                _pct(d.get("vulnerability_rate", 0.0)),
            ]
            for domain, d in by_domain.items()
        ]
        parts.append("<h3>By Domain</h3>")
        parts.append(_html_table(["Domain", "Attacks", "Hits", "ASR"], domain_rows))

    return "\n".join(parts)


def _render_turn_depth_bar_chart(rows: list[dict[str, Any]]) -> str:
    """Render a vertical bar chart of ASR% by turn count."""
    if not _charts_available() or not rows:
        return ""
    import plotly.graph_objects as go

    turn_counts = [str(r["turn_count"]) for r in rows]
    asr_pcts = [r.get("vulnerability_rate", 0.0) * 100 for r in rows]
    totals = [r.get("total_attacks", 0) for r in rows]

    bar_colors = [
        _COLORS["red_400"] if asr >= 50 else (_COLORS["yellow_400"] if asr >= 20 else _COLORS["success_400"])
        for asr in asr_pcts
    ]

    fig = go.Figure(data=[go.Bar(
        x=turn_counts,
        y=asr_pcts,
        marker_color=bar_colors,
        text=[f"{v:.0f}%" for v in asr_pcts],
        textposition="outside",
        customdata=totals,
        hovertemplate="Turn %{x}<br>ASR: %{y:.1f}%<br>Attacks: %{customdata}<extra></extra>",
    )])
    fig.update_layout(
        width=500, height=320,
        margin=dict(t=50, b=50, l=60, r=30),
        title=dict(text="ASR% by Conversation Turn Count", font=dict(size=14)),
        xaxis_title="Number of Turns",
        yaxis_title="Attack Success Rate (%)",
        yaxis=dict(range=[0, max(max(asr_pcts) * 1.2, 10) if asr_pcts else 100]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    svg = _try_render_svg_chart(fig)
    return f'<div class="chart-container">{svg}</div>' if svg else ""


def _render_turn_depth_analysis_html(section: ReportSection) -> str:
    """Render multi-turn depth analysis with a bar chart and table."""
    rows = section.data.get("rows", [])
    if not rows:
        return f"<h2>{_esc(section.title)}</h2>\n<p>No multi-turn data available.</p>"

    chart = _render_turn_depth_bar_chart(rows)
    table_rows = [
        [
            _esc(str(r["turn_count"])),
            _esc(str(r["total_attacks"])),
            _esc(str(r["vulnerabilities_found"])),
            _pct(r["vulnerability_rate"]),
        ]
        for r in rows
    ]
    table = _html_table(["Turn Count", "Attacks", "Hits", "ASR"], table_rows)
    return f"<h2>{_esc(section.title)}</h2>\n{chart}\n{table}"


def _render_token_per_agent_bar_chart(per_agent: list[dict[str, Any]]) -> str:
    """Render a horizontal bar chart of total tokens per agent."""
    if not _charts_available() or not per_agent:
        return ""
    import plotly.graph_objects as go

    labels = [r.get("agent", "?") for r in per_agent]
    values = [r.get("total_tokens", 0) for r in per_agent]

    if not any(values):
        return ""

    fig = go.Figure(data=[go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker_color=_COLORS["teal_400"],
        text=values,
        textposition="outside",
    )])
    fig.update_layout(
        width=500, height=max(200, len(labels) * 40 + 80),
        margin=dict(t=40, b=40, l=120, r=60),
        title=dict(text="Total Tokens per Agent", font=dict(size=14)),
        xaxis_title="Total Tokens",
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    svg = _try_render_svg_chart(fig)
    return f'<div class="chart-container">{svg}</div>' if svg else ""


def _render_token_usage_html(section: ReportSection) -> str:
    """Render token usage summary as metric cards and optional bar chart."""
    data = section.data
    overall: dict[str, Any] = data.get("overall", {})
    per_agent: list[dict[str, Any]] = data.get("per_agent", [])

    if not overall and not per_agent:
        return f"<h2>{_esc(section.title)}</h2>\n<p>No token usage data available.</p>"

    parts = [f"<h2>{_esc(section.title)}</h2>"]

    if overall:
        total_tokens = overall.get("total_tokens", 0)
        prompt_tokens = overall.get("prompt_tokens", 0)
        completion_tokens = overall.get("completion_tokens", 0)
        calls = overall.get("calls", 0)

        cards = (
            f'<div class="kpi-row">'
            f'<div class="kpi-card">'
            f'<div class="kpi-value">{_esc(str(total_tokens))}</div>'
            f'<div class="kpi-label">Total Tokens</div>'
            f"</div>"
            f'<div class="kpi-card">'
            f'<div class="kpi-value">{_esc(str(prompt_tokens))}</div>'
            f'<div class="kpi-label">Prompt Tokens</div>'
            f"</div>"
            f'<div class="kpi-card">'
            f'<div class="kpi-value">{_esc(str(completion_tokens))}</div>'
            f'<div class="kpi-label">Completion Tokens</div>'
            f"</div>"
            f'<div class="kpi-card">'
            f'<div class="kpi-value">{_esc(str(calls))}</div>'
            f'<div class="kpi-label">API Calls</div>'
            f"</div>"
            f"</div>"
        )
        parts.append(cards)

    if per_agent:
        chart = _render_token_per_agent_bar_chart(per_agent)
        if chart:
            parts.append(chart)
        table_rows = [
            [
                _esc(r.get("agent", "?")),
                _esc(str(r.get("total_tokens", 0))),
                _esc(str(r.get("prompt_tokens", 0))),
                _esc(str(r.get("completion_tokens", 0))),
                _esc(str(r.get("calls", 0))),
            ]
            for r in per_agent
        ]
        table = _html_table(
            ["Agent", "Total Tokens", "Prompt Tokens", "Completion Tokens", "API Calls"],
            table_rows,
        )
        parts.append("<h3>Per-Agent Breakdown</h3>")
        parts.append(table)

    return "\n".join(parts)


def _render_source_donut_chart(rows: list[dict[str, Any]]) -> str:
    """Render a donut chart of attack source distribution."""
    if not _charts_available() or not rows:
        return ""
    import plotly.graph_objects as go

    labels = [r["source"] for r in rows]
    values = [r["count"] for r in rows]

    # Use a qualitative palette based on brand colors
    palette = [
        _COLORS["teal_400"], _COLORS["orange_300"], _COLORS["blue_400"],
        _COLORS["success_400"], _COLORS["yellow_400"], _COLORS["red_400"],
    ]
    colors = [palette[i % len(palette)] for i in range(len(labels))]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        marker=dict(colors=colors),
        textinfo="label+percent",
        textfont=dict(size=11),
    )])
    fig.update_layout(
        width=450, height=320,
        margin=dict(t=40, b=30, l=30, r=30),
        showlegend=False,
        title=dict(text="Attack Sources", font=dict(size=14)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    svg = _try_render_svg_chart(fig)
    return f'<div class="chart-container">{svg}</div>' if svg else ""


def _render_source_distribution_html(section: ReportSection) -> str:
    """Render attack source distribution as a donut chart and table."""
    rows = section.data.get("rows", [])
    if not rows:
        return f"<h2>{_esc(section.title)}</h2>\n<p>No source data available.</p>"

    chart = _render_source_donut_chart(rows)
    table_rows = [
        [_esc(r["source"]), _esc(str(r["count"]))]
        for r in rows
    ]
    table = _html_table(["Source", "Count"], table_rows)
    return f"<h2>{_esc(section.title)}</h2>\n{chart}\n{table}"


def _asr_badge_color(asr: float) -> str:
    """Return an inline background color interpolated from green (0%) to red (100%)."""
    asr = max(0.0, min(1.0, asr))  # clamp to [0, 1]
    # Linear interpolation: green (#2ebd85) at 0%, yellow (#f2b600) at 50%, red (#d92d20) at 100%
    if asr <= 0.5:
        t = asr / 0.5
        r = int(46 + (242 - 46) * t)
        g = int(189 + (182 - 189) * t)
        b = int(133 + (0 - 133) * t)
    else:
        t = (asr - 0.5) / 0.5
        r = int(242 + (217 - 242) * t)
        g = int(182 + (45 - 182) * t)
        b = int(0 + (32 - 0) * t)
    return f"#{r:02X}{g:02X}{b:02X}"


# ---------------------------------------------------------------------------
# Phase 3 chart helpers
# ---------------------------------------------------------------------------


def _asr_cell_color(asr_pct: float) -> str:
    """Interpolate a background color for an ASR percentage cell.

    0%   -> green  (#2ebd85)
    50%  -> yellow (#f2b600)
    100% -> red    (#d92d20)
    """
    t = max(0.0, min(100.0, asr_pct)) / 100.0

    def _lerp(a: int, b: int, factor: float) -> int:
        return int(round(a + (b - a) * factor))

    if t <= 0.5:
        n = t * 2
        r = _lerp(0x2E, 0xF2, n)
        g = _lerp(0xBD, 0xB6, n)
        b = _lerp(0x85, 0x00, n)
    else:
        n = (t - 0.5) * 2
        r = _lerp(0xF2, 0xD9, n)
        g = _lerp(0xB6, 0x2D, n)
        b = _lerp(0x00, 0x20, n)

    return f"#{r:02x}{g:02x}{b:02x}"


def _render_agent_comparison_grouped_bar(
    vuln_asr_rows: list[dict[str, Any]],
    agents: list[str],
) -> str:
    """Render a grouped horizontal bar chart of ASR by vulnerability per agent."""
    if not _charts_available() or not vuln_asr_rows or not agents:
        return ""
    import plotly.graph_objects as go

    _qualitative = [
        _COLORS["teal_400"],
        _COLORS["orange_300"],
        _COLORS["blue_400"],
        _COLORS["yellow_400"],
        _COLORS["red_400"],
        _COLORS["success_400"],
    ]

    rows = vuln_asr_rows[:15]
    vuln_labels = [r["vulnerability"] for r in rows]

    traces = []
    for i, agent_name in enumerate(agents):
        asr_values = [r["agents"].get(agent_name, {}).get("asr", 0.0) * 100 for r in rows]
        traces.append(go.Bar(
            name=agent_name,
            y=vuln_labels,
            x=asr_values,
            orientation="h",
            marker_color=_qualitative[i % len(_qualitative)],
            text=[f"{v:.0f}%" for v in asr_values],
            textposition="outside",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode="group",
        width=600,
        height=max(300, len(rows) * len(agents) * 20 + 100),
        margin=dict(t=50, b=40, l=180, r=60),
        title=dict(text="ASR by Vulnerability per Agent", font=dict(size=14)),
        xaxis_title="Attack Success Rate (%)",
        xaxis=dict(range=[0, 110]),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    svg = _try_render_svg_chart(fig)
    return f'<div class="chart-container">{svg}</div>' if svg else ""


def _render_framework_bar_chart(rows: list[dict[str, Any]]) -> str:
    """Render a vertical bar chart of ASR% by framework."""
    if not _charts_available() or not rows:
        return ""
    import plotly.graph_objects as go

    labels = [r["framework"] for r in rows]
    rates = [r["vulnerability_rate"] * 100 for r in rows]
    totals = [r["total_attacks"] for r in rows]

    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=rates,
        marker_color=_COLORS["teal_400"],
        text=[f"{rate:.1f}%<br>n={n}" for rate, n in zip(rates, totals)],
        textposition="outside",
    )])
    fig.update_layout(
        width=500,
        height=350,
        margin=dict(t=50, b=50, l=60, r=30),
        title=dict(text="ASR by Framework", font=dict(size=14)),
        xaxis_title="Framework",
        yaxis_title="ASR (%)",
        yaxis=dict(range=[0, max(max(rates) * 1.25, 10) if rates else 100]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    svg = _try_render_svg_chart(fig)
    return f'<div class="chart-container">{svg}</div>' if svg else ""


# ---------------------------------------------------------------------------
# Phase 3 section renderers
# ---------------------------------------------------------------------------


def _render_agent_comparison_html(section: ReportSection) -> str:
    """Render the multi-agent comparison section.

    Produces:
    - KPI cards (one per agent) showing total attacks, vulnerabilities, and ASR.
    - An HTML heatmap table (vulnerability x agent, color-coded by ASR%).
    - A grouped bar chart of ASR by vulnerability per agent (Plotly SVG when
      available, plain table fallback otherwise).
    """
    data = section.data
    agents: list[str] = data.get("agents", [])
    agent_metrics: list[dict[str, Any]] = data.get("agent_metrics", [])
    vuln_asr_rows: list[dict[str, Any]] = data.get("vuln_asr_rows", [])
    heatmap: dict[str, Any] = data.get("heatmap", {})

    if len(agents) < 2:
        return f"<h2>{_esc(section.title)}</h2>\n<p>Agent comparison requires at least 2 agents.</p>"

    parts = [f"<h2>{_esc(section.title)}</h2>"]

    # Per-agent metric cards (side by side)
    parts.append('<div class="kpi-row">')
    for am in agent_metrics:
        agent_name = _esc(am["agent"])
        asr = am["asr"]
        asr_class = "kpi-alert" if asr >= 0.5 else ("kpi-warn" if asr >= 0.2 else "")
        parts.append(
            f'<div class="kpi-card {asr_class}" style="flex:1 1 160px;">'
            f'<div class="kpi-label" style="font-size:.9em;color:var(--teal-400);">{agent_name}</div>'
            f'<div class="kpi-value">{_pct(asr)}</div>'
            f'<div class="kpi-subtitle">'
            f'{_esc(str(am["vulnerabilities_found"]))} vulns / {_esc(str(am["total_attacks"]))} attacks'
            f"</div>"
            f"</div>"
        )
    parts.append("</div>")

    # Heatmap table: rows=vulnerabilities, cols=agents
    vuln_list: list[str] = heatmap.get("vulnerabilities", [])
    agent_list: list[str] = heatmap.get("agents", [])
    z_matrix: list[list[float]] = heatmap.get("z_matrix", [])
    text_matrix: list[list[str]] = heatmap.get("text_matrix", [])

    if vuln_list and agent_list and z_matrix:
        parts.append("<h3>ASR Heatmap (Vulnerability \u00d7 Agent)</h3>")
        tbl_parts = ['<div style="overflow-x:auto"><table class="heatmap-table">']
        tbl_parts.append("<thead><tr>")
        tbl_parts.append("<th>Vulnerability</th>")
        for ag in agent_list:
            tbl_parts.append(f"<th>{_esc(ag)}</th>")
        tbl_parts.append("</tr></thead><tbody>")
        for row_idx, vuln in enumerate(vuln_list):
            tbl_parts.append("<tr>")
            tbl_parts.append(f"<td><strong>{_esc(vuln)}</strong></td>")
            for col_idx in range(len(agent_list)):
                try:
                    asr_pct = z_matrix[row_idx][col_idx]
                    cell_text = text_matrix[row_idx][col_idx] if text_matrix else f"{asr_pct:.0f}%"
                except IndexError:
                    asr_pct = 0.0
                    cell_text = "\u2014"
                bg = _asr_cell_color(asr_pct)
                text_color = "white" if asr_pct >= 30 else _COLORS["ink_700"]
                tbl_parts.append(
                    f'<td style="text-align:center;">'
                    f'<span class="heatmap-cell" style="background:{bg};color:{text_color};">'
                    f"{_esc(cell_text)}"
                    f"</span></td>"
                )
            tbl_parts.append("</tr>")
        tbl_parts.append("</tbody></table></div>")
        parts.append("\n".join(tbl_parts))

    # Grouped bar chart: ASR by vulnerability per agent (with table fallback)
    if vuln_asr_rows:
        chart = _render_agent_comparison_grouped_bar(vuln_asr_rows, agents)
        if chart:
            parts.append(chart)
        else:
            table_headers = ["Vulnerability"] + agents
            table_rows_data: list[list[str]] = []
            for vrow in vuln_asr_rows[:20]:
                row_cells = [_esc(vrow["vulnerability"])]
                for ag in agents:
                    ag_data = vrow["agents"].get(ag, {"asr": 0.0, "total": 0})
                    row_cells.append(f"{ag_data['asr']:.0%} (n={ag_data['total']})")
                table_rows_data.append(row_cells)
            parts.append(_html_table(table_headers, table_rows_data))

    return "\n".join(parts)


def _render_agent_disagreements_html(section: ReportSection) -> str:
    """Render the agent disagreement viewer as collapsible <details> entries.

    Each entry shows side-by-side agent responses for attacks where agents
    produced different verdicts (one vulnerable, one resistant).
    """
    data = section.data
    agents: list[str] = data.get("agents", [])
    total: int = data.get("total_disagreements", 0)
    disagreements: list[dict[str, Any]] = data.get("disagreements", [])

    if not disagreements:
        agents_str = ", ".join(_esc(a) for a in agents)
        return (
            f"<h2>{_esc(section.title)}</h2>\n"
            f"<p>No disagreements found between agents ({agents_str}).</p>"
        )

    parts = [
        f"<h2>{_esc(section.title)}</h2>",
        f"<p>Found <strong>{_esc(str(total))}</strong> attack(s) where agents disagreed "
        f"(one found vulnerable, another resistant).</p>",
    ]

    for i, dis in enumerate(disagreements, start=1):
        attack_id = _esc(dis.get("attack_id", ""))
        vulnerability = _esc(dis.get("vulnerability", ""))
        technique = _esc(dis.get("technique", ""))
        severity = dis.get("severity", "")
        prompt_snippet = dis.get("prompt_snippet", "")
        per_agent: list[dict[str, Any]] = dis.get("per_agent", [])

        sev_cls = f"severity-{_esc(severity)}" if severity else ""
        summary_label = (
            f"#{i} &nbsp; {attack_id} &mdash; {vulnerability} / {technique}"
            + (f" &nbsp; <span class=\"{sev_cls}\">[{_esc(severity)}]</span>" if severity else "")
        )

        # Build per-agent side-by-side response panels
        panels: list[str] = []
        for agent_entry in per_agent:
            agent_name = _esc(agent_entry["agent"])
            is_vulnerable = agent_entry["vulnerable"]
            verdict_cls = "badge-vulnerable" if is_vulnerable else "badge-resistant"
            verdict_label = "VULNERABLE" if is_vulnerable else "RESISTANT"
            explanation = agent_entry.get("explanation", "")
            response_snippet = agent_entry.get("response_snippet", "")

            panel_parts = [
                '<div style="flex:1 1 45%;min-width:180px;">',
                f'<strong>{agent_name}</strong> '
                f'<span class="badge {verdict_cls}">{verdict_label}</span>',
            ]
            if explanation:
                panel_parts.append(f"<p><em>{_esc(explanation)}</em></p>")
            if response_snippet:
                panel_parts.append(f"<pre>{_esc(_truncate(response_snippet, 500))}</pre>")
            panel_parts.append("</div>")
            panels.append("\n".join(panel_parts))

        panels_html = (
            '<div style="display:flex;gap:1rem;flex-wrap:wrap;margin-top:.5rem;">'
            + "".join(panels)
            + "</div>"
        )

        prompt_block = ""
        if prompt_snippet:
            prompt_block = (
                f"<p><strong>Attack prompt:</strong></p>"
                f"<pre>{_esc(_truncate(prompt_snippet, 300))}</pre>"
            )

        parts.append(
            f"<details>"
            f"<summary>{summary_label}</summary>"
            f'<div style="padding:.5rem 0 .5rem 1rem;">'
            f"{prompt_block}"
            f"{panels_html}"
            f"</div>"
            f"</details>"
        )

    return "\n".join(parts)


def _render_framework_breakdown_html(section: ReportSection) -> str:
    """Render the per-framework breakdown as a bar chart and summary table."""
    rows = section.data.get("rows", [])
    if not rows:
        return f"<h2>{_esc(section.title)}</h2>\n<p>No framework data available.</p>"

    chart = _render_framework_bar_chart(rows)
    table_rows = [
        [
            _esc(r["framework"]),
            _esc(str(r["total_attacks"])),
            _esc(str(r["vulnerabilities_found"])),
            _pct(r["vulnerability_rate"]),
            _pct(r["resistance_rate"]),
        ]
        for r in rows
    ]
    table = _html_table(
        ["Framework", "Attacks", "Hits", "ASR", "Resistance Rate"],
        table_rows,
    )
    return f"<h2>{_esc(section.title)}</h2>\n{chart}\n{table}"


# ---------------------------------------------------------------------------
# Section dispatch
# ---------------------------------------------------------------------------

def _render_severity_definitions_html(section: ReportSection) -> str:
    """Render the severity level definitions as an HTML table."""
    definitions = section.data.get("definitions", [])
    weights = section.data.get("weights", {})
    if not definitions:
        return ""

    table_rows = []
    for d in definitions:
        level = d["level"]
        sev_cls = f"severity-{level}"
        weight = weights.get(level, "")
        table_rows.append([
            f'<span class="{sev_cls}">{_esc(level.capitalize())}</span>',
            _esc(d["description"]),
            str(weight),
        ])

    table = _html_table(
        ["Severity", "Description", "Risk Weight"],
        table_rows,
    )
    return f"<h2>{_esc(section.title)}</h2>\n{table}"


def _render_methodology_html(section: ReportSection) -> str:
    """Render methodology disclosure section as HTML."""
    data = section.data
    rows = []

    pipeline = data.get("pipeline", "unknown")
    scoring = data.get("scoring_method", "llm-as-judge")
    rows.append(("Assessment Type", pipeline.capitalize()))
    rows.append(("Scoring Method", scoring))

    evaluator_model = data.get("evaluator_model")
    if evaluator_model:
        rows.append(("Evaluator Model", evaluator_model))
    attack_model = data.get("attack_model")
    if attack_model:
        rows.append(("Attack Model", attack_model))
    dataset_source = data.get("dataset_source")
    if dataset_source:
        rows.append(("Dataset Source", dataset_source))
    framework = data.get("framework")
    if framework:
        rows.append(("Framework", framework))

    categories = data.get("categories_tested") or []
    vulnerabilities = data.get("vulnerabilities_tested") or []
    total_attacks = data.get("total_attacks", 0)
    coverage = data.get("evaluation_coverage", 0.0)
    max_turns = data.get("max_turns")
    duration = data.get("duration_seconds")

    rows.append(("Categories Tested", str(len(categories))))
    if vulnerabilities:
        rows.append(("Vulnerabilities Tested", str(len(vulnerabilities))))
    rows.append(("Total Attacks", str(total_attacks)))
    rows.append(("Evaluation Coverage", f"{coverage:.0%}"))
    if max_turns:
        rows.append(("Max Conversation Turns", str(max_turns)))
    if duration is not None:
        mins, secs = divmod(int(duration), 60)
        rows.append(("Duration", f"{mins}m {secs}s"))

    agents = data.get("tested_agents", [])
    if agents:
        rows.append(("Agents Tested", ", ".join(agents)))

    table_html = "\n".join(
        f'<tr><td style="font-weight:600;padding:6px 12px;white-space:nowrap">{_esc(k)}</td>'
        f'<td style="padding:6px 12px">{_esc(v)}</td></tr>'
        for k, v in rows
    )

    limitations = data.get("scope_limitations", [])
    limitations_html = ""
    if limitations:
        items = "".join(f"<li>{_esc(lim)}</li>" for lim in limitations)
        limitations_html = f'<h4 style="margin-top:16px">Known Limitations</h4><ul>{items}</ul>'

    untested = data.get("untested_categories", [])
    untested_names = data.get("untested_category_names", {})
    untested_html = ""
    if untested:
        items = "".join(
            f"<li><strong>{_esc(cat)}</strong> — {_esc(untested_names.get(cat, cat))}</li>"
            for cat in untested
        )
        untested_html = (
            f'<div style="margin-top:16px;padding:12px;background:#fff3cd;border-left:3px solid #ffc107;border-radius:4px">'
            f'<strong>{len(untested)} supported categories not tested:</strong>'
            f'<ul style="margin:8px 0 0 0">{items}</ul></div>'
        )

    return (
        f'<section id="methodology"><h2>{_esc(section.title)}</h2>'
        f'<table style="border-collapse:collapse;width:100%;max-width:600px">'
        f'{table_html}</table>'
        f'{limitations_html}'
        f'{untested_html}</section>'
    )


_SECTION_RENDERERS = {
    "summary": _render_summary_html,
    "methodology": _render_methodology_html,
    "severity_definitions": _render_severity_definitions_html,
    "focus_areas": _render_focus_areas_html,
    "vulnerability_breakdown": _render_vulnerability_breakdown_html,
    "category_breakdown": _render_category_breakdown_html,
    "technique_breakdown": _render_technique_breakdown_html,
    "delivery_breakdown": _render_delivery_breakdown_html,
    "error_analysis": _render_error_analysis_html,
    "attack_heatmap": _render_attack_heatmap_html,
    "individual_results": _render_individual_results_html,
    # Phase 2 sections
    "agent_context": _render_agent_context_html,
    "turn_scope_breakdown": _render_turn_scope_breakdown_html,
    "turn_depth_analysis": _render_turn_depth_analysis_html,
    "token_usage": _render_token_usage_html,
    "source_distribution": _render_source_distribution_html,
    # Phase 3 sections
    "agent_comparison": _render_agent_comparison_html,
    "agent_disagreements": _render_agent_disagreements_html,
    "framework_breakdown": _render_framework_breakdown_html,
}


# ---------------------------------------------------------------------------
# Risk verdict & table of contents
# ---------------------------------------------------------------------------


def _risk_level(asr: float, critical_count: int) -> tuple[str, str]:
    """Derive a qualitative risk level from ASR and critical findings.

    Returns (level, css_class) where level is a human-readable label.
    """
    if critical_count > 0 or asr >= 0.5:
        return "Critical Risk", "risk-critical"
    if asr >= 0.25:
        return "High Risk", "risk-high"
    if asr >= 0.10:
        return "Medium Risk", "risk-medium"
    return "Low Risk", "risk-low"


def _render_risk_banner(
    asr: float,
    critical_count: int,
    total_attacks: int,
    confidence: str = "",
    confidence_note: str = "",
) -> str:
    """Render a prominent risk-level verdict banner."""
    level, css_cls = _risk_level(asr, critical_count)
    summary = (
        f"Based on {total_attacks:,} attack simulations, "
        f"the agent demonstrates a {asr:.1%} attack success rate (ASR). "
    )
    if critical_count > 0:
        summary += f"{critical_count} critical-severity vulnerabilities were confirmed."
    elif asr < 0.10:
        summary += "No critical exposures detected."
    else:
        summary += f"Immediate remediation of top-risk areas is recommended."

    confidence_html = ""
    if confidence:
        confidence_html = (
            f'<div style="margin-top:.5rem;font-size:.82em;opacity:.8">'
            f'<strong>Confidence: {_esc(confidence)}</strong> — {_esc(confidence_note)}</div>'
        )

    return (
        f'<div class="risk-banner {css_cls}">'
        f'<div style="font-size:1.4em;margin-bottom:.3rem">{level}</div>'
        f'<div style="font-weight:400;font-size:.9em">{summary}</div>'
        f'{confidence_html}'
        f"</div>"
    )


def _render_toc(sections: list[ReportSection]) -> str:
    """Render a table of contents with anchor links."""
    items: list[str] = []
    for section in sections:
        renderer = _SECTION_RENDERERS.get(section.kind)
        if renderer is None:
            continue
        anchor = f"section-{section.kind}"
        items.append(f'<li><a href="#{anchor}">{_esc(section.title)}</a></li>')

    if not items:
        return ""

    return (
        '<div class="toc">'
        "<h3>Contents</h3>"
        f'<ol>{"".join(items)}</ol>'
        "</div>"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_html(report: RedTeamReport) -> str:
    """Convert a ``RedTeamReport`` to a self-contained HTML string.

    The output is a single HTML file with inline CSS and optional inline SVG
    charts (if ``plotly`` and ``kaleido`` are installed).  No JavaScript is
    required.

    Args:
        report: The red team report to render.

    Returns:
        A complete HTML document string.
    """
    sections = build_report_sections(report)

    summary_section = next((s for s in sections if s.kind == "summary"), None)
    header = _render_header_html(summary_section.data if summary_section else {})

    # Enrich the summary section data with renderer-specific fields that are
    # derived from the full report object.  We do this here to keep the sections
    # layer renderer-agnostic.
    critical_exposure = 0
    if summary_section is not None:
        if report.summary.by_severity:
            summary_section.data["by_severity"] = report.summary.by_severity
        # Compute critical exposure: count of critical-severity vulnerable results
        critical_exposure = sum(
            1 for r in report.results
            if r.vulnerable and r.attack.severity.value == "critical"
        )
        summary_section.data["critical_exposure"] = critical_exposure

    # Risk verdict banner
    asr = report.summary.vulnerability_rate
    total_attacks = report.summary.total_attacks
    confidence = summary_section.data.get("confidence", "") if summary_section else ""
    confidence_note = summary_section.data.get("confidence_note", "") if summary_section else ""
    risk_banner = _render_risk_banner(asr, critical_exposure, total_attacks, confidence, confidence_note)

    # Table of contents
    toc = _render_toc(sections)

    body_parts: list[str] = [header, risk_banner, toc]

    for section in sections:
        renderer = _SECTION_RENDERERS.get(section.kind)
        if renderer is not None:
            # Wrap each section in a div with an anchor ID for TOC navigation
            anchor = f"section-{section.kind}"
            rendered = renderer(section)
            body_parts.append(f'<div id="{anchor}">{rendered}</div>')

    body_parts.append(
        '<p class="footer">Generated by evaluatorq red team suite.</p>'
    )

    body_html = "\n\n".join(body_parts)

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Red Team Security Report</title>\n"
        f"<style>\n{_load_css()}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{body_html}\n"
        "</body>\n"
        "</html>\n"
    )
