"""Markdown renderer for red team reports.

``export_markdown(report)`` converts a ``RedTeamReport`` to a plain-text
Markdown string suitable for sharing in pull requests, wikis, or any text
editor.  Output is text-only — no images, no embedded HTML charts.

Collapsible prompt/response blocks use standard ``<details>``/``<summary>``
HTML tags that GitHub-Flavored Markdown and most modern renderers support.
"""

from __future__ import annotations

import textwrap
from typing import Any

from evaluatorq.redteam.contracts import RedTeamReport
from evaluatorq.redteam.reports.sections import ReportSection, build_report_sections


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _pct(rate: float) -> str:
    """Format a float rate as a percentage string, e.g. ``0.75`` → ``'75%'``."""
    return f"{rate:.0%}"


def _bar(rate: float, width: int = 10) -> str:
    """Render a Unicode block-character progress bar with a numeric percentage.

    Uses U+2588 (full block) for filled segments and U+2591 (light shade) for
    empty segments.  Always ``width`` characters wide, followed by the numeric
    percentage.  Example: ``'████░░░░░░ 40%'``.
    """
    filled = round(rate * width)
    return "\u2588" * filled + "\u2591" * (width - filled) + f" {rate:.0%}"


def _bold_bar(rate: float, threshold: float = 0.5) -> str:
    """Return a Unicode bar, bolded when rate exceeds ``threshold``."""
    cell = _bar(rate)
    return f"**{cell}**" if rate > threshold else cell


def _md_table(
    headers: list[str],
    rows: list[list[str]],
    right_align: set[int] | None = None,
) -> str:
    """Render a Markdown table from headers and string rows.

    Args:
        headers: Column header labels.
        rows: Table data rows; each element is a list of cell values.
        right_align: Optional set of zero-based column indices that should be
            right-aligned (rendered with ``---:`` separator).
    """
    right_align = right_align or set()
    lines: list[str] = []
    lines.append("| " + " | ".join(headers) + " |")
    separators = ["---:" if i in right_align else "---" for i in range(len(headers))]
    lines.append("| " + " | ".join(separators) + " |")
    for row in rows:
        sanitized = [str(cell).replace("|", "\\|").replace("\n", " ") for cell in row]
        lines.append("| " + " | ".join(sanitized) + " |")
    return "\n".join(lines)


def _center_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table with all columns center-aligned."""
    sep = " | ".join(":---:" for _ in headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + sep + " |",
    ]
    for row in rows:
        sanitized = [str(c).replace("|", "\\|").replace("\n", " ") for c in row]
        lines.append("| " + " | ".join(sanitized) + " |")
    return "\n".join(lines)


def _truncate(text: str, max_chars: int = 800) -> str:
    """Truncate long text with an ellipsis indicator."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n*[truncated — full text in report JSON]*"


def _details_block(summary: str, body: str) -> str:
    """Wrap content in a collapsible ``<details>`` block."""
    inner = textwrap.indent(body.strip(), "  ")
    return f"<details>\n<summary>{summary}</summary>\n\n{inner}\n\n</details>"


def _severity_label(risk_score: float) -> str:
    """Map a numeric risk score to a text severity prefix."""
    if risk_score >= 4.0:
        return "CRITICAL"
    if risk_score >= 2.0:
        return "HIGH"
    if risk_score >= 1.0:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_header(data: dict[str, Any]) -> str:
    """Render the document header with run metadata."""
    target = data.get("target", "unknown")
    pipeline = data.get("pipeline", "unknown")
    created_at = data.get("created_at")
    vulnerability_rate = data.get("vulnerability_rate", 0.0)

    date_str = created_at.strftime("%Y-%m-%d %H:%M UTC") if created_at else "unknown"
    lines = [
        "# Red Team Security Report",
        "",
        f"**Target:** {target}  ",
        f"**Mode:** {pipeline}  ",
        f"**Date:** {date_str}  ",
        f"**ASR:** {_pct(vulnerability_rate)}  ",
    ]
    return "\n".join(lines)


def _render_summary_section(section: ReportSection) -> str:
    """Render the executive summary with a risk verdict callout and KPI strip."""
    data = section.data
    asr = data.get("vulnerability_rate", 0.0)
    total_attacks = data.get("total_attacks", 0)
    vulnerabilities_found = data.get("vulnerabilities_found", 0)
    evaluated_attacks = data.get("evaluated_attacks", 0)
    evaluation_coverage = data.get("evaluation_coverage", 0.0)
    total_errors = data.get("total_errors", 0)

    if asr >= 0.50:
        callout_type, risk_label = "CAUTION", "RISK: CRITICAL"
    elif asr >= 0.25:
        callout_type, risk_label = "CAUTION", "RISK: HIGH"
    elif asr >= 0.10:
        callout_type, risk_label = "WARNING", "RISK: MEDIUM"
    else:
        callout_type, risk_label = "NOTE", "RISK: LOW"

    risk_callout = f"> [!{callout_type}]\n> {risk_label} — Attack Success Rate: **{_pct(asr)}**"

    confidence = data.get("confidence", "")
    confidence_note = data.get("confidence_note", "")
    if confidence:
        confidence_line = f"\n> **Confidence: {confidence}** — {confidence_note}"
        risk_callout += confidence_line

    kpi_table = _center_table(
        ["Total Attacks", "Successful Attacks", "ASR", "Coverage", "Errors"],
        [[
            f"**{total_attacks}**",
            f"**{vulnerabilities_found}**",
            f"**{_pct(asr)}**",
            f"**{_pct(evaluation_coverage)}**",
            f"**{total_errors}**",
        ]],
    )

    detail_rows: list[list[str]] = [
        ["Total Attacks", str(total_attacks)],
        ["Evaluated", str(evaluated_attacks)],
        ["Successful Attacks", str(vulnerabilities_found)],
        ["ASR", _pct(asr)],
        ["Evaluation Coverage", _pct(evaluation_coverage)],
    ]
    if total_errors:
        detail_rows.append(["Errors", str(total_errors)])
    duration_seconds = data.get("duration_seconds")
    if duration_seconds is not None:
        mins, secs = divmod(int(duration_seconds), 60)
        detail_rows.append(["Duration", f"{mins}m {secs}s"])

    detail_table = _md_table(["Metric", "Value"], detail_rows)

    lines = [
        f"## {section.title}",
        "",
        risk_callout,
        "",
        kpi_table,
        "",
        detail_table,
    ]
    return "\n".join(lines)


def _render_agent_context_section(section: ReportSection) -> str:
    """Render agent capability cards for all tested agents."""
    agents = section.data.get("agents", [])
    if not agents:
        return f"## {section.title}\n\nNo agent information available."

    lines = [f"## {section.title}", ""]
    for agent in agents:
        display_name = agent.get("display_name") or agent.get("key", "unknown")
        model = agent.get("model", "")
        description = agent.get("description", "")
        tools: list[str] = agent.get("tools", [])
        memory_stores: list[str] = agent.get("memory_stores", [])
        knowledge_bases: list[str] = agent.get("knowledge_bases", [])

        lines.append(f"### {display_name}")
        lines.append("")
        if model:
            lines.append(f"**Model:** {model}  ")
        if description:
            lines.append(f"**Description:** {description}  ")
        if tools:
            lines.append(f"**Tools:** {', '.join(tools)}  ")
        if memory_stores:
            lines.append(f"**Memory:** {', '.join(memory_stores)}  ")
        if knowledge_bases:
            lines.append(f"**Knowledge:** {', '.join(knowledge_bases)}  ")
        lines.append("")

    return "\n".join(lines)


def _render_focus_areas_section(section: ReportSection) -> str:
    """Render the focus areas (top risks) with severity labels and Unicode bars."""
    focus_areas = section.data.get("focus_areas", [])
    if not focus_areas:
        return f"## {section.title}\n\nNo significant risk areas identified."

    lines = [f"## {section.title}", ""]
    for i, area in enumerate(focus_areas, start=1):
        cat = area["category"]
        cat_name = area.get("category_name", cat)
        vuln = area.get("vulnerabilities_found", 0)
        vuln_rate = area.get("vulnerability_rate", 0.0)
        risk_score = area.get("risk_score", 0.0)
        remediation = area.get("remediation", "")
        severity = _severity_label(risk_score)

        lines.append(f"### {i}. [{severity}] {cat} — {cat_name}")
        lines.append("")
        lines.append(
            f"- **Vulnerabilities:** {vuln} ({_bar(vuln_rate)} of attacks succeeded)"
        )
        lines.append(f"- **Risk Score:** {risk_score:.2f}")
        lines.append("")
        if remediation:
            lines.append("**Remediation guidance:**")
            lines.append("")
            lines.append(f"> {remediation}")
            lines.append("")

        agent_remediation = area.get("agent_specific_remediation", "")
        if agent_remediation:
            lines.append("**Agent-specific recommendations:**")
            lines.append("")
            lines.append(f"> {agent_remediation}")
            lines.append("")

        llm_rec = area.get("llm_recommendations")
        if llm_rec:
            patterns = llm_rec.get("patterns_observed", "")
            recs = llm_rec.get("recommendations", [])
            traces_analyzed = llm_rec.get("traces_analyzed", 0)
            if recs:
                lines.append(f"**Actionable recommendations** (based on {traces_analyzed} trace samples):")
                lines.append("")
                for rec_item in recs:
                    lines.append(f"- {rec_item}")
                lines.append("")
            if patterns:
                lines.append(f"*Patterns observed:* {patterns}")
                lines.append("")

    return "\n".join(lines)


def _render_agent_comparison_section(section: ReportSection) -> str:
    """Render multi-agent comparison with overview metrics and ASR heatmap."""
    data = section.data
    agents: list[str] = data.get("agents", [])
    agent_metrics: list[dict[str, Any]] = data.get("agent_metrics", [])
    heatmap: dict[str, Any] = data.get("heatmap", {})

    if not agents:
        return f"## {section.title}\n\nNo multi-agent data available."

    lines = [f"## {section.title}", ""]

    if agent_metrics:
        lines.append("### Agent Overview")
        lines.append("")
        metric_rows = [
            [m["agent"], str(m["total_attacks"]), str(m["vulnerabilities_found"]),
             _bold_bar(m.get("asr", 0.0))]
            for m in agent_metrics
        ]
        lines.append(_md_table(
            ["Agent", "Attacks", "Hits", "ASR"], metric_rows, right_align={1, 2},
        ))
        lines.append("")

    vulns: list[str] = heatmap.get("vulnerabilities", [])
    hm_agents: list[str] = heatmap.get("agents", [])
    z_matrix: list[list[float]] = heatmap.get("z_matrix", [])
    text_matrix: list[list[str]] = heatmap.get("text_matrix", [])

    if vulns and hm_agents and z_matrix:
        lines.append("### ASR by Vulnerability per Agent")
        lines.append("")
        headers = ["Vulnerability"] + hm_agents
        table_rows: list[list[str]] = []
        for i, vuln_name in enumerate(vulns):
            row: list[str] = [vuln_name]
            for j in range(len(hm_agents)):
                pct = z_matrix[i][j] if i < len(z_matrix) and j < len(z_matrix[i]) else 0.0
                cell = text_matrix[i][j] if i < len(text_matrix) and j < len(text_matrix[i]) else f"{pct:.0f}%"
                if pct >= 40:
                    cell = f"**{cell}**"
                row.append(cell)
            table_rows.append(row)

        lines.append(_center_table(headers, table_rows))
        lines.append("")

    return "\n".join(lines)


def _render_agent_disagreements_section(section: ReportSection) -> str:
    """Render agent disagreements as collapsible per-attack blocks."""
    data = section.data
    total = data.get("total_disagreements", 0)
    disagreements: list[dict[str, Any]] = data.get("disagreements", [])

    if total == 0:
        return f"## {section.title}\n\nNo disagreements — all agents agreed on every attack."

    lines = [f"## {section.title}", ""]
    lines.append("> [!NOTE]")
    lines.append(f"> **{total}** disagreements found where agents produced different verdicts")
    lines.append("")

    for d in disagreements:
        attack_id = d.get("attack_id", "unknown")
        vulnerability = d.get("vulnerability", "")
        technique = d.get("technique", "")
        severity = d.get("severity", "")
        per_agent: list[dict[str, Any]] = d.get("per_agent", [])

        verdicts = [
            f"{pa['agent']}: **{'VULNERABLE' if pa['vulnerable'] else 'RESISTANT'}**"
            for pa in per_agent
        ]

        body_lines = [
            f"- **Vulnerability:** {vulnerability}",
            f"- **Technique:** {technique}",
            f"- **Severity:** {severity}",
            f"- **Verdicts:** {' | '.join(verdicts)}",
            "",
        ]
        for pa in per_agent:
            verdict = "VULNERABLE" if pa["vulnerable"] else "RESISTANT"
            explanation = pa.get("explanation", "")
            snippet = pa.get("response_snippet", "")
            body_lines.append(f"**{pa['agent']}** ({verdict}):")
            if explanation:
                body_lines.append(f"> {explanation}")
            if snippet:
                body_lines.append(f"```\n{_truncate(snippet, 300)}\n```")
            body_lines.append("")

        block = _details_block(
            f"[{severity.upper()}] {attack_id} — {vulnerability}",
            "\n".join(body_lines),
        )
        lines.append(block)
        lines.append("")

    return "\n".join(lines)


def _render_vulnerability_breakdown_section(section: ReportSection) -> str:
    """Render the per-vulnerability breakdown table with Unicode ASR bars."""
    rows = section.data.get("rows", [])
    if not rows:
        return f"## {section.title}\n\nNo vulnerability data available."

    table_rows = [
        [
            r["vulnerability_name"] or r["vulnerability"],
            r.get("domain", ""),
            str(r["total_attacks"]),
            str(r["vulnerabilities_found"]),
            _bold_bar(r["vulnerability_rate"]),
        ]
        for r in rows
    ]
    table = _md_table(
        ["Vulnerability", "Domain", "Attacks", "Hits", "ASR"],
        table_rows,
        right_align={2, 3},
    )
    return f"## {section.title}\n\n{table}"


def _render_category_breakdown_section(section: ReportSection) -> str:
    """Render the per-category breakdown table with Unicode ASR bars."""
    rows = section.data.get("rows", [])
    if not rows:
        return f"## {section.title}\n\nNo category data available."

    table_rows = [
        [
            f"{r['category']} — {r['category_name']}",
            str(r["total_attacks"]),
            str(r["vulnerabilities_found"]),
            _bold_bar(r["vulnerability_rate"]),
        ]
        for r in rows
    ]
    table = _md_table(
        ["Category", "Attacks", "Hits", "ASR"],
        table_rows,
        right_align={1, 2},
    )
    return f"## {section.title}\n\n{table}"


def _render_attack_heatmap_section(section: ReportSection) -> str:
    """Render a vulnerability x technique ASR heatmap as a center-aligned table."""
    vulnerabilities: list[str] = section.data.get("vulnerabilities", [])
    techniques: list[str] = section.data.get("techniques", [])
    cells: list[dict[str, Any]] = section.data.get("cells", [])

    if not vulnerabilities or not techniques or not cells:
        return f"## {section.title}\n\nNo heatmap data available."

    cell_map: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in cells:
        cell_map[(cell["vulnerability"], cell["technique"])] = cell

    col_labels = [t[:12] if len(techniques) > 6 else t for t in techniques]
    headers = ["Vulnerability"] + col_labels

    table_rows: list[list[str]] = []
    for vuln in vulnerabilities:
        row: list[str] = [vuln]
        for tech in techniques:
            cell = cell_map.get((vuln, tech))
            if cell is None or cell.get("total_attacks", 0) == 0:
                row.append("-")
            else:
                asr = cell.get("vulnerability_rate", 0.0)
                pct_str = f"{asr:.0%}"
                row.append(f"**{pct_str}**" if asr >= 0.40 else pct_str)
        table_rows.append(row)

    table = _center_table(headers, table_rows)
    legend = (
        "\n*Legend: Values show Attack Success Rate (ASR). "
        "**Bold** = ASR >= 40%. `-` = no attacks attempted.*"
    )
    return f"## {section.title}\n\n{table}{legend}"


def _render_technique_breakdown_section(section: ReportSection) -> str:
    """Render the per-technique breakdown table with Unicode ASR bars."""
    rows = section.data.get("rows", [])
    if not rows:
        return f"## {section.title}\n\nNo technique data available."

    table_rows = [
        [
            r["technique"],
            str(r["total_attacks"]),
            str(r["vulnerabilities_found"]),
            _bold_bar(r["vulnerability_rate"]),
        ]
        for r in rows
    ]
    table = _md_table(
        ["Technique", "Attacks", "Hits", "ASR"],
        table_rows,
        right_align={1, 2},
    )
    return f"## {section.title}\n\n{table}"


def _render_delivery_breakdown_section(section: ReportSection) -> str:
    """Render per-delivery-method ASR breakdown with Unicode bars."""
    rows = section.data.get("rows", [])
    if not rows:
        return ""

    table_rows = [
        [
            r["delivery_method"],
            str(r["total_attacks"]),
            str(r["vulnerabilities_found"]),
            _bold_bar(r["vulnerability_rate"]),
        ]
        for r in rows
    ]
    table = _md_table(
        ["Delivery Method", "Attacks", "Hits", "ASR"],
        table_rows,
        right_align={1, 2},
    )
    return f"## {section.title}\n\n{table}"


def _render_turn_scope_breakdown_section(section: ReportSection) -> str:
    """Render turn-type and domain breakdown as two sub-tables."""
    by_turn_type: dict[str, Any] = section.data.get("by_turn_type", {})
    by_domain: dict[str, Any] = section.data.get("by_domain", {})

    if not by_turn_type and not by_domain:
        return ""

    def _sub_table(label: str, mapping: dict[str, Any]) -> str:
        sub_rows = [
            [
                name,
                str(stats.get("total_attacks", 0)),
                str(stats.get("vulnerabilities_found", 0)),
                _bold_bar(stats.get("vulnerability_rate", 0.0)),
            ]
            for name, stats in mapping.items()
        ]
        return f"### {label}\n\n" + _md_table(
            [label, "Attacks", "Hits", "ASR"], sub_rows, right_align={1, 2},
        )

    parts: list[str] = [f"## {section.title}", ""]
    if by_turn_type:
        parts.append(_sub_table("Turn Type", by_turn_type))
        parts.append("")
    if by_domain:
        parts.append(_sub_table("Domain", by_domain))
        parts.append("")

    return "\n".join(parts)


def _render_turn_depth_analysis_section(section: ReportSection) -> str:
    """Render conversation depth analysis for multi-turn attacks."""
    rows = section.data.get("rows", [])
    if not rows:
        return ""

    table_rows = [
        [
            str(r["turn_count"]),
            str(r["total_attacks"]),
            str(r["vulnerabilities_found"]),
            _bold_bar(r["vulnerability_rate"]),
        ]
        for r in rows
    ]
    table = _md_table(
        ["Turn Depth", "Attacks", "Hits", "ASR"],
        table_rows,
        right_align={0, 1, 2},
    )
    return f"## {section.title}\n\n{table}"


def _render_error_analysis_section(section: ReportSection) -> str:
    """Render error analysis with a warning callout and breakdown tables."""
    data = section.data
    total_errors: int = data.get("total_errors", 0)
    error_rate: float = data.get("error_rate", 0.0)
    errors_by_type: dict[str, int] = data.get("errors_by_type", {})
    detail_rows: list[dict[str, Any]] = data.get("detail_rows", [])

    if total_errors == 0:
        return ""

    lines: list[str] = [f"## {section.title}", ""]

    if error_rate > 0.10:
        lines.append("> [!WARNING]")
        lines.append(
            f"> Error rate is {_pct(error_rate)} ({total_errors} errors). "
            "This may indicate infrastructure problems or misconfigured targets."
        )
        lines.append("")

    lines.append(
        f"**Total errors:** {total_errors} ({_pct(error_rate)} of attacks) "
        f"across {data.get('error_types_count', len(errors_by_type))} error type(s)."
    )
    lines.append("")

    if errors_by_type:
        type_rows = [
            [error_type, str(count)]
            for error_type, count in sorted(
                errors_by_type.items(), key=lambda kv: kv[1], reverse=True
            )
        ]
        lines.append(_md_table(["Error Type", "Count"], type_rows, right_align={1}))
        lines.append("")

    if detail_rows:
        detail_table_rows = [
            [
                r.get("id", ""),
                r.get("category", ""),
                r.get("technique", ""),
                r.get("error_type", ""),
                r.get("stage", ""),
                str(r.get("error", ""))[:200],
            ]
            for r in detail_rows
        ]
        detail_table = _md_table(
            ["ID", "Category", "Technique", "Error Type", "Stage", "Error"],
            detail_table_rows,
        )
        lines.append(_details_block("Individual Error Details", detail_table))
        lines.append("")

    return "\n".join(lines)


def _render_framework_breakdown_section(section: ReportSection) -> str:
    """Render per-framework ASR breakdown with Unicode bars."""
    rows = section.data.get("rows", [])
    if not rows:
        return ""

    table_rows = [
        [
            r["framework"],
            str(r["total_attacks"]),
            str(r["vulnerabilities_found"]),
            _bold_bar(r["vulnerability_rate"]),
        ]
        for r in rows
    ]
    table = _md_table(
        ["Framework", "Attacks", "Hits", "ASR"],
        table_rows,
        right_align={1, 2},
    )
    return f"## {section.title}\n\n{table}"



def _render_source_distribution_section(section: ReportSection) -> str:
    """Render attack source distribution as a table with relative bars."""
    rows = section.data.get("rows", [])
    if not rows:
        return ""

    total = sum(r["count"] for r in rows)
    table_rows = [
        [r["source"], str(r["count"]), _bar(r["count"] / total if total > 0 else 0.0)]
        for r in rows
    ]
    table = _md_table(["Source", "Attacks", "Share"], table_rows, right_align={1})
    return f"## {section.title}\n\n{table}"


def _render_token_usage_section(section: ReportSection) -> str:
    """Render token usage summary with overall and per-agent breakdowns."""
    data = section.data
    overall: dict[str, Any] = data.get("overall", {})
    per_agent: list[dict[str, Any]] = data.get("per_agent", [])

    if not overall and not per_agent:
        return f"## {section.title}\n\nNo token usage data available."

    lines = [f"## {section.title}", ""]

    if overall:
        lines.append(_md_table(
            ["Metric", "Value"],
            [
                ["Total Tokens", f"{overall.get('total_tokens', 0):,}"],
                ["Prompt Tokens", f"{overall.get('prompt_tokens', 0):,}"],
                ["Completion Tokens", f"{overall.get('completion_tokens', 0):,}"],
                ["API Calls", f"{overall.get('calls', 0):,}"],
            ],
            right_align={1},
        ))
        lines.append("")

    if per_agent:
        lines.append("### Per Agent")
        lines.append("")
        agent_rows = [
            [
                a["agent"],
                f"{a['total_tokens']:,}",
                f"{a['prompt_tokens']:,}",
                f"{a['completion_tokens']:,}",
                f"{a['calls']:,}",
            ]
            for a in per_agent
        ]
        lines.append(_md_table(
            ["Agent", "Total", "Prompt", "Completion", "Calls"],
            agent_rows, right_align={1, 2, 3, 4},
        ))
        lines.append("")

    return "\n".join(lines)


def _render_severity_definitions_section(section: ReportSection) -> str:
    """Render the severity level definitions table."""
    definitions = section.data.get("definitions", [])
    weights = section.data.get("weights", {})
    if not definitions:
        return ""

    table_rows = [
        [d["level"].capitalize(), d["description"], str(weights.get(d["level"], ""))]
        for d in definitions
    ]
    table = _md_table(["Severity", "Description", "Risk Weight"], table_rows)
    return f"## {section.title}\n\n{table}"


def _render_methodology_section(section: ReportSection) -> str:
    """Render the methodology disclosure section."""
    data = section.data
    lines = [f"## {section.title}", ""]

    # Pipeline and scoring
    pipeline = data.get("pipeline", "unknown")
    scoring = data.get("scoring_method", "llm-as-judge")
    lines.append(f"**Assessment Type:** {pipeline}  ")
    lines.append(f"**Scoring Method:** {scoring}  ")

    # Models used
    evaluator_model = data.get("evaluator_model")
    attack_model = data.get("attack_model")
    if evaluator_model:
        lines.append(f"**Evaluator Model:** {evaluator_model}  ")
    if attack_model:
        lines.append(f"**Attack Model:** {attack_model}  ")

    # Dataset source
    dataset_source = data.get("dataset_source")
    if dataset_source:
        lines.append(f"**Dataset Source:** {dataset_source}  ")

    # Framework
    framework = data.get("framework")
    if framework:
        lines.append(f"**Framework:** {framework}  ")
    lines.append("")

    # Scope
    categories = data.get("categories_tested") or []
    vulnerabilities = data.get("vulnerabilities_tested") or []
    total_attacks = data.get("total_attacks", 0)
    coverage = data.get("evaluation_coverage", 0.0)
    max_turns = data.get("max_turns")
    duration = data.get("duration_seconds")

    scope_rows = [
        ["Categories Tested", str(len(categories))],
        ["Total Attacks", str(total_attacks)],
        ["Evaluation Coverage", _pct(coverage)],
    ]
    if vulnerabilities:
        scope_rows.insert(1, ["Vulnerabilities Tested", str(len(vulnerabilities))])
    if max_turns:
        scope_rows.append(["Max Conversation Turns", str(max_turns)])
    if duration is not None:
        mins, secs = divmod(int(duration), 60)
        scope_rows.append(["Duration", f"{mins}m {secs}s"])

    lines.append(_md_table(["Parameter", "Value"], scope_rows))
    lines.append("")

    # Agents tested
    agents = data.get("tested_agents", [])
    if agents:
        lines.append(f"**Agents Tested:** {', '.join(agents)}")
        lines.append("")

    # Scope limitations
    limitations = data.get("scope_limitations", [])
    if limitations:
        lines.append("### Known Limitations")
        lines.append("")
        for lim in limitations:
            lines.append(f"- {lim}")
        lines.append("")

    untested = data.get("untested_categories", [])
    untested_names = data.get("untested_category_names", {})
    if untested:
        lines.append("### Untested Categories")
        lines.append("")
        lines.append(f"> [!WARNING]")
        lines.append(f"> {len(untested)} supported categories were not included in this assessment:")
        lines.append("")
        for cat in untested:
            name = untested_names.get(cat, cat)
            lines.append(f"- **{cat}** — {name}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Section dispatch
# ---------------------------------------------------------------------------

_SECTION_RENDERERS = {
    "summary": _render_summary_section,
    "methodology": _render_methodology_section,
    "agent_context": _render_agent_context_section,
    "focus_areas": _render_focus_areas_section,
    "agent_comparison": _render_agent_comparison_section,
    "agent_disagreements": _render_agent_disagreements_section,
    "vulnerability_breakdown": _render_vulnerability_breakdown_section,
    "category_breakdown": _render_category_breakdown_section,
    "attack_heatmap": _render_attack_heatmap_section,
    "technique_breakdown": _render_technique_breakdown_section,
    "delivery_breakdown": _render_delivery_breakdown_section,
    "turn_scope_breakdown": _render_turn_scope_breakdown_section,
    "turn_depth_analysis": _render_turn_depth_analysis_section,
    "error_analysis": _render_error_analysis_section,
    "framework_breakdown": _render_framework_breakdown_section,
    "source_distribution": _render_source_distribution_section,
    "token_usage": _render_token_usage_section,
    "severity_definitions": _render_severity_definitions_section,
}

# Sections wrapped in collapsible <details> blocks to keep the report concise.
_COLLAPSED_SECTIONS = {
    "methodology",
    "severity_definitions",
    "token_usage",
    "source_distribution",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_markdown(report: RedTeamReport) -> str:
    """Convert a ``RedTeamReport`` to a Markdown string.

    The output is text-only (no images, no embedded charts).  Collapsible
    prompt/response blocks use ``<details>`` tags supported by GitHub-Flavored
    Markdown and most modern renderers.

    Verbose sections (individual results, severity definitions, token usage,
    source distribution) are wrapped in collapsible ``<details>`` blocks to
    keep the document scannable.

    Args:
        report: The red team report to render.

    Returns:
        A Markdown string representing the full report.
    """
    sections = build_report_sections(report)

    summary_section = next((s for s in sections if s.kind == "summary"), None)
    header = _render_header(summary_section.data if summary_section else {})

    parts: list[str] = [header, ""]

    for section in sections:
        renderer = _SECTION_RENDERERS.get(section.kind)
        if renderer is not None:
            rendered = renderer(section)
            if not rendered:
                continue
            if section.kind in _COLLAPSED_SECTIONS:
                # Strip the leading ## heading — _details_block adds its own <summary> title
                heading_prefix = f"## {section.title}\n"
                if rendered.startswith(heading_prefix):
                    rendered = rendered[len(heading_prefix):].lstrip("\n")
                rendered = _details_block(section.title, rendered)
            parts.append(rendered)
            parts.append("")

    parts.append("---")
    parts.append("")
    parts.append("*Generated by evaluatorq red team suite.*")

    return "\n".join(parts)
