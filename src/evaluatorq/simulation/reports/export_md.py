"""Markdown renderer for agent simulation reports.

``export_markdown(results)`` converts a list of ``SimulationResult`` into a
plain-text Markdown report suitable for PRs and wikis. Uses shared
formatting helpers and dispatch from ``evaluatorq.common.reports`` so the
report style matches red-team output.
"""

from __future__ import annotations

from datetime import datetime, timezone

from evaluatorq.common.reports import bold_bar as _bold_bar
from evaluatorq.common.reports import center_table as _center_table
from evaluatorq.common.reports import details_block as _details_block
from evaluatorq.common.reports import format_date as _format_date
from evaluatorq.common.reports import md_table as _md_table
from evaluatorq.common.reports import pct as _pct
from evaluatorq.common.reports import render_header_md as _render_header_md
from evaluatorq.common.reports import render_markdown as _render_markdown_doc
from evaluatorq.common.reports import truncate as _truncate
from evaluatorq.contracts import ReportSection
from evaluatorq.simulation.reports.sections import build_report_sections
from evaluatorq.simulation.types import SimulationResult

# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_summary_section(section: ReportSection) -> str:
    data = section.data
    total = data.get("total_conversations", 0)
    achieved = data.get("goals_achieved", 0)
    success_rate = data.get("success_rate", 0.0)
    avg_score = data.get("avg_goal_completion_score", 0.0)
    avg_turns = data.get("avg_turn_count", 0.0)
    errors = data.get("errors", 0)
    total_tokens = data.get("total_tokens", 0)
    confidence = data.get("confidence", "")
    confidence_note = data.get("confidence_note", "")

    if success_rate >= 0.80:
        callout, label = "TIP", "GOAL ACHIEVEMENT: STRONG"
    elif success_rate >= 0.50:
        callout, label = "NOTE", "GOAL ACHIEVEMENT: MIXED"
    elif success_rate >= 0.25:
        callout, label = "WARNING", "GOAL ACHIEVEMENT: LOW"
    else:
        callout, label = "CAUTION", "GOAL ACHIEVEMENT: VERY LOW"

    risk_callout = f"> [!{callout}]\n> {label} — success rate: **{_pct(success_rate)}**"
    if confidence:
        risk_callout += f"\n> **Confidence: {confidence}** — {confidence_note}"

    kpi_table = _center_table(
        ["Conversations", "Goals Achieved", "Success Rate", "Avg Score", "Errors"],
        [[
            f"**{total}**",
            f"**{achieved}**",
            f"**{_pct(success_rate)}**",
            f"**{avg_score:.2f}**",
            f"**{errors}**",
        ]],
    )

    detail_rows: list[list[str]] = [
        ["Total Conversations", str(total)],
        ["Goals Achieved", str(achieved)],
        ["Success Rate", _pct(success_rate)],
        ["Avg Goal Completion Score", f"{avg_score:.2f}"],
        ["Avg Turn Count", f"{avg_turns:.1f}"],
        ["Total Tokens", f"{total_tokens:,}"],
    ]
    if errors:
        detail_rows.append(["Errors", str(errors)])

    detail_table = _md_table(["Metric", "Value"], detail_rows)

    return "\n".join([
        f"## {section.title}",
        "",
        risk_callout,
        "",
        kpi_table,
        "",
        detail_table,
    ])


def _render_persona_breakdown_section(section: ReportSection) -> str:
    rows = section.data.get("rows", [])
    if not rows:
        return f"## {section.title}\n\nNo persona data."
    table_rows = [
        [
            r["persona"],
            str(r["conversations"]),
            str(r["goals_achieved"]),
            _bold_bar(r["success_rate"]),
            f"{r['avg_goal_completion_score']:.2f}",
            f"{r['total_tokens']:,}",
        ]
        for r in rows
    ]
    table = _md_table(
        ["Persona", "Convs", "Achieved", "Success", "Avg Score", "Tokens"],
        table_rows,
        right_align={1, 2, 5},
    )
    return f"## {section.title}\n\n{table}"


def _render_scenario_breakdown_section(section: ReportSection) -> str:
    rows = section.data.get("rows", [])
    if not rows:
        return f"## {section.title}\n\nNo scenario data."
    table_rows = [
        [
            r["scenario"],
            str(r["conversations"]),
            str(r["goals_achieved"]),
            _bold_bar(r["success_rate"]),
            f"{r['avg_goal_completion_score']:.2f}",
            f"{r['avg_turn_count']:.1f}",
        ]
        for r in rows
    ]
    table = _md_table(
        ["Scenario", "Convs", "Achieved", "Success", "Avg Score", "Avg Turns"],
        table_rows,
        right_align={1, 2, 5},
    )
    return f"## {section.title}\n\n{table}"


def _render_judge_verdicts_section(section: ReportSection) -> str:
    data = section.data
    terminated_by = data.get("terminated_by", {})
    rules_broken = data.get("rules_broken", {})
    total_broken = data.get("total_rules_broken_instances", 0)

    lines = [f"## {section.title}", ""]

    if terminated_by:
        term_rows = [
            [reason, str(count)] for reason, count in
            sorted(terminated_by.items(), key=lambda kv: -kv[1])
        ]
        lines.extend((
            "**Terminated By:**",
            "",
            _md_table(["Reason", "Count"], term_rows, right_align={1}),
            "",
        ))

    if rules_broken:
        rule_rows = [
            [rule, str(count)] for rule, count in rules_broken.items()
        ]
        lines.extend((
            f"**Rules Broken** (total instances: {total_broken}):",
            "",
            _md_table(["Rule", "Count"], rule_rows, right_align={1}),
        ))
    else:
        lines.append("No rules broken across any conversation.")

    return "\n".join(lines)


def _render_turn_metrics_section(section: ReportSection) -> str:
    data = section.data
    dist = data.get("turn_count_distribution", {})
    qualities = data.get("avg_quality_metrics", {})

    lines = [f"## {section.title}", ""]

    if dist:
        dist_rows = [[str(t), str(c)] for t, c in dist.items()]
        lines.extend((
            "**Turn Count Distribution:**",
            "",
            _md_table(["Turns", "Conversations"], dist_rows, right_align={1}),
            "",
        ))

    if qualities:
        qual_rows = [[k, f"{v:.2f}"] for k, v in qualities.items()]
        lines.extend((
            "**Average Per-Turn Quality Metrics:**",
            "",
            _md_table(["Metric", "Avg Score"], qual_rows),
        ))

    return "\n".join(lines)


def _render_evaluator_scores_section(section: ReportSection) -> str:
    rows = section.data.get("rows", [])
    if not rows:
        return ""
    table_rows = [
        [
            r["evaluator"],
            str(r["runs"]),
            f"{r['mean_score']:.2f}",
            f"{r['min_score']:.2f}",
            f"{r['max_score']:.2f}",
        ]
        for r in rows
    ]
    table = _md_table(
        ["Evaluator", "Runs", "Mean", "Min", "Max"],
        table_rows,
        right_align={1, 2, 3, 4},
    )
    return f"## {section.title}\n\n{table}"


def _render_token_usage_section(section: ReportSection) -> str:
    data = section.data
    rows = [
        ["Prompt Tokens (total)", f"{data.get('prompt_tokens', 0):,}"],
        ["Completion Tokens (total)", f"{data.get('completion_tokens', 0):,}"],
        ["Total Tokens", f"{data.get('total_tokens', 0):,}"],
        ["Avg Total / Conversation", f"{data.get('avg_total_per_conversation', 0):,.0f}"],
        ["Avg Prompt / Conversation", f"{data.get('avg_prompt_per_conversation', 0):,.0f}"],
        ["Avg Completion / Conversation", f"{data.get('avg_completion_per_conversation', 0):,.0f}"],
    ]
    table = _md_table(["Metric", "Value"], rows)
    return f"## {section.title}\n\n{table}"


def _render_errors_section(section: ReportSection) -> str:
    data = section.data
    total = data.get("total_errored", 0)
    by_message = data.get("by_message", {})
    lines = [f"## {section.title}", "", f"**Total errored conversations:** {total}", ""]
    if by_message:
        rows = [[msg, str(count)] for msg, count in by_message.items()]
        lines.append(_md_table(["Error", "Count"], rows, right_align={1}))
    return "\n".join(lines)


def _render_individual_results_section(section: ReportSection) -> str:
    entries = section.data.get("entries", [])
    if not entries:
        return f"## {section.title}\n\nNo conversations."

    lines = [f"## {section.title}", ""]
    for entry in entries:
        verdict = "ACHIEVED" if entry["goal_achieved"] else "NOT ACHIEVED"
        title = (
            f"#{entry['index'] + 1}: {entry['persona']} / {entry['scenario']} "
            f"— **{verdict}** ({entry['turn_count']} turns, "
            f"score {entry['goal_completion_score']:.2f})"
        )

        body_lines: list[str] = []
        body_lines.extend((
            f"- **Model:** {entry['model']}",
            f"- **Terminated by:** {entry['terminated_by']}",
            f"- **Tokens:** {entry['total_tokens']:,}",
        ))
        if entry["rules_broken"]:
            body_lines.append(
                "- **Rules broken:** " + ", ".join(entry["rules_broken"])
            )
        if entry["evaluator_scores"]:
            scores_str = ", ".join(
                f"{k}={v:.2f}" for k, v in entry["evaluator_scores"].items()
            )
            body_lines.append(f"- **Evaluator scores:** {scores_str}")
        if entry["error"]:
            body_lines.append(f"- **Error:** {entry['error']}")
        body_lines.extend((
            "",
            f"**Judge reason:** {entry['judge_reason']}",
            "",
            "**Transcript:**",
            "",
        ))
        for msg in entry["transcript"]:
            content = _truncate(msg.get("content", ""), 600)
            body_lines.extend((
                f"_{msg['role']}_:",
                "",
                f"> {content.replace(chr(10), chr(10) + '> ')}",
                "",
            ))

        lines.extend((
            _details_block(title, "\n".join(body_lines)),
            "",
        ))

    return "\n".join(lines)


_SECTION_RENDERERS = {
    "summary": _render_summary_section,
    "persona_breakdown": _render_persona_breakdown_section,
    "scenario_breakdown": _render_scenario_breakdown_section,
    "judge_verdicts": _render_judge_verdicts_section,
    "turn_metrics": _render_turn_metrics_section,
    "evaluator_scores": _render_evaluator_scores_section,
    "token_usage": _render_token_usage_section,
    "errors": _render_errors_section,
    "individual_results": _render_individual_results_section,
}

_COLLAPSED_SECTIONS = {
    "token_usage",
    "individual_results",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_markdown(
    results: list[SimulationResult],
    *,
    target: str = "agent",
    run_date: datetime | None = None,
) -> str:
    """Render a list of simulation results as a Markdown report.

    Args:
        results: The simulation results to summarise.
        target: Human-readable target name for the document header.
        run_date: Timestamp shown in the header (defaults to now in UTC).
    """
    sections = build_report_sections(results)
    summary_data = next(
        (s.data for s in sections if s.kind == "summary"),
        {},
    )

    header = _render_header_md(
        title="Agent Simulation Report",
        rows=[
            ("Target", target),
            ("Date", _format_date(run_date or datetime.now(tz=timezone.utc))),
            ("Conversations", str(summary_data.get("total_conversations", 0))),
            ("Success Rate", _pct(summary_data.get("success_rate", 0.0))),
        ],
    )

    return _render_markdown_doc(
        sections,
        renderers=_SECTION_RENDERERS,
        collapsed_kinds=_COLLAPSED_SECTIONS,
        header=header,
        footer="\n---\n\n*Generated by evaluatorq agent simulation suite.*",
    )
