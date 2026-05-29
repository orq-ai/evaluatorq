"""HTML renderer for agent simulation reports.

``export_html(results)`` produces a self-contained HTML document styled with
the shared report CSS. Charts via Plotly + kaleido degrade gracefully when
those packages are unavailable.
"""

from __future__ import annotations

from datetime import datetime, timezone

from evaluatorq.common.reports import COLORS as _COLORS
from evaluatorq.common.reports import STATUS_COLORS as _STATUS_COLORS
from evaluatorq.common.reports import esc as _esc
from evaluatorq.common.reports import format_date as _format_date
from evaluatorq.common.reports import html_table as _html_table
from evaluatorq.common.reports import load_css as _load_css
from evaluatorq.common.reports import pct as _pct
from evaluatorq.common.reports import render_donut_chart as _render_donut_chart_common
from evaluatorq.common.reports import render_horizontal_bar_chart as _render_horizontal_bar_chart
from evaluatorq.common.reports import render_html as _render_html_doc
from evaluatorq.common.reports.html_helpers import truncate as _truncate_html
from evaluatorq.contracts import ReportSection
from evaluatorq.simulation.reports.sections import build_report_sections
from evaluatorq.simulation.types import SimulationResult

# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_summary_html(section: ReportSection) -> str:
    data = section.data
    total = data.get("total_conversations", 0)
    achieved = data.get("goals_achieved", 0)
    failed = data.get("goals_failed", 0)
    errors = data.get("errors", 0)
    success_rate = data.get("success_rate", 0.0)
    avg_score = data.get("avg_goal_completion_score", 0.0)

    donut = _render_donut_chart_common(
        labels=["Achieved", "Not Achieved", "Errored"],
        values=[achieved, failed, errors],
        colors=[
            _STATUS_COLORS["success"],
            _STATUS_COLORS["failure"],
            _STATUS_COLORS["warning"],
        ],
        title="Goal Outcomes",
    )

    kpi_table = _html_table(
        ["Conversations", "Goals Achieved", "Success Rate", "Avg Score", "Errors"],
        [[
            f"<strong>{total}</strong>",
            f"<strong>{achieved}</strong>",
            f"<strong>{_pct(success_rate)}</strong>",
            f"<strong>{avg_score:.2f}</strong>",
            f"<strong>{errors}</strong>",
        ]],
    )

    return f"<section><h2>{_esc(section.title)}</h2>{kpi_table}{donut}</section>"


def _render_persona_breakdown_html(section: ReportSection) -> str:
    rows = section.data.get("rows", [])
    if not rows:
        return f"<section><h2>{_esc(section.title)}</h2><p>No persona data.</p></section>"
    table_rows = [
        [
            _esc(r["persona"]),
            str(r["conversations"]),
            str(r["goals_achieved"]),
            _pct(r["success_rate"]),
            f"{r['avg_goal_completion_score']:.2f}",
            f"{r['total_tokens']:,}",
        ]
        for r in rows
    ]
    table = _html_table(
        ["Persona", "Conversations", "Achieved", "Success", "Avg Score", "Tokens"],
        table_rows,
    )
    return f"<section><h2>{_esc(section.title)}</h2>{table}</section>"


def _render_scenario_breakdown_html(section: ReportSection) -> str:
    rows = section.data.get("rows", [])
    if not rows:
        return f"<section><h2>{_esc(section.title)}</h2><p>No scenario data.</p></section>"
    table_rows = [
        [
            _esc(r["scenario"]),
            str(r["conversations"]),
            str(r["goals_achieved"]),
            _pct(r["success_rate"]),
            f"{r['avg_goal_completion_score']:.2f}",
            f"{r['avg_turn_count']:.1f}",
        ]
        for r in rows
    ]
    table = _html_table(
        ["Scenario", "Conversations", "Achieved", "Success", "Avg Score", "Avg Turns"],
        table_rows,
    )
    return f"<section><h2>{_esc(section.title)}</h2>{table}</section>"


def _render_judge_verdicts_html(section: ReportSection) -> str:
    data = section.data
    terminated_by = data.get("terminated_by", {})
    rules_broken = data.get("rules_broken", {})

    parts = [f"<section><h2>{_esc(section.title)}</h2>"]

    if terminated_by:
        rows = [[_esc(r), str(c)] for r, c in
                sorted(terminated_by.items(), key=lambda kv: -kv[1])]
        parts.append("<h3>Terminated By</h3>")
        parts.append(_html_table(["Reason", "Count"], rows))

    if rules_broken:
        rows = [[_esc(r), str(c)] for r, c in rules_broken.items()]
        parts.append("<h3>Rules Broken</h3>")
        parts.append(_html_table(["Rule", "Count"], rows))
    else:
        parts.append("<p>No rules broken across any conversation.</p>")

    parts.append("</section>")
    return "".join(parts)


def _render_turn_metrics_html(section: ReportSection) -> str:
    data = section.data
    dist = data.get("turn_count_distribution", {})
    qualities = data.get("avg_quality_metrics", {})

    parts = [f"<section><h2>{_esc(section.title)}</h2>"]

    if dist:
        labels = [str(t) for t in dist]
        values = [float(c) for c in dist.values()]
        chart = _render_horizontal_bar_chart(
            labels=labels,
            values=values,
            color=_COLORS["teal_400"],
            title="Conversations by Turn Count",
            x_title="Conversations",
        )
        parts.append(chart)
        parts.append(_html_table(
            ["Turns", "Conversations"],
            [[t, str(c)] for t, c in dist.items()],
        ))

    if qualities:
        parts.append("<h3>Average Per-Turn Quality Metrics</h3>")
        parts.append(_html_table(
            ["Metric", "Avg Score"],
            [[_esc(k), f"{v:.2f}"] for k, v in qualities.items()],
        ))

    parts.append("</section>")
    return "".join(parts)


def _render_evaluator_scores_html(section: ReportSection) -> str:
    rows = section.data.get("rows", [])
    if not rows:
        return ""
    table_rows = [
        [_esc(r["evaluator"]), str(r["runs"]),
         f"{r['mean_score']:.2f}", f"{r['min_score']:.2f}", f"{r['max_score']:.2f}"]
        for r in rows
    ]
    table = _html_table(["Evaluator", "Runs", "Mean", "Min", "Max"], table_rows)
    return f"<section><h2>{_esc(section.title)}</h2>{table}</section>"


def _render_token_usage_html(section: ReportSection) -> str:
    data = section.data
    rows = [
        ["Prompt Tokens (total)", f"{data.get('prompt_tokens', 0):,}"],
        ["Completion Tokens (total)", f"{data.get('completion_tokens', 0):,}"],
        ["Total Tokens", f"{data.get('total_tokens', 0):,}"],
        ["Avg Total / Conversation", f"{data.get('avg_total_per_conversation', 0):,.0f}"],
        ["Avg Prompt / Conversation", f"{data.get('avg_prompt_per_conversation', 0):,.0f}"],
        ["Avg Completion / Conversation",
         f"{data.get('avg_completion_per_conversation', 0):,.0f}"],
    ]
    table = _html_table(["Metric", "Value"], rows)
    return f"<section><h2>{_esc(section.title)}</h2>{table}</section>"


def _render_errors_html(section: ReportSection) -> str:
    data = section.data
    total = data.get("total_errored", 0)
    by_message = data.get("by_message", {})
    parts = [f"<section><h2>{_esc(section.title)}</h2>",
             f"<p>Total errored conversations: <strong>{total}</strong></p>"]
    if by_message:
        rows = [[_esc(m), str(c)] for m, c in by_message.items()]
        parts.append(_html_table(["Error", "Count"], rows))
    parts.append("</section>")
    return "".join(parts)


def _render_individual_results_html(section: ReportSection) -> str:
    entries = section.data.get("entries", [])
    if not entries:
        return f"<section><h2>{_esc(section.title)}</h2><p>No conversations.</p></section>"

    parts = [f"<section><h2>{_esc(section.title)}</h2>"]
    for entry in entries:
        status_class = "status-success" if entry["goal_achieved"] else "status-failure"
        verdict = "ACHIEVED" if entry["goal_achieved"] else "NOT ACHIEVED"
        title = (
            f"#{entry['index'] + 1}: {_esc(entry['persona'])} / "
            f"{_esc(entry['scenario'])} "
            f"— <span class=\"{status_class}\">{verdict}</span>"
            f" ({entry['turn_count']} turns, "
            f"score {entry['goal_completion_score']:.2f})"
        )
        meta_rows = [
            ["Model", _esc(entry["model"])],
            ["Terminated by", _esc(entry["terminated_by"])],
            ["Tokens", f"{entry['total_tokens']:,}"],
        ]
        if entry["rules_broken"]:
            meta_rows.append(["Rules broken", _esc(", ".join(entry["rules_broken"]))])
        if entry["evaluator_scores"]:
            meta_rows.append(["Evaluator scores", _esc(", ".join(
                f"{k}={v:.2f}" for k, v in entry["evaluator_scores"].items()
            ))])
        if entry["error"]:
            meta_rows.append(["Error", _esc(entry["error"])])
        meta_rows.append(["Judge reason", _esc(entry["judge_reason"])])

        transcript_html = []
        for msg in entry["transcript"]:
            role = _esc(msg["role"])
            content = _esc(_truncate_html(msg.get("content", ""), 1500)).replace(
                "\n", "<br>"
            )
            transcript_html.append(
                f'<div class="transcript-message"><em>{role}:</em><br>{content}</div>'
            )

        parts.append(
            f"<details><summary>{title}</summary>"
            f"{_html_table(['Field', 'Value'], meta_rows)}"
            f"<h4>Transcript</h4>{''.join(transcript_html)}"
            f"</details>"
        )

    parts.append("</section>")
    return "".join(parts)


_SECTION_RENDERERS = {
    "summary": _render_summary_html,
    "persona_breakdown": _render_persona_breakdown_html,
    "scenario_breakdown": _render_scenario_breakdown_html,
    "judge_verdicts": _render_judge_verdicts_html,
    "turn_metrics": _render_turn_metrics_html,
    "evaluator_scores": _render_evaluator_scores_html,
    "token_usage": _render_token_usage_html,
    "errors": _render_errors_html,
    "individual_results": _render_individual_results_html,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_html(
    results: list[SimulationResult],
    *,
    target: str = "agent",
    run_date: datetime | None = None,
) -> str:
    """Render a list of simulation results as a self-contained HTML document."""
    sections = build_report_sections(results)
    summary_data = next(
        (s.data for s in sections if s.kind == "summary"),
        {},
    )

    head = (
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Agent Simulation Report</title>\n"
        f"<style>\n{_load_css()}\n</style>"
    )

    header_html = (
        f"<header><h1>Agent Simulation Report</h1>"
        f"<p><strong>Target:</strong> {_esc(target)}<br>"
        f"<strong>Date:</strong> {_format_date(run_date or datetime.now(tz=timezone.utc))}<br>"
        f"<strong>Conversations:</strong> {summary_data.get('total_conversations', 0)}<br>"
        f"<strong>Success Rate:</strong> {_pct(summary_data.get('success_rate', 0.0))}"
        f"</p></header>"
    )

    return _render_html_doc(
        sections,
        renderers=_SECTION_RENDERERS,
        head=head,
        body_header=header_html,
        body_footer='<footer><p class="footer">Generated by evaluatorq agent simulation suite.</p></footer>',
    )
