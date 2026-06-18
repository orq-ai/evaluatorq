"""Interactive Streamlit dashboard for exploring agent-simulation runs.

Launch via CLI:  evaluatorq sim ui /path/to/run.json
Or directly:     streamlit run dashboard.py -- /path/to/run.json

The dashboard is a *renderer* over the shared section layer in
``simulation.reports.sections``: filters are applied to the raw
``SimulationResult`` list, then ``build_report_sections`` recomputes every
aggregate so the panels and the Markdown/HTML exports always agree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pydantic import ValidationError

from evaluatorq.common.messages import coerce_content_text
from evaluatorq.simulation.reports.sections import build_report_sections
from evaluatorq.simulation.types import SimulationResult, SimulationRun
from evaluatorq.simulation.ui.colors import (
    COLORS,
    ORQ_SCALE_HEAT,
    QUALITATIVE,
)

if TYPE_CHECKING:
    from evaluatorq.contracts import ReportSection

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Agent Simulation",
    page_icon="\U0001f4ac",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Per-turn quality metrics, with the colour each gets across charts.
_QUALITY_METRICS: list[tuple[str, str]] = [
    ("response_quality", "Response quality"),
    ("hallucination_risk", "Hallucination risk"),
    ("tone_appropriateness", "Tone"),
    ("factual_accuracy", "Factual accuracy"),
]

_TERMINATED_BY_COLORS = {
    "judge": COLORS["success_400"],
    "max_turns": COLORS["yellow_400"],
    "timeout": COLORS["orange_300"],
    "error": COLORS["red_400"],
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@st.cache_data
def _load_run(path: str, mtime: float) -> dict[str, Any]:
    # mtime is unused in the body but part of the cache key, so overwriting a
    # run file in place (same path, new contents) invalidates the cached parse.
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_run(data: dict[str, Any]) -> SimulationRun:
    return SimulationRun.model_validate(data)


def _section(sections: list[ReportSection], kind: str) -> dict[str, Any]:
    """Return the data dict for a section kind, or an empty dict if absent."""
    for s in sections:
        if s.kind == kind:
            return s.data
    return {}


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


# ---------------------------------------------------------------------------
# Filters (operate on raw results so aggregates recompute reactively)
# ---------------------------------------------------------------------------


def _meta(result: SimulationResult, key: str) -> str:
    return str(result.metadata.get(key, "unknown"))


def _render_sidebar_filters(results: list[SimulationResult]) -> list[SimulationResult]:
    st.sidebar.subheader("Filters")

    personas = sorted({_meta(r, "persona") for r in results})
    scenarios = sorted({_meta(r, "scenario") for r in results})
    terminated = sorted({r.terminated_by.value for r in results})

    sel_personas = st.sidebar.multiselect("Persona", personas, default=personas)
    sel_scenarios = st.sidebar.multiselect("Scenario", scenarios, default=scenarios)
    sel_terminated = st.sidebar.multiselect("Terminated by", terminated, default=terminated)
    goal_filter = st.sidebar.radio(
        "Goal outcome",
        options=["All", "Achieved", "Not achieved"],
        horizontal=True,
    )

    def _keep(r: SimulationResult) -> bool:
        if _meta(r, "persona") not in sel_personas:
            return False
        if _meta(r, "scenario") not in sel_scenarios:
            return False
        if r.terminated_by.value not in sel_terminated:
            return False
        if goal_filter == "Achieved" and not r.goal_achieved:
            return False
        return not (goal_filter == "Not achieved" and r.goal_achieved)

    return [r for r in results if _keep(r)]


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------


def _render_overview(sections: list[ReportSection]) -> None:
    summary = _section(sections, "summary")
    if not summary:
        st.info("No results match the current filters.")
        return

    cols = st.columns(5)
    cols[0].metric("Goal completion", _pct(summary.get("success_rate", 0.0)))
    cols[1].metric("Avg score", f"{summary.get('avg_goal_completion_score', 0.0):.2f}")
    cols[2].metric("Conversations", summary.get("total_conversations", 0))
    cols[3].metric("Avg turns", f"{summary.get('avg_turn_count', 0.0):.1f}")
    cols[4].metric("Errors", summary.get("errors", 0))

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Outcomes")
        labels = ["Achieved", "Not achieved", "Errors"]
        values = [
            summary.get("goals_achieved", 0),
            summary.get("goals_failed", 0),
            summary.get("errors", 0),
        ]
        colors = [COLORS["success_400"], COLORS["red_400"], COLORS["yellow_400"]]
        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker_colors=colors,
                sort=False,
            )
        )
        fig.update_layout(margin={"t": 10, "b": 10, "l": 10, "r": 10}, height=300)
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Tokens")
        tokens = _section(sections, "token_usage")
        st.metric("Total tokens", f"{tokens.get('total_tokens', 0):,}")
        st.caption(
            f"Prompt {tokens.get('prompt_tokens', 0):,} · "
            f"Completion {tokens.get('completion_tokens', 0):,} · "
            f"Avg {tokens.get('avg_total_per_conversation', 0.0):.0f}/conv"
        )

    overview = _section(sections, "overview")
    personas = overview.get("personas", [])
    scenarios = overview.get("scenarios", [])
    if personas or scenarios:
        pcol, scol = st.columns(2)
        with pcol:
            st.subheader(f"Personas ({len(personas)})")
            for p in personas:
                with st.expander(f"{p['name']} · {p['conversations']} conv"):
                    if p.get("background"):
                        st.caption(p["background"])
                    if p.get("traits"):
                        st.json(p["traits"], expanded=False)
        with scol:
            st.subheader(f"Scenarios ({len(scenarios)})")
            for s in scenarios:
                with st.expander(s["name"]):
                    if s.get("goal"):
                        st.markdown(f"**Goal:** {s['goal']}")
                    if s.get("context"):
                        st.caption(s["context"])
                    for c in s.get("criteria", []):
                        st.markdown(f"- _{c['type']}_: {c['description']}")


def _render_breakdown(sections: list[ReportSection]) -> None:
    heat = _section(sections, "persona_scenario_heatmap")
    cells = heat.get("cells", [])
    if cells:
        st.subheader("Persona x Scenario goal completion")
        personas = heat.get("personas", [])
        scenarios = heat.get("scenarios", [])
        lookup = {(c["persona"], c["scenario"]): c for c in cells}
        z = [[lookup.get((p, s), {}).get("success_rate") for s in scenarios] for p in personas]
        text = [
            [
                (f"{cell['success_rate'] * 100:.0f}%\nn={cell['n']}" if (cell := lookup.get((p, s))) else "")
                for s in scenarios
            ]
            for p in personas
        ]
        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=scenarios,
                y=personas,
                text=text,
                texttemplate="%{text}",
                colorscale=ORQ_SCALE_HEAT,
                zmin=0,
                zmax=1,
                colorbar={"title": "Goal rate"},
            )
        )
        fig.update_layout(margin={"t": 10, "b": 10}, height=max(260, 60 * len(personas)))
        st.plotly_chart(fig, width="stretch")

    dist = _section(sections, "score_distribution")
    scores = dist.get("scores", [])
    if scores:
        st.subheader("Goal completion score distribution")
        fig = px.histogram(x=scores, nbins=20, color_discrete_sequence=[COLORS["teal_400"]])
        fig.update_layout(
            xaxis_title="Goal completion score",
            yaxis_title="Conversations",
            margin={"t": 10, "b": 10},
            height=280,
        )
        st.plotly_chart(fig, width="stretch")

    pcol, scol = st.columns(2)
    with pcol:
        rows = _section(sections, "persona_breakdown").get("rows", [])
        if rows:
            st.subheader("Per-persona")
            st.dataframe(
                [
                    {
                        "Persona": r["persona"],
                        "Conv": r["conversations"],
                        "Goal rate": _pct(r["success_rate"]),
                        "Avg score": round(r["avg_goal_completion_score"], 2),
                        "Tokens": r["total_tokens"],
                    }
                    for r in rows
                ],
                width="stretch",
                hide_index=True,
            )
    with scol:
        rows = _section(sections, "scenario_breakdown").get("rows", [])
        if rows:
            st.subheader("Per-scenario")
            st.dataframe(
                [
                    {
                        "Scenario": r["scenario"],
                        "Conv": r["conversations"],
                        "Goal rate": _pct(r["success_rate"]),
                        "Avg score": round(r["avg_goal_completion_score"], 2),
                        "Avg turns": round(r["avg_turn_count"], 1),
                    }
                    for r in rows
                ],
                width="stretch",
                hide_index=True,
            )

    failure = _section(sections, "failure_mode").get("rows", [])
    if failure:
        st.subheader("Top failure modes")
        labels = [row[0] for row in failure]
        counts = [row[1] for row in failure]
        fig = px.bar(
            x=counts,
            y=labels,
            orientation="h",
            color_discrete_sequence=[COLORS["red_400"]],
        )
        fig.update_layout(
            xaxis_title="Failures",
            yaxis_title="",
            yaxis={"autorange": "reversed"},
            margin={"t": 10, "b": 10, "l": 10},
            height=max(240, 28 * len(labels)),
        )
        st.plotly_chart(fig, width="stretch")


def _render_transcripts(sections: list[ReportSection]) -> None:
    entries = _section(sections, "individual_results").get("entries", [])
    if not entries:
        st.info("No conversations match the current filters.")
        return

    table = [
        {
            "#": e["index"] + 1,
            "Persona": e["persona"],
            "Scenario": e["scenario"],
            "Goal": "yes" if e["goal_achieved"] else "no",
            "Score": round(e["goal_completion_score"], 2),
            "Turns": e["turn_count"],
            "Terminated": e["terminated_by"],
            "Tokens": e["total_tokens"],
        }
        for e in entries
    ]

    st.download_button(
        "Download table (JSON)",
        data=json.dumps(entries, indent=2),
        file_name="sim-conversations.json",
        mime="application/json",
    )

    event = st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    rows = event.selection.rows if event and event.selection else []
    if not rows:
        st.caption("Select a row above to view its transcript.")
        return

    entry = entries[rows[0]]
    st.divider()
    st.subheader(f"#{entry['index'] + 1} · {entry['persona']} · {entry['scenario']}")

    m = st.columns(4)
    m[0].metric("Goal achieved", "yes" if entry["goal_achieved"] else "no")
    m[1].metric("Score", f"{entry['goal_completion_score']:.2f}")
    m[2].metric("Turns", entry["turn_count"])
    m[3].metric("Terminated by", entry["terminated_by"])

    if entry.get("judge_reason"):
        st.markdown(f"**Judge:** {entry['judge_reason']}")
    if entry.get("error"):
        st.error(entry["error"])

    criteria = entry.get("criteria", [])
    if criteria:
        st.markdown("**Criteria**")
        for c in criteria:
            icon = "✅" if c["passed"] else ("⛔" if c.get("safety") else "❌")
            ctype = f" _{c['type']}_" if c.get("type") else ""
            st.markdown(f"{icon} {c['description']}{ctype}")

    scores = entry.get("evaluator_scores") or {}
    if scores:
        st.markdown("**Evaluator scores**")
        st.json(scores, expanded=False)

    st.markdown("**Transcript**")
    role_label = {"user": "User (sim)", "assistant": "Target", "system": "System", "tool": "Tool"}
    for msg in entry.get("transcript", []):
        label = role_label.get(msg["role"], msg["role"])
        with st.chat_message("user" if msg["role"] == "user" else "assistant"):
            st.markdown(f"**{label}**")
            st.markdown(coerce_content_text(msg.get("content")) or "_(empty)_")


def _render_turn_quality(sections: list[ReportSection]) -> None:
    timeline = _section(sections, "turn_quality_timeline")
    turns = timeline.get("turns", [])
    series = timeline.get("series", {})
    if turns and series:
        st.subheader("Per-turn quality")
        fig = go.Figure()
        colors = dict(zip([k for k, _ in _QUALITY_METRICS], QUALITATIVE, strict=False))
        labels = dict(_QUALITY_METRICS)
        for metric, values in series.items():
            fig.add_trace(
                go.Scatter(
                    x=turns,
                    y=values,
                    mode="lines+markers",
                    name=labels.get(metric, metric),
                    line={"color": colors.get(metric)},
                    connectgaps=True,
                )
            )
        fig.update_layout(
            xaxis_title="Turn",
            yaxis_title="Score",
            margin={"t": 10, "b": 10},
            height=320,
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("No per-turn quality metrics were recorded for this run.")

    tm = _section(sections, "turn_metrics")
    dist = tm.get("turn_count_distribution", {})
    if dist:
        st.subheader("Turn-count distribution")
        fig = px.bar(
            x=list(dist.keys()),
            y=list(dist.values()),
            color_discrete_sequence=[COLORS["teal_400"]],
        )
        fig.update_layout(
            xaxis_title="Turns", yaxis_title="Conversations", margin={"t": 10, "b": 10}, height=260
        )
        st.plotly_chart(fig, width="stretch")

    avg = tm.get("avg_quality_metrics", {})
    if avg:
        cols = st.columns(len(avg))
        labels = dict(_QUALITY_METRICS)
        for col, (metric, value) in zip(cols, avg.items(), strict=False):
            col.metric(labels.get(metric, metric), f"{value:.2f}")


def _render_tokens(sections: list[ReportSection]) -> None:
    tokens = _section(sections, "token_usage")
    cols = st.columns(3)
    cols[0].metric("Prompt", f"{tokens.get('prompt_tokens', 0):,}")
    cols[1].metric("Completion", f"{tokens.get('completion_tokens', 0):,}")
    cols[2].metric("Total", f"{tokens.get('total_tokens', 0):,}")

    entries = _section(sections, "individual_results").get("entries", [])
    if entries:
        st.subheader("Tokens per conversation")
        per_conv = [e["total_tokens"] for e in entries]
        fig = px.histogram(x=per_conv, nbins=20, color_discrete_sequence=[COLORS["orange_300"]])
        fig.update_layout(
            xaxis_title="Total tokens", yaxis_title="Conversations", margin={"t": 10, "b": 10}, height=280
        )
        st.plotly_chart(fig, width="stretch")


def _render_evaluators(sections: list[ReportSection]) -> None:
    rows = _section(sections, "evaluator_scores").get("rows", [])
    if not rows:
        st.caption("No custom evaluator scores were recorded for this run.")
        return
    st.dataframe(
        [
            {
                "Evaluator": r["evaluator"],
                "Runs": r["runs"],
                "Mean": round(r["mean_score"], 2),
                "Min": round(r["min_score"], 2),
                "Max": round(r["max_score"], 2),
            }
            for r in rows
        ],
        width="stretch",
        hide_index=True,
    )


def _render_judge_errors(sections: list[ReportSection]) -> None:
    verdicts = _section(sections, "judge_verdicts")
    terminated = verdicts.get("terminated_by", {})
    if terminated:
        st.subheader("Termination reasons")
        labels = list(terminated.keys())
        fig = px.bar(
            x=labels,
            y=list(terminated.values()),
            color=labels,
            color_discrete_map=_TERMINATED_BY_COLORS,
        )
        fig.update_layout(
            xaxis_title="", yaxis_title="Conversations", showlegend=False, margin={"t": 10, "b": 10}, height=260
        )
        st.plotly_chart(fig, width="stretch")

    rules = verdicts.get("rules_broken", {})
    if rules:
        st.subheader("Rules broken")
        st.dataframe(
            [{"Rule": k, "Count": v} for k, v in rules.items()],
            width="stretch",
            hide_index=True,
        )

    errors = _section(sections, "errors")
    by_message = errors.get("by_message", {})
    if by_message:
        st.subheader(f"Errors ({errors.get('total_errored', 0)})")
        st.dataframe(
            [{"Error": k, "Count": v} for k, v in by_message.items()],
            width="stretch",
            hide_index=True,
        )
    elif not terminated and not rules:
        st.caption("No judge verdicts or errors recorded.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _render_dashboard() -> None:
    args = sys.argv[1:]
    run_path_str = args[0] if args else ""

    st.sidebar.title("Agent Simulation")

    if run_path_str and Path(run_path_str).exists():
        try:
            run = _parse_run(_load_run(run_path_str, Path(run_path_str).stat().st_mtime))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            st.error(f"Could not load run file: {exc}")
            st.stop()
            return
    else:
        uploaded = st.sidebar.file_uploader("Upload run JSON", type=["json"])
        run = None
        if uploaded is not None:
            try:
                run = _parse_run(json.loads(uploaded.read()))
            except (json.JSONDecodeError, ValidationError) as exc:
                st.error(f"Could not parse uploaded file: {exc}")
                st.stop()
                return

    if run is None:
        st.info(
            "**Agent Simulation Dashboard**\n\n"
            "Upload a run JSON in the sidebar, or launch via CLI:\n\n"
            "```\nevaluatorq sim ui /path/to/run.json\n```"
        )
        st.stop()
        return  # st.stop() halts the script; this keeps the type-checker happy.

    st.sidebar.divider()
    st.sidebar.markdown(f"**Run:** {run.run_name}")
    st.sidebar.markdown(f"**Created:** {run.created_at:%Y-%m-%d %H:%M}")
    st.sidebar.markdown(f"**Mode:** {run.mode} · **Target:** {run.target_kind}")
    st.sidebar.markdown(f"**Conversations:** {run.total_results}")
    st.sidebar.divider()

    filtered = _render_sidebar_filters(run.results)
    st.sidebar.caption(f"{len(filtered)} / {len(run.results)} conversations shown")

    if not filtered:
        st.warning("No conversations match the current filters.")
        st.stop()

    sections = build_report_sections(filtered)

    tabs = st.tabs(
        ["Overview", "Breakdown", "Transcripts", "Turn quality", "Tokens", "Evaluators", "Judge & errors"]
    )
    with tabs[0]:
        _render_overview(sections)
    with tabs[1]:
        _render_breakdown(sections)
    with tabs[2]:
        _render_transcripts(sections)
    with tabs[3]:
        _render_turn_quality(sections)
    with tabs[4]:
        _render_tokens(sections)
    with tabs[5]:
        _render_evaluators(sections)
    with tabs[6]:
        _render_judge_errors(sections)


_render_dashboard()
