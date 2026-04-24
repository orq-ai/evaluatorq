"""Interactive Streamlit dashboard for exploring red team reports.

Launch via CLI:  evaluatorq redteam ui /path/to/report.json
Or directly:     streamlit run dashboard.py
"""

from __future__ import annotations

import json
import operator
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from loguru import logger

from evaluatorq.redteam.contracts import (
    OWASP_CATEGORY_NAMES,
    AgentContext,
    RedTeamReport,
    RedTeamResult,
    ReportSummary,
)
from evaluatorq.redteam.reports.converters import compute_report_summary
from evaluatorq.redteam.reports.export_md import export_markdown
from evaluatorq.redteam.reports.guidance import REMEDIATION_GUIDANCE
from evaluatorq.redteam.ui.colors import (
    COLORS,
    ORQ_SCALE_AGENT,
    ORQ_SCALE_HEAT,
    QUALITATIVE,
    SEVERITY_COLORS,
    SEVERITY_ORDER,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Red Team Report",
    page_icon="\U0001f6e1\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


def _status_icon(rate: float) -> str:
    if rate >= 0.9:
        return "\u2705"
    if rate >= 0.8:
        return "\U0001f7e0"
    return "\u274c"


def _asr_color(asr: float) -> str:
    """Interpolate from success_400 (low ASR) through yellow_400 to red_400 (high ASR)."""
    # success_400: #2ebd85 (46, 189, 133)
    # yellow_400: #f2b600 (242, 182, 0)
    # red_400: #d92d20 (217, 45, 32)
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
# Remediation guidance is imported from reports.guidance (single source of truth).
# REMEDIATION_GUIDANCE is re-exported here for backward compatibility with any
# code that imports it directly from this module.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Report loading
# ---------------------------------------------------------------------------


def _find_reports(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".json":
        return [path]
    if path.is_dir():
        reports = sorted(path.rglob("report.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not reports:
            reports = sorted(path.rglob("*report*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return reports
    return []


@st.cache_data
def _load_report(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def _parse_report(data: dict[str, Any]) -> RedTeamReport:
    return RedTeamReport.model_validate(data)


def _report_cache_key(report: RedTeamReport) -> str:
    """Stable cache key for a report based on identity fields."""
    return f"{report.created_at.isoformat()}_{len(report.results)}"


@st.cache_data
def _cached_compute_summary(_key: str, _results: list[RedTeamResult]) -> ReportSummary:
    return compute_report_summary(_results)


@st.cache_data
def _cached_export_markdown(_key: str, _report: RedTeamReport) -> str:
    return export_markdown(_report)


@st.cache_data
def _cached_export_html(_key: str, _report: RedTeamReport) -> str | None:
    try:
        from evaluatorq.redteam.reports.export_html import export_html

        return export_html(_report)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Helpers for computing per-result aggregations from report.results
# ---------------------------------------------------------------------------


def _weakest_link(results: list[RedTeamResult]) -> tuple[str, int]:
    """Find the vulnerability+technique combo with most successful attacks."""
    combos: Counter[tuple[str, str]] = Counter()
    for r in results:
        if r.vulnerable:
            vuln = r.attack.vulnerability or r.attack.category
            combos[vuln, r.attack.attack_technique.value] += 1
    if not combos:
        return "None", 0
    (vuln, tech), count = combos.most_common(1)[0]
    return f"{_fmt_vulnerability(vuln)} + {tech}", count


def _critical_exposure(results: list[RedTeamResult]) -> int:
    return sum(1 for r in results if r.vulnerable and r.attack.severity.value == "critical")


def _compliant_vulnerabilities(summary: ReportSummary) -> tuple[int, int]:
    """Return (fully_resistant, total) vulnerability counts."""
    total = len(summary.by_vulnerability)
    compliant = sum(1 for v in summary.by_vulnerability.values() if v.vulnerabilities_found == 0)
    return compliant, total


# ---------------------------------------------------------------------------
# Sidebar agent context
# ---------------------------------------------------------------------------

def _esc_html(text: str) -> str:
    """Escape HTML special characters to prevent XSS."""
    import html as _html

    return _html.escape(text)


def _chip_css(color: str) -> str:
    """Return inline CSS for a capability chip using a brand color."""
    return (
        f"display:inline-block;background:{color}1f;border:1px solid {color}4d;"
        f"color:{color};border-radius:14px;padding:2px 9px;font-size:0.76rem;"
        f"font-family:monospace;margin:2px 2px;"
    )


def _render_capability_chips(ctx: AgentContext) -> None:
    """Render tool/memory/knowledge chips for an agent context."""
    label = (
        'font-size:0.7rem;font-weight:600;text-transform:uppercase;'
        'letter-spacing:0.05em;color:#888;margin-bottom:4px;'
    )
    parts: list[str] = []
    if ctx.tools:
        tool_color = COLORS['success_400']
        chips = "".join(
            f'<span style="{_chip_css(tool_color)}">{_esc_html(t.name or "unknown")}</span>'
            for t in ctx.tools
        )
        parts.append(f'<div style="{label}">Tools ({len(ctx.tools)})</div>{chips}')

    if ctx.memory_stores:
        mem_color = COLORS['purple_400']
        chips = "".join(
            f'<span style="{_chip_css(mem_color)}">{_esc_html(m.key or m.id or "unknown")}</span>'
            for m in ctx.memory_stores
        )
        parts.append(f'<div style="{label}margin-top:8px;">Memory ({len(ctx.memory_stores)})</div>{chips}')

    if ctx.knowledge_bases:
        kb_color = COLORS['orange_300']
        chips = "".join(
            f'<span style="{_chip_css(kb_color)}">{_esc_html(kb.name or kb.key or kb.id or "unknown")}</span>'
            for kb in ctx.knowledge_bases
        )
        parts.append(f'<div style="{label}margin-top:8px;">Knowledge ({len(ctx.knowledge_bases)})</div>{chips}')

    if parts:
        st.markdown(
            f'<div style="padding-bottom:12px;">{"".join(parts)}</div>',
            unsafe_allow_html=True,
        )


def _fetch_agent_context_from_orq(agent_key: str) -> AgentContext | None:
    """Attempt to fetch agent context from the ORQ API. Returns None on failure."""
    try:
        import asyncio
        import os
        from concurrent.futures import ThreadPoolExecutor

        if not os.environ.get("ORQ_API_KEY"):
            return None
        from evaluatorq.redteam.backends.orq import _get_orq_api_key, _get_orq_server_url, _orq_cls
        if _orq_cls is None:
            return None
        client = _orq_cls(api_key=_get_orq_api_key(), server_url=_get_orq_server_url())

        from evaluatorq.redteam.adaptive.agent_context import retrieve_agent_context

        # Run in a separate thread with its own event loop to avoid conflicts
        # with Streamlit's running event loop.
        def _run() -> AgentContext:
            return asyncio.run(retrieve_agent_context(client, agent_key))

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(_run).result(timeout=10)
    except Exception as exc:
        logger.warning(f'Failed to fetch agent context for {agent_key!r}: {exc}')
        return None


@st.cache_data(show_spinner="Fetching agent details from orq...", ttl=300)
def _cached_fetch_agent_contexts(agent_keys: tuple[str, ...]) -> dict[str, Any]:
    """Fetch and cache agent contexts from ORQ API. Returns serialized dicts."""
    result: dict[str, Any] = {}
    for key in agent_keys:
        ctx = _fetch_agent_context_from_orq(key)
        if ctx is not None:
            result[key] = ctx.model_dump(mode="json")
    return result


def _agents_from_results(results: list[RedTeamResult]) -> dict[str, AgentContext]:
    """Build agent context stubs from result agent info (key, display_name, model)."""
    agents: dict[str, AgentContext] = {}
    for r in results:
        key = r.agent.key or r.agent.display_name or "unknown"
        if key not in agents:
            agents[key] = AgentContext(
                key=key,
                display_name=r.agent.display_name or r.agent.key,
                model=r.agent.model,
            )
    return agents


def _resolve_agent_contexts(report: RedTeamReport) -> dict[str, AgentContext]:
    """Resolve agent contexts: report data > result metadata > orq API > stub."""
    # 1. Use stored agent_contexts dict (new reports)
    if report.agent_contexts:
        return dict(report.agent_contexts)

    # 2. Always build from results — covers all agents in the report
    from_results = _agents_from_results(report.results)

    # Merge in singular agent_context if it has richer data
    if report.agent_context:
        key = report.agent_context.key or report.agent_context.display_name or "Agent"
        from_results[key] = report.agent_context

    if from_results:
        # Try enriching with orq API (best-effort)
        try:
            api_keys = tuple(sorted(from_results.keys()))
            fetched = _cached_fetch_agent_contexts(api_keys)
            for key, ctx_dict in fetched.items():
                from_results[key] = AgentContext.model_validate(ctx_dict)
        except Exception as exc:
            logger.warning(f'Agent context API enrichment failed: {exc}')
        return from_results

    # 3. Last resort: tested_agents names as stubs
    if report.tested_agents:
        return {k: AgentContext(key=k, display_name=k) for k in report.tested_agents}

    return {}


def _render_agent_card(key: str, ctx: AgentContext) -> None:
    """Render a single agent info card inside a container."""
    name = ctx.display_name or ctx.key or key
    model = ctx.model or "Unknown"
    st.markdown(f"**{_esc_html(name)}**")
    st.markdown(f"Model: **`{_esc_html(model)}`**")
    if ctx.description:
        st.markdown(f"_{_esc_html(ctx.description)}_", unsafe_allow_html=False)
    if ctx.tools or ctx.memory_stores or ctx.knowledge_bases:
        _render_capability_chips(ctx)


def _render_agent_context_section(report: RedTeamReport) -> None:
    """Render agent context as simple side-by-side cards."""
    contexts = _resolve_agent_contexts(report)

    if not contexts:
        # Fall back to tested_agents names only
        if report.tested_agents:
            contexts = {k: AgentContext(key=k, display_name=k) for k in report.tested_agents}
        else:
            return

    st.subheader("Target Agents")
    cols = st.columns(len(contexts))
    for col, (key, ctx) in zip(cols, sorted(contexts.items()), strict=False):
        with col, st.container(border=True):
            _render_agent_card(key, ctx)
    st.divider()


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------


def _render_sidebar_filters(results: list[RedTeamResult]) -> list[RedTeamResult]:
    """Render global filter widgets in a sidebar expander. Returns filtered results."""
    if not results:
        return results

    # Derive available options from full result set
    all_categories = sorted({r.attack.category for r in results})
    all_severities = [s for s in SEVERITY_ORDER if any(r.attack.severity.value == s for r in results)]
    all_techniques = sorted({r.attack.attack_technique.value for r in results})
    all_delivery = sorted({dm.value for r in results for dm in (r.attack.delivery_methods or [])})
    all_vulnerabilities = sorted({r.attack.vulnerability for r in results if r.attack.vulnerability})
    all_agents = sorted({r.agent.key or r.agent.display_name or "unknown" for r in results})

    with st.sidebar.expander("Filters", expanded=False):
        # Result type
        result_filter = st.radio(
            "Result",
            options=["All", "Vulnerable", "Resistant", "Error"],
            index=0,
            horizontal=True,
            key="filter_result",
        )

        # Agent (only show when multi-agent)
        if len(all_agents) > 1:
            sel_agents = st.multiselect(
                "Agent",
                options=all_agents,
                default=all_agents,
                key="filter_agents",
            )
        else:
            sel_agents = all_agents

        # Category
        sel_categories = st.multiselect(
            "Category",
            options=all_categories,
            default=all_categories,
            format_func=lambda c: _fmt_category(c),
            key="filter_categories",
        )

        # Severity
        sel_severities = st.multiselect(
            "Severity",
            options=all_severities,
            default=all_severities,
            key="filter_severities",
        )

        # Technique
        sel_techniques = st.multiselect(
            "Attack Technique",
            options=all_techniques,
            default=all_techniques,
            key="filter_techniques",
        )

        # Delivery method
        if all_delivery:
            sel_delivery = st.multiselect(
                "Delivery Method",
                options=all_delivery,
                default=all_delivery,
                key="filter_delivery",
            )
        else:
            sel_delivery = []

        # Vulnerability
        if all_vulnerabilities:
            sel_vulnerabilities = st.multiselect(
                "Vulnerability",
                options=all_vulnerabilities,
                default=all_vulnerabilities,
                key="filter_vulnerabilities",
            )
        else:
            sel_vulnerabilities = []

        # Reset button
        if st.button("Reset All Filters", width="stretch", key="reset_filters"):
            st.session_state["filter_result"] = "All"
            st.session_state["filter_categories"] = all_categories
            st.session_state["filter_severities"] = all_severities
            st.session_state["filter_techniques"] = all_techniques
            if all_delivery:
                st.session_state["filter_delivery"] = all_delivery
            if all_vulnerabilities:
                st.session_state["filter_vulnerabilities"] = all_vulnerabilities
            if len(all_agents) > 1:
                st.session_state["filter_agents"] = all_agents
            st.rerun()

    # Apply filters
    filtered = results

    if result_filter == "Vulnerable":
        filtered = [r for r in filtered if r.vulnerable]
    elif result_filter == "Resistant":
        filtered = [r for r in filtered if not r.vulnerable and not r.error]
    elif result_filter == "Error":
        filtered = [r for r in filtered if r.error]

    if set(sel_categories) != set(all_categories):
        filtered = [r for r in filtered if r.attack.category in sel_categories]

    if set(sel_severities) != set(all_severities):
        filtered = [r for r in filtered if r.attack.severity.value in sel_severities]

    if set(sel_techniques) != set(all_techniques):
        filtered = [r for r in filtered if r.attack.attack_technique.value in sel_techniques]

    if all_delivery and set(sel_delivery) != set(all_delivery):
        filtered = [r for r in filtered if any(dm.value in sel_delivery for dm in (r.attack.delivery_methods or []))]

    if all_vulnerabilities and set(sel_vulnerabilities) != set(all_vulnerabilities):
        filtered = [r for r in filtered if r.attack.vulnerability in sel_vulnerabilities]

    if len(all_agents) > 1 and set(sel_agents) != set(all_agents):
        filtered = [r for r in filtered if (r.agent.key or r.agent.display_name or "unknown") in sel_agents]

    # Show count in sidebar
    if len(filtered) != len(results):
        st.sidebar.caption(f"Showing {len(filtered)} of {len(results)} results")

    return filtered


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------


def _render_dashboard() -> None:
    # Resolve report path from CLI args (passed after -- by streamlit run)
    args = sys.argv[1:]
    report_path_str = args[0] if args else ""
    report_path = Path(report_path_str) if report_path_str else None

    # Sidebar: report selection
    st.sidebar.title("Red Team Dashboard")

    report_files = _find_reports(report_path) if report_path and report_path.exists() else []

    # File uploader fallback
    if not report_files:
        uploaded = st.sidebar.file_uploader("Upload report JSON", type=["json"])
        if uploaded is not None:
            data = json.loads(uploaded.read())
            report = _parse_report(data)
        else:
            st.info(
                "**Welcome to the Red Team Dashboard**\n\n"
                "Upload a report JSON file in the sidebar, or launch via CLI:\n\n"
                "```\nevaluatorq redteam ui /path/to/report.json\n```"
            )
            st.stop()
    else:
        if len(report_files) == 1:
            selected_path = report_files[0]
        else:
            labels = [str(p.relative_to(report_path)) if report_path and report_path.is_dir() else str(p) for p in report_files]
            idx = st.sidebar.selectbox("Select report", range(len(labels)), format_func=lambda i: labels[i])
            selected_path = report_files[idx]

        data = _load_report(str(selected_path))
        report = _parse_report(data)

    summary = report.summary

    # Sidebar: report metadata
    st.sidebar.divider()
    if report.description:
        st.sidebar.markdown(f"*{report.description}*")
    st.sidebar.markdown(f"**Pipeline:** {report.pipeline.value}")
    if report.framework:
        st.sidebar.markdown(f"**Framework:** {report.framework.value}")
    st.sidebar.markdown(f"**Created:** {report.created_at:%Y-%m-%d %H:%M}")
    agent_keys = sorted({r.agent.key or r.agent.display_name or "unknown" for r in report.results})
    if agent_keys:
        st.sidebar.markdown(f"**Agents:** {', '.join(agent_keys)}")
    if report.duration_seconds is not None:
        mins, secs = divmod(int(report.duration_seconds), 60)
        st.sidebar.markdown(f"**Duration:** {mins}m {secs}s")

    # Markdown export download button
    st.sidebar.divider()
    target = ", ".join(sorted({r.agent.key or r.agent.display_name or "unknown" for r in report.results})) or "report"
    md_content = _cached_export_markdown(_report_cache_key(report), report)
    md_filename = f"redteam-report-{target.replace('/', '-').replace(':', '-')}-{report.created_at:%Y%m%d_%H%M%S}.md"
    st.sidebar.download_button(
        label="Download Markdown Report",
        data=md_content,
        file_name=md_filename,
        mime="text/markdown",
    )

    # HTML export — on-demand to avoid blocking dashboard startup
    html_session_key = f"html_export_{_report_cache_key(report)}"
    if html_session_key not in st.session_state:
        if st.sidebar.button("Download HTML Report"):
            with st.sidebar.spinner("Rendering HTML report..."):
                st.session_state[html_session_key] = _cached_export_html(_report_cache_key(report), report)
            st.rerun()
    else:
        html_content = st.session_state[html_session_key]
        if html_content is not None:
            html_filename = f"redteam-report-{target.replace('/', '-').replace(':', '-')}-{report.created_at:%Y%m%d_%H%M%S}.html"
            st.sidebar.download_button(
                label="Download HTML Report",
                data=html_content,
                file_name=html_filename,
                mime="text/html",
            )

    # Global filters
    st.sidebar.divider()
    filtered_results = _render_sidebar_filters(report.results)

    # Recompute summary from filtered results when filters are active
    is_filtered = len(filtered_results) != len(report.results)
    if is_filtered:
        filter_key = f"{_report_cache_key(report)}_filtered_{len(filtered_results)}"
        summary = _cached_compute_summary(filter_key, filtered_results)
    else:
        summary = report.summary

    # Build a filtered report for render functions that need report.results
    filtered_report = report.model_copy(update={"results": filtered_results, "summary": summary})

    # Title
    framework_name = report.framework.value if report.framework else "Security"
    st.title("Red Team Security Report")

    if is_filtered:
        st.caption(f"Showing {len(filtered_results)} of {len(report.results)} results (filtered)")

    # Detect multi-agent
    unique_agents = sorted({r.agent.key or r.agent.display_name or "unknown" for r in filtered_results})
    has_multi_agent = len(unique_agents) >= 2

    # Build tabs
    tab_names = ["\U0001f4ca Summary", "\U0001f50d Breakdown", "\U0001f4c2 Explorer"]
    has_usage = summary.token_usage_total is not None or any(
        r.execution and r.execution.token_usage for r in filtered_results
    )
    if has_usage:
        tab_names.append("\U0001f4b0 Usage")
    if summary.total_errors > 0:
        tab_names.append("\u26a0\ufe0f Error Analysis")
    if has_multi_agent:
        tab_names.append("\U0001f916 Comparison")
    tab_names.append("\U0001f4d1 Methodology")
    tabs = st.tabs(tab_names)

    tab_idx = 0
    with tabs[tab_idx]:
        _render_executive_summary(filtered_report, summary, framework_name)
    tab_idx += 1

    with tabs[tab_idx]:
        _render_technical_analysis(filtered_report, summary)
    tab_idx += 1

    with tabs[tab_idx]:
        _render_data_explorer(filtered_report, summary)
    tab_idx += 1

    if has_usage:
        with tabs[tab_idx]:
            _render_usage_tab(filtered_report, summary, unique_agents if has_multi_agent else [])
        tab_idx += 1

    if summary.total_errors > 0:
        with tabs[tab_idx]:
            _render_errors_tab(summary, filtered_report)
        tab_idx += 1

    if has_multi_agent:
        with tabs[tab_idx]:
            _render_agent_comparison(filtered_report, summary, unique_agents)
        tab_idx += 1

    with tabs[tab_idx]:
        _render_methodology_tab(filtered_report, summary, unique_agents if has_multi_agent else [])


# ---------------------------------------------------------------------------
# Focus Areas (top-5 risk vulnerabilities with remediation guidance)
# ---------------------------------------------------------------------------

_SEVERITY_WEIGHTS: dict[str, int] = {
    'critical': 8,
    'high': 4,
    'medium': 2,
    'low': 1,
}


def _dominant_severity_for_vulnerability(results: list[RedTeamResult], vuln_id: str) -> str:
    """Return the most common severity among vulnerable results for the given vulnerability.

    Falls back to the most common severity among all results for the vulnerability if no
    vulnerable results exist, and finally to 'medium' if there are no results.
    """
    from collections import Counter as _Counter

    vuln_results_all = [r for r in results if r.attack.vulnerability == vuln_id]
    vuln_results = [r for r in vuln_results_all if r.vulnerable]
    source = vuln_results or vuln_results_all
    if not source:
        return 'medium'
    counts = _Counter(r.attack.severity.value for r in source)
    return counts.most_common(1)[0][0]


def _render_focus_areas(report: RedTeamReport) -> None:
    """Render top-5 risk focus areas with remediation guidance and LLM recommendations.

    Risk score formula: vulnerability_rate * severity_weight
    Severity weights: low=1, medium=2, high=4, critical=8
    """
    summary = report.summary
    if not summary.by_vulnerability:
        return

    # Build recommendation lookup by category (recommendations may still be keyed by category)
    rec_lookup: dict[str, Any] = {}
    if report.focus_area_recommendations:
        for rec in report.focus_area_recommendations:
            rec_lookup[rec.category] = rec

    # Compute risk scores per vulnerability
    risk_items = []
    for vuln_id, vuln_summary in summary.by_vulnerability.items():
        if vuln_summary.total_attacks == 0:
            continue
        vuln_rate = 1.0 - vuln_summary.resistance_rate
        dominant_sev = _dominant_severity_for_vulnerability(report.results, vuln_id)
        weight = _SEVERITY_WEIGHTS.get(dominant_sev, 2)
        risk_score = vuln_rate * weight
        risk_items.append((vuln_summary, dominant_sev, risk_score))

    # Sort by risk score descending, take top 5
    risk_items.sort(key=operator.itemgetter(2), reverse=True)
    top_items = risk_items[:5]

    if not top_items:
        return

    st.subheader("Focus Areas")
    st.caption("Top-5 vulnerabilities by risk score (ASR × severity weight). Prioritize these for remediation.")  # noqa: RUF001

    for vuln_summary, dominant_sev, risk_score in top_items:
        vuln_name = _esc_html(vuln_summary.vulnerability_name or _fmt_vulnerability(vuln_summary.vulnerability))
        accent_color = SEVERITY_COLORS.get(dominant_sev, COLORS['orange_300'])
        vuln_rate = 1.0 - vuln_summary.resistance_rate

        # Look up remediation by vulnerability id, then by mapped framework categories
        remediation = REMEDIATION_GUIDANCE.get(vuln_summary.vulnerability)
        if not remediation:
            # Try mapped OWASP category codes
            for cats in vuln_summary.framework_categories.values():
                for cat_code in cats:
                    remediation = REMEDIATION_GUIDANCE.get(cat_code)
                    if remediation:
                        break
                if remediation:
                    break
        if not remediation:
            remediation = (
                "Review attack logs for this vulnerability and apply the principle of least privilege. "
                "Consult the relevant framework documentation for specific countermeasures."
            )
        remediation = _esc_html(remediation)

        # Build LLM recommendations HTML if available
        llm_rec_html = ""
        rec = rec_lookup.get(vuln_summary.vulnerability)
        if not rec:
            for cats in vuln_summary.framework_categories.values():
                for cat_code in cats:
                    rec = rec_lookup.get(cat_code)
                    if rec:
                        break
                if rec:
                    break
        if rec and rec.recommendations:
            rec_items = "".join(
                f'<li style="margin-bottom: 4px;">{_esc_html(item)}</li>'
                for item in rec.recommendations
            )
            patterns_html = ""
            if rec.patterns_observed:
                patterns_html = (
                    f'<div style="margin-top: 6px; font-style: italic; font-size: 0.8em; opacity: 0.6;">'
                    f'Patterns observed: {_esc_html(rec.patterns_observed)}</div>'
                )
            llm_rec_html = (
                f'<div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid {accent_color}30;">'
                f'<div style="font-size: 0.78rem; font-weight: 600; color: {COLORS["teal_400"]}; '
                f'margin-bottom: 4px;">Based on {rec.traces_analyzed} trace samples</div>'
                f'<ul style="margin: 4px 0 0 16px; padding: 0; font-size: 0.83em; opacity: 0.85;">'
                f'{rec_items}</ul>'
                f'{patterns_html}'
                f'</div>'
            )

        # Domain badge
        domain_badge = f'<span style="font-size: 0.75rem; opacity: 0.5; margin-left: 8px;">{_esc_html(vuln_summary.domain)}</span>'

        st.markdown(
            f"""
<div style="background: {accent_color}10; border-left: 4px solid {accent_color}; \
padding: 16px; border-radius: 8px; margin-bottom: 12px;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <strong style="font-size: 0.95rem;">{vuln_name}{domain_badge}</strong>
        <span style="color: {accent_color}; font-weight: bold; font-size: 0.85rem;">
            Risk: {risk_score:.1f} &nbsp;|&nbsp; {dominant_sev.upper()}
        </span>
    </div>
    <div style="margin-top: 8px; font-size: 0.9em; opacity: 0.6;">
        ASR: {vuln_rate:.0%} &nbsp;|&nbsp; Samples: {vuln_summary.total_attacks:,}
    </div>
    <div style="margin-top: 8px; font-size: 0.85em; opacity: 0.75;">
        {remediation}
    </div>
    {llm_rec_html}
</div>
""",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Executive Summary
# ---------------------------------------------------------------------------


def _render_executive_summary(report: RedTeamReport, summary: ReportSummary, framework_name: str) -> None:
    critical_count = _critical_exposure(report.results)
    critical_msg = (
        "**Critical vulnerabilities detected** - immediate action required."
        if critical_count > 0
        else "No critical exposures detected."
    )

    st.markdown(
        f"## Executive Summary\n\n"
        f"This dashboard analyzes the security posture of AI agents against the **{framework_name}** "
        f"vulnerability framework. Based on **{summary.total_attacks:,}** attack simulations, your agent "
        f"demonstrates a **{summary.vulnerability_rate:.1%} attack success rate (ASR)**. {critical_msg}"
    )

    # KPI Cards
    _compliant, _total_vulns = _compliant_vulnerabilities(summary)
    error_rate = (summary.total_errors / summary.total_attacks * 100) if summary.total_attacks else 0
    num_agents = len(report.tested_agents) if report.tested_agents else 1

    kpi_cols = st.columns(6 if num_agents > 1 else 5)
    col_idx = 0
    if num_agents > 1:
        kpi_cols[col_idx].metric("Agents Tested", str(num_agents))
        col_idx += 1
    kpi_cols[col_idx].metric("ASR", f"{summary.vulnerability_rate:.1%}", help="Attack Success Rate — target is 0%")
    col_idx += 1
    kpi_cols[col_idx].metric(
        "Critical Exposure", str(critical_count),
        help="Number of critical-severity vulnerabilities found",
    )
    col_idx += 1
    kpi_cols[col_idx].metric("Eval Coverage", f"{summary.evaluation_coverage:.1%}")
    col_idx += 1
    kpi_cols[col_idx].metric("Errors", f"{summary.total_errors:,}", help=f"{error_rate:.1f}% of attacks")
    col_idx += 1
    if summary.average_turns_per_attack > 0:
        kpi_cols[col_idx].metric("Avg Turns/Attack", f"{summary.average_turns_per_attack:.1f}")
    else:
        kpi_cols[col_idx].metric("Unevaluated", f"{summary.unevaluated_attacks:,}")

    if summary.total_errors > 0:
        st.warning(f"{summary.total_errors} attacks errored and were excluded from ASR calculations.")

    # Datapoint breakdown (hybrid runs)
    if summary.datapoint_breakdown:
        bp = summary.datapoint_breakdown
        parts = []
        if bp.get("static", 0):
            parts.append(f"{bp['static']} static")
        if bp.get("template_dynamic", 0):
            parts.append(f"{bp['template_dynamic']} template")
        if bp.get("generated_dynamic", 0):
            parts.append(f"{bp['generated_dynamic']} generated")
        if parts:
            st.caption(f"Datapoint breakdown: {' | '.join(parts)}")

    # Agent context section
    _render_agent_context_section(report)

    # Severity legend cards
    st.subheader("Criticality")
    st.caption("Vulnerabilities found by severity level. Hover over each card for details.")
    sev_short_defs = {
        'critical': 'Full system compromise or unauthorized data access — immediate fix required',
        'high': 'Significant harm such as data exfiltration or privilege escalation — prioritize remediation',
        'medium': 'Moderate impact like non-sensitive info leakage or minor policy violations — normal dev cycle',
        'low': 'Limited impact such as edge-case behaviors or cosmetic safety issues — address as time permits',
    }
    sev_cols = st.columns(4)
    for i, lvl in enumerate(SEVERITY_ORDER):
        clr = SEVERITY_COLORS[lvl]
        count = summary.by_severity[lvl].vulnerabilities_found if lvl in summary.by_severity else 0
        desc = sev_short_defs[lvl]
        sev_cols[i].markdown(
            f"<div class='sev-card' style='border-left:4px solid {clr};padding:8px 12px;"
            f"background:{clr}10;border-radius:0 6px 6px 0;position:relative;cursor:default'>"
            f"<div style='display:flex;align-items:baseline;gap:8px'>"
            f"<span style='font-size:1.8rem;font-weight:800;color:{clr};line-height:1'>{count}</span>"
            f"<span style='font-size:0.82rem;font-weight:600;color:{clr}'>{lvl.capitalize()}</span>"
            f"</div>"
            f"<div class='sev-tooltip' style='display:none;position:absolute;left:0;right:0;top:100%;"
            f"z-index:999;padding:8px 12px;background:#1e1e1e;border:1px solid {clr}40;"
            f"border-radius:6px;font-size:0.78rem;color:#ccc;margin-top:4px;"
            f"box-shadow:0 4px 12px rgba(0,0,0,0.3)'>{desc}</div>"
            f"</div>"
            f"<style>.sev-card:hover .sev-tooltip {{display:block!important}}</style>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Overall outcome donut + Vulnerability ASR table
    col_donut, col_cat = st.columns([1, 1.2])

    with col_donut:
        st.subheader("Overall Outcome")
        resistant = summary.evaluated_attacks - summary.vulnerabilities_found
        vulnerable = summary.vulnerabilities_found
        errors = summary.total_errors
        donut_labels = []
        donut_values = []
        donut_colors = []
        if resistant > 0:
            donut_labels.append("Resistant")
            donut_values.append(resistant)
            donut_colors.append(COLORS['success_400'])
        if vulnerable > 0:
            donut_labels.append("Vulnerable")
            donut_values.append(vulnerable)
            donut_colors.append(COLORS['red_400'])
        if errors > 0:
            donut_labels.append("Error")
            donut_values.append(errors)
            donut_colors.append(COLORS['sand_400'])

        if donut_values:
            fig = go.Figure(go.Pie(
                labels=donut_labels, values=donut_values,
                marker_colors=donut_colors,
                textinfo="label+percent", textposition="outside",
                hole=0.5,
            ))
            fig.update_layout(
                height=400, showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No results to display.")

    with col_cat:
        st.subheader("Vulnerability ASR")
        if summary.by_vulnerability:
            vuln_table_html = (
                '<table style="width:100%;border-collapse:collapse;font-size:0.85rem">'
                '<thead><tr>'
                '<th style="text-align:left;padding:8px 10px;border-bottom:1px solid rgba(255,255,255,0.1)">Vulnerability</th>'
                '<th style="text-align:right;padding:8px 10px;border-bottom:1px solid rgba(255,255,255,0.1);width:110px">ASR</th>'
                '</tr></thead><tbody>'
            )
            for vs in sorted(summary.by_vulnerability.values(), key=lambda v: v.resistance_rate):
                asr = 1 - vs.resistance_rate
                color = _asr_color(asr)
                vuln_table_html += (
                    f'<tr style="border-bottom:1px solid rgba(255,255,255,0.05)">'
                    f'<td style="padding:6px 10px">{vs.vulnerability_name or vs.vulnerability}</td>'
                    f'<td style="text-align:right;padding:6px 10px">'
                    f'<span style="background:{color}18;color:{color};border:1px solid {color}40;'
                    f'border-radius:12px;padding:2px 10px;font-weight:600;font-size:0.82rem">'
                    f'{asr:.1%}</span></td>'
                    f'</tr>'
                )
            vuln_table_html += '</tbody></table>'
            st.markdown(vuln_table_html, unsafe_allow_html=True)

    # Focus Areas section — top-3 risk categories with remediation guidance
    st.divider()
    _render_focus_areas(report)


# ---------------------------------------------------------------------------
# Methodology & Scope tab
# ---------------------------------------------------------------------------


def _render_methodology_tab(
    report: RedTeamReport, summary: ReportSummary, agents: list[str],
) -> None:
    """Render the Methodology & Scope tab with assessment details."""
    st.header("Methodology & Scope")
    methodology = getattr(report, 'methodology', None)

    # --- Assessment Configuration ---
    st.subheader("Assessment Configuration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Pipeline:** {report.pipeline.value}")
        st.markdown("**Scoring Method:** LLM-as-Judge")
        st.markdown(f"**Framework:** {report.framework.value if report.framework else 'OWASP LLM + ASI'}")
        if methodology and getattr(methodology, 'dataset_source', None):
            st.markdown(f"**Dataset Source:** {methodology.dataset_source}")
    with col2:
        if methodology and getattr(methodology, 'attack_model', None):
            st.markdown(f"**Attack Model:** {methodology.attack_model}")
        if methodology and getattr(methodology, 'evaluator_model', None):
            st.markdown(f"**Evaluator Model:** {methodology.evaluator_model}")
        if methodology and getattr(methodology, 'max_turns', None):
            st.markdown(f"**Max Turns:** {methodology.max_turns}")
        if report.duration_seconds is not None:
            mins, secs = divmod(int(report.duration_seconds), 60)
            st.markdown(f"**Duration:** {mins}m {secs}s")

    # --- Run Statistics ---
    st.divider()
    st.subheader("Run Statistics")
    stat_cols = st.columns(4)
    stat_cols[0].metric("Total Attacks", summary.total_attacks)
    stat_cols[1].metric("Evaluated", summary.evaluated_attacks)
    stat_cols[2].metric("Coverage", f"{summary.evaluation_coverage:.0%}")
    stat_cols[3].metric("Errors", summary.total_errors)

    # --- Categories Tested ---
    st.divider()
    st.subheader("Categories Tested")
    cats = report.categories_tested or []
    if cats:
        cat_cols = st.columns(min(len(cats), 4))
        for i, cat in enumerate(sorted(cats)):
            cat_name = OWASP_CATEGORY_NAMES.get(cat, cat)
            cat_summary = summary.by_category.get(cat)
            with cat_cols[i % len(cat_cols)]:
                if cat_summary:
                    asr = cat_summary.vulnerability_rate
                    color = _asr_color(asr)
                    st.markdown(
                        f'<span style="color:{color};font-weight:600">{cat}</span> '
                        f'— {cat_name}<br/>'
                        f'<span style="font-size:0.85rem;color:#888">'
                        f'{cat_summary.total_attacks} attacks, {asr:.0%} ASR</span>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(f"**{cat}** — {cat_name}")

        all_supported = {k for k in OWASP_CATEGORY_NAMES if not k.startswith("OWASP-")}
        untested = sorted(all_supported - set(cats))
        if untested:
            st.markdown(f"**{len(untested)} categories not tested:**")
            st.caption(", ".join(f"{c} ({OWASP_CATEGORY_NAMES.get(c, c)})" for c in untested))

    # --- Agents Tested ---
    if report.tested_agents:
        st.divider()
        st.subheader("Agents Tested")
        for agent_key in report.tested_agents:
            ctx = report.agent_contexts.get(agent_key) or report.agent_context
            if ctx:
                tools = [t.name for t in ctx.tools] if ctx.tools else []
                memory = [m.key or m.id for m in ctx.memory_stores] if ctx.memory_stores else []
                kbs = [kb.name or kb.key or kb.id for kb in ctx.knowledge_bases] if ctx.knowledge_bases else []
                st.markdown(f"#### {ctx.display_name or agent_key}")
                if ctx.description:
                    st.caption(ctx.description)
                if ctx.model:
                    st.markdown(f"**Model:** {ctx.model}")
                st.markdown(f"**Tools ({len(tools)}):** {', '.join(tools) if tools else 'none'}")
                st.markdown(f"**Memory ({len(memory)}):** {', '.join(memory) if memory else 'none'}")
                if kbs:
                    st.markdown(f"**Knowledge Bases ({len(kbs)}):** {', '.join(kbs)}")
            else:
                st.markdown(f"- {agent_key}")

    # --- Severity Definitions ---
    st.divider()
    st.subheader("Severity Definitions")
    sev_data = {
        "Severity": ["Critical", "High", "Medium", "Low"],
        "Weight": [8, 4, 2, 1],
        "Description": [
            "Immediate, exploitable vulnerability with high impact",
            "Significant vulnerability likely exploitable in practice",
            "Moderate vulnerability with limited impact or exploitability",
            "Minor issue, low risk or unlikely to be exploited",
        ],
    }
    st.table(sev_data)


# ---------------------------------------------------------------------------
# Technical Analysis
# ---------------------------------------------------------------------------


def _render_technical_analysis(report: RedTeamReport, summary: ReportSummary) -> None:
    st.header("Breakdown")

    if not report.results:
        st.info("No results to analyze. Run a red teaming session first.")
        return

    # KPIs
    weakest, weakest_count = _weakest_link(report.results)

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Attack Success Rate (ASR)", f"{summary.vulnerability_rate:.1%}",
        delta_color="inverse",
        help="Lower is better. Measures how often attacks succeed against defenses.",
    )
    col2.metric("Weakest Link", weakest, f"{weakest_count} breaches", delta_color="inverse")
    col3.metric("Total Samples Tested", f"{summary.total_attacks:,}")

    st.divider()

    # Interactive breakdown chart
    st.subheader("Interactive Breakdown")
    st.caption(
        "Use the dropdowns to explore ASR across different dimensions. "
        "**Group by** sets the Y-axis, **Stack by** adds a color breakdown."
    )

    _render_interactive_breakdown(report.results, summary)

    st.divider()

    # Technique and Delivery Method charts side by side
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("ASR by Technique")
        if summary.by_technique:
            items = sorted(summary.by_technique.items(), key=lambda t: t[1].vulnerability_rate, reverse=True)
            names = [k for k, _ in items]
            vuln_rates = [v.vulnerability_rate * 100 for _, v in items]
            totals = [v.total_attacks for _, v in items]

            fig = px.bar(
                x=vuln_rates, y=names, orientation="h",
                labels={"x": "ASR (%)", "y": "Technique"},
                text=[f"n={n}" for n in totals],
            )
            fig.update_traces(
                textposition="inside", textfont=dict(color="white", size=10),
                marker_color=QUALITATIVE[:len(names)],
            )
            fig.update_layout(
                height=max(300, len(names) * 35), showlegend=False,
                margin=dict(l=20, r=20, t=10, b=20),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No technique breakdown available.")

    with col_right:
        st.subheader("ASR by Delivery Method")
        if summary.by_delivery_method:
            items = sorted(summary.by_delivery_method.items(), key=lambda t: t[1].vulnerability_rate, reverse=True)
            names = [k for k, _ in items]
            vuln_rates = [v.vulnerability_rate * 100 for _, v in items]
            totals = [v.total_attacks for _, v in items]

            fig = px.bar(
                x=vuln_rates, y=names, orientation="h",
                labels={"x": "ASR (%)", "y": "Delivery Method"},
                text=[f"n={n}" for n in totals],
            )
            fig.update_traces(
                textposition="inside", textfont=dict(color="white", size=10),
                marker_color=QUALITATIVE[:len(names)],
            )
            fig.update_layout(
                height=max(300, len(names) * 35), showlegend=False,
                margin=dict(l=20, r=20, t=10, b=20),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No delivery method breakdown available.")

    # Vulnerability breakdown
    if summary.by_vulnerability:
        st.divider()
        st.subheader("ASR by Vulnerability")
        v_items = sorted(summary.by_vulnerability.items(), key=lambda t: (1.0 - t[1].resistance_rate), reverse=True)
        v_names = [v.vulnerability_name or k for k, v in v_items]
        v_vuln_rates = [(1.0 - v.resistance_rate) * 100 for _, v in v_items]
        v_totals = [v.total_attacks for _, v in v_items]

        fig = px.bar(
            x=v_vuln_rates, y=v_names, orientation="h",
            labels={"x": "ASR (%)", "y": "Vulnerability"},
            text=[f"n={n}" for n in v_totals],
        )
        fig.update_traces(
            textposition="inside", textfont=dict(color="white", size=10),
            marker_color=QUALITATIVE[:len(v_names)],
        )
        fig.update_layout(
            height=max(300, len(v_names) * 35), showlegend=False,
            margin=dict(l=20, r=20, t=10, b=20),
        )
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # Severity + Turn Type + Domain (only show panels with data)
    breakdown_panels = []
    if summary.by_severity:
        breakdown_panels.append("severity")
    if summary.by_turn_type:
        breakdown_panels.append("turn_type")
    if summary.by_domain:
        breakdown_panels.append("domain")

    if breakdown_panels:
        bd_cols = st.columns(len(breakdown_panels))
        for bd_i, bd_key in enumerate(breakdown_panels):
            with bd_cols[bd_i]:
                if bd_key == "severity":
                    st.subheader("By Severity")
                    items = [(k, summary.by_severity[k]) for k in SEVERITY_ORDER if k in summary.by_severity]
                    names = [k for k, _ in items]
                    vuln_rates = [v.vulnerability_rate * 100 for _, v in items]
                    totals = [v.total_attacks for _, v in items]
                    colors = [SEVERITY_COLORS.get(k, COLORS['sand_400']) for k in names]

                    fig = go.Figure(go.Bar(
                        x=names, y=vuln_rates, marker_color=colors,
                        text=[f"{r:.1f}%<br>n={n}" for r, n in zip(vuln_rates, totals, strict=False)],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        height=350, xaxis_title="Severity", yaxis_title="ASR (%)",
                        margin=dict(l=20, r=20, t=10, b=20),
                    )
                    st.plotly_chart(fig, width="stretch")

                elif bd_key == "turn_type":
                    st.subheader("By Turn Type")
                    names = list(summary.by_turn_type.keys())
                    totals = [v.total_attacks for v in summary.by_turn_type.values()]
                    fig = go.Figure(go.Pie(
                        labels=names, values=totals,
                        marker_colors=QUALITATIVE[:len(names)],
                        textinfo="label+value", hole=0.4,
                    ))
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                    st.plotly_chart(fig, width="stretch")
                    for name, tt in summary.by_turn_type.items():
                        st.caption(f"{name}: {tt.vulnerability_rate:.1%} vuln rate")

                elif bd_key == "domain":
                    st.subheader("By Domain")
                    names = list(summary.by_domain.keys())
                    totals = [v.total_attacks for v in summary.by_domain.values()]
                    fig = go.Figure(go.Pie(
                        labels=names, values=totals,
                        marker_colors=QUALITATIVE[:len(names)],
                        textinfo="label+value", hole=0.4,
                    ))
                    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                    st.plotly_chart(fig, width="stretch")
                    for name, sc in summary.by_domain.items():
                        st.caption(f"{name}: {sc.vulnerability_rate:.1%} vuln rate")

    # Framework breakdown (for mixed reports)
    if summary.by_framework and len(summary.by_framework) > 1:
        st.divider()
        st.subheader("By Framework")
        items = sorted(summary.by_framework.items(), key=lambda t: t[1].vulnerability_rate, reverse=True)
        names = [k for k, _ in items]
        vuln_rates = [v.vulnerability_rate * 100 for _, v in items]
        totals = [v.total_attacks for _, v in items]

        fig = px.bar(
            x=names, y=vuln_rates,
            labels={"x": "Framework", "y": "ASR (%)"},
            text=[f"{r:.1f}% (n={n})" for r, n in zip(vuln_rates, totals, strict=False)],
        )
        fig.update_traces(textposition="outside", marker_color=QUALITATIVE[:len(names)])
        fig.update_layout(
            height=350, showlegend=False,
            margin=dict(l=20, r=20, t=10, b=20),
        )
        st.plotly_chart(fig, width="stretch")

    # Turn depth analysis
    _render_turn_depth_analysis(report.results)

    # Attack failure treemap
    _render_attack_failure_treemap(report.results)


# ---------------------------------------------------------------------------
# Attack Failure Treemap
# ---------------------------------------------------------------------------


def _render_interactive_breakdown(results: list[RedTeamResult], summary: ReportSummary) -> None:
    """Interactive bar chart with Group-by and Stack-by dropdowns."""
    if not results:
        st.info("No results to display.")
        return

    # Available dimensions — vulnerability first
    dim_labels: dict[str, str] = {
        "vulnerability": "Vulnerability",
        "category": "Framework Category",
        "severity": "Severity",
        "attack_technique": "Attack Technique",
        "delivery_method": "Delivery Method",
        "turn_type": "Turn Type",
        "source": "Source",
    }
    dimensions = list(dim_labels.keys())

    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        group_by = st.selectbox(
            "Group by (Y-Axis)",
            options=dimensions,
            format_func=lambda x: dim_labels.get(x, x),
            key="breakdown_group_by",
        )
    with ctrl2:
        stack_options = ["None"] + [d for d in dimensions if d != group_by]
        stack_label = st.selectbox(
            "Stack / Color by",
            options=stack_options,
            format_func=lambda x: "None" if x == "None" else dim_labels.get(x, x),
            key="breakdown_stack_by",
        )
        stack_by: str | None = None if stack_label == "None" else stack_label

    # Extract per-result dimension values
    def _dim_value(r: RedTeamResult, dim: str) -> str:
        if dim == "category":
            return _fmt_category(r.attack.category)
        if dim == "vulnerability":
            return _fmt_vulnerability(r.attack.vulnerability) if r.attack.vulnerability else "unknown"
        if dim == "severity":
            return r.attack.severity.value
        if dim == "attack_technique":
            return r.attack.attack_technique.value
        if dim == "delivery_method":
            return r.attack.delivery_methods[0].value if r.attack.delivery_methods else "unknown"
        if dim == "turn_type":
            return r.attack.turn_type.value if r.attack.turn_type else "unknown"
        if dim == "source":
            return r.attack.source
        return "unknown"

    # Build chart data
    from collections import defaultdict

    if stack_by is None:
        # Simple grouped bar: dimension -> {vuln, total}
        groups: dict[str, dict[str, int]] = defaultdict(lambda: {"vuln": 0, "total": 0})
        for r in results:
            key = _dim_value(r, group_by)
            groups[key]["total"] += 1
            if r.vulnerable:
                groups[key]["vuln"] += 1

        chart_rows = []
        for name, counts in groups.items():
            asr = (counts["vuln"] / counts["total"] * 100) if counts["total"] else 0
            chart_rows.append({"dimension": name, "asr": round(asr, 1), "n": counts["total"]})

        chart_rows.sort(key=operator.itemgetter("asr"), reverse=True)
        names = [r["dimension"] for r in chart_rows]

        fig = go.Figure(go.Bar(
            y=names, x=[r["asr"] for r in chart_rows],
            orientation="h",
            marker_color=QUALITATIVE[:len(names)],
            text=[f'{r["asr"]:.1f}% (n={r["n"]})' for r in chart_rows],
            textposition="inside",
            textfont=dict(color="white", size=10),
        ))
        fig.update_layout(
            height=max(350, len(names) * 40),
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis_title="ASR (%)",
            yaxis_title=dim_labels.get(group_by, group_by),
            showlegend=False,
        )
    else:
        # Stacked bar: dimension x stack_dim
        groups_stacked: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"vuln": 0, "total": 0})
        for r in results:
            g = _dim_value(r, group_by)
            s = _dim_value(r, stack_by)
            groups_stacked[g, s]["total"] += 1
            if r.vulnerable:
                groups_stacked[g, s]["vuln"] += 1

        chart_rows_s = []
        for (g, s), counts in groups_stacked.items():
            asr = (counts["vuln"] / counts["total"] * 100) if counts["total"] else 0
            chart_rows_s.append({
                "dimension": g, "stack": s,
                "asr": round(asr, 1), "n": counts["total"],
            })

        # Sort dimensions by average ASR descending
        dim_asr: dict[str, list[float]] = defaultdict(list)
        for r in chart_rows_s:
            dim_asr[r["dimension"]].append(r["asr"])
        dim_order = sorted(dim_asr.keys(), key=lambda d: sum(dim_asr[d]) / len(dim_asr[d]), reverse=True)

        # Use severity colors if stacking by severity, otherwise qualitative
        stack_vals = sorted({r["stack"] for r in chart_rows_s})
        if stack_by == "severity":
            color_map = SEVERITY_COLORS
        else:
            color_map = {v: QUALITATIVE[i % len(QUALITATIVE)] for i, v in enumerate(stack_vals)}

        fig = go.Figure()
        for sv in stack_vals:
            sv_rows = {r["dimension"]: r for r in chart_rows_s if r["stack"] == sv}
            y_vals = dim_order
            x_vals = [sv_rows[d]["asr"] if d in sv_rows else 0 for d in dim_order]
            n_vals = [sv_rows[d]["n"] if d in sv_rows else 0 for d in dim_order]

            fig.add_trace(go.Bar(
                y=y_vals, x=x_vals, name=sv,
                orientation="h",
                marker_color=color_map.get(sv, COLORS['sand_400']),
                text=[f"n={n}" for n in n_vals],
                textposition="inside",
                textfont=dict(color="white", size=9),
            ))

        fig.update_layout(
            barmode="stack",
            height=max(400, len(dim_order) * 55),
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis_title="ASR (%)",
            yaxis_title=dim_labels.get(group_by, group_by),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, title_text=dim_labels.get(stack_by, stack_by)),
        )

    st.plotly_chart(fig, width="stretch")

    # Vulnerability details expander (always available)
    if summary.by_vulnerability:
        with st.expander("Vulnerability Details"):
            rows = [{
                    "Vulnerability": v.vulnerability_name or _fmt_vulnerability(v.vulnerability),
                    "Domain": v.domain,
                    "Attacks": v.total_attacks,
                    "Vulnerable": v.vulnerabilities_found,
                    "ASR": f"{1 - v.resistance_rate:.1%}",
                    "Strategies": ", ".join(v.strategies_used) if v.strategies_used else "-",
                } for v in sorted(summary.by_vulnerability.values(), key=lambda v: v.resistance_rate)]
            st.dataframe(rows, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Turn Depth Analysis
# ---------------------------------------------------------------------------


def _render_turn_depth_analysis(results: list[RedTeamResult]) -> None:
    """Show when vulnerabilities are found across conversation turns."""
    multi_turn = [r for r in results if r.execution and r.execution.turns and r.execution.turns > 1]
    if not multi_turn:
        return

    st.divider()
    st.subheader("Conversation Depth Analysis")
    st.caption("How attack success varies with conversation length. Only includes multi-turn attacks.")

    # Group by turn count
    turn_data: dict[int, dict[str, int]] = {}
    for r in multi_turn:
        turns = r.execution.turns  # type: ignore[union-attr]
        if turns not in turn_data:
            turn_data[turns] = {"total": 0, "vulnerable": 0}
        turn_data[turns]["total"] += 1
        if r.vulnerable:
            turn_data[turns]["vulnerable"] += 1

    col_left, col_right = st.columns(2)

    with col_left:
        sorted_turns = sorted(turn_data.keys())
        totals = [turn_data[t]["total"] for t in sorted_turns]
        vulns = [turn_data[t]["vulnerable"] for t in sorted_turns]
        asrs = [(v / t * 100) if t else 0 for v, t in zip(vulns, totals, strict=False)]

        fig = go.Figure()
        bar_width = 0.4 if len(sorted_turns) <= 3 else None
        fig.add_trace(go.Bar(
            x=[str(t) for t in sorted_turns], y=asrs,
            marker_color=[_asr_color(a / 100) for a in asrs],
            text=[f"{a:.0f}%<br>n={n}" for a, n in zip(asrs, totals, strict=False)],
            textposition="outside",
            width=bar_width,
        ))
        fig.update_layout(
            height=350, xaxis_title="Conversation Turns", yaxis_title="ASR (%)",
            xaxis_type="category",
            margin=dict(l=20, r=20, t=10, b=20),
        )
        st.plotly_chart(fig, width="stretch")

    with col_right:
        # Cumulative vulnerability discovery curve
        vuln_results = [r for r in multi_turn if r.vulnerable]
        if vuln_results:
            total_vulns = len(vuln_results)
            sorted_turns = sorted(turn_data.keys())
            cumulative = []
            running = 0
            for t in sorted_turns:
                running += turn_data[t]["vulnerable"]
                cumulative.append(running / total_vulns * 100)

            fig = go.Figure(go.Scatter(
                x=[str(t) for t in sorted_turns], y=cumulative,
                mode="lines+markers+text",
                marker=dict(size=8, color=COLORS['orange_300']),
                line=dict(color=COLORS['orange_300'], width=2),
                text=[f"{c:.0f}%" for c in cumulative],
                textposition="top center",
                textfont=dict(size=10),
            ))
            fig.update_layout(
                height=350,
                xaxis_title="Conversation Turns",
                yaxis_title="% of Vulnerabilities Found",
                xaxis_type="category",
                yaxis_range=[0, 105],
                margin=dict(l=20, r=20, t=10, b=20),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No multi-turn vulnerabilities found.")


# ---------------------------------------------------------------------------
# Attack Failure Treemap
# ---------------------------------------------------------------------------


def _render_attack_failure_treemap(results: list[RedTeamResult]) -> None:
    """Treemap showing attack failures: Vulnerability -> Technique, size=failures, color=ASR%."""
    vulnerable_results = [r for r in results if r.vulnerable]
    if not vulnerable_results:
        return

    st.divider()
    st.subheader("Attack Failure Treemap")
    st.caption("Block size = number of successful attacks. Color intensity = attack success rate. Use the sidebar filters to narrow down by agent, category, or severity.")

    # Build grouped data: (vulnerability, technique) -> {failures, total}
    from collections import defaultdict

    groups: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"failures": 0, "total": 0})
    for r in results:
        vuln_label = _fmt_vulnerability(r.attack.vulnerability) if r.attack.vulnerability else _fmt_category(r.attack.category)
        key = (vuln_label, r.attack.attack_technique.value)
        groups[key]["total"] += 1
        if r.vulnerable:
            groups[key]["failures"] += 1

    rows = []
    for (vuln, tech), counts in groups.items():
        if counts["failures"] > 0:
            rows.append({
                "vulnerability": vuln,
                "technique": tech,
                "failures": counts["failures"],
                "total": counts["total"],
                "asr": round(counts["failures"] / counts["total"] * 100, 1),
            })

    if not rows:
        return

    fig = px.treemap(
        rows,
        path=["vulnerability", "technique"],
        values="failures",
        color="asr",
        color_continuous_scale=ORQ_SCALE_AGENT,
        range_color=[0, 100],
        custom_data=["total", "asr", "vulnerability", "technique"],
    )
    # Build per-node hovertext, omitting fields that resolve to "?" on parent nodes
    hovertexts = []
    for cd, label, value in zip(
        fig.data[0].customdata,
        fig.data[0].labels,
        fig.data[0].values, strict=False,
    ):
        parts = [f"<b>{label}</b>"]
        vuln = cd[2] if cd[2] is not None and str(cd[2]) != "?" else None
        tech = cd[3] if cd[3] is not None and str(cd[3]) != "?" else None
        if vuln:
            parts.append(f"<i>Vulnerability:</i> {vuln}")
        if tech:
            parts.append(f"<i>Attack Technique:</i> {tech}")
        total = cd[0] if cd[0] is not None and str(cd[0]) != "?" else None
        asr = cd[1] if cd[1] is not None and str(cd[1]) != "?" else None
        parts.append(f"Failures: {value}")
        if total is not None:
            parts.append(f"Total: {total}")
        if asr is not None:
            parts.append(f"ASR: {asr:.1f}%")
        hovertexts.append("<br>".join(parts))
    fig.data[0].hovertext = hovertexts
    fig.data[0].hoverinfo = "text"
    fig.update_traces(
        texttemplate=f"<b>%{label}</b><br>%{value} failures",
        hovertemplate=None,
    )
    fig.update_layout(
        height=600,
        coloraxis_colorbar=dict(title="ASR (%)"),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Data Explorer
# ---------------------------------------------------------------------------


def _render_data_explorer(report: RedTeamReport, summary: ReportSummary) -> None:
    st.header("Dataset Overview")

    results = report.results
    if not results:
        st.info("No results to explore.")
        return

    # Overview metrics from summary breakdowns
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Samples", f"{summary.total_attacks:,}")
    col2.metric("Vulnerabilities", f"{len(summary.by_vulnerability)}")
    col3.metric("Attack Techniques", f"{len(summary.by_technique)}")
    col4.metric("Delivery Methods", f"{len(summary.by_delivery_method)}")
    sources = {r.attack.source for r in results}
    col5.metric("Sources", f"{len(sources)}")

    st.divider()

    # Distribution charts row 1
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Vulnerabilities by Severity")
        if summary.by_vulnerability and summary.by_severity:
            # Build from results for cross-dimension
            vuln_sev: Counter[tuple[str, str]] = Counter()
            for r in results:
                vuln_label = _fmt_vulnerability(r.attack.vulnerability) if r.attack.vulnerability else r.attack.category
                vuln_sev[vuln_label, r.attack.severity.value] += 1

            rows = [{"vulnerability": vuln, "severity": sev, "count": cnt} for (vuln, sev), cnt in vuln_sev.items()]
            if rows:
                fig = px.bar(
                    rows, y="vulnerability", x="count", color="severity",
                    orientation="h", title="",
                    color_discrete_map=SEVERITY_COLORS,
                    category_orders={"severity": SEVERITY_ORDER},
                )
                fig.update_layout(
                    barmode="stack", height=max(350, len(summary.by_vulnerability) * 35),
                    margin=dict(l=20, r=20, t=10, b=20), legend_title_text="Severity",
                )
                st.plotly_chart(fig, width="stretch")

    with col_right:
        st.subheader("Attack Technique Distribution")
        if summary.by_technique:
            items = sorted(summary.by_technique.items(), key=lambda t: t[1].total_attacks)
            names = [k for k, _ in items]
            counts = [v.total_attacks for _, v in items]

            fig = px.bar(
                x=counts, y=names, orientation="h",
                labels={"x": "Sample Count", "y": "Technique"},
                text=counts,
            )
            fig.update_traces(textposition="outside", marker_color=COLORS['success_400'])
            fig.update_layout(
                showlegend=False, height=max(350, len(names) * 30),
                margin=dict(l=20, r=20, t=10, b=20),
            )
            st.plotly_chart(fig, width="stretch")

    # Distribution charts row 2
    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("Delivery Method Distribution")
        if summary.by_delivery_method:
            items = sorted(summary.by_delivery_method.items(), key=lambda t: t[1].total_attacks, reverse=True)
            names = [k for k, _ in items]
            counts = [v.total_attacks for _, v in items]

            fig = px.bar(
                x=names, y=counts,
                labels={"x": "Delivery Method", "y": "Sample Count"},
                text=counts,
            )
            fig.update_traces(textposition="outside", marker_color=COLORS['success_400'])
            fig.update_layout(
                showlegend=False, height=350,
                margin=dict(l=20, r=20, t=10, b=20),
            )
            st.plotly_chart(fig, width="stretch")

    with col_right2:
        st.subheader("Source Distribution")
        if sources:
            source_counts = Counter(r.attack.source for r in results)
            fig = go.Figure(go.Pie(
                labels=list(source_counts.keys()),
                values=list(source_counts.values()),
                marker_colors=QUALITATIVE[:len(source_counts)],
                hole=0.4, textinfo="label+value",
            ))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=10, b=20))
            st.plotly_chart(fig, width="stretch")

    # Dimension counts
    st.divider()
    st.subheader("Dimension Counts")

    dimensions: list[tuple[str, str, dict[str, int]]] = []
    if summary.by_vulnerability:
        dimensions.append(("Vulnerability", f"{len(summary.by_vulnerability)} unique", {(v.vulnerability_name or k): v.total_attacks for k, v in summary.by_vulnerability.items()}))
    if summary.by_technique:
        dimensions.append(("Technique", f"{len(summary.by_technique)} unique", {k: v.total_attacks for k, v in summary.by_technique.items()}))
    if summary.by_delivery_method:
        dimensions.append(("Delivery Method", f"{len(summary.by_delivery_method)} unique", {k: v.total_attacks for k, v in summary.by_delivery_method.items()}))
    if summary.by_severity:
        dimensions.append(("Severity", f"{len(summary.by_severity)} unique", {k: v.total_attacks for k, v in summary.by_severity.items()}))
    if summary.by_turn_type:
        dimensions.append(("Turn Type", f"{len(summary.by_turn_type)} unique", {k: v.total_attacks for k, v in summary.by_turn_type.items()}))
    if summary.by_domain:
        dimensions.append(("Domain", f"{len(summary.by_domain)} unique", {k: v.total_attacks for k, v in summary.by_domain.items()}))

    cols = st.columns(3)
    for i, (label, subtitle, counts) in enumerate(dimensions):
        with cols[i % 3], st.expander(f"{label} ({subtitle})", expanded=False):
            total = sum(counts.values())
            rows = [
                {label: name, "Count": cnt, "%": round(cnt / total * 100, 1) if total else 0}
                for name, cnt in sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
            ]
            st.dataframe(rows, width="stretch", hide_index=True)

    # Sample Explorer
    st.divider()
    st.subheader("Sample Explorer")

    filtered = results
    st.caption(f"Showing {len(filtered)} results")

    if not filtered:
        st.info("No results match the current filters.")
        return

    # Results table — extended columns including delivery_method, turn_type, vulnerability_domain
    table_rows = []
    for r in filtered:
        # delivery_methods is a list in AttackInfo; join for display
        dms = getattr(r.attack, "delivery_methods", None)
        delivery_str = ", ".join(dm.value if hasattr(dm, "value") else str(dm) for dm in dms) if dms else "-"
        domain_val = r.attack.vulnerability_domain.value if r.attack.vulnerability_domain else "-"
        table_rows.append({
            "ID": r.attack.id,
            "Category": r.attack.category,
            "Vulnerability": r.attack.vulnerability or "-",
            "Technique": r.attack.attack_technique.value,
            "Delivery Method": delivery_str,
            "Turn Type": r.attack.turn_type.value if r.attack.turn_type else "-",
            "Domain": domain_val,
            "Severity": r.attack.severity.value,
            "Result": "VULNERABLE" if r.vulnerable else "RESISTANT",
            "Source": r.attack.source,
        })

    # Try row-click selection (Streamlit >= 1.35 supports on_select="rerun")
    st.caption("Click a row to view its conversation below.")
    selected_row_idx: int | None = None
    try:
        selection_state = st.dataframe(
            table_rows,
            width="stretch",
            height=300,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="de_results_table",
        )
        rows_selected = (
            selection_state.selection.get("rows", [])
            if hasattr(selection_state, "selection") and selection_state.selection
            else []
        )
        if rows_selected:
            selected_row_idx = rows_selected[0]
    except TypeError:
        # Fallback for older Streamlit versions that do not support on_select
        st.dataframe(table_rows, width="stretch", height=300, hide_index=True)

    # Export buttons
    export_cols = st.columns([1, 1, 4])
    with export_cols[0]:
        import csv
        import io

        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)
        st.download_button(
            "Download CSV",
            csv_buf.getvalue(),
            file_name="redteam_results.csv",
            mime="text/csv",
        )
    with export_cols[1]:
        json_data = json.dumps(
            [r.model_dump(mode="json") for r in filtered],
            indent=2, default=str,
        )
        st.download_button(
            "Download JSON",
            json_data,
            file_name="redteam_results.json",
            mime="application/json",
        )

    # Conversation viewer — driven by row click, with dropdown fallback
    st.subheader("Conversation Viewer")

    # Sync session state from row-click selection
    if selected_row_idx is not None:
        st.session_state["de_conv_select"] = selected_row_idx

    options = []
    for r in filtered:
        status = "VULN" if r.vulnerable else "SAFE"
        label = f"[{status}] {r.attack.id} / {r.attack.category} / {r.attack.attack_technique.value}"
        options.append(label)

    conv_idx = st.selectbox(
        "Select a sample",
        range(len(options)),
        format_func=lambda i: options[i],
        key="de_conv_select",
    )
    if conv_idx is not None:
        result = filtered[conv_idx]
        _render_result_detail(result)


def _render_result_detail(result: RedTeamResult) -> None:
    """Render detailed view for a single result."""
    atk = result.attack

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.markdown(f"**Vulnerability:** {_fmt_vulnerability(atk.vulnerability) if atk.vulnerability else '-'}")
    mc2.markdown(f"**Category:** {_fmt_category(atk.category)}")
    mc3.markdown(f"**Technique:** {atk.attack_technique.value}")
    mc4.markdown(f"**Severity:** {atk.severity.value}")
    mc5.markdown(f"**Turn Type:** {atk.turn_type.value}")

    if atk.delivery_methods:
        st.markdown(f"**Delivery Methods:** {', '.join(dm.value for dm in atk.delivery_methods)}")

    # Execution details
    if result.execution:
        ex = result.execution
        ec1, ec2, ec3 = st.columns(3)
        ec1.markdown(f"**Turns:** {ex.turns}" + (f" / {ex.max_turns}" if ex.max_turns else ""))
        if ex.duration_seconds is not None:
            ec2.markdown(f"**Duration:** {ex.duration_seconds:.1f}s")
        if ex.objective_achieved is not None:
            ec3.markdown(f"**Objective Achieved:** {ex.objective_achieved}")

    # Evaluation
    if result.evaluation:
        ev = result.evaluation
        if ev.passed is True:
            st.success(f"**RESISTANT** - {ev.explanation}")
        elif ev.passed is False:
            st.error(f"**VULNERABLE** - {ev.explanation}")
        else:
            st.warning(f"**Unevaluated** - {ev.explanation}")
        evaluator_parts = []
        if ev.evaluator_name:
            evaluator_parts.append(ev.evaluator_name)
        if ev.evaluator_id:
            evaluator_parts.append(f"`{ev.evaluator_id}`")
        if evaluator_parts:
            st.caption(f"Evaluator: {' — '.join(evaluator_parts)}")

    # Error
    if result.error:
        st.error(f"**Error ({result.error_type or 'unknown'}):** {result.error}")

    # Conversation
    if result.messages:
        st.markdown("---")
        st.markdown("**Conversation:**")
        for msg in result.messages:
            role = msg.role
            content = msg.content or ""

            if role == "system":
                with st.expander("System prompt", expanded=False):
                    st.code(content, language=None)
            elif role == "user":
                st.markdown("**User:**")
                st.code(content, language=None)
            elif role == "assistant":
                st.markdown("**Assistant:**")
                if content:
                    st.code(content, language=None)
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        st.code(
                            f"Tool call: {tc.function.name}({tc.function.arguments})",
                            language="json",
                        )
            elif role == "tool":
                name = msg.name or "tool"
                with st.expander(f"Tool response: {name}", expanded=False):
                    st.code(content, language=None)


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


def _aggregate_token_usage(results: list[RedTeamResult]) -> dict[str, dict[str, float | int]]:
    """Aggregate token usage per agent from results."""
    agents: dict[str, dict[str, float | int]] = {}
    for r in results:
        key = r.agent.key or r.agent.display_name or "unknown"
        if key not in agents:
            agents[key] = {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "attacks": 0}
        agents[key]["attacks"] += 1

        # Collect from execution-level token usage
        if r.execution and r.execution.token_usage:
            tu = r.execution.token_usage
            agents[key]["total_tokens"] += tu.total_tokens
            agents[key]["prompt_tokens"] += tu.prompt_tokens
            agents[key]["completion_tokens"] += tu.completion_tokens
            agents[key]["calls"] += tu.calls

        # Also add evaluation token usage
        if r.evaluation and r.evaluation.token_usage:
            eu = r.evaluation.token_usage
            agents[key]["total_tokens"] += eu.total_tokens
            agents[key]["prompt_tokens"] += eu.prompt_tokens
            agents[key]["completion_tokens"] += eu.completion_tokens
            agents[key]["calls"] += eu.calls
    return agents


def _render_usage_tab(report: RedTeamReport, summary: ReportSummary, agents: list[str]) -> None:
    st.header("Token Usage")

    # Overall totals
    tu = summary.token_usage_total
    if tu:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tokens", f"{tu.total_tokens:,}")
        col2.metric("Prompt Tokens", f"{tu.prompt_tokens:,}")
        col3.metric("Completion Tokens", f"{tu.completion_tokens:,}")
        col4.metric("API Calls", f"{tu.calls:,}")
    else:
        st.info("No aggregate token usage data available.")

    st.divider()

    # Per-agent breakdown
    agent_usage = _aggregate_token_usage(report.results)

    if agent_usage:
        st.subheader("Token Usage per Agent")

        if len(agent_usage) > 1:
            names = list(agent_usage.keys())
            tokens = [agent_usage[n]["total_tokens"] for n in names]

            fig = go.Figure(go.Bar(
                x=names, y=tokens,
                marker_color=QUALITATIVE[:len(names)],
                text=[f"{t:,}" for t in tokens], textposition="outside",
            ))
            fig.update_layout(
                height=350, yaxis_title="Total Tokens",
                margin=dict(l=20, r=20, t=10, b=20),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption(f"Single agent: **{next(iter(agent_usage.keys()))}**")

        # Detail table
        rows = []
        for name, usage in agent_usage.items():
            rows.append({
                "Agent": name,
                "Attacks": int(usage["attacks"]),
                "Total Tokens": f"{int(usage['total_tokens']):,}",
                "Prompt Tokens": f"{int(usage['prompt_tokens']):,}",
                "Completion Tokens": f"{int(usage['completion_tokens']):,}",
                "API Calls": int(usage["calls"]),
            })
        st.dataframe(rows, width="stretch", hide_index=True)

    # Token distribution histogram
    st.divider()
    st.subheader("Token Distribution per Attack")

    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    for r in report.results:
        if r.execution and r.execution.token_usage:
            tu = r.execution.token_usage
            if tu.prompt_tokens > 0:
                prompt_tokens.append(tu.prompt_tokens)
            if tu.completion_tokens > 0:
                completion_tokens.append(tu.completion_tokens)

    if prompt_tokens or completion_tokens:
        col_left, col_right = st.columns(2)
        with col_left:
            if prompt_tokens:
                fig = go.Figure(go.Histogram(
                    x=prompt_tokens, nbinsx=30,
                    marker_color=COLORS['success_400'],
                    name="Prompt Tokens",
                ))
                fig.update_layout(
                    height=300, xaxis_title="Tokens", yaxis_title="Frequency",
                    margin=dict(l=20, r=20, t=30, b=20),
                    title=dict(text="Prompt Tokens", font=dict(size=14)),
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No prompt token data available.")

        with col_right:
            if completion_tokens:
                fig = go.Figure(go.Histogram(
                    x=completion_tokens, nbinsx=30,
                    marker_color=COLORS['teal_400'],
                    name="Completion Tokens",
                ))
                fig.update_layout(
                    height=300, xaxis_title="Tokens", yaxis_title="Frequency",
                    margin=dict(l=20, r=20, t=30, b=20),
                    title=dict(text="Completion Tokens", font=dict(size=14)),
                )
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No completion token data available.")
    else:
        st.info("No per-attack token data available.")


# ---------------------------------------------------------------------------
# Error Analysis
# ---------------------------------------------------------------------------


def _render_errors_tab(summary: ReportSummary, report: RedTeamReport) -> None:
    st.header("Error Analysis")

    error_rate = (summary.total_errors / summary.total_attacks * 100) if summary.total_attacks else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Errors", f"{summary.total_errors:,}")
    col2.metric("Error Rate", f"{error_rate:.1f}%")
    col3.metric("Error Types", f"{len(summary.errors_by_type):,}")

    st.divider()

    if summary.errors_by_type:
        st.subheader("Errors by Type")
        items = sorted(summary.errors_by_type.items(), key=operator.itemgetter(1), reverse=True)
        names = [k for k, _ in items]
        counts = [v for _, v in items]

        fig = px.bar(
            x=counts, y=names, orientation="h",
            labels={"x": "Count", "y": "Error Type"},
            text=counts,
        )
        fig.update_traces(textposition="outside", marker_color=QUALITATIVE[:len(names)])
        fig.update_layout(
            height=max(250, len(names) * 40), showlegend=False,
            margin=dict(l=20, r=20, t=10, b=20),
        )
        st.plotly_chart(fig, width="stretch")

    # Error detail table
    error_results = [r for r in report.results if r.error]
    if error_results:
        st.subheader("Error Details")
        has_details = any(r.error_details for r in error_results)
        rows = []
        for r in error_results:
            row: dict[str, Any] = {
                "ID": r.attack.id,
                "Category": r.attack.category,
                "Technique": r.attack.attack_technique.value,
                "Error Type": r.error_type or "unknown",
                "Stage": r.error_stage or "-",
                "Error": r.error or "",
            }
            if has_details:
                row["Details"] = json.dumps(r.error_details, default=str) if r.error_details else "-"
            rows.append(row)
        st.dataframe(rows, width="stretch", hide_index=True)

    # Error impact
    st.divider()
    st.subheader("Error Impact on Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("ASR (including errors)", f"{summary.vulnerability_rate:.1%}")
    with col2:
        evaluated = summary.evaluated_attacks
        if evaluated > 0:
            vuln_of_evaluated = summary.vulnerabilities_found / evaluated
            clean_resistance = 1.0 - vuln_of_evaluated
            st.metric("ASR (errors excluded)", f"{vuln_of_evaluated:.1%}")


# ---------------------------------------------------------------------------
# Agent Comparison
# ---------------------------------------------------------------------------


def _render_agent_comparison(report: RedTeamReport, summary: ReportSummary, agents: list[str]) -> None:
    st.header("Multi-Agent Comparison")

    if len(agents) < 2:
        st.info("Agent comparison requires at least 2 agents in the report.")
        return

    # Group results by agent
    agent_results: dict[str, list[RedTeamResult]] = {a: [] for a in agents}
    for r in report.results:
        key = r.agent.key or r.agent.display_name or "unknown"
        if key in agent_results:
            agent_results[key].append(r)

    # Overall metrics per agent
    st.subheader("Overall Metrics")
    cols = st.columns(len(agents))
    for i, agent_name in enumerate(agents):
        ar = agent_results[agent_name]
        total = len(ar)
        vulns = sum(1 for r in ar if r.vulnerable)
        asr = (vulns / total * 100) if total else 0
        with cols[i]:
            st.markdown(f"### {agent_name}")
            st.metric("Attacks", f"{total:,}")
            st.metric("ASR", f"{asr:.1f}%")
            st.metric("Vulnerabilities", f"{vulns:,}")

    st.divider()

    # Agent comparison heatmap — ASR% by grouping dimension × agent  # noqa: RUF003
    st.subheader("Agent Heatmap")

    dim_options: dict[str, str] = {
        "vulnerability": "Vulnerability",
        "category": "Category",
        "technique": "Technique",
        "severity": "Severity",
    }
    heatmap_dim = st.selectbox(
        "Group by",
        options=list(dim_options.keys()),
        format_func=lambda x: dim_options.get(x, x),
        key="comp_heatmap_dim",
    )

    def _heatmap_dim_value(r: RedTeamResult, dim: str) -> str:
        if dim == "vulnerability":
            return _fmt_vulnerability(r.attack.vulnerability) if r.attack.vulnerability else "unknown"
        if dim == "category":
            return _fmt_category(r.attack.category)
        if dim == "technique":
            return r.attack.attack_technique.value
        if dim == "severity":
            return r.attack.severity.value
        return "unknown"

    # Build pivot: rows=dim_values, cols=agents, values=ASR%
    from collections import defaultdict as _defaultdict

    pivot_data: dict[str, dict[str, dict[str, int]]] = _defaultdict(
        lambda: _defaultdict(lambda: {"total": 0, "vuln": 0})
    )
    for agent_name in agents:
        for r in agent_results[agent_name]:
            dv = _heatmap_dim_value(r, heatmap_dim)
            pivot_data[dv][agent_name]["total"] += 1
            if r.vulnerable:
                pivot_data[dv][agent_name]["vuln"] += 1

    # Sort rows
    if heatmap_dim == "severity":
        all_dim_vals = [s for s in SEVERITY_ORDER if s in pivot_data]
        other_vals = sorted(v for v in pivot_data if v not in SEVERITY_ORDER)
        all_dim_vals = all_dim_vals + other_vals
    else:
        all_dim_vals = sorted(pivot_data.keys())

    if all_dim_vals:
        # Build z-matrix (rows=dim_vals, cols=agents), text for annotations
        z_matrix = []
        text_matrix = []
        for dv in all_dim_vals:
            row_z = []
            row_text = []
            for agent_name in agents:
                counts = pivot_data[dv].get(agent_name, {"total": 0, "vuln": 0})
                asr = (counts["vuln"] / counts["total"] * 100) if counts["total"] else 0
                row_z.append(round(asr, 1))
                row_text.append(f"{asr:.0f}%<br>n={counts['total']}")
            z_matrix.append(row_z)
            text_matrix.append(row_text)

        heatmap_fig = go.Figure(go.Heatmap(
            z=z_matrix,
            x=agents,
            y=all_dim_vals,
            colorscale=ORQ_SCALE_HEAT,
            zmin=0,
            zmax=100,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(size=11),
            hoverongaps=False,
            colorbar=dict(title="ASR (%)"),
        ))
        heatmap_fig.update_layout(
            height=max(350, len(all_dim_vals) * 45 + 80),
            xaxis_title="Agent",
            yaxis_title=dim_options.get(heatmap_dim, heatmap_dim),
            margin=dict(l=20, r=20, t=10, b=40),
        )
        st.plotly_chart(heatmap_fig, width="stretch")

    st.divider()

    # Per-vulnerability comparison (grouped bar chart)
    st.subheader("ASR by Vulnerability")

    vuln_data: dict[str, dict[str, float]] = {}
    for agent_name in agents:
        for r in agent_results[agent_name]:
            vuln = r.attack.vulnerability or r.attack.category
            if vuln not in vuln_data:
                vuln_data[vuln] = {}
            vuln_data[vuln].setdefault(f"{agent_name}_total", 0)
            vuln_data[vuln].setdefault(f"{agent_name}_vuln", 0)
            vuln_data[vuln][f"{agent_name}_total"] += 1
            if r.vulnerable:
                vuln_data[vuln][f"{agent_name}_vuln"] += 1

    if vuln_data:
        chart_rows = []
        for vuln in sorted(vuln_data.keys()):
            for agent_name in agents:
                total = vuln_data[vuln].get(f"{agent_name}_total", 0)
                v_count = vuln_data[vuln].get(f"{agent_name}_vuln", 0)
                asr = (v_count / total * 100) if total else 0
                chart_rows.append({
                    "vulnerability": _fmt_vulnerability(vuln),
                    "agent": agent_name,
                    "asr": asr,
                    "n": total,
                })

        fig = px.bar(
            chart_rows, y="vulnerability", x="asr", color="agent",
            orientation="h", barmode="group",
            labels={"asr": "ASR (%)", "vulnerability": "Vulnerability"},
            color_discrete_sequence=QUALITATIVE,
        )
        fig.update_layout(
            height=max(350, len(vuln_data) * 50),
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, width="stretch")

    st.divider()

    # Agreement analysis
    st.subheader("Agent Agreement")

    # Build attack-id -> {agent: vulnerable} map
    attack_verdicts: dict[str, dict[str, bool]] = {}
    for r in report.results:
        aid = r.attack.id
        agent_key = r.agent.key or r.agent.display_name or "unknown"
        if aid not in attack_verdicts:
            attack_verdicts[aid] = {}
        attack_verdicts[aid][agent_key] = r.vulnerable

    # Count agreements/disagreements for pairs with both agents
    if len(agents) >= 2:
        a1, a2 = agents[0], agents[1]
        both_pass = 0
        both_fail = 0
        only_a1_fail = 0
        only_a2_fail = 0
        shared = 0

        for verdicts in attack_verdicts.values():
            if a1 in verdicts and a2 in verdicts:
                shared += 1
                v1, v2 = verdicts[a1], verdicts[a2]
                if not v1 and not v2:
                    both_pass += 1
                elif v1 and v2:
                    both_fail += 1
                elif v1 and not v2:
                    only_a1_fail += 1
                else:
                    only_a2_fail += 1

        if shared > 0:
            agreement = (both_pass + both_fail) / shared * 100
            ac1, ac2, ac3 = st.columns(3)
            ac1.metric("Shared Samples", f"{shared:,}")
            ac2.metric("Agreement Rate", f"{agreement:.1f}%")
            ac3.metric("Disagreements", f"{only_a1_fail + only_a2_fail:,}")

            fig = go.Figure(go.Bar(
                x=["Both Resist", f"Only {a1} Fails", f"Only {a2} Fails", "Both Fail"],
                y=[both_pass, only_a1_fail, only_a2_fail, both_fail],
                marker_color=[COLORS['success_400'], COLORS['orange_300'], COLORS['orange_300'], COLORS['red_400']],
                text=[both_pass, only_a1_fail, only_a2_fail, both_fail],
                textposition="outside",
            ))
            fig.update_layout(
                height=350, yaxis_title="Count",
                margin=dict(l=20, r=20, t=10, b=20),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No shared attack samples between agents to compare.")

    # Disagreement viewer
    if len(agents) >= 2:
        _render_disagreement_viewer(report.results, agents)


def _render_disagreement_viewer(results: list[RedTeamResult], agents: list[str]) -> None:
    """Side-by-side sample viewer for disagreements between two agents.

    Shows attacks where one agent is vulnerable and another is resistant.
    Paginated at 10 disagreements per page.
    """
    if len(agents) < 2:
        return

    st.divider()
    st.subheader("Disagreement Analysis")
    st.caption(
        "Attacks where agents disagree — one found vulnerable, the other resistant. "
        "Select agent pair to compare."
    )

    # Allow selecting any two agents
    if len(agents) == 2:
        a1, a2 = agents[0], agents[1]
    else:
        pair_col1, pair_col2 = st.columns(2)
        with pair_col1:
            a1 = st.selectbox("Agent A", options=agents, index=0, key="dis_agent_a")
        with pair_col2:
            remaining = [ag for ag in agents if ag != a1]
            a2 = st.selectbox("Agent B", options=remaining, index=0, key="dis_agent_b")

    # Build attack-id -> {agent: result} map
    from collections import defaultdict as _dd2

    attack_results_map: dict[str, dict[str, RedTeamResult]] = _dd2(dict)
    for r in results:
        agent_key = r.agent.key or r.agent.display_name or "unknown"
        if agent_key in (a1, a2):
            attack_results_map[r.attack.id][agent_key] = r

    # Find disagreements
    disagreements: list[tuple[RedTeamResult, RedTeamResult]] = []
    for agent_map in attack_results_map.values():
        if a1 not in agent_map or a2 not in agent_map:
            continue
        r1, r2 = agent_map[a1], agent_map[a2]
        if r1.vulnerable != r2.vulnerable:
            disagreements.append((r1, r2))

    if not disagreements:
        st.info(f"No disagreements found between **{a1}** and **{a2}**.")
        return

    PAGE_SIZE = 10
    total_pages = max(1, (len(disagreements) + PAGE_SIZE - 1) // PAGE_SIZE)

    st.caption(f"Found {len(disagreements)} disagreements between **{a1}** and **{a2}** ({PAGE_SIZE} per page).")

    if "dis_page" not in st.session_state:
        st.session_state.dis_page = 1
    if st.session_state.dis_page > total_pages:
        st.session_state.dis_page = 1

    prev_col, info_col, next_col = st.columns([1, 2, 1])
    with prev_col:
        if st.button("← Previous", key="dis_prev", disabled=st.session_state.dis_page <= 1):
            st.session_state.dis_page -= 1
            st.rerun()
    with info_col:
        st.markdown(
            f"<div style='text-align:center; padding-top:6px;'>Page <b>{st.session_state.dis_page}</b> of <b>{total_pages}</b></div>",
            unsafe_allow_html=True,
        )
    with next_col:
        if st.button("Next →", key="dis_next", disabled=st.session_state.dis_page >= total_pages):
            st.session_state.dis_page += 1
            st.rerun()

    page = st.session_state.dis_page

    start = (page - 1) * PAGE_SIZE
    end = start + PAGE_SIZE
    page_items = disagreements[start:end]

    for idx, (r1, r2) in enumerate(page_items, start=start + 1):
        with st.expander(
            f"#{idx} {r1.attack.id} — {_fmt_vulnerability(r1.attack.vulnerability) if r1.attack.vulnerability else _fmt_category(r1.attack.category)} / {r1.attack.attack_technique.value}",
            expanded=(idx == start + 1),  # expand first item by default
        ):
            col_a, col_b = st.columns(2)

            for col, r, agent_name in [(col_a, r1, a1), (col_b, r2, a2)]:
                with col:
                    verdict = "VULNERABLE" if r.vulnerable else "RESISTANT"
                    verdict_color = COLORS['red_400'] if r.vulnerable else COLORS['success_400']
                    st.markdown(
                        f"**{_esc_html(agent_name)}** "
                        f"<span style='color:{verdict_color}; font-weight:bold;'>[{verdict}]</span>",
                        unsafe_allow_html=True,
                    )

                    # Show the last user message as the attack prompt
                    user_msgs = [m for m in r.messages if m.role == "user"]
                    if user_msgs:
                        st.markdown("**Attack prompt:**")
                        prompt_text = user_msgs[-1].content or ""
                        st.code(
                            prompt_text[:600] + ("…" if len(prompt_text) > 600 else ""),
                            language=None,
                        )

                    # Agent response
                    if r.response:
                        st.markdown("**Response:**")
                        resp = r.response
                        st.code(
                            resp[:600] + ("…" if len(resp) > 600 else ""),
                            language=None,
                        )

                    # Evaluator explanation
                    if r.evaluation and r.evaluation.explanation:
                        st.markdown("**Evaluator:**")
                        expl = r.evaluation.explanation
                        st.caption(expl[:400] + ("…" if len(expl) > 400 else ""))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_render_dashboard()
