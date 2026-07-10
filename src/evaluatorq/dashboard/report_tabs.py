"""Tabbed report bodies for the combined dashboard.

The standalone HTML exports render every section as one long scroll. The
dashboard instead groups those same sections into tabs that mirror the Streamlit
dashboards (``redteam/ui/dashboard.py`` and ``simulation/ui/dashboard.py``), so
the in-app report reads like the Streamlit UI rather than the export.

Each surface already computes its sections via ``build_report_sections`` and
renders them with a ``_SECTION_RENDERERS`` dispatch table; this module reuses
both and only decides which sections land in which tab. Interactive (HTMX)
panels are slotted into the tab they belong to.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from evaluatorq.common.reports import esc

if TYPE_CHECKING:
    from evaluatorq.redteam.contracts import RedTeamReport
    from evaluatorq.simulation.types import SimulationRun


def _render_sections(
    by_kind: dict[str, Any],
    renderers: dict[str, Any],
    kinds: tuple[str, ...],
) -> str:
    """Render the given section kinds, in order, each wrapped in the same
    ``<div id="section-{kind}">`` anchor the flat HTML export uses (so anchors
    and id-based assertions keep working). Unknown/absent kinds are skipped."""
    out: list[str] = []
    for kind in kinds:
        section = by_kind.get(kind)
        renderer = renderers.get(kind)
        if section is not None and renderer is not None:
            out.append(f'<div id="section-{esc(kind)}">{renderer(section)}</div>')
    return ''.join(out)


def _tabs(group: str, items: list[tuple[str, str] | tuple[str, str, str]]) -> str:
    """Render a CSS-only tab group.

    ``items`` is an ordered list of ``(label, panel_html)`` or ``(label,
    panel_html, label_html)``. In the 2-tuple form the label is HTML-escaped;
    in the 3-tuple form ``label_html`` is rendered raw inside the ``<label>``
    (the caller is responsible for escaping it), which lets a surface inject
    e.g. a count pill next to the label. Tabs whose panel is empty are dropped
    so a surface that lacks (say) error or comparison data simply shows fewer
    tabs, matching the Streamlit conditional-tab behaviour. The first
    surviving tab is checked. Switching is pure CSS (see ``styles.py``
    ``_TAB_RULES``): the Nth radio toggles the Nth label and Nth panel.
    """
    live = [it for it in items if it[1] and it[1].strip()]
    if not live:
        return ''
    radios: list[str] = []
    labels: list[str] = []
    panels: list[str] = []
    for i, it in enumerate(live):
        label, html = it[0], it[1]
        label_html = it[2] if len(it) > 2 else esc(label)
        tid = f'{group}-{i}'
        checked = ' checked' if i == 0 else ''
        radios.append(f'<input class="tab-radio" type="radio" name="{esc(group)}" id="{esc(tid)}"{checked}>')
        labels.append(f'<label class="tab-label" for="{esc(tid)}">{label_html}</label>')
        panels.append(f'<section class="tab-panel">{html}</section>')
    return (
        f'<div class="tabs">{"".join(radios)}'
        f'<div class="tab-bar">{"".join(labels)}</div>'
        f'<div class="tab-panels">{"".join(panels)}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Agent simulation
# ---------------------------------------------------------------------------


def sim_report_tabs(rid: str, run: SimulationRun, results: list[Any] | None = None) -> str:
    """Render the Agent Sim report body as Streamlit-aligned tabs.

    Tabs: Overview, Breakdown, Transcripts, Turn quality, Tokens, Evaluators,
    Judge & errors — each populated from the precomputed report sections (empty
    tabs drop out). Pass ``results`` to render a filtered subset (the filter
    round-trip); it defaults to the run's full result list.
    """
    from evaluatorq.dashboard.view import sim_interactive_panels
    from evaluatorq.simulation.reports.export_html import _SECTION_RENDERERS
    from evaluatorq.simulation.reports.sections import build_report_sections, individual_entries

    rows = run.results if results is None else results
    sections = build_report_sections(rows)
    by_kind: dict[str, Any] = {}
    for s in sections:
        by_kind.setdefault(s.kind, s)

    def render(*kinds: str) -> str:
        return _render_sections(by_kind, _SECTION_RENDERERS, kinds)

    hero = _sim_hero(by_kind.get('summary'), run)

    entries = individual_entries(rows)

    # Folded 7→4 to curb tab sprawl: Turn quality → Breakdown; Evaluators +
    # Judge & errors → Transcripts (all per-conversation verdicts); Tokens → Config.
    tabs = _tabs(
        'simtab',
        [
            ('Overview', _sim_overview(by_kind, rows)),
            (
                'Breakdown',
                render(
                    'persona_breakdown',
                    'scenario_breakdown',
                    'persona_scenario_heatmap',
                    'score_distribution',
                    'failures_first',
                    'turn_metrics',
                    'turn_quality_timeline',
                ),
            ),
            (
                'Transcripts',
                sim_interactive_panels(rid, entries)
                + render('evaluator_scores', 'judge_verdicts', 'failure_mode', 'errors'),
                f'Transcripts <span class="tab-count">{len(entries)}</span>',
            ),
            ('Config', render('token_usage')),
        ],
    )
    return f'<div class="sim-report">{hero}{tabs}</div>'


def _sim_overview(by_kind: dict[str, Any], rows: list[Any]) -> str:
    """Overview tab body: exec summary, 5-card KPI band, then two 2-col grids
    (donut + tokens; personas + scenarios). Spec §Overview."""
    from evaluatorq.dashboard.report_kit import exec_summary

    summary_section = by_kind.get('summary')
    overview_section = by_kind.get('overview')
    heatmap_section = by_kind.get('persona_scenario_heatmap')
    tokens_section = by_kind.get('token_usage')

    summary_data = summary_section.data if summary_section is not None else {}
    overview_data = overview_section.data if overview_section is not None else {}
    heatmap_data = heatmap_section.data if heatmap_section is not None else {}
    tokens_data = tokens_section.data if tokens_section is not None else {}

    summary_html = exec_summary(
        summary_data=summary_data,
        heatmap_data=heatmap_data,
        confidence=summary_data.get('confidence'),
    )
    kpi_html = _sim_kpi_band(summary_data)
    donut_html = _sim_outcomes_donut(rows)
    tokens_html = _sim_tokens_panel(tokens_data)
    personas_html = _sim_personas_panel(overview_data.get('personas', []))
    scenarios_html = _sim_scenarios_panel(overview_data.get('scenarios', []))

    return (
        f'{summary_html}{kpi_html}'
        f'<div class="sim-overview-grid-2">{donut_html}{tokens_html}</div>'
        f'<div class="sim-overview-grid-2">{personas_html}{scenarios_html}</div>'
    )


def _sim_kpi_band(summary_data: dict[str, Any]) -> str:
    """5-card KPI band (spec §Overview.2). Goal-completion status is the
    summary verdict (pass/warn/fail) — never an ad-hoc threshold."""
    from evaluatorq.common.reports.html_helpers import kpi_cards, pct

    verdict = summary_data.get('verdict', 'neutral')
    goal_status = verdict if verdict in {'pass', 'warn', 'fail'} else 'neutral'
    errors = summary_data.get('errors', 0)
    return kpi_cards([
        {'label': 'Goal completion', 'value': pct(summary_data.get('success_rate', 0.0)), 'status': goal_status},
        {
            'label': 'Avg score',
            'value': f'{summary_data.get("avg_goal_completion_score", 0.0):.2f}',
            'status': 'neutral',
        },
        {'label': 'Conversations', 'value': str(summary_data.get('total_conversations', 0)), 'status': 'neutral'},
        {'label': 'Avg turns', 'value': f'{summary_data.get("avg_turn_count", 0.0):.1f}', 'status': 'neutral'},
        {'label': 'Errors', 'value': str(errors), 'status': 'fail' if errors else 'pass'},
    ])


def _sim_tokens_panel(data: dict[str, Any]) -> str:
    """Token usage panel: Input/Output bar rows (spec §Overview.3)."""
    from evaluatorq.dashboard.report_kit import bar_rows, panel

    total = data.get('total_tokens', 0)
    if not total:
        return ''
    prompt = data.get('prompt_tokens', 0)
    completion = data.get('completion_tokens', 0)
    avg = data.get('avg_total_per_conversation', 0.0)
    rows = bar_rows(
        [('Input', prompt), ('Output', completion)],
        width=360,
        label_w=70,
        color='var(--chart-2)',
        fmt=lambda v: f'{int(v):,}',
    )
    return panel('Tokens', rows, sub=f'{total:,} total · {avg:,.0f}/conv')


_TRAIT_LABELS: tuple[str, ...] = ('patience', 'assertiveness', 'politeness', 'technical_level')


def _sim_personas_panel(personas: list[dict[str, Any]]) -> str:
    """Personas panel: name + style tag + conv count, plus a 2-col trait-bar
    grid when traits are present (spec §Overview.4)."""
    from evaluatorq.dashboard.report_kit import panel, tag

    if not personas:
        return ''
    rows: list[str] = []
    for p in personas:
        name = esc(p.get('name', ''))
        conv = p.get('conversations', 0)
        traits = p.get('traits')
        style = traits.get('communication_style') if isinstance(traits, dict) else None
        header = (
            f'<div class="sim-persona-row"><span class="sim-persona-name">{name}</span>'
            f'{tag(str(style), tone="teal") if style else ""}'
            f'<span class="sim-persona-count">{conv} conv</span></div>'
        )
        traits_html = ''
        if isinstance(traits, dict):
            bars = ''.join(
                (
                    f'<div class="sim-trait-bar"><span class="sim-trait-label">{esc(label)}</span>'
                    '<span class="sim-trait-track">'
                    f'<span class="sim-trait-fill" style="width:{max(0.0, min(1.0, float(v))) * 100:.0f}%"></span>'
                    '</span></div>'
                )
                for label in _TRAIT_LABELS
                if (v := traits.get(label)) is not None
            )
            if bars:
                traits_html = f'<div class="sim-trait-grid">{bars}</div>'
        rows.append(f'<div class="sim-persona-item">{header}{traits_html}</div>')
    return panel(f'Personas ({len(personas)})', ''.join(rows))


def _sim_scenarios_panel(scenarios: list[dict[str, Any]]) -> str:
    """Scenarios panel: name, goal, and typed criteria lines (spec §Overview.4)."""
    from evaluatorq.dashboard.report_kit import panel

    if not scenarios:
        return ''
    rows: list[str] = []
    for s in scenarios:
        name = esc(s.get('name', ''))
        goal = s.get('goal')
        goal_html = f'<div class="sim-scenario-goal">{esc(goal)}</div>' if goal else ''
        criteria_html = ''.join(
            (
                '<div class="sim-criterion">'
                f'<span class="sim-criterion-type" style="color:var(--{"red-600" if c.get("type") == "must_not_happen" else "teal-600"})">'
                f'{esc((c.get("type") or "").replace("_", " "))}</span>'
                f'<span class="sim-criterion-desc">{esc(c.get("description", ""))}</span>'
                '</div>'
            )
            for c in s.get('criteria', [])
        )
        rows.append(
            f'<div class="sim-scenario-item"><div class="sim-scenario-name">{name}</div>{goal_html}{criteria_html}</div>'
        )
    return panel(f'Scenarios ({len(scenarios)})', ''.join(rows))


_DONUT_SEGMENTS = (
    ('Achieved', 'var(--teal-600)'),
    ('Not achieved', 'var(--amber-600)'),
    ('Errors', 'var(--red-600)'),
)


def _sim_outcomes_donut(rows: list[Any]) -> str:
    """Three-segment outcomes donut (achieved / not achieved / errors) for the
    sim report Overview tab. Parity with the Streamlit dashboard (RES-1022).

    Self-contained SVG (no vl-convert dependency), mirroring the landing donut.
    Returns '' for an empty run so the Overview section renders unchanged.
    """
    achieved = not_achieved = errors = 0
    for r in rows:
        if str(getattr(r, 'terminated_by', '') or '') == 'error':
            errors += 1
        elif getattr(r, 'goal_achieved', False):
            achieved += 1
        else:
            not_achieved += 1
    counts = (achieved, not_achieved, errors)
    total = sum(counts)
    if total == 0:
        return ''

    radius = 60
    circ = 2 * math.pi * radius
    arcs: list[str] = []
    offset = 0.0
    for (_, color), value in zip(_DONUT_SEGMENTS, counts, strict=True):
        if value <= 0:
            continue
        length = circ * value / total
        arcs.append(
            f'<circle cx="75" cy="75" r="{radius}" fill="none" stroke="{color}" stroke-width="18"'
            f' stroke-dasharray="{length:.1f} {circ - length:.1f}" stroke-dashoffset="{-offset:.1f}"/>'
        )
        offset += length
    pct_achieved = round(achieved / total * 100)
    legend = ''.join(
        f'<li><span class="donut-key" style="background:{color}"></span>{esc(label)} · {value}</li>'
        for (label, color), value in zip(_DONUT_SEGMENTS, counts, strict=True)
        if value > 0
    )
    return (
        '<figure class="chart-card"><figcaption>Outcomes</figcaption>'
        '<div class="donut-wrap"><div class="donut">'
        f'<svg width="150" height="150" viewBox="0 0 150 150">{"".join(arcs)}</svg>'
        f'<div class="donut-center"><span class="donut-value">{pct_achieved}%</span>'
        '<span class="donut-label">achieved</span></div></div>'
        f'<ul class="donut-legend">{legend}</ul></div></figure>'
    )


def _sim_hero(summary_section: Any, run: SimulationRun) -> str:
    from evaluatorq.common.reports.html_helpers import kpi_cards, pct

    data = summary_section.data if summary_section is not None else {}
    verdict = data.get('verdict', 'neutral')
    success_status = 'pass' if verdict == 'pass' else ('warn' if verdict == 'warn' else 'fail')
    errors = data.get('errors', 0)
    cards = kpi_cards([
        {'label': 'Success Rate', 'value': pct(data.get('success_rate', 0.0)), 'status': success_status},
        {'label': 'Avg Score', 'value': f'{data.get("avg_goal_completion_score", 0.0):.2f}', 'status': 'neutral'},
        {'label': 'Conversations', 'value': str(data.get('total_conversations', 0)), 'status': 'neutral'},
        {'label': 'Runtime Errors', 'value': str(errors), 'status': 'warn' if errors else 'neutral'},
    ])
    return (
        f'<header class="report-hero"><h1 class="report-hero-title">Agent Simulation</h1>'
        f'<p class="report-hero-sub">{esc(run.run_name)} · target {esc(run.target_kind)}</p>'
        f'{cards}</header>'
    )


# ---------------------------------------------------------------------------
# Red team
# ---------------------------------------------------------------------------


def redteam_report_tabs(rid: str, report: RedTeamReport) -> str:
    """Render the Red Team report body as Streamlit-aligned tabs.

    Tabs: Summary, Breakdown, Explorer, Usage, Error Analysis, Comparison
    (multi-agent only), Methodology — each populated from the precomputed report
    sections plus the HTMX interactive panels (empty tabs drop out).
    """
    from evaluatorq.dashboard.view import (
        rt_panel_agent_heatmap,
        rt_panel_breakdown,
        rt_panel_conversation,
        rt_panel_disagreement,
    )
    from evaluatorq.redteam.reports.export_html import _SECTION_RENDERERS
    from evaluatorq.redteam.reports.sections import build_report_sections

    sections = build_report_sections(report)
    by_kind: dict[str, Any] = {}
    for s in sections:
        by_kind.setdefault(s.kind, s)

    def render(*kinds: str) -> str:
        return _render_sections(by_kind, _SECTION_RENDERERS, kinds)

    multi_agent = len(report.tested_agents) > 1
    hero = _redteam_hero(by_kind.get('summary'), report)

    # Folded 7→5: Comparison (agent heatmap + disagreements) → Evidence; Usage +
    # Methodology → Config. Error Analysis stays its own tab — it's where users
    # go to understand where the agent went wrong, not run metadata.
    comparison = (
        rt_panel_agent_heatmap(rid) + rt_panel_disagreement(rid) + render('agent_comparison', 'agent_disagreements')
        if multi_agent
        else ''
    )
    tabs = _tabs(
        'rttab',
        [
            ('Overview', render('summary', 'focus_areas')),
            (
                'Breakdowns',
                rt_panel_breakdown(rid)
                + render(
                    'vulnerability_breakdown',
                    'category_breakdown',
                    'technique_breakdown',
                    'delivery_breakdown',
                    'turn_scope_breakdown',
                    'turn_depth_analysis',
                    'attack_heatmap',
                    'framework_breakdown',
                ),
            ),
            (
                'Evidence',
                rt_panel_conversation(rid) + render('individual_results', 'source_distribution') + comparison,
            ),
            ('Error Analysis', render('error_analysis')),
            ('Config', render('token_usage', 'methodology', 'agent_context', 'severity_definitions')),
        ],
    )
    return f'{hero}{tabs}'


def _redteam_hero(summary_section: Any, report: RedTeamReport) -> str:
    from evaluatorq.common.reports.html_helpers import kpi_cards, pct

    data = summary_section.data if summary_section is not None else {}
    asr = data.get('vulnerability_rate', 0.0)
    resistance = data.get('resistance_rate', 0.0)
    vulns = data.get('vulnerabilities_found', 0)
    critical = data.get('critical_exposure', 0)
    errors = data.get('total_errors', 0)
    cards = kpi_cards([
        {
            'label': 'Attack Success Rate',
            'value': pct(asr),
            'status': 'fail' if asr >= 0.25 else ('warn' if asr > 0 else 'pass'),
        },
        {'label': 'Resistance', 'value': pct(resistance), 'status': 'pass' if resistance >= 0.8 else 'warn'},
        {'label': 'Vulnerabilities', 'value': str(vulns), 'status': 'fail' if vulns else 'pass'},
        {'label': 'Critical', 'value': str(critical), 'status': 'fail' if critical else 'neutral'},
        {'label': 'Errors', 'value': str(errors), 'status': 'warn' if errors else 'neutral'},
    ])
    return (
        f'<header class="report-hero"><h1 class="report-hero-title">Red Team</h1>'
        f'<p class="report-hero-sub">{esc(report.description or "Red teaming report")}</p>'
        f'{cards}</header>'
    )
