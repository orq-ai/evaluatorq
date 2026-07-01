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


def _tabs(group: str, items: list[tuple[str, str]]) -> str:
    """Render a CSS-only tab group.

    ``items`` is an ordered list of ``(label, panel_html)``. Tabs whose panel is
    empty are dropped so a surface that lacks (say) error or comparison data
    simply shows fewer tabs, matching the Streamlit conditional-tab behaviour.
    The first surviving tab is checked. Switching is pure CSS (see ``styles.py``
    ``_TAB_RULES``): the Nth radio toggles the Nth label and Nth panel.
    """
    live = [(label, html) for label, html in items if html and html.strip()]
    if not live:
        return ''
    radios: list[str] = []
    labels: list[str] = []
    panels: list[str] = []
    for i, (label, html) in enumerate(live):
        tid = f'{group}-{i}'
        checked = ' checked' if i == 0 else ''
        radios.append(f'<input class="tab-radio" type="radio" name="{esc(group)}" id="{esc(tid)}"{checked}>')
        labels.append(f'<label class="tab-label" for="{esc(tid)}">{esc(label)}</label>')
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

    tabs = _tabs(
        'simtab',
        [
            ('Overview', _sim_outcomes_donut(rows) + render('overview')),
            (
                'Breakdown',
                render(
                    'persona_breakdown',
                    'scenario_breakdown',
                    'persona_scenario_heatmap',
                    'score_distribution',
                    'failures_first',
                ),
            ),
            ('Transcripts', sim_interactive_panels(rid, entries)),
            ('Turn quality', render('turn_metrics', 'turn_quality_timeline')),
            ('Tokens', render('token_usage')),
            ('Evaluators', render('evaluator_scores')),
            ('Judge & errors', render('judge_verdicts', 'failure_mode', 'errors')),
        ],
    )
    return f'{hero}{tabs}'


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
    cards = kpi_cards(
        [
            {'label': 'Success Rate', 'value': pct(data.get('success_rate', 0.0)), 'status': success_status},
            {'label': 'Avg Score', 'value': f'{data.get("avg_goal_completion_score", 0.0):.2f}', 'status': 'neutral'},
            {'label': 'Conversations', 'value': str(data.get('total_conversations', 0)), 'status': 'neutral'},
            {'label': 'Runtime Errors', 'value': str(errors), 'status': 'warn' if errors else 'neutral'},
        ]
    )
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

    tabs = _tabs(
        'rttab',
        [
            ('Summary', render('summary', 'focus_areas')),
            (
                'Breakdown',
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
            ('Explorer', rt_panel_conversation(rid) + render('individual_results', 'source_distribution')),
            ('Usage', render('token_usage')),
            ('Error Analysis', render('error_analysis')),
            (
                'Comparison',
                (rt_panel_agent_heatmap(rid) + rt_panel_disagreement(rid) + render('agent_comparison', 'agent_disagreements'))
                if multi_agent
                else '',
            ),
            ('Methodology', render('methodology', 'agent_context', 'severity_definitions')),
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
    cards = kpi_cards(
        [
            {'label': 'Attack Success Rate', 'value': pct(asr), 'status': 'fail' if asr >= 0.25 else ('warn' if asr > 0 else 'pass')},
            {'label': 'Resistance', 'value': pct(resistance), 'status': 'pass' if resistance >= 0.8 else 'warn'},
            {'label': 'Vulnerabilities', 'value': str(vulns), 'status': 'fail' if vulns else 'pass'},
            {'label': 'Critical', 'value': str(critical), 'status': 'fail' if critical else 'neutral'},
            {'label': 'Errors', 'value': str(errors), 'status': 'warn' if errors else 'neutral'},
        ]
    )
    return (
        f'<header class="report-hero"><h1 class="report-hero-title">Red Team</h1>'
        f'<p class="report-hero-sub">{esc(report.description or "Red teaming report")}</p>'
        f'{cards}</header>'
    )
