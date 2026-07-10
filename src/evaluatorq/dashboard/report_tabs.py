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

from typing import TYPE_CHECKING, Any

from evaluatorq.common.reports import esc

if TYPE_CHECKING:
    from collections.abc import Callable

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

    Tabs: Overview · Breakdown · Transcripts · Turn quality · Config — each
    populated from the precomputed report sections (empty tabs drop out; Turn
    quality drops when a run carries no ``turn_metrics``). Config folds job-level
    metadata (run configuration, personas, scenarios) plus the kept token_usage
    table. Pass ``results`` to render a filtered subset (the filter round-trip);
    it defaults to the run's full result list.
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

    # Folded 7→5 to curb tab sprawl: Evaluators + Judge & errors → Transcripts
    # (all per-conversation verdicts); Tokens → Config. Turn quality is its own
    # tab (unfolded from Breakdown) and drops out when a run has no turn_metrics.
    tabs = _tabs(
        'simtab',
        [
            ('Overview', _sim_overview(by_kind, rows)),
            ('Breakdown', _sim_breakdown(by_kind, render)),
            (
                'Transcripts',
                sim_interactive_panels(rid, entries) + render('evaluator_scores', 'judge_verdicts', 'errors'),
                f'Transcripts <span class="tab-count">{len(entries)}</span>',
            ),
            ('Turn quality', _sim_turn_quality(by_kind)),
            ('Config', _sim_config(by_kind, run) + render('token_usage')),
        ],
    )
    return f'<div class="sim-report">{hero}{tabs}</div>'


def _sim_config(by_kind: dict[str, Any], run: SimulationRun) -> str:
    """Config tab body: run-configuration meta grid → personas panel →
    scenarios panel (spec §Config.1-3). The kept ``token_usage`` table is
    appended by the caller."""
    from evaluatorq.dashboard.report_kit import meta_grid, panel

    overview_section = by_kind.get('overview')
    overview_data = overview_section.data if overview_section is not None else {}
    personas: list[dict[str, Any]] = overview_data.get('personas', [])
    scenarios: list[dict[str, Any]] = overview_data.get('scenarios', [])

    generated = run.created_at
    generated_str = generated.date().isoformat() if hasattr(generated, 'date') else str(generated)[:10]

    config_html = panel(
        'Run configuration',
        meta_grid([
            ('Target', run.target),
            ('Run name', run.run_name),
            ('Model', run.target_model),
            ('Mode', run.mode),
            ('Target kind', run.target_kind),
            ('Evaluators', ', '.join(run.evaluator_names) if run.evaluator_names else None),
            ('Personas', str(len(personas)) if personas else None),
            ('Scenarios', str(len(scenarios)) if scenarios else None),
            ('Conversations', str(run.total_results)),
            ('Generated', generated_str),
        ]),
        sub='Job-level metadata',
    )

    personas_html = ''
    if personas:
        rows = ''.join(_sim_config_persona_row(p) for p in personas)
        personas_html = panel('Personas', rows, sub='Simulated user profiles')

    scenarios_html = ''
    if scenarios:
        rows = ''.join(_sim_config_scenario_row(s) for s in scenarios)
        scenarios_html = panel('Scenarios', rows, sub='Goals + pass/fail criteria')

    return f'{config_html}{personas_html}{scenarios_html}'


def _sim_config_persona_row(persona: dict[str, Any]) -> str:
    """One Config-panel persona row: name · communication style · background
    (spec §Config.2). Missing style/background fields are simply omitted."""
    name = esc(persona.get('name', ''))
    traits = persona.get('traits')
    style = traits.get('communication_style') if isinstance(traits, dict) else None
    background = persona.get('background') or (traits.get('background') if isinstance(traits, dict) else None)
    style_html = f'<span class="sim-config-persona-style">{esc(str(style))}</span>' if style else ''
    background_html = f'<span class="sim-config-persona-bg">{esc(str(background))}</span>' if background else ''
    return (
        '<div class="sim-config-persona-row">'
        f'<span class="sim-config-persona-name">{name}</span>{style_html}{background_html}'
        '</div>'
    )


def _sim_config_scenario_row(scenario: dict[str, Any]) -> str:
    """One Config-panel scenario row: name + goal, then criteria chips
    (spec §Config.3). ``must_not_happen`` → red "✗" prefix, else green "✓"."""
    name = esc(scenario.get('name', ''))
    goal = scenario.get('goal')
    goal_html = f'<div class="sim-config-scenario-goal">{esc(str(goal))}</div>' if goal else ''
    chips: list[str] = []
    for c in scenario.get('criteria', []):
        is_negative = c.get('type') == 'must_not_happen'
        tone = 'red-600' if is_negative else 'green-600'
        mark = '✗' if is_negative else '✓'
        chips.append(
            f'<span class="sim-config-criterion" style="color:var(--{tone})">'
            f'{mark} {esc(c.get("description", ""))}</span>'
        )
    chips_html = f'<div class="sim-config-criteria">{"".join(chips)}</div>' if chips else ''
    return (
        '<div class="sim-config-scenario-row">'
        f'<div class="sim-config-scenario-name">{name}</div>{goal_html}{chips_html}'
        '</div>'
    )


def _sim_breakdown(by_kind: dict[str, Any], render: Callable[..., str]) -> str:
    """Breakdown tab body: heatmap → score-distribution → per-persona/scenario
    tables → top failure modes → failures table (spec §Breakdown)."""
    from evaluatorq.dashboard.report_kit import bar_rows, heatmap, histogram, panel

    heatmap_section = by_kind.get('persona_scenario_heatmap')
    heatmap_html = ''
    if heatmap_section is not None:
        d = heatmap_section.data
        heatmap_html = panel(
            'Goal completion — persona × scenario',  # noqa: RUF001 (mockup wording — spec §Breakdown.1)
            heatmap(d.get('personas', []), d.get('scenarios', []), d.get('cells', [])),
            sub='Red → yellow → green as the goal-completion rate rises',
        )

    dist_section = by_kind.get('score_distribution')
    dist_html = ''
    if dist_section is not None:
        scores = dist_section.data.get('scores', [])
        if scores:
            dist_html = panel(
                'Score distribution',
                histogram(scores),
                sub=f'{len(scores)} conversations · dashed line = mean',
            )

    persona_html = render('persona_breakdown')
    scenario_html = render('scenario_breakdown')
    tables_html = (
        f'<div class="sim-breakdown-grid-2">{persona_html}{scenario_html}</div>'
        if persona_html or scenario_html
        else ''
    )

    failure_mode_section = by_kind.get('failure_mode')
    failure_html = ''
    if failure_mode_section is not None:
        rows = [(str(label), float(count)) for label, count in failure_mode_section.data.get('rows', [])]
        if rows:
            failure_html = panel(
                'Top failure modes',
                bar_rows(rows, width=520, label_w=220, color='var(--red-600)', fmt=lambda v: str(int(v))),
            )

    failures_html = render('failures_first')

    return f'{heatmap_html}{dist_html}{tables_html}{failure_html}{failures_html}'


_TURN_METRIC_LABELS: dict[str, str] = {
    'response_quality': 'response quality',
    'hallucination_risk': 'hallucination risk',
    'tone_appropriateness': 'tone appropriateness',
    'factual_accuracy': 'factual accuracy',
}
# Metrics where a rising value is bad (risk), vs. the default where rising is good (quality).
_TURN_RISK_METRICS = frozenset({'hallucination_risk'})


def _turn_delta_callout(series: dict[str, list[float | None]]) -> str:
    """Templated first-to-last-turn delta callout, no confidence pill (spec
    §Turn.1). A clause renders only for series with >= 2 non-None points;
    absent/short metrics are dropped. Returns '' when nothing qualifies."""
    clauses: list[str] = []
    for name, values in series.items():
        points = [v for v in values if v is not None]
        if len(points) < 2:
            continue
        delta = points[-1] - points[0]
        label = esc(_TURN_METRIC_LABELS.get(name, name.replace('_', ' ')))
        if abs(delta) < 0.005:
            clauses.append(f'{label} held steady around <strong>{points[-1]:.2f}</strong>')
            continue
        if name in _TURN_RISK_METRICS:
            verb = 'rose' if delta > 0 else 'fell'
        else:
            verb = 'improved' if delta > 0 else 'declined'
        clauses.append(f'{label} {verb} by <strong>{abs(delta):.2f}</strong> from turn 1 to the last turn')
    if not clauses:
        return ''
    body = '; '.join(clauses) + '.'
    sentence = body[0].upper() + body[1:]
    return (
        '<div class="exec-summary">'
        '<div class="es-head"><span class="es-label">Turn quality trend</span></div>'
        f'<p class="es-body">{sentence}</p>'
        '</div>'
    )


def _sim_turn_count_bar(turn_count_distribution: dict[int, int]) -> str:
    """Turn-count distribution bar rows, sorted by turn count (spec §Turn.3)."""
    from evaluatorq.dashboard.report_kit import bar_rows

    if not turn_count_distribution:
        return ''
    rows = [(f'{n} turns', float(count)) for n, count in sorted(turn_count_distribution.items())]
    return bar_rows(rows, width=420, label_w=70, color='var(--chart-2)', fmt=lambda v: str(int(v)))


def _sim_avg_quality_tiles(avg_quality_metrics: dict[str, float]) -> str:
    """Average quality metric stat tiles, 2-col grid (spec §Turn.3)."""
    if not avg_quality_metrics:
        return ''
    tiles = ''.join(
        f'<div class="sim-stat-tile"><div class="sim-stat-value">{value:.2f}</div>'
        f'<div class="sim-stat-label">{esc(name.replace("_", " "))}</div></div>'
        for name, value in avg_quality_metrics.items()
    )
    return f'<div class="sim-stat-grid">{tiles}</div>'


def _sim_turn_quality(by_kind: dict[str, Any]) -> str:
    """Turn quality tab body: delta callout → per-turn line chart → 2-col grid
    of turn-count distribution + avg-quality stat tiles (spec §Turn).

    ``turn_metrics``/``turn_quality_timeline`` sections are always built (even
    for runs with no per-turn measurements — e.g. ``turn_count_distribution``
    is derived from every result's ``turn_count``), so presence of the section
    itself isn't a signal. What matters is whether there's any actual
    turn-*quality* data (a non-empty timeline series or avg quality metric);
    absent that, this returns '' and the whole tab drops out.
    """
    from evaluatorq.dashboard.report_kit import line_chart, panel

    timeline_section = by_kind.get('turn_quality_timeline')
    metrics_section = by_kind.get('turn_metrics')
    timeline_data = timeline_section.data if timeline_section is not None else {}
    metrics_data = metrics_section.data if metrics_section is not None else {}
    series = timeline_data.get('series', {})
    avg_quality_metrics = metrics_data.get('avg_quality_metrics', {})
    if not series and not avg_quality_metrics:
        return ''

    callout_html = _turn_delta_callout(series)

    chart_html = ''
    chart = line_chart(timeline_data.get('turns', []), series)
    if chart:
        chart_html = panel(
            'Per-turn quality',
            chart,
            sub='Average across conversations, by turn index (0–1)',  # noqa: RUF001 (mockup wording — spec §Turn.2)
        )

    dist_body = _sim_turn_count_bar(metrics_data.get('turn_count_distribution', {}))
    dist_html = panel('Turn-count distribution', dist_body) if dist_body else ''

    tiles_body = _sim_avg_quality_tiles(avg_quality_metrics)
    tiles_html = panel('Average quality metrics', tiles_body) if tiles_body else ''

    grid_html = f'<div class="sim-breakdown-grid-2">{dist_html}{tiles_html}</div>' if dist_html or tiles_html else ''

    return f'{callout_html}{chart_html}{grid_html}'


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

    pct_achieved = round(achieved / total * 100)
    # One donut implementation: delegate the ring/center/legend markup to
    # report_kit.donut(); this function owns only the chart-card wrapper.
    segments = [
        {'label': label, 'value': value, 'color': color}
        for (label, color), value in zip(_DONUT_SEGMENTS, counts, strict=True)
    ]
    from evaluatorq.dashboard.report_kit import donut

    inner = donut(segments, f'{pct_achieved}%', 'goal met')
    return f'<figure class="chart-card"><figcaption>Outcomes</figcaption>{inner}</figure>'


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
