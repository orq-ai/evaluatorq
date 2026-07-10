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

    from evaluatorq.redteam.contracts import RedTeamReport, RedTeamResult
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
    return f'<div class="report-aligned sim-report">{hero}{tabs}</div>'


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


def _rt_by_kind(report: RedTeamReport) -> dict[str, Any]:
    """``build_report_sections(report)`` collapsed to ``{kind: section}``
    (first section wins per kind), shared by the hero, tab wiring, and tests."""
    from evaluatorq.redteam.reports.sections import build_report_sections

    by_kind: dict[str, Any] = {}
    for s in build_report_sections(report):
        by_kind.setdefault(s.kind, s)
    return by_kind


def _rt_agent_stats(report: RedTeamReport) -> dict[str, dict[str, Any]]:
    """Per-agent stats keyed by ``r.agent.key`` (spec §Run header / §Data
    sources — ``agent_comparison`` is populated from the same ``report.results``
    grouping, so card and table numbers always agree; this also makes
    single-agent runs work since ``agent_comparison`` is None for < 2 agents).

    ``display_name``/``model`` are taken from the **first** result matching
    each key (not ``results[0]`` globally, which would attach one agent's
    model to every card). ``asr = vulns / attacks`` guards ``attacks == 0``.
    """
    stats: dict[str, dict[str, Any]] = {}
    for r in report.results:
        key = r.agent.key or r.agent.display_name or r.agent.model or 'unknown'
        entry: dict[str, Any] = stats.setdefault(
            key,
            {
                'display_name': r.agent.display_name or key,
                'model': r.agent.model or '',
                'attacks': 0,
                'vulns': 0,
                'critical': 0,
                'errors': 0,
            },
        )
        entry['attacks'] += 1
        if r.vulnerable:
            entry['vulns'] += 1
            if r.attack.severity == 'critical':
                entry['critical'] += 1
        if r.error is not None:
            entry['errors'] += 1
    for entry in stats.values():
        attacks = entry['attacks']
        vulns = entry['vulns']
        entry['asr'] = vulns / attacks if attacks else 0.0
        entry['resistance'] = 1.0 - entry['asr']
    return stats


def _rt_exec_summary(summary_data: dict[str, Any], by_kind: dict[str, Any]) -> str:
    """Templated exec-summary sentence + fallbacks (spec §Overview.1).
    Empty-run guard: ``total_attacks`` falsy -> ``''``, nothing below runs."""
    from evaluatorq.common.reports.html_helpers import pct
    from evaluatorq.dashboard.report_kit import callout

    total = summary_data.get('total_attacks', 0)
    if not total:
        return ''

    category_section = by_kind.get('category_breakdown')
    rows = category_section.data.get('rows', []) if category_section is not None else []
    k = len(rows)

    vulns = summary_data.get('vulnerabilities_found', 0)
    critical = summary_data.get('critical_exposure', 0)
    resistance_rate = summary_data.get('resistance_rate', 0.0)

    if vulns:
        critical_clause = f', including <strong>{critical} critical</strong>' if critical else ''
        sentence = (
            f'Across <strong>{total}</strong> adversarial attacks spanning {k} OWASP '
            f'categor{"y" if k == 1 else "ies"}, the agent resisted <strong>{pct(resistance_rate)}</strong> '
            f'and exposed <strong>{vulns}</strong> vulnerabilit{"y" if vulns == 1 else "ies"}{critical_clause}.'
        )
    else:
        sentence = (
            f'Across <strong>{total}</strong> adversarial attacks spanning {k} OWASP '
            f'categor{"y" if k == 1 else "ies"}, the agent resisted all of them.'
        )

    if rows and rows[0].get('vulnerability_rate', 0.0) > 0:
        top = rows[0]
        sentence += (
            f' <strong>{esc(top.get("category_name", ""))}</strong> is the weakest area '
            f'({pct(top.get("vulnerability_rate", 0.0))} attack success rate).'
        )

    turn_section = by_kind.get('turn_depth_analysis')
    if turn_section is not None:
        turn_rows = turn_section.data.get('rows', [])
        if len(turn_rows) >= 2 and turn_rows[-1].get('vulnerability_rate', 0.0) > turn_rows[0].get(
            'vulnerability_rate', 0.0
        ):
            first, last = turn_rows[0], turn_rows[-1]
            sentence += (
                f' Attack success climbs with conversation depth — from '
                f'{pct(first.get("vulnerability_rate", 0.0))} at {first.get("turn_count")} turns to '
                f'{pct(last.get("vulnerability_rate", 0.0))} at {last.get("turn_count")} turns.'
            )

    total_errors = summary_data.get('total_errors', 0)
    if total_errors:
        sentence += f' {total_errors} attack{"s" if total_errors != 1 else ""} errored and were not evaluated.'

    return callout(sentence, confidence=summary_data.get('confidence'))


def _rt_kpi_band(s: dict[str, Any]) -> str:
    """5-card KPI band: Attacks run / Vulnerabilities / Attack success rate /
    Resistance rate / Critical findings (spec §Overview.2)."""
    from evaluatorq.common.reports.html_helpers import kpi_cards, pct

    asr = s.get('vulnerability_rate', 0.0)
    resistance = s.get('resistance_rate', 0.0)
    vulns = s.get('vulnerabilities_found', 0)
    critical = s.get('critical_exposure', 0)
    return kpi_cards([
        {'label': 'Attacks run', 'value': str(s.get('total_attacks', 0)), 'status': 'neutral'},
        {'label': 'Vulnerabilities', 'value': str(vulns), 'status': 'fail' if vulns else 'pass'},
        {
            'label': 'Attack success rate',
            'value': pct(asr),
            'status': 'fail' if asr >= 0.25 else ('warn' if asr > 0 else 'pass'),
        },
        {'label': 'Resistance rate', 'value': pct(resistance), 'status': 'pass' if resistance >= 0.8 else 'warn'},
        {'label': 'Critical findings', 'value': str(critical), 'status': 'fail' if critical else 'pass'},
    ])


def _rt_agent_row(stats: dict[str, Any]) -> str:
    """One agents-under-test table row: dot + name/model, hit count, ASR track
    bar, ASR value (spec §Overview.4)."""
    from evaluatorq.common.reports.html_helpers import pct

    critical = stats.get('critical', 0)
    vulns = stats.get('vulns', 0)
    dot_cls = 'rt-hero-dot--critical' if critical else ('rt-hero-dot--vuln' if vulns else 'rt-hero-dot--clean')
    asr = stats.get('asr', 0.0)
    bar_pct = max(0.0, min(1.0, asr)) * 100
    bar_color = 'var(--red-600)' if critical else 'var(--orange-500)'
    asr_color = 'var(--orange-600)' if vulns else 'var(--green-600)'
    model = stats.get('model', '')
    model_html = f'<div class="rt-agent-row-model">{esc(model)}</div>' if model else ''
    return (
        '<div class="rt-agent-row">'
        f'<div class="rt-agent-row-name"><span class="rt-hero-dot {dot_cls}"></span>'
        f'<span>{esc(stats.get("display_name", ""))}</span>{model_html}</div>'
        f'<div class="rt-agent-row-count">{vulns}/{stats.get("attacks", 0)}</div>'
        '<div class="rt-agent-row-track">'
        f'<div class="rt-agent-row-fill" style="width:{bar_pct:.1f}%;background:{bar_color}"></div>'
        '</div>'
        f'<div class="rt-agent-row-asr" style="color:{asr_color}">{pct(asr)}</div>'
        '</div>'
    )


def _rt_agents_under_test(report: RedTeamReport) -> str:
    """Agents-under-test panel, weakest (highest ASR) first (spec §Overview.4)."""
    from evaluatorq.dashboard.report_kit import panel

    stats = _rt_agent_stats(report)
    if not stats:
        return ''
    ranked = sorted(stats.values(), key=lambda st: st.get('asr', 0.0), reverse=True)
    rows_html = ''.join(_rt_agent_row(st) for st in ranked)
    return panel(
        'Agents under test',
        f'<div class="rt-agents-table">{rows_html}</div>',
        sub=f'{len(ranked)} agents · per-agent attack success, weakest first',
    )


def _rt_overview(by_kind: dict[str, Any], report: RedTeamReport) -> str:
    """Overview tab body: exec summary → 5-card KPI band → 2-col grid
    (outcome donut + vulnerabilities-by-severity bars) → agents-under-test
    panel (multi-agent only). Spec §Overview."""
    from evaluatorq.common.reports.html_helpers import pct
    from evaluatorq.dashboard.report_kit import bar_rows, donut, panel

    summary_section = by_kind.get('summary')
    summary_data = summary_section.data if summary_section is not None else {}

    exec_html = _rt_exec_summary(summary_data, by_kind)
    kpi_html = _rt_kpi_band(summary_data)

    evaluated = summary_data.get('evaluated_attacks', 0)
    vulns = summary_data.get('vulnerabilities_found', 0)
    total_errors = summary_data.get('total_errors', 0)
    resistant = max(evaluated - vulns, 0)
    resistance_rate = summary_data.get('resistance_rate', 0.0)

    outcome_html = donut(
        [
            {'label': 'Resistant', 'value': resistant, 'color': 'var(--green-600)'},
            {'label': 'Vulnerable', 'value': vulns, 'color': 'var(--red-600)'},
            {'label': 'Error', 'value': total_errors, 'color': 'var(--amber-600)'},
        ],
        pct(resistance_rate),
        'resistant',
    )
    outcome_panel = panel('Outcome', outcome_html)

    by_severity: dict[str, Any] = summary_data.get('by_severity', {})

    def _sev_count(name: str) -> int:
        entry = by_severity.get(name) or {}
        return entry.get('vulnerabilities_found', 0)

    high_count = _sev_count('high')
    max_value = max(high_count, 3)
    severity_bars = ''.join(
        bar_rows(
            [(label, float(count))],
            width=420,
            label_w=84,
            color=color,
            fmt=lambda v: str(int(v)),
            max_value=max_value,
        )
        for label, count, color in (
            ('Critical', _sev_count('critical'), 'var(--red-600)'),
            ('High', high_count, 'var(--orange-500)'),
            ('Medium', _sev_count('medium'), 'var(--text-muted)'),
            ('Low', _sev_count('low'), 'var(--green-600)'),
        )
    )
    severity_panel = panel('Vulnerabilities by severity', severity_bars)

    grid_html = f'<div class="rt-overview-grid-2">{outcome_panel}{severity_panel}</div>'
    agents_html = _rt_agents_under_test(report) if len(report.tested_agents) > 1 else ''

    return f'{exec_html}{kpi_html}{grid_html}{agents_html}'


_RISK_MAX = 8  # risk_score = vulnerability_rate x avg_severity_weight; SEVERITY_WEIGHTS tops out at critical=8


def _rt_focus_tier(risk_score: float) -> tuple[str, str, str]:
    """Tier code/label/color from ``risk_score`` (spec §Focus areas): >=2 -> P1
    Critical priority (red-600); >=1 -> P2 High priority (orange-600); else P3
    Medium priority (amber-600)."""
    if risk_score >= 2:
        return 'P1', 'Critical priority', 'var(--red-600)'
    if risk_score >= 1:
        return 'P2', 'High priority', 'var(--orange-600)'
    return 'P3', 'Medium priority', 'var(--amber-600)'


def _rt_focus_pattern_chips(area: dict[str, Any], color: str) -> str:
    """Pattern chips from ``llm_recommendations.patterns_observed`` — row
    omitted when the key is absent (static pipeline; never subscript, spec
    §Data sources)."""
    patterns = area.get('llm_recommendations', {}).get('patterns_observed')
    if not patterns:
        return ''
    dot = f'<span class="rt-focus-pattern-dot" style="background:{color}"></span>'
    return f'<div class="rt-focus-patterns">{dot}<span>{esc(patterns)}</span></div>'


def _rt_focus_remediation_box(area: dict[str, Any]) -> str:
    """Recommended-fix box from ``remediation`` — omitted when empty (spec §Focus areas)."""
    remediation = area.get('remediation')
    if not remediation:
        return ''
    return (
        '<div class="rt-focus-fixbox">'
        '<div class="rt-focus-fixbox-label">Recommended fix</div>'
        f'<div class="rt-focus-fixbox-body">{esc(remediation)}</div>'
        '</div>'
    )


def _rt_focus_area_card(area: dict[str, Any]) -> str:
    """One focus-area panel: tier + category header, pattern chips,
    remediation box (main column) and risk dial + ASR/hits mini-stats
    (fixed 100px right column). Spec §Focus areas."""
    from evaluatorq.common.reports.html_helpers import pct
    from evaluatorq.dashboard.report_kit import dial

    risk_score = area.get('risk_score', 0.0)
    tier_code, tier_label, color = _rt_focus_tier(risk_score)

    header = (
        '<div class="rt-focus-tier-row">'
        f'<span class="rt-focus-tier-dot" style="background:{color}"></span>'
        f'<span class="rt-focus-tier-label" style="color:{color}">{esc(tier_code)} · {esc(tier_label)}</span>'
        '</div>'
        f'<div class="rt-focus-category-name">{esc(area.get("category_name", ""))}</div>'
        f'<div class="rt-focus-category-code">{esc(area.get("category", ""))}</div>'
    )

    patterns_html = _rt_focus_pattern_chips(area, color)
    fixbox_html = _rt_focus_remediation_box(area)

    main_col = f'<div class="rt-focus-main">{header}{patterns_html}{fixbox_html}</div>'

    vulnerability_rate = area.get('vulnerability_rate', 0.0)
    vulnerabilities_found = area.get('vulnerabilities_found', 0)
    risk_dial = dial(f'{risk_score:.1f}', risk_score / _RISK_MAX, radius=38, stroke=9, color=color, sub='RISK')
    right_col = (
        '<div class="rt-focus-right">'
        f'{risk_dial}'
        '<div class="rt-focus-mini-stats">'
        '<div class="rt-focus-mini-stat">'
        '<span class="rt-focus-mini-key">ASR</span>'
        f'<span class="rt-focus-mini-value">{pct(vulnerability_rate)}</span></div>'
        '<div class="rt-focus-mini-stat">'
        '<span class="rt-focus-mini-key">Hits</span>'
        f'<span class="rt-focus-mini-value" style="color:{color}">{vulnerabilities_found}</span></div>'
        '</div>'
        '</div>'
    )

    return f'<div class="rk-panel rt-focus-card">{main_col}{right_col}</div>'


def _rt_focus(by_kind: dict[str, Any]) -> str:
    """Focus areas tab body: intro copy + one card per top-risk area (worst
    first, section list is already top-5). Empty list (clean run) -> ``''``
    so the tab drops entirely (spec §Focus areas)."""
    section = by_kind.get('focus_areas')
    areas: list[dict[str, Any]] = section.data.get('focus_areas', []) if section is not None else []
    if not areas:
        return ''

    intro = (
        '<p class="rt-focus-intro">Prioritized fixes, ranked by '
        '<code>risk = success rate × avg severity</code>. Start at the top — P1 first.</p>'  # noqa: RUF001
    )
    cards = ''.join(_rt_focus_area_card(area) for area in areas)
    return f'{intro}{cards}'


def _rt_agent_card_chip_row(label: str, items: list[str]) -> str:
    """One TOOLS/KNOWLEDGE chip row: mono faint label + ``tag()`` per item,
    "—" when empty (spec §Agents)."""
    from evaluatorq.dashboard.report_kit import tag

    body = ''.join(tag(item) for item in items) if items else '<span class="rt-agent-card-chip-empty">—</span>'
    return (
        '<div class="rt-agent-card-chiprow">'
        f'<span class="rt-agent-card-chip-label">{esc(label)}</span>'
        f'<div class="rt-agent-card-chips">{body}</div>'
        '</div>'
    )


def _rt_agent_card(agent_ctx: dict[str, Any] | None, key: str, stats: dict[str, Any]) -> str:
    """One agent card: ASR dial column + main column (name/critical chip,
    model, description, stat strip, TOOLS/KNOWLEDGE chip rows). Agents
    present in results but missing ``agent_context`` still render via
    ``stats``-only fallback (spec §Agents)."""
    from evaluatorq.common.reports.html_helpers import pct
    from evaluatorq.dashboard.report_kit import dial

    ctx = agent_ctx or {}
    display_name = ctx.get('display_name') or stats.get('display_name') or key
    model = ctx.get('model') or stats.get('model') or ''
    description = ctx.get('description') or ''
    tools = ctx.get('tools') or []
    knowledge_bases = ctx.get('knowledge_bases') or []

    attacks = stats.get('attacks', 0)
    vulns = stats.get('vulns', 0)
    critical = stats.get('critical', 0)
    asr = stats.get('asr', 0.0)
    resistance = stats.get('resistance', 1.0 - asr)

    dial_color = 'var(--red-600)' if critical else ('var(--orange-500)' if vulns else 'var(--green-600)')
    dial_html = dial(pct(asr), asr, radius=24, stroke=6, color=dial_color, sub='ASR')

    critical_chip = f'<span class="rt-agent-card-critical">{critical} critical</span>' if critical else ''
    model_html = f'<div class="rt-agent-card-model">{esc(model)}</div>' if model else ''
    description_html = f'<div class="rt-agent-card-desc">{esc(description)}</div>' if description else ''

    critical_style = 'color:var(--red-600)' if critical else ''
    stat_strip = (
        '<div class="rt-agent-card-stats">'
        '<div class="rt-agent-card-stat">'
        '<span class="rt-agent-card-stat-key">Found</span>'
        f'<span class="rt-agent-card-stat-value">{vulns}/{attacks}</span></div>'
        '<div class="rt-agent-card-stat">'
        '<span class="rt-agent-card-stat-key">Critical</span>'
        f'<span class="rt-agent-card-stat-value" style="{critical_style}">{critical}</span></div>'
        '<div class="rt-agent-card-stat">'
        '<span class="rt-agent-card-stat-key">Resisted</span>'
        f'<span class="rt-agent-card-stat-value" style="color:var(--green-600)">{pct(resistance)}</span></div>'
        '</div>'
    )

    chips_html = _rt_agent_card_chip_row('TOOLS', tools) + _rt_agent_card_chip_row('KNOWLEDGE', knowledge_bases)

    return (
        '<div class="rk-panel rt-agent-card">'
        f'<div class="rt-agent-card-dial">{dial_html}</div>'
        '<div class="rt-agent-card-main">'
        f'<div class="rt-agent-card-name-row">'
        f'<span class="rt-agent-card-name">{esc(display_name)}</span>{critical_chip}</div>'
        f'{model_html}{description_html}{stat_strip}{chips_html}'
        '</div>'
        '</div>'
    )


def _rt_agents_intro(*, multi_agent: bool, n_agents: int) -> str:
    """Intro copy above the agent cards (spec §Agents)."""
    if multi_agent:
        text = (
            f'The job targeted a <strong>{n_agents}-agent system</strong>. When an orchestrator '
            'delegates to sub-agents that trust its routing context, a breach upstream propagates '
            'downstream.'
        )
    else:
        text = 'Single agent under assessment.'
    return f'<p class="rt-agents-intro">{text}</p>'


def _rt_agents(by_kind: dict[str, Any], report: RedTeamReport, rid: str) -> str:
    """Agents tab body: intro copy -> one card per agent (``agent_context``
    joined with ``_rt_agent_stats`` by key) -> multi-agent-only server-rendered
    agent_comparison (ASR heatmap) + agent_disagreements. Spec §Agents."""
    from evaluatorq.redteam.reports.export_html import _SECTION_RENDERERS

    agent_ctx_section = by_kind.get('agent_context')
    agents_ctx: list[dict[str, Any]] = agent_ctx_section.data.get('agents', []) if agent_ctx_section else []
    ctx_by_key = {a['key']: a for a in agents_ctx}
    stats = _rt_agent_stats(report)

    ordered_keys = [*ctx_by_key.keys(), *(k for k in stats if k not in ctx_by_key)]
    if not ordered_keys:
        return ''

    multi_agent = len(report.tested_agents) > 1
    intro = _rt_agents_intro(multi_agent=multi_agent, n_agents=len(report.tested_agents))
    cards = ''.join(_rt_agent_card(ctx_by_key.get(key), key, stats.get(key, {})) for key in ordered_keys)

    tail = ''
    if multi_agent:
        # Server-rendered agent_comparison (ASR heatmap) + agent_disagreements only.
        # The lazy vega Agent-Heatmap / Disagreement-Viewer panels duplicated these.
        tail = _render_sections(by_kind, _SECTION_RENDERERS, ('agent_comparison', 'agent_disagreements'))

    return f'{intro}{cards}{tail}'


def _rt_attack_row(r: RedTeamResult, rid: str, idx: int) -> str:
    """One `<details class="rt-attack-row">` evidence row: design header grid
    summary (title/agent/vector/severity/outcome/chevron) + lazy fragment
    body (spec §Attacks)."""
    from evaluatorq.dashboard.redteam_charts import fmt_vulnerability
    from evaluatorq.dashboard.report_kit import outcome_pill, severity_pill

    atk = r.attack
    title_name = fmt_vulnerability(atk.vulnerability) if atk.vulnerability else esc(atk.category)
    outcome = 'error' if r.error else ('vulnerable' if r.vulnerable else 'resistant')
    safe_rid = esc(rid)

    summary_html = (
        '<summary class="rt-attack-row-summary">'
        '<div class="rt-attack-row-title">'
        f'<strong>{esc(title_name)}</strong>'
        f'<span class="rt-attack-row-id">{esc(atk.id)}</span>'
        '</div>'
        f'<div class="rt-attack-row-agent">{esc(r.agent.display_name or r.agent.key)}</div>'
        f'<div class="rt-attack-row-vector">{esc(atk.attack_technique.value)}</div>'
        f'<div class="rt-attack-row-severity">{severity_pill(atk.severity.value)}</div>'
        f'<div class="rt-attack-row-outcome">{outcome_pill(outcome)}</div>'
        '<div class="rt-attack-row-chevron">&#9660;</div>'
        '</summary>'
    )
    # Lazy-load on the <details> `toggle` event, not a click on the (empty,
    # collapsed) body div — clicking the summary opens the details but never
    # delivers a click to the inner div, so a click trigger there never fires.
    return (
        '<details class="rt-attack-row"'
        f' hx-get="/r/{safe_rid}/redteam/attack?idx={idx}"'
        ' hx-include="#filter-form"'
        ' hx-trigger="toggle once"'
        ' hx-target="find .rt-attack-row-body"'
        ' hx-swap="innerHTML"'
        f'>{summary_html}<div class="rt-attack-row-body"></div></details>'
    )


def _rt_attacks(report: RedTeamReport, rid: str) -> str:
    """Attacks tab body: evidence table of one ``<details>`` row per
    (already-filtered) result, lazy-loading its fragment body once on click,
    then the kept ``source_distribution`` render (spec §Attacks).

    Ponytail: this list is intentionally unpaginated — scale honesty means
    the collapsed row list itself renders in full while each row's
    transcript payload stays lazy (deviation #7).
    """
    from evaluatorq.redteam.reports.export_html import _SECTION_RENDERERS

    results = report.results
    n = len(results)
    intro = (
        f'<p class="rt-attacks-intro">Evidence &mdash; {n} attack{"s" if n != 1 else ""}. '
        'Click a row to expand the evaluator verdict and full transcript.</p>'
        if results
        else '<p class="rt-attacks-intro">No attacks match the current filters.</p>'
    )

    header = (
        '<div class="rt-attack-row-header">'
        '<div>Attack</div><div>Agent</div><div>Vector</div>'
        '<div>Severity</div><div>Outcome</div><div></div>'
        '</div>'
    )
    rows = ''.join(_rt_attack_row(r, rid, i) for i, r in enumerate(results))
    table_html = f'<div class="rk-panel rt-attack-table">{header}{rows}</div>' if results else ''

    # `report` is already the *filtered* report the caller composed, so
    # rebuild its sections here (not the outer tab's unfiltered `by_kind`)
    # to keep `source_distribution` in sync with the visible rows.
    kept = _render_sections(_rt_by_kind(report), _SECTION_RENDERERS, ('source_distribution',))
    return f'{intro}{table_html}{kept}'


def _rt_config(by_kind: dict[str, Any], report: RedTeamReport) -> str:
    """Config tab body: run-configuration meta grid, methodology panel
    (TESTED/NOT TESTED tags + jury-reliability block), kept
    ``severity_definitions``/``error_analysis`` renders. ``agent_context`` is
    deliberately excluded here — it now renders exclusively in the Agents
    tab (spec §Agents, CRITICAL: no double-render)."""
    from evaluatorq.common.reports.html_helpers import humanize_duration
    from evaluatorq.dashboard.report_kit import meta_grid, panel, tag
    from evaluatorq.redteam.reports.export_html import _SECTION_RENDERERS

    summary_section = by_kind.get('summary')
    summary_data: dict[str, Any] = summary_section.data if summary_section is not None else {}
    methodology_section = by_kind.get('methodology')
    method_data: dict[str, Any] = methodology_section.data if methodology_section is not None else {}

    created_at = summary_data.get('created_at')
    generated = created_at.date().isoformat() if hasattr(created_at, 'date') else (str(created_at) or None)

    run_config_html = panel(
        'Run configuration',
        meta_grid([
            ('Target', summary_data.get('target')),
            ('Pipeline', summary_data.get('pipeline')),
            ('Framework', method_data.get('framework')),
            ('Scoring method', method_data.get('scoring_method')),
            ('Agents', str(len(report.tested_agents)) if report.tested_agents else None),
            (
                'Attacks',
                str(summary_data['total_attacks']) if summary_data.get('total_attacks') is not None else None,
            ),
            ('Generated', generated),
            ('Duration', humanize_duration(summary_data.get('duration_seconds')) or None),
        ]),
    )

    tested = method_data.get('categories_tested') or []
    untested = method_data.get('untested_categories') or []
    untested_names = method_data.get('untested_category_names') or {}

    tested_html = (
        '<div class="rt-config-methodology-row">'
        '<span class="rt-config-methodology-label">TESTED</span>'
        f'<div class="rt-config-methodology-tags">{"".join(tag(c) for c in tested)}</div>'
        '</div>'
        if tested
        else ''
    )
    untested_html = (
        '<div class="rt-config-methodology-row">'
        '<span class="rt-config-methodology-label">NOT TESTED</span>'
        f'<div class="rt-config-methodology-tags">'
        f'{"".join(tag(untested_names.get(c, c)) for c in untested)}</div>'
        '</div>'
        if untested
        else ''
    )

    jury = summary_data.get('jury_reliability')
    jury_html = ''
    if jury is not None:
        alpha = jury.get('krippendorff_alpha')
        jury_html = (
            '<div class="rt-config-methodology-row">'
            '<span class="rt-config-methodology-label">JURY RELIABILITY</span>'
            + meta_grid([
                ('Krippendorff alpha', f'{alpha:.2f}' if alpha is not None else None),
                ('Samples', str(jury.get('samples')) if jury.get('samples') is not None else None),
                ('Method', jury.get('method')),
            ])
            + '</div>'
        )

    pipeline = method_data.get('pipeline', '')
    scoring = method_data.get('scoring_method', '')
    sub = f'{pipeline} pipeline · {scoring}' if pipeline or scoring else None
    methodology_html = panel('Methodology', f'{tested_html}{untested_html}{jury_html}', sub=sub)

    kept = _render_sections(by_kind, _SECTION_RENDERERS, ('severity_definitions', 'error_analysis'))
    return f'{run_config_html}{methodology_html}{kept}' or '<p class="rk-empty">No config data available.</p>'


def _rt_breakdowns_category_table(rows: list[dict[str, Any]]) -> str:
    """Attack success by OWASP category table, design-styled (spec §Breakdowns.1):
    Cat (tag) · Category (strong) · Run (right) · Found (right, red-600
    semibold when >0 else muted) · ASR (right, mono pct)."""
    from evaluatorq.common.reports import html_table
    from evaluatorq.common.reports.html_helpers import pct
    from evaluatorq.dashboard.report_kit import tag

    table_rows: list[list[str]] = []
    for r in rows:
        found = r.get('vulnerabilities_found', 0)
        found_html = (
            f'<span style="color:var(--red-600);font-weight:600">{found}</span>'
            if found
            else f'<span style="color:var(--text-muted)">{found}</span>'
        )
        table_rows.append([
            tag(str(r.get('category', ''))),
            f'<strong>{esc(r.get("category_name", ""))}</strong>',
            f'<span style="font-variant-numeric:tabular-nums">{r.get("total_attacks", 0)}</span>',
            found_html,
            f'<span style="font-family:var(--font-mono)">{pct(r.get("vulnerability_rate", 0.0))}</span>',
        ])
    return html_table(['Cat', 'Category', 'Run', 'Found', 'ASR'], table_rows)


def _rt_breakdowns_depth_footnote(rows: list[dict[str, Any]]) -> str:
    """Mono faint footnote: ``2t: 1/12 · 3t: 3/11 · …`` per depth row (spec §Breakdowns.4)."""
    parts = [f'{r.get("turn_count")}t: {r.get("vulnerabilities_found", 0)}/{r.get("total_attacks", 0)}' for r in rows]
    return f'<div class="rt-breakdowns-footnote">{esc(" · ".join(parts))}</div>'


def _rt_breakdowns_depth_leadin(rows: list[dict[str, Any]]) -> str:
    """Lead-in sentence above the depth bars, only when >=2 rows and ASR
    climbs from first to last depth (spec §Breakdowns.4)."""
    from evaluatorq.common.reports.html_helpers import pct

    if len(rows) < 2:
        return ''
    first, last = rows[0], rows[-1]
    if last.get('vulnerability_rate', 0.0) <= first.get('vulnerability_rate', 0.0):
        return ''
    return (
        '<p class="rt-breakdowns-leadin">Attack success climbs from '
        f'<strong>{pct(first.get("vulnerability_rate", 0.0))}</strong> at {first.get("turn_count")} turns to '
        f'<strong>{pct(last.get("vulnerability_rate", 0.0))}</strong> at {last.get("turn_count")} turns — '
        'single-turn defenses are not enough.</p>'
    )


def _rt_breakdowns_depth_section(by_kind: dict[str, Any]) -> str:
    """Attack success by conversation depth panel: only rendered when
    ``turn_depth_analysis`` exists (execution-derived; static-pipeline and
    single-turn-only runs simply omit this panel) — spec §Breakdowns.4."""
    from evaluatorq.common.reports.html_helpers import pct
    from evaluatorq.dashboard.report_kit import bar_rows, panel

    section = by_kind.get('turn_depth_analysis')
    if section is None:
        return ''
    rows = section.data.get('rows', [])
    if not rows:
        return ''
    max_rate = max((r.get('vulnerability_rate', 0.0) for r in rows), default=0.0) or 1.0
    bars = bar_rows(
        [(f'{r.get("turn_count")} turns', r.get('vulnerability_rate', 0.0)) for r in rows],
        width=520,
        label_w=70,
        color='var(--orange-500)',
        fmt=pct,
        max_value=max_rate,
    )
    leadin = _rt_breakdowns_depth_leadin(rows)
    footnote = _rt_breakdowns_depth_footnote(rows)
    return panel(
        'Attack success by conversation depth',
        f'{leadin}{bars}{footnote}',
        # Lowercase (not "Multi-turn") — no separate Multi-turn tab exists
        # (deviation #1, spec §Breakdowns.4); keeping the literal capitalized
        # phrase out of the folded Breakdowns tab avoids implying one.
        sub='multi-turn results only',
    )


def _rt_breakdowns_turn_scope_grid(by_kind: dict[str, Any]) -> str:
    """By turn type / by domain 2-col grid, omitted entirely when the
    ``turn_scope_breakdown`` section is None; either half omitted when its
    dict is empty (spec §Breakdowns.4)."""
    from evaluatorq.common.reports.html_helpers import pct
    from evaluatorq.dashboard.report_kit import SCALE_HEAT_RT, bar_rows, panel

    section = by_kind.get('turn_scope_breakdown')
    if section is None:
        return ''
    data = section.data

    def _grid_panel(title: str, entries: dict[str, dict[str, Any]]) -> str:
        if not entries:
            return ''
        rows = [(name.replace('_', ' '), stats.get('vulnerability_rate', 0.0)) for name, stats in entries.items()]
        bars = bar_rows(rows, width=420, label_w=110, color_scale=SCALE_HEAT_RT, fmt=pct, max_value=1.0)
        return panel(title, bars)

    turn_type_html = _grid_panel('By turn type', data.get('by_turn_type', {}))
    domain_html = _grid_panel('By domain', data.get('by_domain', {}))
    if not turn_type_html and not domain_html:
        return ''
    return f'<div class="rt-breakdowns-grid-2">{turn_type_html}{domain_html}</div>'


def _rt_breakdowns(by_kind: dict[str, Any]) -> str:
    """Breakdowns tab body: category table -> attack-success heatmap -> 2-col
    technique/delivery bar rows -> folded multi-turn section (depth panel +
    turn-scope grid, conditional) -> kept framework/vulnerability renders
    (spec §Breakdowns)."""
    from evaluatorq.dashboard.report_kit import SCALE_HEAT_RT, bar_rows, heatmap, panel

    category_section = by_kind.get('category_breakdown')
    category_html = ''
    if category_section is not None:
        rows = category_section.data.get('rows', [])
        if rows:
            category_html = panel(
                'Attack success by OWASP category',
                _rt_breakdowns_category_table(rows),
                sub='Worst first',
            )

    heatmap_section = by_kind.get('attack_heatmap')
    heatmap_html = ''
    if heatmap_section is not None and heatmap_section.data.get('cells'):
        data = heatmap_section.data
        table_html = heatmap(
            data.get('vulnerabilities', []),
            data.get('techniques', []),
            data.get('cells', []),
            row_key='vulnerability',
            col_key='technique',
            value_key='vulnerability_rate',
            color_scale=SCALE_HEAT_RT,
        )
        heatmap_html = panel(
            'Attack success heatmap',
            table_html,
            sub='Category × technique — sand → orange → red as ASR rises',  # noqa: RUF001
        )

    from evaluatorq.common.reports.html_helpers import pct

    def _rate_panel(title: str, kind: str, row_key: str) -> str:
        section = by_kind.get(kind)
        if section is None:
            return ''
        rows = section.data.get('rows', [])
        if not rows:
            return ''
        bars = bar_rows(
            [(str(r.get(row_key, '')), r.get('vulnerability_rate', 0.0)) for r in rows],
            width=420,
            label_w=150,
            color_scale=SCALE_HEAT_RT,
            fmt=pct,
            max_value=1.0,
        )
        return panel(title, bars)

    technique_html = _rate_panel('By technique', 'technique_breakdown', 'technique')
    delivery_html = _rate_panel('By delivery method', 'delivery_breakdown', 'delivery_method')
    rate_grid_html = (
        f'<div class="rt-breakdowns-grid-2">{technique_html}{delivery_html}</div>'
        if technique_html or delivery_html
        else ''
    )

    multiturn_html = f'{_rt_breakdowns_depth_section(by_kind)}{_rt_breakdowns_turn_scope_grid(by_kind)}'

    return f'{category_html}{heatmap_html}{rate_grid_html}{multiturn_html}'


def redteam_report_tabs(rid: str, report: RedTeamReport) -> str:
    """Render the Red Team report body as Streamlit-aligned tabs.

    7 tabs: Overview, Agents (N), Focus areas (N), Breakdowns, Attacks (N),
    Usage, Config — each populated from the precomputed report sections plus
    the HTMX interactive panels (empty tabs drop out).
    """
    from evaluatorq.redteam.reports.export_html import _SECTION_RENDERERS

    by_kind = _rt_by_kind(report)

    def render(*kinds: str) -> str:
        return _render_sections(by_kind, _SECTION_RENDERERS, kinds)

    hero = _redteam_hero(by_kind.get('summary'), report)

    focus_section = by_kind.get('focus_areas')
    focus_areas_list = focus_section.data.get('focus_areas', []) if focus_section is not None else []
    n_agents = len(report.tested_agents)
    n_focus = len(focus_areas_list)
    n_attacks = len(report.results)

    agents_tab = _rt_agents(by_kind, report, rid)

    focus_tab = _rt_focus(by_kind)

    breakdowns_tab = _rt_breakdowns(by_kind) + render('framework_breakdown', 'vulnerability_breakdown')

    attacks_tab = _rt_attacks(report, rid)

    usage_tab = render('token_usage') or '<p class="rk-empty">No token usage data recorded for this run.</p>'
    config_tab = _rt_config(by_kind, report)

    tabs = _tabs(
        'rttab',
        [
            ('Overview', _rt_overview(by_kind, report)),
            ('Agents', agents_tab, f'Agents <span class="tab-count">{n_agents}</span>'),
            ('Focus areas', focus_tab, f'Focus areas <span class="tab-count">{n_focus}</span>'),
            ('Breakdowns', breakdowns_tab),
            ('Attacks', attacks_tab, f'Attacks <span class="tab-count">{n_attacks}</span>'),
            ('Usage', usage_tab),
            ('Config', config_tab),
        ],
    )
    return f'<div class="report-aligned rt-report">{hero}{tabs}</div>'


def _rt_agent_pill(stats: dict[str, Any]) -> str:
    """One hero agent pill: dot (critical→red/vuln→orange/clean→green) + name
    + mono faint ``{n} vuln`` / ``clean`` (spec §Run header)."""
    vulns = stats.get('vulns', 0)
    critical = stats.get('critical', 0)
    dot_cls = 'rt-hero-dot--critical' if critical else ('rt-hero-dot--vuln' if vulns else 'rt-hero-dot--clean')
    sub = f'{vulns} vuln' if vulns else 'clean'
    return (
        '<span class="rt-hero-pill">'
        f'<span class="rt-hero-dot {dot_cls}"></span>'
        f'<span class="rt-hero-pill-name">{esc(stats.get("display_name", ""))}</span>'
        f'<span class="rt-hero-pill-sub">{esc(sub)}</span>'
        '</span>'
    )


def _redteam_hero(summary_section: Any, report: RedTeamReport) -> str:
    """Title row + `N agents` pill (multi only) + per-agent pill row (multi
    only). The 5-card KPI band moves to the Overview tab (spec §Run header) —
    no double KPI band."""
    multi_agent = len(report.tested_agents) > 1
    agents_pill = f'<span class="rt-hero-agents-pill">{len(report.tested_agents)} agents</span>' if multi_agent else ''
    agent_pills_html = ''
    if multi_agent:
        agent_stats = _rt_agent_stats(report)
        pills = ''.join(_rt_agent_pill(stats) for stats in agent_stats.values())
        agent_pills_html = f'<div class="rt-hero-agent-row">{pills}</div>'
    return (
        '<header class="report-hero rt-hero">'
        f'<h1 class="rt-hero-title">Red Team{agents_pill}</h1>'
        f'<p class="report-hero-sub">{esc(report.description or "Red teaming report")}</p>'
        f'{agent_pills_html}'
        '</header>'
    )
