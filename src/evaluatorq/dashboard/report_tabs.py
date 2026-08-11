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

import re
from typing import TYPE_CHECKING, Any, cast

from evaluatorq.common.reports import esc
from evaluatorq.dashboard.trace_links import run_trace_url, trace_link_button
from evaluatorq.simulation.metrics import TURN_METRICS

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

# Shown in place of any data-driven sim tab/block when the active filter matches
# zero conversations, so the report keeps its structure instead of collapsing.
_SIM_NO_MATCH_NOTE = '<p class="sim-empty-note">No conversations match the current filter.</p>'


def _stable_entries(run: SimulationRun, rows: list[Any]) -> list[Any]:
    """Build entries for *rows* while retaining their full-run indexes."""
    from evaluatorq.simulation.reports.sections import individual_entries

    full_indexes = {id(result): index for index, result in enumerate(run.results)}
    return [
        entry.model_copy(update={'index': full_indexes[id(result)]})
        for result, entry in zip(rows, individual_entries(rows), strict=True)
    ]


def sim_report_tabs(rid: str, run: SimulationRun, results: list[Any] | None = None, compare_html: str = '') -> str:
    """Render the Agent Sim report body as Streamlit-aligned tabs.

    Tabs: Overview · Breakdown · Transcripts · Turn quality · Config — each
    populated from the precomputed report sections (empty tabs drop out; Turn
    quality drops when a run carries no ``turn_metrics``). Config folds job-level
    metadata (run configuration, personas, scenarios) plus the kept token_usage
    table. Pass ``results`` to render a filtered subset (the filter round-trip);
    it defaults to the run's full result list. ``compare_html`` is an optional
    pre-rendered compare-with-another-run control shown in the hero actions.
    """
    from evaluatorq.dashboard.view import sim_interactive_panels
    from evaluatorq.simulation.reports.export_html import _SECTION_RENDERERS
    from evaluatorq.simulation.reports.sections import build_report_sections

    rows = run.results if results is None else results
    # The executive summary describes the complete run and always stays visible,
    # even on filtered views — it is whole-run context, not a subset metric.
    sections = build_report_sections(rows, executive_summary=run.executive_summary)
    by_kind: dict[str, Any] = {}
    for s in sections:
        by_kind.setdefault(s.kind, s)

    # Filtering affects report metrics and Breakdown rows, but Config must
    # remain a faithful registry of every configured persona and scenario.
    # Build its drawer templates from the full run so a cohort that has no
    # matching filtered conversations remains reachable and keeps its stable
    # DOM ID. Only the cohort conversation lists below use filtered entries.
    entity_sections = sections if results is None else build_report_sections(run.results)
    entity_by_kind: dict[str, Any] = {}
    for s in entity_sections:
        entity_by_kind.setdefault(s.kind, s)

    def render(*kinds: str) -> str:
        return _render_sections(by_kind, _SECTION_RENDERERS, kinds)

    hero = _sim_hero(run, compare_html)

    entries = _stable_entries(run, rows)

    entity_context = _sim_entity_context(entity_by_kind, entries, rows, rid)
    _set_failure_full_run_indexes(by_kind.get('failures_first'), entries)

    # When a filter matches no conversations, the data-driven tabs would
    # collapse to blank/dropped. Keep them present with an explicit "no matches"
    # note so the report structure stays stable (Config still shows the full-run
    # registry; Overview handles its own empty state internally).
    no_matches = results is not None and not rows

    breakdown_body = _SIM_NO_MATCH_NOTE if no_matches else _sim_breakdown(by_kind, rid, entity_context)
    transcripts_body = (
        _SIM_NO_MATCH_NOTE
        if no_matches
        else sim_interactive_panels(rid, entries) + render('evaluator_scores', 'judge_verdicts', 'errors')
    )
    turn_quality_body = _SIM_NO_MATCH_NOTE if no_matches else _sim_turn_quality(by_kind)

    # Folded 7→5 to curb tab sprawl: Evaluators + Judge & errors → Transcripts
    # (all per-conversation verdicts); Tokens → Config. Turn quality is its own
    # tab (unfolded from Breakdown) and drops out when a run has no turn_metrics.
    tabs = _tabs(
        'simtab',
        [
            ('Overview', _sim_overview(rid, by_kind, entity_by_kind, rows, run, filtered=results is not None)),
            ('Breakdown', breakdown_body),
            (
                'Transcripts',
                transcripts_body,
                f'Transcripts <span class="tab-count">{len(entries)}</span>',
            ),
            ('Turn quality', turn_quality_body),
            ('Config', _sim_config(by_kind, run, entity_context) + render('token_usage')),
        ],
    )
    return f'<div class="report-aligned sim-report">{hero}{tabs}{_sim_entity_modal(entity_context)}</div>'


def _section_rows(by_kind: dict[str, Any], section_kind: str, key: str) -> list[dict[str, Any]]:
    section = by_kind.get(section_kind)
    data = section.data if section is not None else {}
    rows = data.get(key, [])
    return rows if isinstance(rows, list) else []


def _group_entries_by_cohort(
    entries: list[Any], rows: list[Any], personas: list[dict[str, Any]], scenarios: list[dict[str, Any]]
) -> dict[str, dict[str, list[Any]]]:
    """Associate transcript entries with Task 1's collision-safe cohort IDs."""
    from evaluatorq.simulation.reports.sections import _persona_cohort_id, _scenario_cohort_id

    cohorts = {
        'persona': {str(row['id']): [] for row in personas},
        'scenario': {str(row['id']): [] for row in scenarios},
    }
    for entry, result in zip(entries, rows, strict=True):
        persona_id = _persona_cohort_id(result)
        scenario_id = _scenario_cohort_id(result)
        if persona_id in cohorts['persona']:
            cohorts['persona'][persona_id].append(entry)
        if scenario_id in cohorts['scenario']:
            cohorts['scenario'][scenario_id].append(entry)
    return cohorts


def _sim_entity_context(
    by_kind: dict[str, Any], entries: list[Any] | None = None, rows: list[Any] | None = None, rid: str = ''
) -> dict[str, Any]:
    personas = _section_rows(by_kind, 'overview', 'personas')
    scenarios = _section_rows(by_kind, 'overview', 'scenarios')
    persona_rows = _section_rows(by_kind, 'persona_breakdown', 'rows')
    scenario_rows = _section_rows(by_kind, 'scenario_breakdown', 'rows')
    entries = entries or []
    rows = rows or []
    return {
        'rid': rid,
        'personas': personas,
        'scenarios': scenarios,
        'persona_stats': {str(row['id']): row for row in persona_rows},
        'scenario_stats': {str(row['id']): row for row in scenario_rows},
        'cohorts': _group_entries_by_cohort(entries, rows, personas, scenarios),
        'persona_dom_ids': {str(row['id']): f'persona-{index}' for index, row in enumerate(personas)},
        'scenario_dom_ids': {str(row['id']): f'scenario-{index}' for index, row in enumerate(scenarios)},
    }


def _set_failure_full_run_indexes(section: Any, entries: list[Any]) -> None:
    """Replace filtered section offsets with the entries' stable run indexes."""
    if section is None:
        return
    failures = [
        entry for entry in entries if not entry.goal_achieved and not entry.error and entry.terminated_by != 'error'
    ]
    for row, entry in zip(section.data.get('rows', []), failures, strict=True):
        row['index'] = entry.index + 1


def _sim_config(by_kind: dict[str, Any], run: SimulationRun, entity_context: dict[str, Any] | None = None) -> str:
    """Config tab body: run-configuration meta grid → personas panel →
    scenarios panel (spec §Config.1-3). The kept ``token_usage`` table is
    appended by the caller."""
    from evaluatorq.dashboard.report_kit import meta_grid, panel

    entity_context = entity_context or _sim_entity_context(by_kind)
    personas: list[dict[str, Any]] = entity_context.get('personas', [])
    scenarios: list[dict[str, Any]] = entity_context.get('scenarios', [])
    persona_dom_ids: dict[str, str] = entity_context.get('persona_dom_ids', {})
    scenario_dom_ids: dict[str, str] = entity_context.get('scenario_dom_ids', {})

    generated = run.created_at
    # Human month + time-of-day (e.g. "Jul 6, 2026 · 16:42 UTC"); created_at is
    # stored UTC-aware. Fall back to a raw slice for anything without strftime.
    if hasattr(generated, 'strftime'):
        generated_str = f'{generated:%b} {generated.day}, {generated:%Y · %H:%M} UTC'
    else:
        generated_str = str(generated)[:16]

    # Run scale (persona/scenario/conversation counts) reads as "how big was
    # this run"; promote it to stat tiles above the textual config so it pops.
    scope = [
        t
        for t in (
            (str(len(personas)), 'Personas') if personas else None,
            (str(len(scenarios)), 'Scenarios') if scenarios else None,
            (str(run.total_results), 'Conversations'),
        )
        if t
    ]
    stat_html = ''
    if scope:
        tiles = ''.join(
            f'<div class="rk-stat"><span class="rk-stat-num">{esc(num)}</span>'
            f'<span class="rk-stat-cap">{esc(cap)}</span></div>'
            for num, cap in scope
        )
        stat_html = f'<div class="rk-stat-row">{tiles}</div>'

    # Config split into semantic groups (target vs run metadata) instead of one
    # flat sweep, so the panel is scannable.
    target_grid = meta_grid([
        ('Target', run.target),
        ('Run name', run.run_name),
        ('Model', run.target_model),
        ('Target kind', run.target_kind),
        ('Mode', run.mode),
        ('Max turns', str(run.max_turns) if run.max_turns is not None else None),
    ])
    run_grid = meta_grid([
        ('Evaluators', ', '.join(run.evaluator_names) if run.evaluator_names else None),
        ('Generated', generated_str),
    ])
    groups = (
        f'<div class="rk-meta-group"><div class="rk-meta-group-title">Target</div>{target_grid}</div>'
        f'<div class="rk-meta-group"><div class="rk-meta-group-title">Run</div>{run_grid}</div>'
    )
    config_html = panel(
        'Run configuration',
        f'{stat_html}{groups}',
        sub='Job-level metadata',
    )

    personas_html = ''
    if personas:
        rows = _sim_config_persona_header() + ''.join(
            _sim_config_persona_row(p, persona_dom_ids.get(str(p.get('id', '')))) for p in personas
        )
        personas_html = panel('Personas', rows, sub='Simulated user profiles')

    scenarios_html = ''
    if scenarios:
        rows = _sim_config_scenario_header() + ''.join(
            _sim_config_scenario_row(s, scenario_dom_ids.get(str(s.get('id', '')))) for s in scenarios
        )
        scenarios_html = panel('Scenarios', rows, sub='Goals + pass/fail criteria')

    return f'{config_html}{personas_html}{scenarios_html}'


def _sim_config_persona_header() -> str:
    """Column headers for the persona grid: Persona · Tone · one per trait."""
    traits = ''.join(f'<span>{esc(label)}</span>' for _key, label in _SIM_TRAIT_LABELS)
    return f'<div class="sim-config-persona-head"><span>Persona</span><span>Tone</span>{traits}</div>'


def _sim_config_persona_row(persona: dict[str, Any], entity_id: str | None = None) -> str:
    """One compact Config-panel persona row: name · tone · trait mini-bars (grid columns)."""
    name = esc(persona.get('name', ''))
    traits = persona.get('traits')
    style = traits.get('communication_style') if isinstance(traits, dict) else None
    style_html = f'<span class="sim-config-persona-style">{esc(str(style)) if style else "—"}</span>'
    traits_html = _sim_trait_minis(traits) if isinstance(traits, dict) else ''
    trigger_attrs = _sim_entity_trigger_attrs('persona', entity_id)
    return (
        f'<button type="button" class="sim-config-persona-row sim-entity-row" {trigger_attrs}>'
        f'<span class="sim-config-persona-name">{name}</span>{style_html}{traits_html}'
        '</button>'
    )


def _sim_config_scenario_header() -> str:
    """Column headers for the scenario grid: Scenario · Criteria · Pass rate
    (aligned with the row grid below)."""
    return (
        '<div class="sim-config-scenario-head"><span>Scenario</span><span>Criteria</span><span>Pass rate</span></div>'
    )


def _sim_config_scenario_row(scenario: dict[str, Any], entity_id: str | None = None) -> str:
    """One compact Config-panel scenario row: name + goal · criteria split · pass-rate badge
    (three grid columns aligned with the header)."""
    name = esc(scenario.get('name', ''))
    goal = scenario.get('goal')
    goal_html = f'<div class="sim-config-scenario-goal">{esc(str(goal))}</div>' if goal else ''
    count_html = _sim_config_criteria_split(scenario.get('criteria') or [])
    badge_html = _sim_pass_rate_badge(scenario.get('pass_rate'))
    trigger_attrs = _sim_entity_trigger_attrs('scenario', entity_id)
    return (
        f'<button type="button" class="sim-config-scenario-row sim-entity-row" {trigger_attrs}>'
        f'<div class="sim-config-scenario-main">'
        f'<span class="sim-config-scenario-name">{name}</span>{goal_html}'
        '</div>'
        f'<span class="sim-config-scenario-checks">{count_html}</span>'
        f'<span class="sim-config-scenario-rate-cell">{badge_html}</span>'
        '</button>'
    )


def _sim_config_criteria_split(criteria: list[dict[str, Any]]) -> str:
    """`N must-happen · M must-not` label, split by criterion type. Falls back
    to the flat `N checks` label when no criterion carries a recognised type."""
    must_happen = sum(1 for c in criteria if c.get('type') == 'must_happen')
    must_not = sum(1 for c in criteria if c.get('type') == 'must_not_happen')
    if must_happen == 0 and must_not == 0:
        count = len(criteria)
        label = f'{count} {"check" if count == 1 else "checks"}'
        return f'<span class="sim-config-check-count">{label}</span>'

    parts: list[str] = []
    if must_happen:
        parts.append(f'{must_happen} must-happen')
    if must_not:
        parts.append(f'{must_not} must-not')
    return f'<span class="sim-config-check-count">{" · ".join(parts)}</span>'


def _sim_pass_rate_badge(pass_rate: float | None) -> str:
    """Small pill with a colored dot showing the scenario's pass rate. Omitted
    entirely when the pass rate is unknown."""
    if pass_rate is None:
        return ''
    if pass_rate >= 0.8:
        severity = 'high'
    elif pass_rate >= 0.5:
        severity = 'mid'
    else:
        severity = 'low'
    pct = round(pass_rate * 100)
    return (
        f'<span class="sim-scenario-rate sim-scenario-rate--{severity}">'
        f'<span class="sim-scenario-rate-dot"></span>{pct}%'
        '</span>'
    )


_SIM_TRAIT_LABELS: tuple[tuple[str, str], ...] = (
    ('patience', 'Patience'),
    ('assertiveness', 'Assertiveness'),
    ('politeness', 'Politeness'),
    ('technical_level', 'Technical'),
)


def _sim_trait_minis(traits: dict[str, Any]) -> str:
    # One cell per label (empty placeholder if missing) so cells stay column-aligned under the headers.
    # data-tip drives an instant CSS tooltip (native title has a ~1s browser delay).
    bars: list[str] = []
    for key, label in _SIM_TRAIT_LABELS:
        raw = traits.get(key)
        if not isinstance(raw, int | float):
            bars.append('<span class="sim-trait-mini sim-trait-mini--empty" aria-hidden="true"></span>')
            continue
        value = max(0.0, min(1.0, float(raw)))
        bars.append(
            f'<span class="sim-trait-mini" data-tip="{esc(label)} {value:.2f}" aria-label="{esc(label)} {value:.2f}">'
            f'<span class="sim-trait-mini-fill" style="width:{value * 100:.0f}%"></span>'
            f'</span>'
        )
    return f'<span class="sim-trait-minis">{"".join(bars)}</span>'


def _sim_entity_trigger_attrs(kind: str, entity_id: str | None) -> str:
    if not entity_id:
        return 'disabled aria-disabled="true"'
    return f'data-sim-entity-trigger data-entity-kind="{esc(kind)}" data-entity-id="{esc(entity_id)}"'


def _sim_entity_modal(entity_context: dict[str, Any]) -> str:
    personas: list[dict[str, Any]] = entity_context.get('personas', [])
    scenarios: list[dict[str, Any]] = entity_context.get('scenarios', [])
    persona_stats: dict[str, dict[str, Any]] = entity_context.get('persona_stats', {})
    scenario_stats: dict[str, dict[str, Any]] = entity_context.get('scenario_stats', {})
    cohorts: dict[str, dict[str, list[Any]]] = entity_context.get('cohorts', {})
    persona_dom_ids: dict[str, str] = entity_context.get('persona_dom_ids', {})
    scenario_dom_ids: dict[str, str] = entity_context.get('scenario_dom_ids', {})
    rid = str(entity_context.get('rid', ''))
    if not personas and not scenarios:
        return ''

    templates: list[str] = []
    for i, persona in enumerate(personas):
        cohort_id = str(persona['id'])
        templates.append(
            _sim_persona_template(
                persona,
                persona_dom_ids[cohort_id],
                i,
                len(personas),
                persona_stats.get(cohort_id),
                cohorts.get('persona', {}).get(cohort_id, []),
                rid,
            )
        )
    for i, scenario in enumerate(scenarios):
        cohort_id = str(scenario['id'])
        templates.append(
            _sim_scenario_template(
                scenario,
                scenario_dom_ids[cohort_id],
                i,
                len(scenarios),
                scenario_stats.get(cohort_id),
                cohorts.get('scenario', {}).get(cohort_id, []),
                rid,
            )
        )

    return (
        '<dialog class="sim-entity-dialog" aria-label="Simulation entity detail">'
        '<div class="sim-entity-modal-shell">'
        '<div class="sim-entity-modal-content" data-sim-entity-content></div>'
        '<div class="sim-entity-modal-actions">'
        '<button type="button" class="sim-entity-back" data-sim-entity-back hidden aria-label="Back to cohort">&larr; Back</button>'
        '<button type="button" class="sim-entity-nav" data-sim-entity-prev aria-label="Previous entity (k)" title="Previous entity (k)">&larr;<kbd class="sim-entity-kbd">k</kbd></button>'
        '<button type="button" class="sim-entity-nav" data-sim-entity-next aria-label="Next entity (j)" title="Next entity (j)">&rarr;<kbd class="sim-entity-kbd">j</kbd></button>'
        '<button type="button" class="sim-entity-close" data-sim-entity-close>Close</button>'
        '</div>'
        '</div>'
        '</dialog>'
        f'<div class="sim-entity-templates" hidden>{"".join(templates)}</div>'
    )


def _sim_persona_template(
    persona: dict[str, Any],
    entity_id: str,
    index: int,
    total: int,
    stats: dict[str, Any] | None = None,
    conversations: list[Any] | None = None,
    rid: str = '',
) -> str:
    name = esc(persona.get('name', ''))
    raw_traits = persona.get('traits')
    traits: dict[str, Any] = raw_traits if isinstance(raw_traits, dict) else {}
    style = traits.get('communication_style')
    background = persona.get('background') or traits.get('background')
    conversation_count = persona.get('conversations')
    trait_rows = ''.join(
        _sim_trait_detail(label, traits.get(key))
        for key, label in _SIM_TRAIT_LABELS
        if isinstance(traits.get(key), int | float)
    )
    meta = ''.join(
        item
        for item in [
            f'<span class="sim-entity-pill">{esc(str(style))}</span>' if style else '',
            f'<span class="sim-entity-pill">{esc(str(conversation_count))} conversations</span>'
            if conversation_count
            else '',
        ]
        if item
    )
    background_html = f'<p class="sim-entity-prose">{esc(str(background))}</p>' if background else ''
    body = (
        f'<div class="sim-entity-detail">'
        f'<div class="sim-entity-kicker">Persona {index + 1} / {total}</div>'
        f'<h2>{name}</h2>'
        f'<div class="sim-entity-pills">{meta}</div>'
        f'<div class="sim-entity-traits">{trait_rows}</div>'
        f'{background_html}'
        f'{_sim_cohort_stats(stats)}{_sim_cohort_list(conversations or [], rid)}'
        f'</div>'
    )
    return (
        f'<template id="{esc(entity_id)}" data-sim-entity-template data-entity-kind="persona" '
        f'data-entity-id="{esc(entity_id)}" data-entity-index="{index}">{body}</template>'
    )


def _sim_scenario_template(
    scenario: dict[str, Any],
    entity_id: str,
    index: int,
    total: int,
    stats: dict[str, Any] | None = None,
    conversations: list[Any] | None = None,
    rid: str = '',
) -> str:
    name = esc(scenario.get('name', ''))
    goal = scenario.get('goal')
    context = scenario.get('context')
    criteria = scenario.get('criteria') or []
    criteria_html = ''.join(_sim_criterion_detail(c) for c in criteria)
    goal_html = f'<p class="sim-entity-prose">{esc(str(goal))}</p>' if goal else ''
    context_html = f'<p class="sim-entity-prose">{esc(str(context))}</p>' if context else ''
    goal_section = f'<section><h3>Goal</h3>{goal_html}</section>' if goal_html else ''
    context_section = f'<section><h3>Context</h3>{context_html}</section>' if context_html else ''
    criteria_section = (
        f'<section><h3>Criteria</h3><ul class="sim-entity-criteria">{criteria_html}</ul></section>'
        if criteria_html
        else ''
    )
    checks = f'{len(criteria)} {"check" if len(criteria) == 1 else "checks"}'
    body = (
        f'<div class="sim-entity-detail">'
        f'<div class="sim-entity-kicker">Scenario {index + 1} / {total}</div>'
        f'<h2>{name}</h2>'
        f'<div class="sim-entity-pills"><span class="sim-entity-pill">{checks}</span></div>'
        f'{goal_section}{context_section}{criteria_section}'
        f'{_sim_cohort_stats(stats)}{_sim_cohort_list(conversations or [], rid)}'
        f'</div>'
    )
    return (
        f'<template id="{esc(entity_id)}" data-sim-entity-template data-entity-kind="scenario" '
        f'data-entity-id="{esc(entity_id)}" data-entity-index="{index}">{body}</template>'
    )


def _sim_cohort_stats(stats: dict[str, Any] | None) -> str:
    if not stats:
        return ''
    from evaluatorq.common.reports.html_helpers import pct

    rate = float(stats.get('success_rate', 0.0))
    score = float(stats.get('avg_goal_completion_score', 0.0))
    tokens = stats.get('total_tokens')
    token_text = f'{int(tokens):,}' if isinstance(tokens, int | float) and tokens else '—'
    return (
        '<div class="sim-cohort-stats">'
        f'<span><b>Goal rate</b>{esc(pct(rate))}</span>'
        f'<span><b>Avg score</b>{score:.2f}</span>'
        f'<span><b>Tokens</b>{token_text}</span>'
        '</div>'
    )


def _sim_cohort_list(entries: list[Any], rid: str) -> str:
    if not entries:
        return '<p class="sim-cohort-empty">No conversations.</p>'

    from evaluatorq.common.reports.html_helpers import status_badge

    items: list[str] = []
    for entry in entries:
        outcome = status_badge('Passed' if entry.goal_achieved else 'Failed', 'pass' if entry.goal_achieved else 'fail')
        items.append(
            f'<button type="button" class="sim-cohort-conversation" '
            f'data-sim-entity-trigger data-entity-kind="conversation" '
            f'data-drawer-url="/r/{esc(rid)}/sim/transcript?idx={int(entry.index)}">'
            f'<span>#{int(entry.index) + 1}</span><span>{esc(entry.persona)}</span>'
            f'<span>{esc(entry.scenario)}</span><span>{float(entry.goal_completion_score):.2f}</span>{outcome}'
            '</button>'
        )
    return (
        '<section class="sim-cohort-conversations"><h3>Conversations</h3>'
        f'<div class="sim-cohort-list">{"".join(items)}</div></section>'
    )


def _sim_trait_detail(label: str, raw: Any) -> str:
    value = max(0.0, min(1.0, float(raw)))
    return (
        '<div class="sim-entity-trait-row">'
        f'<span>{esc(label)}</span><span class="sim-entity-trait-track">'
        f'<span class="sim-entity-trait-fill" style="width:{value * 100:.0f}%"></span></span>'
        f'<span class="sim-entity-trait-value">{value:.2f}</span>'
        '</div>'
    )


def _sim_criterion_detail(criterion: dict[str, Any]) -> str:
    ctype = criterion.get('type')
    is_negative = ctype == 'must_not_happen'
    mark = '&#x2717;' if is_negative else '&#x2713;'
    cls = 'sim-entity-criterion--negative' if is_negative else 'sim-entity-criterion--positive'
    type_html = f'<span class="sim-entity-criterion-type">{esc(str(ctype).replace("_", " "))}</span>' if ctype else ''
    return (
        f'<li class="sim-entity-criterion {cls}">'
        f'<span class="sim-entity-criterion-mark">{mark}</span>'
        f'<span>{esc(str(criterion.get("description", "")))}</span>{type_html}'
        '</li>'
    )


def _tinted(text: str, value: float) -> str:
    """Wrap a cell value in a span tinted red->yellow->green by ``value`` (0-1,
    higher = better - both goal rate and avg score are goal-completion metrics)."""
    from evaluatorq.dashboard.report_kit import _interp_color

    color = _interp_color(max(0.0, min(1.0, value)))
    return f'<span class="sim-td-tint" style="color:{color}">{esc(text)}</span>'


def _sim_failures_table(section: Any, rid: str) -> str:
    """Dashboard-only failure rows, which open their transcripts in the drawer."""
    from evaluatorq.common.reports.html_helpers import html_table
    from evaluatorq.simulation.reports.export_html import _cap, _criteria_dots

    rows = section.data.get('rows', []) if section is not None else []
    if not rows:
        return '<section class="report-card"><h2>Failures</h2><p>No failed conversations.</p></section>'
    headers = ['Scenario', 'Persona', 'Why', 'Criteria', 'Score']
    table_rows = [
        [
            esc(str(row['scenario'])),
            esc(str(row['persona'])),
            (
                f'<span class="fail-why" data-no-drawer title="{esc(str(row.get("reason", "")))}">'
                f'{esc(_cap(str(row.get("reason", ""))))}</span>'
            ),
            # data-no-drawer so unfolding the dots foldout doesn't open the drawer.
            f'<span data-no-drawer>{_criteria_dots(row.get("criteria", []))}</span>',
            f'{float(row["score"]):.2f}',
        ]
        for row in rows
    ]
    row_attrs = [
        'class="sim-drawer-row sim-failure-row" role="button" tabindex="0" '
        'data-sim-entity-trigger data-entity-kind="conversation" '
        f'data-drawer-url="/r/{esc(rid)}/sim/transcript?idx={int(row["index"]) - 1}"'
        for row in rows
    ]
    return f'<section class="report-card"><h2>Failures</h2>{html_table(headers, table_rows, row_attrs)}</section>'


def _sim_breakdown_table(
    section: Any,
    title: str,
    key: str,
    headers: list[str],
    cols: Callable[[dict[str, Any]], list[str]],
    kind: str,
    entity_ids: dict[str, str],
) -> str:
    """One Breakdown per-persona/per-scenario table (mockup columns), wrapped in
    a serif-titled panel. ``key`` is the label column field; ``cols`` maps a row
    to the remaining (numeric) cells. Empty section → ''."""
    from evaluatorq.common.reports.html_helpers import html_table
    from evaluatorq.dashboard.report_kit import panel

    rows = section.data.get('rows', []) if section is not None else []
    if not rows:
        return ''
    table_rows = [[esc(str(row[key])), *cols(row)] for row in rows]
    row_attrs = [
        'class="sim-drawer-row" role="button" tabindex="0" '
        f'{_sim_entity_trigger_attrs(kind, entity_ids.get(str(row.get("id", ""))))}'
        for row in rows
    ]
    table = html_table(headers, table_rows, row_attrs)
    return panel(title, f'<div class="sim-bd-table">{table}</div>')


def _sim_breakdown(
    by_kind: dict[str, Any],
    rid: str,
    entity_context: dict[str, Any] | None = None,
) -> str:
    """Breakdown tab body: heatmap → failures table → top failure modes →
    score-distribution → per-persona/scenario tables (spec §Breakdown)."""
    from evaluatorq.common.reports.html_helpers import pct
    from evaluatorq.dashboard.report_kit import heatmap, histogram, panel

    entity_context = entity_context or _sim_entity_context(by_kind)
    persona_dom_ids: dict[str, str] = entity_context.get('persona_dom_ids', {})
    scenario_dom_ids: dict[str, str] = entity_context.get('scenario_dom_ids', {})

    heatmap_section = by_kind.get('persona_scenario_heatmap')
    heatmap_html = ''
    if heatmap_section is not None:
        d = heatmap_section.data
        heatmap_html = panel(
            'Goal completion — persona × scenario',  # noqa: RUF001 (mockup wording — spec §Breakdown.1)
            heatmap(d.get('personas', []), d.get('scenarios', []), d.get('cells', []), value_key='avg_score'),
            sub='Red → yellow → green as the average goal-completion score rises',
            info=(
                'Each cell is the average goal-completion score (0–100%) the judge gave the '  # noqa: RUF001
                "agent across that persona × scenario's conversations — a continuous measure of "  # noqa: RUF001
                'how fully the goal was met, not a count of goals achieved. 100% = the goal was '
                'fully accomplished in every conversation; a single half-met conversation reads '
                '~50%. The binary pass/fail rate is shown separately as "Goal rate" in the tables below.'
            ),
        )

    dist_section = by_kind.get('score_distribution')
    dist_html = ''
    if dist_section is not None:
        scores = dist_section.data.get('scores', [])
        if scores:
            dist_html = panel(
                'Goal-completion score distribution',
                histogram(scores),
                sub=f'{len(scores)} conversations · dashed line = mean',
            )

    # Dashboard-specific tables (mockup columns), not the flat-export renderer
    # (which carries extra Achieved/Success columns for the standalone HTML).
    # Full-width stacked (not 2-col) so long persona/scenario names don't wrap and
    # cram; Goal rate + Avg score are tinted red→green by value.
    persona_html = _sim_breakdown_table(
        by_kind.get('persona_breakdown'),
        'Per-persona',
        'persona',
        ['Persona', 'Conv', 'Goal rate', 'Avg score', 'Tokens'],
        lambda r: [
            str(r['conversations']),
            _tinted(pct(r['success_rate']), r['success_rate']),
            _tinted(f'{r["avg_goal_completion_score"]:.2f}', r['avg_goal_completion_score']),
            f'{r["total_tokens"]:,}' if r.get('total_tokens') else '—',
        ],
        'persona',
        persona_dom_ids,
    )
    scenario_html = _sim_breakdown_table(
        by_kind.get('scenario_breakdown'),
        'Per-scenario',
        'scenario',
        ['Scenario', 'Conv', 'Goal rate', 'Avg score', 'Avg turns', 'Tokens'],
        lambda r: [
            str(r['conversations']),
            _tinted(pct(r['success_rate']), r['success_rate']),
            _tinted(f'{r["avg_goal_completion_score"]:.2f}', r['avg_goal_completion_score']),
            f'{r["avg_turn_count"]:.1f}',
            f'{r["total_tokens"]:,}' if r.get('total_tokens') else '—',
        ],
        'scenario',
        scenario_dom_ids,
    )
    tables_html = f'{persona_html}{scenario_html}'

    failure_mode_section = by_kind.get('failure_mode')
    failure_html = ''
    if failure_mode_section is not None:
        rows = [(str(label), int(count)) for label, count in failure_mode_section.data.get('rows', [])]
        failure_html = _sim_failure_modes(rows)

    failures_html = f'<div id="section-failures_first">{_sim_failures_table(by_kind.get("failures_first"), rid)}</div>'

    return f'{heatmap_html}{failures_html}{failure_html}{dist_html}{tables_html}'


def _sim_failure_modes(rows: list[tuple[str, int]]) -> str:
    """'Top failure modes' as HTML bars with a client-side min-count slider.

    Each failed ``scenario: criterion`` pair aggregates across the personas the
    scenario ran against; the count is how many conversations tripped it. Bars
    below the slider threshold are hidden (default 2 — hide one-offs, which are
    just the failures-first list). Labels truncate at the *end* via CSS ellipsis
    so the scenario/criterion prefix stays readable. Returns '' when no rows."""
    if not rows:
        return ''
    max_c = max(c for _, c in rows)
    # Default: hide singletons when any repeat exists; otherwise show everything
    # (a max of 1 means every failure is a one-off and the slider is a no-op).
    default = 2 if max_c >= 2 else 1
    bars: list[str] = []
    for label, count in rows:
        pct = (count / max_c) * 100 if max_c else 0
        hidden = ' hidden' if count < default else ''
        bars.append(
            f'<div class="sim-fm-row" data-count="{count}"{hidden}>'
            f'<span class="sim-fm-label" title="{esc(label)}">{esc(label)}</span>'
            f'<span class="sim-fm-track"><span class="sim-fm-fill" style="width:{pct:.1f}%"></span></span>'
            f'<span class="sim-fm-count">{count}</span>'
            '</div>'
        )
    slider = (
        '<span class="sim-fm-filter">'
        '<label for="sim-fm-slider">min count</label>'
        f'<input id="sim-fm-slider" type="range" class="sim-fm-slider" data-fm-slider '
        f'min="1" max="{max_c}" value="{default}" aria-label="Minimum failure count">'
        f'<output class="sim-fm-out" data-fm-out>{default}</output>'
        '</span>'
    )
    return (
        '<div class="rk-panel sim-fm" data-fm-panel>'
        f'<div class="rk-panel-title sim-fm-head"><span>Top failure modes</span>{slider}</div>'
        '<div class="rk-panel-body">'
        f'<div class="sim-fm-bars">{"".join(bars)}</div>'
        '<p class="sim-fm-empty" data-fm-empty hidden>No failure modes at or above this count.</p>'
        '</div></div>'
    )


def _turn_delta_callout(series: dict[str, list[float | None]]) -> str:
    """Templated first-to-last-turn delta callout, no confidence pill (spec
    §Turn.1). A clause renders only for series with >= 2 non-None points;
    absent/short metrics are dropped. Returns '' when nothing qualifies."""
    clauses: list[str] = []
    for metric in TURN_METRICS:
        values = series.get(metric.key, [])
        points = [v for v in values if v is not None]
        if len(points) < 2:
            continue
        delta = points[-1] - points[0]
        label = esc(metric.label)
        if abs(delta) < 0.005:
            clauses.append(f'{label} held steady around <strong>{points[-1]:.2f}</strong>')
            continue
        verb = ('rose' if delta > 0 else 'fell') if metric.high_is_risky else ('improved' if delta > 0 else 'declined')
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
    """Average-quality metric cells — editorial 2-col grid with a per-metric
    accent tick colored by a "goodness" score (risk metrics are inverted)."""
    from evaluatorq.dashboard.report_kit import _interp_color

    if not avg_quality_metrics:
        return ''
    cells = []
    for metric in TURN_METRICS:
        value = avg_quality_metrics.get(metric.key)
        if value is None:
            continue
        score = (1.0 - value) if metric.high_is_risky else value
        color = _interp_color(score)
        label = esc(metric.label)
        cells.append(
            f'<div class="sim-aq-cell"><div class="sim-aq-label">{label}</div>'
            f'<div class="sim-aq-value" style="--aq-accent:{color}">{value:.2f}</div></div>'
        )
    return f'<div class="sim-aq-grid">{"".join(cells)}</div>'


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

    # Single-turn runs (every conversation has ≤1 turn) have no per-turn story to
    # tell — no trend, a one-bar distribution — so drop the whole tab.
    turn_dist = metrics_data.get('turn_count_distribution', {})
    if turn_dist and max(turn_dist, default=0) <= 1:
        return ''

    callout_html = _turn_delta_callout(series)

    chart_html = ''
    # Mockup: x-axis reads "Turn N" and the legend uses human labels, not raw keys.
    turns = timeline_data.get('turns', [])
    # A trend line needs ≥2 turns; with a single turn the chart is an empty plot
    # (dots at one x, no lines) that leaves a large void above the metrics — skip
    # it, the avg-quality tiles below already show the single-turn values.
    if len(turns) >= 2:
        x_labels = [f'Turn {t}' for t in turns]
        pretty_series = {
            metric.label.capitalize(): series[metric.key] for metric in TURN_METRICS if metric.key in series
        }
        chart = line_chart(x_labels, pretty_series)
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


def _sim_overview(
    rid: str,
    by_kind: dict[str, Any],
    full_by_kind: dict[str, Any],
    rows: list[Any],
    run: SimulationRun,
    *,
    filtered: bool,
) -> str:
    """Overview tab body: agent info card, exec summary, KPI band, and a
    two-column outcomes/quality grid. Persona and scenario input live in Config.

    The executive summary is whole-run context, so it is built from
    ``full_by_kind`` (the unfiltered sections) and always renders — a filter,
    even one matching zero conversations, never makes it vanish. When a filter
    is active its label is qualified with "· whole run" so the reader knows the
    prose is not describing the narrowed subset below it. The KPI band, outcomes
    donut, and quality tiles remain filtered-subset metrics.
    """
    from evaluatorq.dashboard.report_kit import callout, exec_summary, panel

    agent_card_html = _sim_agent_section(rid, run)
    summary_section = by_kind.get('summary')
    heatmap_section = by_kind.get('persona_scenario_heatmap')
    tokens_section = by_kind.get('token_usage')
    metrics_section = by_kind.get('turn_metrics')

    summary_data = summary_section.data if summary_section is not None else {}
    heatmap_data = heatmap_section.data if heatmap_section is not None else {}
    tokens_data = tokens_section.data if tokens_section is not None else {}
    metrics_data = metrics_section.data if metrics_section is not None else {}

    # One executive-summary card, always whole-run. The saved narrative (richer
    # prose) wins; otherwise the computed stat sentence from the full run. Using
    # the unfiltered sections means it never collapses under a zero-match filter.
    full_summary_data = s.data if (s := full_by_kind.get('summary')) is not None else {}
    full_heatmap_data = s.data if (s := full_by_kind.get('persona_scenario_heatmap')) is not None else {}
    es_label = 'Executive summary · whole run' if filtered else 'Executive summary'
    confidence = full_summary_data.get('confidence')
    if run.executive_summary:
        summary_html = callout(esc(str(run.executive_summary)), label=es_label, confidence=confidence)
    else:
        summary_html = exec_summary(
            summary_data=full_summary_data,
            heatmap_data=full_heatmap_data,
            confidence=confidence,
            label=es_label,
        )
    kpi_html = _sim_kpi_band(
        summary_data,
        n_personas=len(heatmap_data.get('personas', [])),
        n_scenarios=len(heatmap_data.get('scenarios', [])),
    )
    # Outcomes + Average quality metrics. When a filter matches no conversations
    # both blocks would otherwise collapse to empty; render an explicit "no
    # matches" state instead so the Overview keeps its structure.
    if not rows:
        donut_html = f'<figure class="chart-card"><figcaption>Outcomes</figcaption>{_SIM_NO_MATCH_NOTE}</figure>'
        second_html = panel('Average quality metrics', _SIM_NO_MATCH_NOTE)
    else:
        donut_html = _sim_outcomes_donut(rows)
        # Average quality metrics (turn metrics); fall back to the token-usage
        # summary for runs that carry no per-turn quality data.
        quality_tiles = _sim_avg_quality_tiles(metrics_data.get('avg_quality_metrics', {}))
        second_html = (
            panel('Average quality metrics', quality_tiles) if quality_tiles else _sim_tokens_panel(tokens_data)
        )

    return f'{summary_html}{agent_card_html}{kpi_html}<div class="sim-overview-grid-2">{donut_html}{second_html}</div>'


# Scalar identity fields whose absence means the run's snapshot is incomplete
# (empty list fields like tools/sub_agents are legitimately empty, so they do
# NOT trigger a live fetch on their own).
_AGENT_CORE_FIELDS = ('key', 'model', 'description', 'url')
# An agent description belongs in the overview, but it must remain a compact
# identity cue rather than a second copy of prompt/configuration content.
_AGENT_DESCRIPTION_PREVIEW_LIMIT = 280
# Process-lifetime cache keyed by agent key. Failed requests are deliberately
# not cached so a temporary outage can recover on the next report visit.
_AGENT_INFO_CACHE: dict[str, dict[str, Any] | None] = {}


async def _orq_agent_info_cached(agent_key: str) -> dict[str, Any] | None:
    """Fetch and cache agent details without blocking the dashboard event loop."""
    if agent_key in _AGENT_INFO_CACHE:
        return _AGENT_INFO_CACHE[agent_key]

    result: dict[str, Any] | None = None
    try:
        from evaluatorq.simulation.utils.run_store import fetch_agent_info

        # fetch_agent_info returns a typed AgentInfoSnapshot; this cache and the
        # merge logic below treat it as a plain dict, so widen the type here.
        result = cast('dict[str, Any] | None', await fetch_agent_info(agent_key))
    except Exception as exc:  # never let a live fetch break a page render
        from loguru import logger

        # fetch_agent_info owns fetch/network/auth errors (logs at warning, returns
        # None), so this only fires if the import itself breaks — a real regression
        # worth surfacing, hence warning rather than a silent debug line.
        logger.warning('live agent_info enrichment unavailable for {}: {}', agent_key, exc)
    if result is not None:
        _AGENT_INFO_CACHE[agent_key] = result
    return result


def _agent_key_for(run: SimulationRun) -> str | None:
    """Best-effort agent key for a live fetch: the captured key, else the run's
    ``target`` (minus an ``agent:`` prefix), else the ``<key>`` segment parsed
    from a ``sim:<key>:...`` run_name. The last case recovers legacy runs that
    ran against an Orq agent but never persisted a ``target`` — a wrong guess
    just 404s and falls back to no card."""
    captured = run.agent_info if isinstance(run.agent_info, dict) else None
    if captured and captured.get('key'):
        return str(captured.get('key'))
    if run.target:
        return run.target.removeprefix('agent:')
    rn = run.run_name or ''
    if rn.startswith('sim:'):
        return rn[len('sim:') :].split(':', 1)[0].strip() or None
    return None


def _stored_agent_info(run: SimulationRun) -> dict[str, Any] | None:
    """Build the minimum card from the values persisted in the run JSON.

    This is deliberately local-only: it gives a report a stable identity when
    Orq cannot enrich an older run (for example, after an agent was deleted).
    """
    agent_key = _agent_key_for(run)
    if not agent_key:
        return None
    info: dict[str, Any] = {'key': agent_key}
    if run.target_model:
        info['model'] = run.target_model
    return info


async def _resolve_agent_info(
    run: SimulationRun,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    """Resolve the agent card's data + provenance as ``(display, original, source)``.

    ``source`` is one of:
    - ``captured``  — the run's own snapshot is complete; no Orq call.
    - ``augmented`` — snapshot present but missing a core field; gaps filled live.
    - ``fetched``   — nothing captured; whole card loaded live from Orq.
    - ``stored``    — Orq could not enrich the run; card uses saved target data.
    - ``none``      — no data and nothing to fetch.

    We only call Orq when the target is an Orq agent AND the captured snapshot is
    absent or missing a core field.  Captured (as-run) values always win
    per-field, so a live fetch only fills gaps — it never overwrites what the run
    actually recorded.
    """
    captured = run.agent_info if isinstance(run.agent_info, dict) and run.agent_info else None
    if run.target_kind != 'orq_agent':
        return captured, captured, ('captured' if captured else 'none')

    missing_core = captured is None or any(not captured.get(f) for f in _AGENT_CORE_FIELDS)
    if not missing_core:
        return captured, captured, 'captured'

    agent_key = _agent_key_for(run)
    if not agent_key:
        return captured, captured, ('captured' if captured else 'none')
    fetched = await _orq_agent_info_cached(agent_key)
    if not fetched:
        return captured or _stored_agent_info(run), captured, ('captured' if captured else 'stored')
    if captured:
        # Fill only missing attributes; captured values win per-field.
        merged = {**fetched, **{k: v for k, v in captured.items() if v}}
        return merged, captured, 'augmented'
    return fetched, None, 'fetched'


def _sim_agent_card_with_source(
    display: dict[str, Any] | None,
    original: dict[str, Any] | None,
    source: str,
    experiment_url: str | None = None,
) -> str:
    """Render an agent card once its captured/live provenance is known."""
    if source in ('captured', 'none'):
        return _sim_agent_card(display, experiment_url=experiment_url)

    if source == 'augmented':
        note = 'Missing fields loaded live from Orq — as-run values kept where captured.'
        original_body = (
            _sim_agent_card(original, bare=True, experiment_url=experiment_url)
            or '<p class="sim-agent-empty">Nothing captured.</p>'
        )
        toggle = (
            '<details class="sim-agent-original"><summary>Show captured snapshot</summary>'
            f'<div class="sim-agent-original-body">{original_body}</div></details>'
        )
    elif source == 'fetched':
        note = 'Loaded live from Orq — no agent config was captured in this run.'
        toggle = ''
    else:  # stored
        note = 'Showing the target recorded in this run; live Orq details are unavailable.'
        toggle = ''
    footer = f'<div class="sim-agent-source">{note}</div>{toggle}'
    return _sim_agent_card(display, footer_html=footer, experiment_url=experiment_url)


def _sim_agent_section(rid: str, run: SimulationRun) -> str:
    """Initial card HTML, with live enrichment deferred until after page load."""
    captured = run.agent_info if isinstance(run.agent_info, dict) and run.agent_info else None
    if run.target_kind != 'orq_agent':
        return _sim_agent_card(captured, experiment_url=run.experiment_url)

    missing_core = captured is None or any(not captured.get(f) for f in _AGENT_CORE_FIELDS)
    agent_key = _agent_key_for(run)
    if not missing_core or not agent_key:
        return _sim_agent_card(captured, experiment_url=run.experiment_url)

    stored = captured or _stored_agent_info(run)
    if not stored:
        return ''
    initial_card = _sim_agent_card(stored, experiment_url=run.experiment_url)
    return (
        f'<div class="sim-agent-async" hx-get="/r/{esc(rid)}/sim/agent-card" '
        f'hx-trigger="load" hx-swap="outerHTML">{initial_card}</div>'
    )


async def sim_agent_card_fragment(run: SimulationRun) -> str:
    """Async fragment for live agent details; the report itself stays local-only."""
    display, original, source = await _resolve_agent_info(run)
    return _sim_agent_card_with_source(display, original, source, experiment_url=run.experiment_url)


def _sim_agent_card(
    agent_info: dict[str, Any] | None,
    *,
    bare: bool = False,
    footer_html: str = '',
    experiment_url: str | None = None,
) -> str:
    """Agent-under-test card: name/role/model/description, sub-agent
    delegates, and tools/knowledge/memory chip groups (Task 2).

    ``bare`` drops the outer ``.rk-panel`` chrome (for nesting inside the
    "show captured snapshot" disclosure); ``footer_html`` is appended inside the
    card (provenance note + disclosure)."""
    if not agent_info:
        return ''

    key = agent_info.get('key') or ''
    model = agent_info.get('model')
    description = _agent_description_preview(agent_info.get('description'))
    # Host + workspace come from the run's experiment_url when available (the web
    # app resolves that for anyone with access); otherwise fall back to the
    # captured snapshot's workspace/host, then env. Repairs older snapshots that
    # captured a UUID-based URL without altering run JSON.
    from evaluatorq.dashboard.orq_links import orq_studio_url
    from evaluatorq.dashboard.view import _TARGET_ICONS

    url = orq_studio_url(
        target_kind='agent',
        entity_id=agent_info.get('id'),
        experiment_url=experiment_url,
        workspace_id=agent_info.get('workspace_key'),
        base_url=agent_info.get('base_url') or None,
    )
    sub_agents = agent_info.get('sub_agents') or []
    tools = agent_info.get('tools') or []
    knowledge_bases = agent_info.get('knowledge_bases') or []
    memory_stores = agent_info.get('memory_stores') or []

    agent_icon = _TARGET_ICONS['agent'].replace('<svg ', '<svg class="sim-agent-icon" aria-hidden="true" ', 1)
    open_html = (
        f'<a class="sim-agent-open" href="{esc(url)}" target="_blank" rel="noopener">Open in ORQ ↗</a>' if url else ''
    )
    model_html = f'<div class="sim-agent-model">{esc(model)}</div>' if model else ''
    desc_html = f'<p class="sim-agent-desc">{esc(description)}</p>' if description else ''

    def _section(label: str, items: list[str]) -> str:
        if not items:
            return ''
        chips = ''.join(f'<span class="sim-agent-chip">{esc(v)}</span>' for v in items)
        return (
            f'<div class="sim-agent-group"><span class="sim-agent-group-label">{esc(label)}</span>'
            f'<div class="sim-agent-chips">{chips}</div></div>'
        )

    # The composition of the agent under test: what it delegates to, calls, and
    # reads from. Only populated groups render — a bare agent shows none.
    sections = (
        _section('Sub-agents', sub_agents)
        + _section('Tools', tools)
        + _section('Knowledge', knowledge_bases)
        + _section('Memory', memory_stores)
    )
    groups_wrap = f'<div class="sim-agent-groups">{sections}</div>' if sections else ''

    inner = (
        '<div class="sim-agent-head">'
        '<div class="sim-agent-identity">'
        f'{agent_icon}<span class="sim-agent-name">{esc(key)}</span>'
        '</div>'
        f'{open_html}'
        '</div>'
        f'{model_html}{desc_html}{groups_wrap}{footer_html}'
    )
    outer_class = 'sim-agent-card' if bare else 'rk-panel sim-agent-card'
    return f'<div class="{outer_class}">{inner}</div>'


def _agent_description_preview(description: Any) -> str | None:
    """Return a compact first-paragraph description safe for the overview.

    Agent ``instructions`` are never part of the card data. Some agent
    descriptions nevertheless contain extensive configuration, so render only
    the first paragraph and cap it at a readable overview length.
    """
    if not isinstance(description, str):
        return None
    first_paragraph = description.split('\n\n', 1)[0].strip()
    if not first_paragraph:
        return None
    compact = ' '.join(first_paragraph.split())
    if len(compact) <= _AGENT_DESCRIPTION_PREVIEW_LIMIT:
        return compact
    return f'{compact[:_AGENT_DESCRIPTION_PREVIEW_LIMIT].rstrip()}...'


def _sim_kpi_band(summary_data: dict[str, Any], n_personas: int = 0, n_scenarios: int = 0) -> str:
    """6-card KPI band (spec §Overview.2). Leads with the persona/scenario
    matrix dimensions; goal-completion status is the summary verdict."""
    from evaluatorq.common.reports.html_helpers import kpi_cards

    errors = summary_data.get('errors', 0)
    return kpi_cards([
        {'label': 'Personas', 'value': str(n_personas), 'status': 'neutral'},
        {'label': 'Scenarios', 'value': str(n_scenarios), 'status': 'neutral'},
        {'label': 'Conversations', 'value': str(summary_data.get('total_conversations', 0)), 'status': 'neutral'},
        {
            'label': 'Avg score',
            'value': f'{summary_data.get("avg_goal_completion_score", 0.0):.2f}',
            'status': 'neutral',
        },
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
    ('Not achieved', 'var(--outcome-vulnerable)'),
    ('Errors', 'var(--outcome-error)'),
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


def _sim_hero(run: SimulationRun, compare_html: str = '') -> str:
    # KPI cards intentionally omitted: the same metrics render in the 5-card
    # band inside the Overview tab (_sim_kpi_band), so a hero band duplicated
    # them above the tabs. Hero is now just title + subtitle.
    run_btn = trace_link_button(run_trace_url(run.run_id, run.experiment_url), 'View all run traces ↗')
    inner = run_btn + compare_html
    actions = f'<div class="report-hero-actions">{inner}</div>' if inner else ''
    return (
        '<header class="report-hero">'
        '<p class="report-hero-kicker">Agent Simulation</p>'
        f'<h2 class="report-hero-title">{esc(run.run_name)}</h2>'
        f'<p class="report-hero-sub">target {esc(run.target_kind)}</p>'
        f'{actions}</header>'
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
    model to every card). ``asr = vulns / evaluated`` guards ``evaluated == 0``
    — ``r.vulnerable is None`` means the attack couldn't be judged and must
    count in neither ``vulns`` nor the resistant remainder, so an all-
    unevaluated agent reports ``asr``/``resistance`` as ``None`` rather than
    a false 0% ASR / 100% resistance.
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
                'evaluated': 0,
                'vulns': 0,
                'critical': 0,
                'errors': 0,
            },
        )
        entry['attacks'] += 1
        if r.vulnerable is not None:
            entry['evaluated'] += 1
            if r.vulnerable:
                entry['vulns'] += 1
                if r.attack.severity == 'critical':
                    entry['critical'] += 1
        if r.error is not None:
            entry['errors'] += 1
    for entry in stats.values():
        evaluated = entry['evaluated']
        vulns = entry['vulns']
        entry['asr'] = vulns / evaluated if evaluated else None
        entry['resistance'] = (1.0 - entry['asr']) if entry['asr'] is not None else None
    return stats


def _rt_exec_summary(summary_data: dict[str, Any], by_kind: dict[str, Any]) -> str:
    """Templated exec-summary sentence + fallbacks (spec §Overview.1).
    Empty-run guard: ``total_attacks`` falsy -> ``''``, nothing below runs.
    Zero-evaluated guard: nothing could be judged -> a distinct no-verdict
    sentence, never "the agent resisted all of them" (same rule as
    ``_rt_kpi_band``'s resistance card — an untested run must not read as a
    clean pass)."""
    from evaluatorq.common.reports.html_helpers import pct
    from evaluatorq.dashboard.metrics import zero_evaluated_attacks
    from evaluatorq.dashboard.report_kit import callout

    total = summary_data.get('total_attacks', 0)
    if not total:
        return ''

    if zero_evaluated_attacks(summary_data):
        sentence = (
            f'Across <strong>{total}</strong> adversarial attacks, none could be evaluated '
            '(target or judge errors) — no verdict is available for this run.'
        )
        return callout(sentence, confidence=summary_data.get('confidence'))

    category_section = by_kind.get('category_breakdown')
    rows = category_section.data.get('rows', []) if category_section is not None else []
    k = len(rows)

    vulns = summary_data.get('vulnerabilities_found', 0)
    critical = summary_data.get('critical_exposure', 0)
    resistance_rate = summary_data.get('resistance_rate')

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

    top_rate = rows[0].get('vulnerability_rate') if rows else None
    if top_rate is not None and top_rate > 0:
        top = rows[0]
        sentence += (
            f' <strong>{esc(top.get("category_name", ""))}</strong> is the weakest area '
            f'({pct(top_rate)} attack success rate).'
        )

    turn_section = by_kind.get('turn_depth_analysis')
    if turn_section is not None:
        turn_rows = turn_section.data.get('rows', [])
        if len(turn_rows) >= 2:
            first, last = turn_rows[0], turn_rows[-1]
            first_rate = first.get('vulnerability_rate')
            last_rate = last.get('vulnerability_rate')
            if first_rate is not None and last_rate is not None and last_rate > first_rate:
                sentence += (
                    f' Attack success climbs with conversation depth — from '
                    f'{pct(first_rate)} at {first.get("turn_count")} turns to '
                    f'{pct(last_rate)} at {last.get("turn_count")} turns.'
                )

    total_errors = summary_data.get('total_errors', 0)
    if total_errors:
        sentence += f' {total_errors} attack{"s" if total_errors != 1 else ""} errored and were not evaluated.'

    return callout(sentence, confidence=summary_data.get('confidence'))


def _rt_kpi_band(s: dict[str, Any]) -> str:
    """5-card KPI band: Attacks run / Vulnerabilities / Attack success rate /
    Resistance rate / Critical findings (spec §Overview.2)."""
    from evaluatorq.common.reports.html_helpers import kpi_cards, pct
    from evaluatorq.dashboard.metrics import zero_evaluated_attacks

    asr = s.get('vulnerability_rate', 0.0)
    resistance = s.get('resistance_rate', 0.0)
    vulns = s.get('vulnerabilities_found', 0)
    critical = s.get('critical_exposure', 0)
    if zero_evaluated_attacks(s) or resistance is None or asr is None:
        # No verdict to show: either nothing was evaluated (the rate is only its
        # schema default — same no-score rule as the landing rows and the red-team
        # overview) or the rate is explicitly null. Both rate cards share the guard.
        resistance_card = {'label': 'Resistance rate', 'value': 'n/a', 'status': 'neutral'}
        asr_card = {'label': 'Attack success rate', 'value': 'n/a', 'status': 'neutral'}
    else:
        resistance_card = {
            'label': 'Resistance rate',
            'value': pct(resistance),
            'status': 'pass' if resistance >= 0.8 else 'warn',
        }
        asr_card = {
            'label': 'Attack success rate',
            'value': pct(asr),
            'status': 'fail' if asr >= 0.25 else ('warn' if asr > 0 else 'pass'),
        }
    return kpi_cards([
        {'label': 'Attacks run', 'value': str(s.get('total_attacks', 0)), 'status': 'neutral'},
        {'label': 'Vulnerabilities', 'value': str(vulns), 'status': 'fail' if vulns else 'pass'},
        asr_card,
        resistance_card,
        {'label': 'Critical findings', 'value': str(critical), 'status': 'fail' if critical else 'pass'},
    ])


def _rt_agent_row(stats: dict[str, Any]) -> str:
    """One agents-under-test table row: dot + name/model, hit count, ASR track
    bar, ASR value (spec §Overview.4)."""
    from evaluatorq.common.reports.html_helpers import pct

    critical = stats.get('critical', 0)
    vulns = stats.get('vulns', 0)
    dot_cls = 'rt-hero-dot--critical' if critical else ('rt-hero-dot--vuln' if vulns else 'rt-hero-dot--clean')
    asr = stats.get('asr')
    # None (no evaluated attacks for this agent): empty track, neutral color, pct() renders 'n/a'.
    bar_pct = max(0.0, min(1.0, asr)) * 100 if asr is not None else 0.0
    bar_color = 'var(--red-600)' if critical else 'var(--orange-500)'
    asr_color = 'var(--gray-500)' if asr is None else ('var(--orange-600)' if vulns else 'var(--green-600)')
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
    # None (no evaluated attacks) sorts last, not as a false 0.0 tied with resistant agents.
    ranked = sorted(
        stats.values(),
        key=lambda st: (st.get('asr') is None, -(st.get('asr') or 0.0)),
    )
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


def _rt_agent_card(
    agent_ctx: dict[str, Any] | None, key: str, stats: dict[str, Any], experiment_url: str | None = None
) -> str:
    """One agent card: ASR dial column + main column (name/critical chip,
    model, description, stat strip, TOOLS/KNOWLEDGE chip rows). Agents
    present in results but missing ``agent_context`` still render via
    ``stats``-only fallback (spec §Agents)."""
    from evaluatorq.common.reports.html_helpers import pct
    from evaluatorq.dashboard.orq_links import orq_studio_url
    from evaluatorq.dashboard.report_kit import dial
    from evaluatorq.fetch_data import _resolve_orq_base_url

    ctx = agent_ctx or {}
    display_name = ctx.get('display_name') or stats.get('display_name') or key
    model = ctx.get('model') or stats.get('model') or ''
    version = ctx.get('version') or ''
    description = ctx.get('description') or ''
    tools = ctx.get('tools') or []
    skills = ctx.get('skills') or []
    knowledge_bases = ctx.get('knowledge_bases') or []

    # Optional Studio deep-link — only for orq agent/deployment targets that
    # carry id + workspace_id; hidden otherwise (returns None).
    studio_url = orq_studio_url(
        target_kind=ctx.get('target_kind'),
        entity_id=ctx.get('id'),
        experiment_url=experiment_url,
        workspace_id=ctx.get('workspace_id'),
        base_url=_resolve_orq_base_url(None),
    )
    studio_html = (
        f'<a class="rt-agent-card-studio" href="{esc(studio_url)}" target="_blank" rel="noopener">Open in Studio</a>'
        if studio_url
        else ''
    )

    attacks = stats.get('attacks', 0)
    vulns = stats.get('vulns', 0)
    critical = stats.get('critical', 0)
    asr = stats.get('asr', 0.0)
    resistance = stats.get('resistance', 1.0 - asr)

    dial_color = 'var(--red-600)' if critical else ('var(--orange-500)' if vulns else 'var(--green-600)')
    dial_html = dial(pct(asr), asr, radius=24, stroke=6, color=dial_color, sub='ASR')

    critical_chip = f'<span class="rt-agent-card-critical">{critical} critical</span>' if critical else ''
    # Version rides the model line: it answers "what exactly did we attack".
    model_line = ' · '.join(p for p in (esc(model), f'v{esc(version)}' if version else '') if p)
    model_html = f'<div class="rt-agent-card-model">{model_line}</div>' if model_line else ''
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

    chips_html = (
        _rt_agent_card_chip_row('TOOLS', tools)
        + _rt_agent_card_chip_row('SKILLS', skills)
        + _rt_agent_card_chip_row('KNOWLEDGE', knowledge_bases)
    )

    return (
        '<div class="rk-panel rt-agent-card">'
        f'<div class="rt-agent-card-dial">{dial_html}</div>'
        '<div class="rt-agent-card-main">'
        f'<div class="rt-agent-card-name-row">'
        f'<span class="rt-agent-card-name">{esc(display_name)}</span>{critical_chip}{studio_html}</div>'
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
    cards = ''.join(
        _rt_agent_card(ctx_by_key.get(key), key, stats.get(key, {}), report.experiment_url) for key in ordered_keys
    )

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
    # Raw text either way — esc() is applied once at the interpolation site below.
    title_name = fmt_vulnerability(atk.vulnerability) if atk.vulnerability else atk.category
    # Three-way, not two-way: r.vulnerable is None means the attack couldn't be judged
    # (target/judge failure) and must render as its own neutral state, never 'resistant'
    # (outcome_pill() falls back to a neutral tone for any status it doesn't recognize).
    if r.error:
        outcome = 'error'
    elif r.vulnerable is None:
        outcome = 'Not evaluated'
    elif r.vulnerable:
        outcome = 'vulnerable'
    else:
        outcome = 'resistant'
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
    generated = (
        created_at.date().isoformat()
        if created_at is not None and hasattr(created_at, 'date')
        else (str(created_at) or None)
    )

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
    first_rate = first.get('vulnerability_rate')
    last_rate = last.get('vulnerability_rate')
    # Either endpoint unevaluated (no verdict for that depth) -> nothing to compare, skip the lead-in.
    if first_rate is None or last_rate is None or last_rate <= first_rate:
        return ''
    return (
        '<p class="rt-breakdowns-leadin">Attack success climbs from '
        f'<strong>{pct(first_rate)}</strong> at {first.get("turn_count")} turns to '
        f'<strong>{pct(last_rate)}</strong> at {last.get("turn_count")} turns — '
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
    # bar_rows() takes float, not float | None — unevaluated depths (no verdict
    # at that turn count) are excluded from the bars rather than drawn as a
    # misleading 0% bar; the footnote below still lists every depth's raw counts.
    evaluated_rows = [r for r in rows if r.get('vulnerability_rate') is not None]
    bars = ''
    if evaluated_rows:
        max_rate = max((r['vulnerability_rate'] for r in evaluated_rows), default=0.0) or 1.0
        bars = bar_rows(
            [(f'{r.get("turn_count")} turns', r['vulnerability_rate']) for r in evaluated_rows],
            width=520,
            label_w=70,
            color='var(--orange-500)',
            fmt=pct,
            max_value=max_rate,
        )
    leadin = _rt_breakdowns_depth_leadin(evaluated_rows)
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
        # bar_rows() takes float, not float | None — unevaluated slices are omitted.
        rows = [
            (name.replace('_', ' '), stats['vulnerability_rate'])
            for name, stats in entries.items()
            if stats.get('vulnerability_rate') is not None
        ]
        if not rows:
            return ''
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
        # heatmap() does `rate <= 0.55` on the cell value — an unevaluated (vulnerability_rate
        # is None) cell must not reach it; dropping it falls through to the existing
        # "missing cell" sunken "—" rendering, which already reads as "no data".
        cells = [c for c in data.get('cells', []) if c.get('vulnerability_rate') is not None]
        table_html = heatmap(
            data.get('vulnerabilities', []),
            data.get('techniques', []),
            cells,
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
        # bar_rows() takes float, not float | None — unevaluated rows are omitted.
        bar_data = [
            (str(r.get(row_key, '')), r['vulnerability_rate']) for r in rows if r.get('vulnerability_rate') is not None
        ]
        if not bar_data:
            return ''
        bars = bar_rows(
            bar_data,
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
    + mono faint ``{model} · {n} vuln`` / ``clean`` (spec §Run header). The
    model rides in the sub so "what was tested" is readable without opening
    the Config tab — the run name alone never said which model answered. It
    is dropped from the sub when the results carry no model (custom
    ``AgentTarget`` backends don't report one)."""
    vulns = stats.get('vulns', 0)
    critical = stats.get('critical', 0)
    dot_cls = 'rt-hero-dot--critical' if critical else ('rt-hero-dot--vuln' if vulns else 'rt-hero-dot--clean')
    sub = f'{vulns} vuln' if vulns else 'clean'
    model = stats.get('model') or ''
    if model:
        sub = f'{model} · {sub}'
    return (
        '<span class="rt-hero-pill">'
        f'<span class="rt-hero-dot {dot_cls}"></span>'
        f'<span class="rt-hero-pill-name">{esc(stats.get("display_name", ""))}</span>'
        f'<span class="rt-hero-pill-sub">{esc(sub)}</span>'
        '</span>'
    )


_RT_TRAILING_PAREN = re.compile(r'\s*\(([^()]*)\)\s*$')


def _rt_run_name(report: RedTeamReport) -> str:
    """The run name on its own, with the runner's trailing ``(target)`` /
    ``(dynamic)`` / ``(2 targets)`` parentheticals removed.

    ``runner`` builds sub-report descriptions as
    ``f'{description} ({target}) (dynamic)'``, so the hero title used to carry
    the run name, the agent and the pipeline welded into one string. Each of
    those now has its own slot (title / agent pill / kicker), so the suffixes
    are stripped here rather than shown three times. Only suffixes we
    recognise are dropped — a user-written ``"Q3 sweep (post-patch)"`` keeps
    its parenthetical.

    The suffix carries ``PreparedTarget.target``, which is the *full* target
    string (``"agent:my-key"``), while ``tested_agents`` holds bare keys
    (``"my-key"``) — so both sides are compared with any ``kind:`` prefix
    dropped, or ``(agent:my-key)`` would survive into the title.
    """

    def _bare(s: str) -> str:
        return s.rsplit(':', 1)[-1].strip().lower()

    name = (report.description or '').strip()
    known = {'static', 'dynamic', 'hybrid'} | {_bare(a) for a in report.tested_agents}
    while (m := _RT_TRAILING_PAREN.search(name)) is not None:
        inner = _bare(m.group(1))
        if inner in known or re.fullmatch(r'\d+ targets?', inner):
            name = name[: m.start()].rstrip()
        else:
            break
    return name or 'Red teaming report'


def _redteam_hero(summary_section: Any, report: RedTeamReport) -> str:
    """Kicker (`Red Team · {pipeline}`) + run-name title + `N agents` pill
    (multi only) + per-agent pill row carrying agent name and, when the
    results report one, model. The
    5-card KPI band moves to the Overview tab (spec §Run header) — no double
    KPI band."""
    multi_agent = len(report.tested_agents) > 1
    agents_pill = f'<span class="rt-hero-agents-pill">{len(report.tested_agents)} agents</span>' if multi_agent else ''
    # Rendered for single-agent runs too: the pill is where the agent name and
    # model live now that the title is the run name alone.
    agent_stats = _rt_agent_stats(report)
    pills = ''.join(_rt_agent_pill(stats) for stats in agent_stats.values())
    agent_pills_html = f'<div class="rt-hero-agent-row">{pills}</div>' if pills else ''
    pipeline = str(getattr(report.pipeline, 'value', report.pipeline) or '').strip()
    kicker = f'Red Team · {pipeline.capitalize()}' if pipeline else 'Red Team'
    run_btn = trace_link_button(run_trace_url(report.run_id, report.experiment_url), 'View all run traces ↗')
    actions = f'<div class="report-hero-actions">{run_btn}</div>' if run_btn else ''
    return (
        '<header class="report-hero rt-hero">'
        f'<p class="report-hero-kicker">{esc(kicker)}</p>'
        f'<h2 class="report-hero-title rt-hero-title">{esc(_rt_run_name(report))}{agents_pill}</h2>'
        f'{agent_pills_html}'
        f'{actions}'
        '</header>'
    )
