"""Tabbed report bodies — both surfaces render Streamlit-aligned tabs, and
empty tabs (no data) drop out (RES-974)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id
from evaluatorq.dashboard.report_tabs import _tabs

from tests.dashboard.test_downloads import _make_rt_report, _make_sim_run


def _tab_labels(html: str) -> list[str]:
    import html as _html
    import re

    return [_html.unescape(m) for m in re.findall(r'class="tab-label" for="[^"]*">([^<]+)<', html)]


def _headers(html: str) -> list[str]:
    import re

    header_row = html.split('</tr>', 1)[0]
    return re.findall(r'<th>([^<]+)</th>', header_row)


def _template(html: str, prefix: str) -> str:
    import re

    match = re.search(rf'<template id="{prefix}[^"]+"[^>]*>(.*?)</template>', html)
    assert match is not None
    return match.group(1)


@pytest.fixture()
def sim_run():
    return _make_sim_run(personas=['alice', 'bob'], goal_achieved_flags=[True, False])


@pytest.fixture()
def sim_run_with_turn_metrics():
    from evaluatorq.contracts import TokenUsage
    from evaluatorq.simulation.types import TurnMetrics

    def _tm(turn_number: int, response_quality: float, hallucination_risk: float) -> TurnMetrics:
        return TurnMetrics(
            turn_number=turn_number,
            token_usage=TokenUsage(input_tokens=5, output_tokens=5, total_tokens=10),
            response_quality=response_quality,
            hallucination_risk=hallucination_risk,
            judge_reason='ok',
        )

    return _make_sim_run(
        personas=['alice', 'bob'],
        goal_achieved_flags=[True, False],
        turn_metrics_by_result=[
            [_tm(1, 0.6, 0.2), _tm(2, 0.8, 0.1)],
            [_tm(1, 0.5, 0.2), _tm(2, 0.7, 0.1)],
        ],
    )


@pytest.fixture()
def roots(tmp_path: Path) -> list[Path]:
    rt = tmp_path / 'runs'
    sim = tmp_path / 'sim-runs'
    rt.mkdir()
    sim.mkdir()
    (rt / 'rt.json').write_text(_make_rt_report().model_dump_json())
    (sim / 'sim.json').write_text(
        _make_sim_run(personas=['alice', 'bob'], goal_achieved_flags=[True, False]).model_dump_json()
    )
    return [rt, sim]


@pytest.fixture()
def client(roots: list[Path]) -> TestClient:
    return TestClient(build_app(roots=roots), raise_server_exceptions=True)


def test_sim_report_renders_tabs(client: TestClient, roots: list[Path]) -> None:
    rid = report_id(roots[1] / 'sim.json')
    labels = _tab_labels(client.get(f'/r/{rid}').text)
    # Folded to 4: Overview / Breakdown / Transcripts (+ Config when tokens exist).
    # Transcripts carries a raw-HTML count pill, so _tab_labels (which stops at
    # the first "<") only captures the text up to the pill's opening tag.
    assert labels[0] == 'Overview'
    assert labels[1] == 'Breakdown'
    assert labels[2].startswith('Transcripts')
    # Evaluators / Judge & errors / Turn quality / Tokens folded into the above.
    assert 'Judge & errors' not in labels
    assert 'Turn quality' not in labels
    assert 'Evaluators' not in labels


def test_redteam_report_renders_tabs(client: TestClient, roots: list[Path]) -> None:
    rid = report_id(roots[0] / 'rt.json')
    labels = _tab_labels(client.get(f'/r/{rid}').text)
    # 7-tab set: Overview / Agents / Focus areas / Breakdowns / Attacks / Usage / Config.
    assert 'Overview' in labels
    assert 'Breakdowns' in labels
    assert 'Config' in labels
    # Old (pre-alignment) tab names are gone.
    assert 'Summary' not in labels
    assert 'Methodology' not in labels
    assert 'Evidence' not in labels


def test_single_agent_report_has_no_comparison_tab(client: TestClient, roots: list[Path]) -> None:
    """The Comparison tab is multi-agent only — a single-agent report drops it."""
    rid = report_id(roots[0] / 'rt.json')
    labels = _tab_labels(client.get(f'/r/{rid}').text)
    assert 'Comparison' not in labels


def test_clean_run_drops_error_tab(client: TestClient, roots: list[Path]) -> None:
    """No runtime errors → no Error Analysis tab (empty tabs drop out)."""
    rid = report_id(roots[0] / 'rt.json')
    labels = _tab_labels(client.get(f'/r/{rid}').text)
    assert 'Error Analysis' not in labels


def test_tab_panels_match_tab_count(client: TestClient, roots: list[Path]) -> None:
    """Every tab label has exactly one matching panel (no orphans)."""
    rid = report_id(roots[1] / 'sim.json')
    html = client.get(f'/r/{rid}').text
    assert html.count('class="tab-label"') == html.count('class="tab-panel"')


def test_filter_post_preserves_tabs(client: TestClient, roots: list[Path]) -> None:
    """The filter round-trip re-renders the tabbed body, not the flat export."""
    rid = report_id(roots[1] / 'sim.json')
    r = client.post(f'/r/{rid}/filter', data={'persona': 'alice', 'goal_outcome': 'All'})
    assert r.status_code == 200
    assert 'tab-label' in r.text
    assert 'filter-swap' in r.text


def test_tabs_two_tuple_escapes_label() -> None:
    html = _tabs('g', [('<x>', '<p>body</p>')])
    assert '&lt;x&gt;' in html
    assert '<x>' not in html


def test_tabs_three_tuple_renders_raw_label_html() -> None:
    html = _tabs('g', [('Transcripts', '<p>body</p>', 'Transcripts <span class="pill">5</span>')])
    assert '<span class="pill">5</span>' in html


def test_sim_overview_has_exec_summary_and_five_kpis(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    assert 'class="report-aligned sim-report"' in html
    assert 'Executive summary' in html
    # 5-card KPI band incl. Avg turns
    assert 'Avg turns' in html
    assert 'Goal completion' in html
    assert 'goal met' in html
    # Persona/scenario configuration now lives in the Config tab.
    assert 'sim-overview-grid-2--top' not in html


def test_sim_overview_single_exec_summary_card_with_narrative(sim_run) -> None:
    # Regression: a saved narrative + the computed sentence used to render two
    # "Executive summary" cards. With a narrative present, show only the
    # narrative in the callout — not the computed "average score of ..." line.
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    sim_run.executive_summary = 'NARRATIVE_MARKER: the agent held the line.'
    html = sim_report_tabs('rid', sim_run)
    assert 'NARRATIVE_MARKER' in html
    # Exactly one exec-summary callout shell in the overview.
    assert html.count('<div class="exec-summary">') == 1
    # The computed stat sentence must not render alongside the narrative.
    assert 'goal-completion rate at an average score' not in html


def test_sim_overview_falls_back_to_computed_summary(sim_run) -> None:
    # With no saved narrative, the computed stat sentence is the summary.
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    sim_run.executive_summary = None
    html = sim_report_tabs('rid', sim_run)
    assert 'goal-completion rate at an average score' in html
    assert html.count('<div class="exec-summary">') == 1


def test_sim_kpi_goal_status_uses_verdict(sim_run) -> None:
    # Goal-completion KPI status must equal summary verdict (pass/warn/fail), not an ad-hoc threshold.
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    # verdict-driven class present on the goal-completion card
    assert 'kpi-card--' in html


def test_sim_transcripts_tab_has_count_pill(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    assert 'class="tab-count"' in html  # count pill span class


def test_sim_breakdown_has_heatmap_and_histogram(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    assert 'Goal completion —' in html
    assert 'dashed line = mean' in html
    assert 'class="rk-heatmap"' in html
    assert 'class="rk-histogram"' in html


def test_sim_breakdown_no_vlconvert_heatmap_or_histogram(sim_run) -> None:
    """The Breakdown tab must use report_kit's SVG/HTML charts, never vl-convert."""
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    assert 'vega' not in html.lower()


def test_sim_breakdown_scenario_table_has_tokens_column(sim_run) -> None:
    """Per-scenario breakdown table (spec §Breakdown.3) carries a Tokens column
    alongside Conv/Goal rate/Avg score/Avg turns, formatted with thousands
    separators from each result's total token usage."""
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    scenario_html = html.split('Per-scenario')[-1].split('Transcripts')[0]
    assert '<th>Tokens</th>' in scenario_html
    # sim_run fixture: two results, each with total_tokens=15 -> scenario sum 30.
    assert 'data-label="Tokens">30<' in scenario_html


def test_sim_transcripts_tab_drops_failure_mode(sim_run) -> None:
    """failure_mode moved to Breakdown (spec §Breakdown.4); Transcripts keeps
    only evaluator_scores / judge_verdicts / errors below the cards."""
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    assert 'id="section-failure_mode"' not in html


def test_sim_failure_modes_empty_rows_returns_empty_string() -> None:
    from evaluatorq.dashboard.report_tabs import _sim_failure_modes

    assert _sim_failure_modes([]) == ''


def test_sim_failure_modes_default_threshold_hides_singletons() -> None:
    from evaluatorq.dashboard.report_tabs import _sim_failure_modes

    rows = [
        ('Scenario A: criterion x', 3),
        ('Scenario A: criterion y', 2),
        ('Scenario B: criterion z', 1),
    ]
    html = _sim_failure_modes(rows)

    assert 'data-fm-panel' in html
    assert 'data-fm-empty' in html
    assert 'data-fm-slider' in html
    assert 'max="3"' in html
    assert 'value="2"' in html

    # The count-1 row is hidden by the default threshold; count-2/3 rows are not.
    row_3 = html.split('data-count="3"')[1].split('</div>')[0]
    row_2 = html.split('data-count="2"')[1].split('</div>')[0]
    row_1 = html.split('data-count="1"')[1].split('</div>')[0]
    assert 'hidden' not in row_3.split('>')[0]
    assert 'hidden' not in row_2.split('>')[0]
    assert 'hidden' in row_1.split('>')[0]


def test_sim_failure_modes_all_singletons_no_hidden_rows() -> None:
    from evaluatorq.dashboard.report_tabs import _sim_failure_modes

    rows = [('a: b', 1), ('c: d', 1)]
    html = _sim_failure_modes(rows)

    assert 'value="1"' in html
    assert 'max="1"' in html
    bars_html = html.split('sim-fm-bars')[1].split('sim-fm-empty')[0]
    assert 'hidden' not in bars_html


def test_dashboard_failures_use_four_columns_and_drawer_rows(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    failures = html.split('id="section-failures_first"', 1)[1].split('</section>', 1)[0]
    assert ['Scenario', 'Persona', 'Why', 'Score'] == _headers(failures)
    assert 'Criteria' not in failures
    assert 'href="#conv-' not in failures
    assert 'data-entity-kind="conversation"' in failures
    assert 'data-drawer-url="/r/rid/sim/transcript?idx=1"' in failures
    assert 'data-no-drawer' in failures


def test_cohort_template_contains_stats_and_compact_conversation_triggers(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    template = _template(html, 'persona-')
    assert 'Goal rate' in template and 'Avg score' in template and 'Tokens' in template
    assert 'sim-cohort-conversations' in template
    assert 'data-entity-kind="conversation"' in template


def test_duplicate_persona_names_keep_conversation_cohorts_separate() -> None:
    from evaluatorq.contracts import TokenUsage
    from evaluatorq.dashboard.report_tabs import sim_report_tabs
    from evaluatorq.simulation.types import SimulationResult, SimulationRun, TerminatedBy

    def result(*, patience: float, score: float) -> SimulationResult:
        return SimulationResult(
            messages=[],
            terminated_by=TerminatedBy.judge,
            reason='done',
            goal_achieved=score == 1.0,
            goal_completion_score=score,
            rules_broken=[],
            turn_count=1,
            token_usage=TokenUsage(total_tokens=10),
            turn_metrics=[],
            metadata={
                'persona': 'Customer',
                'scenario': 'Billing',
                'persona_traits': {'patience': patience},
            },
            criteria_results={},
        )

    run = SimulationRun(
        run_name='duplicate-cohort-run',
        created_at=datetime.now(tz=timezone.utc),
        mode='run',
        target_kind='orq_agent',
        evaluator_names=[],
        total_results=2,
        scorer_averages={},
        results=[result(patience=0.1, score=1.0), result(patience=0.9, score=0.0)],
    )

    html = sim_report_tabs('rid', run)
    import re

    templates = re.findall(r'<template id="persona-[^"]+"[^>]*>(.*?)</template>', html)
    assert len(templates) == 2
    conversation_urls = [re.findall(r'data-drawer-url="([^"]+)"', template) for template in templates]
    assert conversation_urls == [
        ['/r/rid/sim/transcript?idx=0'],
        ['/r/rid/sim/transcript?idx=1'],
    ]


def test_sim_turn_quality_tab_present_when_data(sim_run_with_turn_metrics) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run_with_turn_metrics)
    assert 'Turn quality' in html
    assert 'by turn index' in html.lower()


def test_sim_turn_quality_tab_absent_without_turn_metrics(sim_run) -> None:
    """No turn_metrics on any result -> the tab drops out (empty panel)."""
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    assert 'Turn quality' not in html


def test_sim_turn_quality_delta_callout_and_chart(sim_run_with_turn_metrics) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run_with_turn_metrics)
    # Delta callout: response quality improved, hallucination risk fell.
    assert 'response quality' in html
    assert 'hallucination risk' in html
    assert 'class="rk-line-chart"' in html
    assert 'class="rk-legend"' in html
    # No confidence pill on the turn-quality callout.
    assert 'CONFIDENCE' not in html.split('Turn quality trend')[-1].split('</div>')[0]


def test_sim_turn_quality_stat_tiles_and_bar_rows(sim_run_with_turn_metrics) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run_with_turn_metrics)
    assert 'class="sim-aq-grid"' in html
    assert 'class="sim-aq-cell"' in html
    assert 'class="rk-bar-rows"' in html
    assert '2 turns' in html


def test_sim_turn_quality_single_turn_drops_tab() -> None:
    """When every conversation is single-turn there's no per-turn story (no
    trend, a one-bar distribution), so the whole Turn quality tab drops out."""
    from evaluatorq.contracts import TokenUsage
    from evaluatorq.dashboard.report_tabs import sim_report_tabs
    from evaluatorq.simulation.types import TurnMetrics

    def _tm() -> TurnMetrics:
        return TurnMetrics(
            turn_number=1,
            token_usage=TokenUsage(input_tokens=5, output_tokens=5, total_tokens=10),
            response_quality=0.6,
            hallucination_risk=0.2,
            judge_reason='ok',
        )

    run = _make_sim_run(
        personas=['alice', 'bob'],
        goal_achieved_flags=[True, False],
        turn_metrics_by_result=[[_tm()], [_tm()]],
    )
    html = sim_report_tabs('rid', run)
    assert 'Turn quality' not in _tab_labels(html)


def test_sim_config_tab_has_metagrid(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    assert 'Job-level metadata' in html
    assert sim_run.run_name in html


def test_sim_config_tab_has_personas_and_scenarios(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    assert 'Simulated user profiles' in html
    assert 'Goals + pass/fail criteria' in html
    assert 'alice' in html
    assert 'billing' in html


def test_sim_overview_omits_persona_and_scenario_config_panels(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    overview_html = html.split('Breakdown')[0]
    assert 'Simulated user profiles' not in overview_html
    assert 'Goals + pass/fail criteria' not in overview_html
    assert 'sim-config-persona-row' not in overview_html
    assert 'sim-config-scenario-row' not in overview_html


def test_sim_config_compacts_entities_and_prerenders_modal_details() -> None:
    from evaluatorq.contracts import TokenUsage
    from evaluatorq.dashboard.report_tabs import sim_report_tabs
    from evaluatorq.simulation.types import SimulationResult, SimulationRun, TerminatedBy

    traits = {
        'patience': 0.2,
        'assertiveness': 0.8,
        'politeness': 0.6,
        'technical_level': 0.4,
        'communication_style': 'terse',
        'background': 'Long persona background should only appear in modal detail.',
    }
    result = SimulationResult(
        messages=[],
        terminated_by=TerminatedBy.judge,
        reason='done',
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=2,
        token_usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        turn_metrics=[],
        metadata={
            'persona': 'Impatient buyer',
            'scenario': 'Refund request',
            'persona_traits': traits,
            'scenario_goal': 'Resolve the refund without unnecessary back and forth.',
            'scenario_context': 'Customer was charged twice.',
        },
        criteria_results={'Confirm refund eligibility': True, 'Do not invent a policy': False},
    )
    run = SimulationRun(
        run_name='entity-detail-run',
        created_at=datetime.now(tz=timezone.utc),
        mode='run',
        target_kind='orq_agent',
        evaluator_names=['goal_achieved'],
        total_results=1,
        scorer_averages={'goal_achieved': 1.0},
        results=[result],
    )

    html = sim_report_tabs('rid', run)
    assert '<dialog class="sim-entity-dialog"' in html
    assert 'data-sim-entity-trigger' in html
    assert 'data-entity-kind="persona"' in html
    assert 'data-entity-kind="scenario"' in html
    assert 'class="sim-trait-mini"' in html
    assert 'data-tip="Patience 0.20"' in html
    assert '2 checks' in html
    assert 'Long persona background should only appear in modal detail.' in html
    assert 'Customer was charged twice.' in html
    assert 'Confirm refund eligibility' in html

    from evaluatorq.dashboard.report_tabs import _sim_config_persona_row, _sim_config_scenario_row

    persona_row = _sim_config_persona_row({'name': 'Impatient buyer', 'traits': traits}, 'persona-0')
    scenario_row = _sim_config_scenario_row(
        {
            'name': 'Refund request',
            'goal': 'Resolve the refund without unnecessary back and forth.',
            'criteria': [
                {'description': 'Confirm refund eligibility', 'type': 'must_happen'},
                {'description': 'Do not invent a policy', 'type': 'must_not_happen'},
            ],
            'pass_rate': 0.5,
        },
        'scenario-0',
    )
    assert 'Long persona background should only appear in modal detail.' not in persona_row
    # Goal one-liner, criteria split, and a colored pass-rate badge now render in the row.
    assert 'Resolve the refund without unnecessary back and forth.' in scenario_row
    assert 'must-happen' in scenario_row and 'must-not' in scenario_row
    assert '50%' in scenario_row


def test_sim_drawer_has_back_nav_close_controls(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    assert 'data-sim-entity-back' in html
    assert 'data-sim-entity-prev' in html
    assert 'data-sim-entity-next' in html
    assert 'data-sim-entity-close' in html
    assert 'aria-label="Back to cohort"' in html
    assert 'data-entity-kind="conversation"' in html
    assert 'data-drawer-url="/r/rid/sim/transcript?idx=0"' in html


def test_sim_drawer_runtime_dispatches_conversations_without_anchor_handler() -> None:
    source = (Path(__file__).parents[2] / 'src/evaluatorq/dashboard/static/dashboard.js').read_text()
    keyboard_handler = source.split("document.body.addEventListener('keydown'", 1)[1].split('});', 1)[0]

    assert 'function openConversation(trigger, pushCurrent)' in source
    assert "trigger.getAttribute('data-drawer-url')" in source
    assert "a[href^=\"#conv-\"]" not in source
    assert "evt.target.closest('[data-no-drawer]')" in keyboard_handler


def test_sim_breakdown_entity_names_open_shared_modal(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    breakdown_html = html.split('Per-persona')[-1].split('Transcripts')[0]
    assert 'data-sim-entity-trigger' in breakdown_html
    assert 'data-entity-kind="persona"' in breakdown_html
    assert 'data-entity-kind="scenario"' in breakdown_html


def test_sim_config_tab_keeps_token_usage(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    config_html = html.split('Job-level metadata')[-1]
    assert 'Tokens' in config_html or 'token' in config_html.lower()


def test_sim_outcomes_donut_still_wraps_in_chart_card(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import _sim_outcomes_donut

    # NB: _sim_outcomes_donut returns '' for zero rows (early-out). Pass rows
    # shaped like real sim results so the wrapper renders.
    html = _sim_outcomes_donut(sim_run.results)
    assert 'class="chart-card"' in html


def test_sim_report_has_shared_aligned_class(sim_run):
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    assert 'class="report-aligned sim-report"' in html


def test_redteam_hero_has_no_kpi_band(rt_report_single):
    from evaluatorq.dashboard.report_tabs import _redteam_hero, _rt_by_kind

    by = _rt_by_kind(rt_report_single)
    html = _redteam_hero(by.get('summary'), rt_report_single)
    assert 'report-hero' in html
    assert 'kpi-band' not in html and 'kpi-card' not in html


def test_redteam_hero_shows_agent_pills_multi(rt_report_multi):
    from evaluatorq.dashboard.report_tabs import _redteam_hero, _rt_by_kind

    by = _rt_by_kind(rt_report_multi)
    html = _redteam_hero(by.get('summary'), rt_report_multi)
    assert 'agents' in html  # "N agents" pill


def test_rt_overview_has_exec_summary_and_five_kpis(rt_report_multi):
    from evaluatorq.dashboard.report_tabs import redteam_report_tabs

    html = redteam_report_tabs('rid', rt_report_multi)
    assert 'class="report-aligned rt-report"' in html
    assert 'Executive summary' in html
    # 5 KPI cards
    assert html.count('kpi-card') >= 5
    assert 'Attack success rate' in html and 'Resistance rate' in html


def test_rt_exec_summary_zero_vuln_fallback(rt_report_clean):
    from evaluatorq.dashboard.report_tabs import _rt_by_kind, _rt_exec_summary

    by = _rt_by_kind(rt_report_clean)
    html = _rt_exec_summary(by['summary'].data, by)
    assert 'resisted' in html.lower()
    assert 'vulnerabilit' not in html.lower() or 'resisted all' in html.lower()


def test_rt_tabs_seven_labels(rt_report_multi):
    from evaluatorq.dashboard.report_tabs import redteam_report_tabs

    html = redteam_report_tabs('rid', rt_report_multi)
    for label in ['Overview', 'Agents', 'Focus areas', 'Breakdowns', 'Attacks', 'Usage', 'Config']:
        assert label in html
    assert 'Multi-turn' not in html  # folded into Breakdowns


def test_rt_tab_count_pills(rt_report_multi):
    # Count pills appear ONLY on Agents / Focus areas / Attacks (mockup parity).
    from evaluatorq.dashboard.report_tabs import redteam_report_tabs

    html = redteam_report_tabs('rid', rt_report_multi)
    assert 'class="tab-count"' in html  # the raw-label 3-tuple pill


def test_rt_exec_summary_multiturn_clause(rt_report_multi):
    # rt_report_multi must include multi-turn results with rising ASR by depth.
    from evaluatorq.dashboard.report_tabs import _rt_by_kind, _rt_exec_summary

    by = _rt_by_kind(rt_report_multi)
    html = _rt_exec_summary(by['summary'].data, by)
    assert 'conversation depth' in html.lower()


def test_rt_overview_end_to_end_wires_all_tabs(rt_report_multi):
    # Guard against a helper (esp. Focus areas) being built but never spliced in.
    from evaluatorq.dashboard.report_tabs import redteam_report_tabs

    html = redteam_report_tabs('rid', rt_report_multi)
    assert 'RISK' in html  # Focus-areas risk dial actually reaches the page
    assert 'ASR' in html  # Agents ASR dial actually reaches the page
    assert 'rk-heatmap' in html  # Breakdowns heatmap actually reaches the page


def test_rt_breakdowns_has_heatmap_and_multiturn(rt_report_multi):
    from evaluatorq.dashboard.report_tabs import _rt_breakdowns, _rt_by_kind

    html = _rt_breakdowns(_rt_by_kind(rt_report_multi))
    assert 'rk-heatmap' in html
    assert 'category' in html.lower()
    # multi-turn depth section present when turn_depth_analysis exists
    assert 'conversation depth' in html.lower()


def test_rt_breakdowns_omits_depth_when_absent(rt_report_single_turn):
    from evaluatorq.dashboard.report_tabs import _rt_breakdowns, _rt_by_kind

    html = _rt_breakdowns(_rt_by_kind(rt_report_single_turn))
    assert 'conversation depth' not in html.lower()


def test_rt_agents_single_agent_card(rt_report_single):
    from evaluatorq.dashboard.report_tabs import _rt_agents, _rt_by_kind

    html = _rt_agents(_rt_by_kind(rt_report_single), rt_report_single, 'rid')
    assert 'ASR' in html  # dial sub-label
    assert 'Single agent under assessment' in html


def test_rt_agent_card_studio_link_present_for_orq_target(monkeypatch):
    from evaluatorq.dashboard.report_tabs import _rt_agent_card

    monkeypatch.delenv('ORQ_WORKSPACE', raising=False)
    ctx = {
        'display_name': 'Support Bot',
        'id': 'abc123',
        'workspace_id': 'ws9',
        'target_kind': 'agent',
        'tools': [],
        'knowledge_bases': [],
    }
    html = _rt_agent_card(ctx, 'k', {'attacks': 5, 'vulns': 2, 'critical': 1, 'asr': 0.4})
    assert 'Open in Studio' in html
    assert 'ws9/agents/abc123' in html


def test_rt_agent_card_studio_link_absent_without_ids():
    from evaluatorq.dashboard.report_tabs import _rt_agent_card

    # Non-orq / missing id+workspace_id → no Studio deep-link.
    ctx = {'display_name': 'X', 'target_kind': 'agent', 'tools': [], 'knowledge_bases': []}
    html = _rt_agent_card(ctx, 'k', {})
    assert 'Open in Studio' not in html


def test_rt_config_no_agent_context(rt_report_single):
    # The Config tab itself must not render agent_context chips (moved to Agents).
    from evaluatorq.dashboard.report_tabs import _rt_by_kind, _rt_config

    config_html = _rt_config(_rt_by_kind(rt_report_single), rt_report_single)
    assert 'KNOWLEDGE' not in config_html  # tools/knowledge chip labels live only in Agents cards
    assert 'Methodology' in config_html  # but Config still has its own content


def test_rt_focus_tiers_and_dials(rt_report_multi):
    from evaluatorq.dashboard.report_tabs import _rt_by_kind, _rt_focus

    html = _rt_focus(_rt_by_kind(rt_report_multi))
    assert 'RISK' in html  # risk dial sub-label
    assert 'Recommended fix' in html or 'remediation' in html.lower()


def test_rt_focus_empty_on_clean_run(rt_report_clean):
    from evaluatorq.dashboard.report_tabs import _rt_by_kind, _rt_focus

    assert _rt_focus(_rt_by_kind(rt_report_clean)) == ''


def test_rt_focus_handles_absent_llm_recs(rt_report_static):
    from evaluatorq.dashboard.report_tabs import _rt_by_kind, _rt_focus

    # must not KeyError when 'llm_recommendations' key is absent
    _rt_focus(_rt_by_kind(rt_report_static))


def test_rt_report_empty_run_does_not_crash(rt_report_empty):
    # 0-attack run: exec summary '', KPI zeros, no crash in any helper.
    from evaluatorq.dashboard.report_tabs import redteam_report_tabs

    html = redteam_report_tabs('rid', rt_report_empty)
    assert 'class="report-aligned rt-report"' in html  # renders, doesn't raise


def test_rt_attacks_rows_are_details(rt_report_multi):
    from evaluatorq.dashboard.report_tabs import _rt_attacks

    html = _rt_attacks(rt_report_multi, 'rid')
    assert 'class="rt-attack-row"' in html
    # lazy-load fires on the <details> toggle (open), targeting the inner body —
    # a click trigger on the collapsed inner div never fires.
    assert 'hx-trigger="toggle once"' in html
    assert 'hx-target="find .rt-attack-row-body"' in html
    assert '/r/rid/redteam/attack?idx=' in html
    assert 'source' in html.lower()  # kept source_distribution renders below the table


def test_rt_config_has_metagrid_and_jury(rt_report_multi):
    # rt_report_multi fixture must set summary.jury_reliability so the block renders.
    from evaluatorq.dashboard.report_tabs import _rt_by_kind, _rt_config

    html = _rt_config(_rt_by_kind(rt_report_multi), rt_report_multi)
    assert 'Run configuration' in html or 'rk-meta' in html
    assert 'Methodology' in html
    assert 'JURY RELIABILITY' in html  # deviation #12: jury block replaces mockup's JURY string


_AGENT_INFO = {
    'key': 'support-orchestrator',
    'id': '01K8N...',
    'role': 'Router',
    'description': 'Front-door agent that triages requests.',
    'model': 'openai/gpt-4o',
    'tools': ['route_request', 'summarize'],
    'knowledge_bases': ['acme-help-center'],
    'memory_stores': [],
    'sub_agents': ['billing-agent'],
    'workspace_id': '624ccbbd-000',
    'base_url': 'https://my.orq.ai',
    'url': 'https://my.orq.ai/project/agents/01K8N...',
}


def test_sim_overview_agent_card_full(sim_run, monkeypatch) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    monkeypatch.setenv('ORQ_WORKSPACE', 'research')
    run = sim_run.model_copy(update={'agent_info': _AGENT_INFO})
    html = sim_report_tabs('rid', run)
    assert 'class="rk-panel sim-agent-card"' in html
    assert 'support-orchestrator' in html
    assert 'Router' in html
    assert 'Front-door agent that triages requests.' in html
    assert 'route_request' in html
    assert 'href="https://my.orq.ai/research/agents/01K8N..."' in html
    assert 'target="_blank"' in html


def test_sim_overview_agent_card_absent_when_no_agent_info(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    run = sim_run.model_copy(update={'agent_info': None})
    html = sim_report_tabs('rid', run)
    assert 'sim-agent-card' not in html


def test_sim_overview_agent_card_omits_open_link_without_url(sim_run, monkeypatch) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    monkeypatch.delenv('ORQ_WORKSPACE', raising=False)
    agent_info = dict(_AGENT_INFO, url=None)
    run = sim_run.model_copy(update={'agent_info': agent_info})
    html = sim_report_tabs('rid', run)
    assert 'class="rk-panel sim-agent-card"' in html
    assert 'sim-agent-open' not in html


def test_sim_overview_agent_card_uses_configured_workspace_for_legacy_uuid_link(sim_run, monkeypatch) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    monkeypatch.setenv('ORQ_WORKSPACE', 'research')
    agent_info = dict(_AGENT_INFO, workspace_id='624ccbbd-a482-40e2-b3d9-3621e09da1f8')
    html = sim_report_tabs('rid', sim_run.model_copy(update={'agent_info': agent_info}))
    assert 'href="https://my.orq.ai/research/agents/01K8N..."' in html


def test_sim_overview_agent_card_escapes_description(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    agent_info = dict(_AGENT_INFO, description='<script>alert(1)</script>')
    run = sim_run.model_copy(update={'agent_info': agent_info})
    html = sim_report_tabs('rid', run)
    assert '<script>' not in html
    assert '&lt;script&gt;' in html


def test_sim_overview_agent_card_uses_compact_description_not_prompt_content(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    agent_info = dict(
        _AGENT_INFO,
        description=(
            'A concise description of the agent.\n\n'
            'INSTRUCTIONS: this prompt-like configuration and its secrets must not appear in the overview.'
        ),
        instructions='This field must never be rendered.',
    )
    html = sim_report_tabs('rid', sim_run.model_copy(update={'agent_info': agent_info}))
    assert 'A concise description of the agent.' in html
    assert 'prompt-like configuration' not in html
    assert 'This field must never be rendered.' not in html


def test_agent_description_preview_caps_a_single_long_paragraph() -> None:
    from evaluatorq.dashboard.report_tabs import _AGENT_DESCRIPTION_PREVIEW_LIMIT, _agent_description_preview

    preview = _agent_description_preview('x' * (_AGENT_DESCRIPTION_PREVIEW_LIMIT + 10))
    assert preview is not None
    assert preview.endswith('...')
    assert len(preview) == _AGENT_DESCRIPTION_PREVIEW_LIMIT + 3


# --- agent-card live Orq fallback (_resolve_agent_info) ----------------------


def _orq_run(sim_run, **updates):
    """A sim_run posing as an Orq-agent target, with agent_info overrides."""
    base = {'target_kind': 'orq_agent', 'target': 'agent:support-orchestrator'}
    base.update(updates)
    return sim_run.model_copy(update=base)


@pytest.mark.asyncio
async def test_resolve_agent_info_complete_captured_skips_fetch(sim_run, monkeypatch) -> None:
    """A complete captured snapshot never triggers a live Orq fetch."""
    import evaluatorq.dashboard.report_tabs as rt

    def _boom(_key):  # fetch must not be called
        raise AssertionError('should not fetch when snapshot is complete')

    monkeypatch.setattr(rt, '_orq_agent_info_cached', _boom)
    run = _orq_run(sim_run, agent_info=dict(_AGENT_INFO))
    display, original, source = await rt._resolve_agent_info(run)
    assert source == 'captured'
    assert display is original


@pytest.mark.asyncio
async def test_resolve_agent_info_missing_core_augments_filling_only_gaps(sim_run, monkeypatch) -> None:
    """Missing a core field → fetch; captured values win, only gaps get filled."""
    import evaluatorq.dashboard.report_tabs as rt

    captured = dict(_AGENT_INFO, model='')  # model missing → incomplete
    captured['description'] = 'AS-RUN description'
    fetched = dict(_AGENT_INFO, model='openai/gpt-5', description='CURRENT description')
    async def _fetch(_key):
        return fetched

    monkeypatch.setattr(rt, '_orq_agent_info_cached', _fetch)
    display, original, source = await rt._resolve_agent_info(_orq_run(sim_run, agent_info=captured))
    assert source == 'augmented'
    assert display is not None
    assert display['model'] == 'openai/gpt-5'  # gap filled from Orq
    assert display['description'] == 'AS-RUN description'  # captured value kept
    assert original is captured


@pytest.mark.asyncio
async def test_resolve_agent_info_none_fetches_whole_card(sim_run, monkeypatch) -> None:
    """Nothing captured → whole card loaded live; no 'original' to show."""
    import evaluatorq.dashboard.report_tabs as rt

    async def _fetch(_key):
        return dict(_AGENT_INFO)

    monkeypatch.setattr(rt, '_orq_agent_info_cached', _fetch)
    display, original, source = await rt._resolve_agent_info(_orq_run(sim_run, agent_info=None))
    assert source == 'fetched'
    assert original is None
    assert display is not None
    assert display['key'] == _AGENT_INFO['key']


@pytest.mark.asyncio
async def test_resolve_agent_info_404_falls_back_to_saved_target(sim_run, monkeypatch) -> None:
    """A deleted or inaccessible agent still renders the run's saved target."""
    import evaluatorq.dashboard.report_tabs as rt

    async def _missing(_key):
        return None

    monkeypatch.setattr(rt, '_orq_agent_info_cached', _missing)
    run = _orq_run(sim_run, agent_info=None, target='agent:flight-delay-analyst', target_model='gpt-4o')
    display, original, source = await rt._resolve_agent_info(run)
    assert source == 'stored'
    assert original is None
    assert display == {'key': 'flight-delay-analyst', 'model': 'gpt-4o'}
    html = await rt.sim_agent_card_fragment(run)
    assert 'flight-delay-analyst' in html
    assert 'live Orq details are unavailable' in html


@pytest.mark.asyncio
async def test_resolve_agent_info_non_orq_target_never_fetches(sim_run, monkeypatch) -> None:
    """A non-Orq target is never enriched, even with an incomplete snapshot."""
    import evaluatorq.dashboard.report_tabs as rt

    async def _boom(_key):
        raise AssertionError('no fetch')

    monkeypatch.setattr(rt, '_orq_agent_info_cached', _boom)
    run = sim_run.model_copy(update={'target_kind': 'openai_model', 'target': 'gpt-4o', 'agent_info': None})
    _display, _original, source = await rt._resolve_agent_info(run)
    assert source == 'none'


@pytest.mark.asyncio
async def test_sim_overview_augmented_card_shows_source_note_and_toggle(sim_run, monkeypatch) -> None:
    import evaluatorq.dashboard.report_tabs as rt

    async def _fetch(_key):
        return dict(_AGENT_INFO)

    monkeypatch.setattr(rt, '_orq_agent_info_cached', _fetch)
    run = _orq_run(sim_run, agent_info=None)
    html = await rt.sim_agent_card_fragment(run)
    assert 'sim-agent-source' in html
    assert 'Loaded live from Orq' in html


def test_sim_overview_defers_incomplete_agent_details(sim_run, monkeypatch) -> None:
    import evaluatorq.dashboard.report_tabs as rt

    async def _boom(_key):
        raise AssertionError('initial report render must not fetch Orq')

    monkeypatch.setattr(rt, '_orq_agent_info_cached', _boom)
    run = _orq_run(sim_run, agent_info=None, target='agent:flight-delay-analyst')
    html = rt.sim_report_tabs('run-id', run)
    assert 'hx-get="/r/run-id/sim/agent-card"' in html
    assert 'flight-delay-analyst' in html


def test_sim_agent_card_endpoint_returns_deferred_fragment(client, roots, monkeypatch) -> None:
    import evaluatorq.dashboard.report_tabs as rt

    async def _fragment(run):
        assert run.run_name == 'test-sim'
        return '<div class="sim-agent-card">enriched</div>'

    monkeypatch.setattr(rt, 'sim_agent_card_fragment', _fragment)
    rid = report_id(roots[1] / 'sim.json')
    response = client.get(f'/r/{rid}/sim/agent-card')
    assert response.status_code == 200
    assert response.text == '<div class="sim-agent-card">enriched</div>'


def test_agent_key_recovered_from_run_name_when_target_missing(sim_run) -> None:
    """Legacy orq_agent runs with target=None recover the key from run_name
    (sim:<key>:...), so the live fallback can still fetch."""
    import evaluatorq.dashboard.report_tabs as rt

    run = sim_run.model_copy(
        update={'target_kind': 'orq_agent', 'target': None, 'agent_info': None, 'run_name': 'sim:refund-agent-fixed:tailscale-openai'}
    )
    assert rt._agent_key_for(run) == 'refund-agent-fixed'


def test_agent_key_prefers_captured_then_target(sim_run) -> None:
    import evaluatorq.dashboard.report_tabs as rt

    assert rt._agent_key_for(sim_run.model_copy(update={'agent_info': {'key': 'k1'}, 'target': 'agent:k2'})) == 'k1'
    assert rt._agent_key_for(sim_run.model_copy(update={'agent_info': None, 'target': 'agent:k2'})) == 'k2'
