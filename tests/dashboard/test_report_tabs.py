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


@pytest.fixture(autouse=True)
def _clean_workspace_env(monkeypatch):
    """Studio deep-links resolve host+workspace from the run's experiment_url,
    then fall back to env. Clear env so the agent-card tests toggle on the
    experiment_url / captured-key / explicit ORQ_WORKSPACE alone."""
    for var in ('ORQ_WORKSPACE', 'ORQ_WORKSPACE_SLUG', 'ORQ_BASE_URL', 'ORQ_API_KEY'):
        monkeypatch.delenv(var, raising=False)


def test_turn_metric_descriptor_has_keys_and_directions() -> None:
    from evaluatorq.simulation.metrics import TURN_METRICS

    assert [(m.key, m.high_is_risky) for m in TURN_METRICS] == [
        ('response_quality', False),
        ('hallucination_risk', True),
        ('tone_appropriateness', False),
        ('factual_accuracy', False),
    ]


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


def test_sim_overview_has_exec_summary_and_six_kpis(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    assert 'class="report-aligned sim-report"' in html
    assert 'Executive summary' in html
    # Six KPI cards: the overview no longer has a Goal-completion card.
    kpi_band = html.split('<div class="kpi-band">', 1)[1].split('</div><div class="sim-overview-grid-2">', 1)[0]
    assert kpi_band.count('class="kpi-card ') == 6
    for label in ('Personas', 'Scenarios', 'Conversations', 'Avg score', 'Avg turns', 'Errors'):
        assert f'<div class="kpi-label">{label}</div>' in kpi_band
    assert 'Goal completion' not in kpi_band
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


def test_sim_kpi_errors_status_reflects_error_count(sim_run) -> None:
    """The Error KPI is pass with no errors and fail when errors are present."""
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    kpi_band = html.split('<div class="kpi-band">', 1)[1].split('</div><div class="sim-overview-grid-2">', 1)[0]
    errors_card = kpi_band.split('<div class="kpi-label">Errors</div>', 1)[0].rsplit('<div class="kpi-card ', 1)[1]
    assert errors_card.startswith('kpi-card--pass">')

    sim_run.results[0].metadata['error'] = 'judge request timed out'
    html_with_error = sim_report_tabs('rid', sim_run)
    error_kpi_band = html_with_error.split('<div class="kpi-band">', 1)[1].split(
        '</div><div class="sim-overview-grid-2">', 1
    )[0]
    error_card = error_kpi_band.split('<div class="kpi-label">Errors</div>', 1)[0].rsplit('<div class="kpi-card ', 1)[1]
    assert error_card.startswith('kpi-card--fail">')


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


def test_dashboard_failures_use_five_columns_and_drawer_rows(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    failures = html.split('id="section-failures_first"', 1)[1].split('</section>', 1)[0]
    assert ['Scenario', 'Persona', 'Why', 'Criteria', 'Score'] == _headers(failures)
    assert 'crit-' in failures  # criteria dots cell rendered (dots or empty-dash)
    assert 'href="#conv-' not in failures
    assert 'data-entity-kind="conversation"' in failures
    assert 'data-drawer-url="/r/rid/sim/transcript?idx=1"' in failures
    assert 'data-no-drawer' in failures


def test_sim_dashboard_no_longer_emits_anchor_or_foldout_drilldown(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)

    assert 'href="#conv-' not in html
    assert 'id="conv-' not in html
    assert '<details class="sim-conv-card"' not in html
    assert 'toggle once from:closest details' not in html


def test_cohort_template_contains_stats_and_compact_conversation_triggers(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    template = _template(html, 'persona-')
    assert 'Goal rate' in template and 'Avg score' in template and 'Tokens' in template
    assert 'sim-cohort-conversations' in template
    assert 'data-entity-kind="conversation"' in template


def test_filtered_cohort_template_shows_empty_message_when_no_conversations_match() -> None:
    from evaluatorq.dashboard.report_tabs import _sim_persona_template

    template = _sim_persona_template(
        {'name': 'No-match persona'},
        'persona-0',
        0,
        1,
        conversations=[],
        rid='rid',
    )

    assert 'data-sim-entity-template' in template
    assert '<p class="sim-cohort-empty">No conversations.</p>' in template


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


def test_sim_entity_dialog_is_a_right_side_drawer() -> None:
    from evaluatorq.dashboard.styles import DASHBOARD_CSS

    css = DASHBOARD_CSS
    assert 'inset: 0 0 0 auto' in css
    assert 'width: 50vw' in css
    assert 'height: 100vh' in css
    assert '.sim-entity-dialog::backdrop' in css
    assert '@media (max-width: 480px)' in css


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

    assert "trigger.getAttribute('data-drawer-url')" in source
    assert 'a[href^="#conv-"]' not in source
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


def test_redteam_hero_separates_run_agent_and_model(rt_report_single):
    """Run name, agent name and model each get their own slot: the title is the
    run name with the runner's `(target) (pipeline)` suffixes stripped, the
    pipeline sits in the kicker, and the agent pill carries name + model."""
    from evaluatorq.dashboard.report_tabs import _redteam_hero, _rt_by_kind

    report = rt_report_single.model_copy(
        update={'description': 'Nightly sweep (agent-a) (static)'},
    )
    for r in report.results:
        r.agent.display_name = 'Support agent'
        r.agent.model = 'gpt-4o'

    html = _redteam_hero(_rt_by_kind(report).get('summary'), report)
    assert '>Nightly sweep<' in html or 'Nightly sweep</h2>' in html
    assert '(agent-a)' not in html and '(static)' not in html
    assert 'Red Team · Static' in html
    assert 'Support agent' in html
    assert 'gpt-4o' in html


def test_redteam_hero_keeps_unrecognised_parenthetical(rt_report_single):
    from evaluatorq.dashboard.report_tabs import _rt_run_name

    report = rt_report_single.model_copy(update={'description': 'Q3 sweep (post-patch)'})
    assert _rt_run_name(report) == 'Q3 sweep (post-patch)'


@pytest.mark.parametrize(
    'description',
    [
        # The runner interpolates PreparedTarget.target, which is the full
        # target string, not the bare key held in tested_agents.
        'Nightly sweep (agent:agent-a) (dynamic)',
        'Nightly sweep (agent-a)',
        'Nightly sweep (2 targets)',
        'Nightly sweep (1 target)',
    ],
)
def test_rt_run_name_strips_runner_suffixes(rt_report_single, description):
    from evaluatorq.dashboard.report_tabs import _rt_run_name

    report = rt_report_single.model_copy(update={'description': description})
    assert _rt_run_name(report) == 'Nightly sweep'


def test_redteam_hero_pill_carries_own_model_per_agent(rt_report_multi):
    """Each pill shows its own agent's model — a global lookup would paste the
    first agent's model onto every pill."""
    import re as _re

    from evaluatorq.dashboard.report_tabs import _redteam_hero, _rt_by_kind

    models: dict[str, str] = {}
    for r in rt_report_multi.results:
        r.agent.display_name = f'Agent {r.agent.key}'
        r.agent.model = f'model-{r.agent.key}'
        models[r.agent.display_name] = r.agent.model

    html = _redteam_hero(_rt_by_kind(rt_report_multi).get('summary'), rt_report_multi)
    pills = _re.findall(
        r'rt-hero-pill-name">([^<]*)</span><span class="rt-hero-pill-sub">([^<]*)<',
        html,
    )
    assert len(pills) == len(models) >= 2
    for name, sub in pills:
        assert sub.startswith(f'{models[name]} · ')


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


def test_rt_exec_summary_reports_pre_execution_rows() -> None:
    from evaluatorq.dashboard.report_tabs import _rt_exec_summary

    html = _rt_exec_summary({'total_attacks': 0, 'pre_execution_errors': 2}, {})

    assert 'rows failed before execution' in html
    assert '<strong>2</strong>' in html


def test_rt_overview_outcome_buckets_conserve_attack_total(monkeypatch, rt_report_clean) -> None:
    from types import SimpleNamespace

    from evaluatorq.dashboard import report_kit
    from evaluatorq.dashboard.report_tabs import _rt_overview

    captured: dict[str, object] = {}

    def capture_donut(segments, center_label, center_caption):
        captured['segments'] = segments
        return ''

    monkeypatch.setattr(report_kit, 'donut', capture_donut)
    summary = SimpleNamespace(
        data={
            'total_attacks': 3,
            'evaluated_attacks': 2,
            'unevaluated_attacks': 1,
            'vulnerabilities_found': 1,
            'resistance_rate': 0.5,
            'vulnerability_rate': 0.5,
            'total_errors': 2,
            'pre_execution_errors': 1,
            'by_severity': {},
        }
    )
    category = SimpleNamespace(data={'rows': [{'total_attacks': 3}]})

    _rt_overview({'summary': summary, 'category_breakdown': category}, rt_report_clean)

    segments = captured['segments']
    assert sum(segment['value'] for segment in segments) == summary.data['total_attacks']
    assert {segment['label']: segment['value'] for segment in segments} == {
        'Resistant': 1,
        'Vulnerable': 1,
        'Error': 1,
    }
    assert sum(row['total_attacks'] for row in category.data['rows']) == summary.data['total_attacks']


def test_redteam_error_analysis_includes_pre_execution_rows(rt_report_clean) -> None:
    from evaluatorq.dashboard.report_tabs import redteam_report_tabs
    from evaluatorq.redteam.contracts import RunError

    report = rt_report_clean.model_copy(
        update={
            'errors': [
                RunError(
                    message='strategy generation failed',
                    error_type='unknown',
                    stage='datapoint_generation',
                    code='datapoint_error',
                )
            ],
            'summary': rt_report_clean.summary.model_copy(
                update={'total_errors': 1, 'pre_execution_errors': 1}
            ),
        }
    )

    html = redteam_report_tabs('rid', report)

    assert 'strategy generation failed' in html


def test_rt_kpi_band_zero_evaluated_shows_na_not_perfect(rt_report_clean):
    """A zero-evaluated run's detail view must not render the schema-default
    resistance as a perfect 100% — same no-score rule as the landing rows."""
    from evaluatorq.dashboard.report_tabs import redteam_report_tabs

    report = rt_report_clean.model_copy(
        update={
            'summary': rt_report_clean.summary.model_copy(
                update={'evaluated_attacks': 0, 'resistance_rate': 1.0}
            )
        }
    )
    html = redteam_report_tabs('rid', report)
    # kpi cards render value before label, so the card's value sits just
    # before the 'Resistance rate' text.
    idx = html.index('Resistance rate')
    card = html[max(idx - 200, 0) : idx]
    assert 'n/a' in card
    assert '100%' not in card


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

    html = _rt_focus(_rt_by_kind(rt_report_multi), 'rid', rt_report_multi)
    assert 'RISK' in html  # risk dial sub-label
    assert 'Recommended fix' in html or 'remediation' in html.lower()


def test_rt_focus_empty_on_clean_run(rt_report_clean):
    from evaluatorq.dashboard.report_tabs import _rt_by_kind, _rt_focus

    assert _rt_focus(_rt_by_kind(rt_report_clean), 'rid', rt_report_clean) == ''


def test_rt_focus_handles_absent_llm_recs(rt_report_static):
    from evaluatorq.dashboard.report_tabs import _rt_by_kind, _rt_focus

    # must not KeyError when 'llm_recommendations' key is absent
    _rt_focus(_rt_by_kind(rt_report_static), 'rid', rt_report_static)


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
    assert 'Router' not in html  # role/"Assistant" marker dropped — redundant next to the name
    assert 'Front-door agent that triages requests.' in html
    assert 'route_request' in html
    assert 'billing-agent' in html  # sub-agents now render as a labelled section
    assert 'Sub-agents' in html
    assert 'href="https://my.orq.ai/research/agents/01K8N..."' in html
    assert 'target="_blank"' in html


def test_sim_overview_agent_card_has_decorative_bot_icon(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run.model_copy(update={'agent_info': _AGENT_INFO}))

    assert 'class="sim-agent-identity"' in html
    assert 'class="sim-agent-icon"' in html
    assert 'aria-hidden="true"' in html


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
        update={
            'target_kind': 'orq_agent',
            'target': None,
            'agent_info': None,
            'run_name': 'sim:refund-agent-fixed:tailscale-openai',
        }
    )
    assert rt._agent_key_for(run) == 'refund-agent-fixed'


def test_agent_key_prefers_captured_then_target(sim_run) -> None:
    import evaluatorq.dashboard.report_tabs as rt

    assert rt._agent_key_for(sim_run.model_copy(update={'agent_info': {'key': 'k1'}, 'target': 'agent:k2'})) == 'k1'
    assert rt._agent_key_for(sim_run.model_copy(update={'agent_info': None, 'target': 'agent:k2'})) == 'k2'
