"""Tabbed report bodies — both surfaces render Streamlit-aligned tabs, and
empty tabs (no data) drop out (RES-974)."""

from __future__ import annotations

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
    # Personas+scenarios row hugs content (no stretch void) — distinct from the
    # donut row's intentional equal-height stretch.
    assert 'sim-overview-grid-2--top' in html


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


def test_sim_transcripts_tab_drops_failure_mode(sim_run) -> None:
    """failure_mode moved to Breakdown (spec §Breakdown.4); Transcripts keeps
    only evaluator_scores / judge_verdicts / errors below the cards."""
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    html = sim_report_tabs('rid', sim_run)
    assert 'id="section-failure_mode"' not in html


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


def test_sim_turn_quality_single_turn_drops_empty_chart() -> None:
    """A single-turn run has no trend to plot, so the per-turn line chart is
    omitted (it would render an empty plot) while the avg-quality tiles remain."""
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
    assert 'class="rk-line-chart"' not in html  # empty single-turn chart omitted
    assert 'Per-turn quality' not in html
    assert 'class="sim-aq-grid"' in html  # avg-quality tiles still shown
    assert '1 turns' in html


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


def test_rt_agent_card_studio_link_present_for_orq_target():
    from evaluatorq.dashboard.report_tabs import _rt_agent_card

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


def test_sim_overview_agent_card_full(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    run = sim_run.model_copy(update={'agent_info': _AGENT_INFO})
    html = sim_report_tabs('rid', run)
    assert 'class="rk-panel sim-agent-card"' in html
    assert 'support-orchestrator' in html
    assert 'Router' in html
    assert 'Front-door agent that triages requests.' in html
    assert 'route_request' in html
    assert 'href="https://my.orq.ai/project/agents/01K8N..."' in html
    assert 'target="_blank"' in html


def test_sim_overview_agent_card_absent_when_no_agent_info(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    run = sim_run.model_copy(update={'agent_info': None})
    html = sim_report_tabs('rid', run)
    assert 'sim-agent-card' not in html


def test_sim_overview_agent_card_omits_open_link_without_url(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    agent_info = dict(_AGENT_INFO, url=None)
    run = sim_run.model_copy(update={'agent_info': agent_info})
    html = sim_report_tabs('rid', run)
    assert 'class="rk-panel sim-agent-card"' in html
    assert 'sim-agent-open' not in html


def test_sim_overview_agent_card_escapes_description(sim_run) -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs

    agent_info = dict(_AGENT_INFO, description='<script>alert(1)</script>')
    run = sim_run.model_copy(update={'agent_info': agent_info})
    html = sim_report_tabs('rid', run)
    assert '<script>' not in html
    assert '&lt;script&gt;' in html
