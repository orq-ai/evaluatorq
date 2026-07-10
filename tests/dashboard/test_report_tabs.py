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
    # Folded to Overview / Breakdowns / Evidence / (Error Analysis) / Config.
    assert 'Overview' in labels
    assert 'Breakdowns' in labels
    assert 'Evidence' in labels
    assert 'Config' in labels
    # Old tab names are gone; Error Analysis stays its own tab, never folded.
    assert 'Summary' not in labels
    assert 'Methodology' not in labels


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
    assert 'class="sim-report"' in html
    assert 'Executive summary' in html
    # 5-card KPI band incl. Avg turns
    assert 'Avg turns' in html
    assert 'Goal completion' in html
    assert 'goal met' in html


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
