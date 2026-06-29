"""Tabbed report bodies — both surfaces render Streamlit-aligned tabs, and
empty tabs (no data) drop out (RES-974)."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id

from tests.dashboard.test_downloads import _make_rt_report, _make_sim_run


def _tab_labels(html: str) -> list[str]:
    import html as _html
    import re

    return [_html.unescape(m) for m in re.findall(r'class="tab-label" for="[^"]*">([^<]+)<', html)]


@pytest.fixture()
def roots(tmp_path: Path) -> list[Path]:
    rt = tmp_path / "runs"
    sim = tmp_path / "sim-runs"
    rt.mkdir()
    sim.mkdir()
    (rt / "rt.json").write_text(_make_rt_report().model_dump_json())
    (sim / "sim.json").write_text(
        _make_sim_run(personas=["alice", "bob"], goal_achieved_flags=[True, False]).model_dump_json()
    )
    return [rt, sim]


@pytest.fixture()
def client(roots: list[Path]) -> TestClient:
    return TestClient(build_app(roots=roots), raise_server_exceptions=True)


def test_sim_report_renders_tabs(client: TestClient, roots: list[Path]) -> None:
    rid = report_id(roots[1] / "sim.json")
    labels = _tab_labels(client.get(f"/r/{rid}").text)
    # Overview / Breakdown / Transcripts always present for a non-empty sim run.
    assert labels[:3] == ["Overview", "Breakdown", "Transcripts"]
    assert "Judge & errors" in labels


def test_redteam_report_renders_tabs(client: TestClient, roots: list[Path]) -> None:
    rid = report_id(roots[0] / "rt.json")
    labels = _tab_labels(client.get(f"/r/{rid}").text)
    assert "Summary" in labels
    assert "Breakdown" in labels
    assert "Explorer" in labels
    assert "Methodology" in labels


def test_single_agent_report_has_no_comparison_tab(client: TestClient, roots: list[Path]) -> None:
    """The Comparison tab is multi-agent only — a single-agent report drops it."""
    rid = report_id(roots[0] / "rt.json")
    labels = _tab_labels(client.get(f"/r/{rid}").text)
    assert "Comparison" not in labels


def test_clean_run_drops_error_tab(client: TestClient, roots: list[Path]) -> None:
    """No runtime errors → no Error Analysis tab (empty tabs drop out)."""
    rid = report_id(roots[0] / "rt.json")
    labels = _tab_labels(client.get(f"/r/{rid}").text)
    assert "Error Analysis" not in labels


def test_tab_panels_match_tab_count(client: TestClient, roots: list[Path]) -> None:
    """Every tab label has exactly one matching panel (no orphans)."""
    rid = report_id(roots[1] / "sim.json")
    html = client.get(f"/r/{rid}").text
    assert html.count('class="tab-label"') == html.count('class="tab-panel"')


def test_filter_post_preserves_tabs(client: TestClient, roots: list[Path]) -> None:
    """The filter round-trip re-renders the tabbed body, not the flat export."""
    rid = report_id(roots[1] / "sim.json")
    r = client.post(f"/r/{rid}/filter", data={"persona": "alice", "goal_outcome": "All"})
    assert r.status_code == 200
    assert "tab-label" in r.text
    assert "filter-swap" in r.text
