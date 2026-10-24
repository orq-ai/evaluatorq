"""Simulation overview keeps goal outcomes and recorded spend distinct."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard import metrics, view
from evaluatorq.dashboard.app import build_app


def _store(tmp_path: Path, results: list[dict]) -> list[Path]:
    sim = tmp_path / 'sim-runs'
    sim.mkdir()
    (sim / 'run.json').write_text(json.dumps({
        'mode': 'run', 'run_name': 'Goal test', 'created_at': '2026-10-24T10:00:00',
        'total_results': len(results), 'scorer_averages': {'quality': 1.0}, 'results': results,
    }))
    return [sim]


def test_recorded_total_is_not_average_and_missing_cost_is_unknown(tmp_path: Path) -> None:
    roots = _store(tmp_path, [
        {'goal_achieved': True, 'token_usage': {'cost_usd': 4.0, 'calls': 2, 'priced_calls': 2}},
        {'goal_achieved': False, 'token_usage': {'cost_usd': 2.0, 'calls': 2, 'priced_calls': 2}},
        {'goal_achieved': False, 'token_usage': {'cost_usd': None, 'calls': 2, 'priced_calls': 0}},
    ])
    data = metrics.sim_overview(roots)
    assert data.total_cost == pytest.approx(6.0)
    assert data.avg_cost == pytest.approx(3.0)
    html = view.sim_overview_body(data)
    assert 'Total spend' in html and '$6.00' in html
    assert '4 of 6 calls priced' in html
    assert 'Avg cost/sim' not in html


def test_goal_rate_is_independent_of_scorer_average_and_retains_errors(tmp_path: Path) -> None:
    roots = _store(tmp_path, [
        {'goal_achieved': True}, {'goal_achieved': False},
        {'goal_achieved': True, 'terminated_by': 'error'},
    ])
    data = metrics.sim_overview(roots)
    assert data.goal_completion == pytest.approx(1 / 3)
    assert data.recent[0].score == 1.0
    assert data.recent[0].achieved == 1
    html = TestClient(build_app(roots=roots)).get('/?surface=sim').text
    assert '1/3' in html and '33%' in html
    assert '1 errored' in html
    assert 'Goal rate' in html
    assert 'run-score rg-score fail' in html
    assert 'Compare runs' not in html


def test_all_errors_have_no_goal_verdict_and_missing_spend_stays_unknown(tmp_path: Path) -> None:
    roots = _store(tmp_path, [{'goal_achieved': True, 'terminated_by': 'error'}])
    data = metrics.sim_overview(roots)
    assert data.goal_completion is None
    assert data.total_cost is None
    html = view.sim_overview_body(data)
    assert 'No goal verdict available' in html
    assert 'run-score rg-score neutral' in html
    assert '1 errored' in html
    assert '$0.00' not in html


def test_zero_cost_is_known_and_out_of_range_page_keeps_rows(tmp_path: Path) -> None:
    roots = _store(tmp_path, [{'goal_achieved': True, 'token_usage': {'cost_usd': 0.0}}])
    data = metrics.sim_overview(roots, page=999)
    assert data.page == 1 and len(data.recent) == 1
    assert data.total_cost == 0.0
    assert '$0.0000' in view.sim_overview_body(data)
