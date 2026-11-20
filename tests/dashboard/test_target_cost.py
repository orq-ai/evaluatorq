"""Target cost in the sim and red-team report headers and on the sim compare page.

The target agent's own cost sits next to the run's total cost. A run saved before
target usage was recorded shows "unknown", never $0, and a partly priced figure is
marked as a lower bound.
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from starlette.testclient import TestClient

from evaluatorq.contracts import TokenUsage
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id
from evaluatorq.redteam.contracts import ExecutionDetails, RedTeamReport
from evaluatorq.redteam.reports.converters import compute_report_summary
from evaluatorq.simulation.contracts import SimulationResult, SimulationRun, TerminatedBy
from tests.dashboard.conftest import _make_rt_result


def _usage(cost: float, *, calls: int = 2, priced: int = 2) -> TokenUsage:
    return TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120, total_cost=cost, calls=calls, priced_calls=priced)


def _sim_run(name: str, target_cost: float | None) -> SimulationRun:
    results = [
        SimulationResult(
            messages=[],
            terminated_by=TerminatedBy.judge,
            reason='done',
            goal_achieved=True,
            goal_completion_score=1.0,
            rules_broken=[],
            turn_count=2,
            turn_metrics=[],
            token_usage=_usage(0.5),
            target_token_usage=None if target_cost is None else _usage(target_cost),
            metadata={'persona': persona, 'scenario': 'billing'},
        )
        for persona in ('alice', 'bob')
    ]
    return SimulationRun(
        run_name=name,
        created_at=datetime(2026, 11, 1, tzinfo=timezone.utc),
        mode='run',
        target_kind='callback',
        evaluator_names=[],
        total_results=2,
        scorer_averages={},
        results=results,
        token_usage_total=_usage(1.0, calls=4, priced=4),
        target_token_usage=None if target_cost is None else _usage(2 * target_cost, calls=4, priced=4),
    )


def _write_legacy(run: SimulationRun, path: Path) -> None:
    """Save *run* the way a version without target usage did: no target fields at all."""
    data = run.model_dump(mode='json')
    data.pop('target_token_usage')
    for r in data['results']:
        r.pop('target_token_usage')
    path.write_text(json.dumps(data))


def _hero_cost(page: str) -> str:
    match = re.search(r'<p class="report-hero-cost">(.*?)</p>', page, re.S)
    assert match, 'hero cost line missing'
    return html.unescape(re.sub(r'<[^>]+>', ' ', match.group(1)))


def _client(root: Path) -> TestClient:
    return TestClient(build_app(roots=[root]), raise_server_exceptions=True)


def test_sim_header_shows_total_and_target_cost(tmp_path: Path) -> None:
    path = tmp_path / 'a.json'
    path.write_text(_sim_run('A', 0.01).model_dump_json())

    text = _hero_cost(_client(tmp_path).get(f'/r/{report_id(path)}').text)

    assert 'Total cost $1.0000' in ' '.join(text.split())
    assert 'Target cost $0.0200' in ' '.join(text.split())


def test_sim_header_shows_unknown_target_cost_for_an_old_run(tmp_path: Path) -> None:
    path = tmp_path / 'old.json'
    _write_legacy(_sim_run('Old', 0.01), path)

    page = _client(tmp_path).get(f'/r/{report_id(path)}').text
    text = ' '.join(_hero_cost(page).split())

    assert 'Target cost unknown' in text
    assert '$0.0000' not in text
    assert 'did not record the target agent' in html.unescape(page)


def test_rt_header_marks_a_partly_priced_target_cost_as_a_lower_bound(tmp_path: Path) -> None:
    results = [
        _make_rt_result(
            attack_id='ASI01-1',
            category='ASI01',
            execution=ExecutionDetails(turns=1, token_usage=_usage(0.3), token_usage_target=_usage(0.004, calls=2, priced=1)),
        )
    ]
    report = RedTeamReport(
        created_at=datetime(2026, 11, 1, tzinfo=timezone.utc),
        description='rt',
        pipeline='dynamic',
        categories_tested=['ASI01'],
        tested_agents=['agent-a'],
        total_results=1,
        results=results,
        summary=compute_report_summary(results),
    )
    path = tmp_path / 'rt.json'
    path.write_text(report.model_dump_json())

    text = ' '.join(_hero_cost(_client(tmp_path).get(f'/r/{report_id(path)}').text).split())

    assert 'Total cost $0.3000' in text
    assert 'Target cost $0.0040 (1 of 2 calls)' in text


def test_rt_header_shows_unknown_target_cost_for_an_old_report(tmp_path: Path) -> None:
    results = [_make_rt_result(attack_id='ASI01-1', category='ASI01', execution=ExecutionDetails(turns=1, token_usage=_usage(0.3)))]
    report = RedTeamReport(
        created_at=datetime(2026, 11, 1, tzinfo=timezone.utc),
        description='rt',
        pipeline='dynamic',
        categories_tested=['ASI01'],
        tested_agents=['agent-a'],
        total_results=1,
        results=results,
        summary=compute_report_summary(results),
    )
    path = tmp_path / 'rt.json'
    path.write_text(report.model_dump_json())

    text = ' '.join(_hero_cost(_client(tmp_path).get(f'/r/{report_id(path)}').text).split())

    assert 'Target cost unknown' in text


def _target_tile(page: str) -> str:
    match = re.search(r'<div class="kpi-card[^"]*"><div class="kpi-value">([^<]*)</div><div class="kpi-label">Target cost</div>(.*?)</div></div>', page, re.S)
    assert match, 'target cost tile missing'
    return ' '.join(html.unescape(match.group(1) + ' ' + re.sub(r'<[^>]+>', ' ', match.group(2))).split())


def test_compare_shows_the_target_cost_change_from_a_to_b(tmp_path: Path) -> None:
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_text(_sim_run('A', 0.01).model_dump_json())
    b.write_text(_sim_run('B', 0.004).model_dump_json())

    tile = _target_tile(_client(tmp_path).get(f'/compare/sim?a={report_id(a)}&b={report_id(b)}').text)

    assert tile.startswith('$0.0080')
    assert '$0.0200 → $0.0080' in tile
    assert '▼ -$0.0120' in tile


def test_compare_with_an_old_run_shows_unknown_and_no_delta(tmp_path: Path) -> None:
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    _write_legacy(_sim_run('Old', 0.01), a)
    b.write_text(_sim_run('B', 0.004).model_dump_json())

    tile = _target_tile(_client(tmp_path).get(f'/compare/sim?a={report_id(a)}&b={report_id(b)}').text)

    assert 'unknown → $0.0080' in tile
    assert 'n/a' in tile
    assert '▲' not in tile and '▼' not in tile
