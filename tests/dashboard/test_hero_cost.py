"""The report header's total cost agrees with the runs list when only part of the cost is known.

A run saved before the whole-run total existed (or a red-team summary without one)
still carries per-conversation costs. The runs list sums those; the report header
must show the same sum, marked as a lower bound with the conversations it covers,
rather than "Total cost unknown".
"""

from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from starlette.testclient import TestClient

from evaluatorq.common.reports import fmt_cost
from evaluatorq.contracts import TokenUsage
from evaluatorq.dashboard import metrics
from evaluatorq.dashboard.app import build_app
from evaluatorq.dashboard.library import report_id
from evaluatorq.redteam.contracts import ExecutionDetails, RedTeamReport
from evaluatorq.redteam.reports.converters import compute_report_summary
from evaluatorq.simulation.contracts import SimulationResult, SimulationRun, TerminatedBy
from tests.dashboard.conftest import _make_rt_result


def _legacy_usage(cost: float | None) -> TokenUsage:
    """Usage as saved before ``priced_calls`` existed: a call count, a cost or ``None``."""
    return TokenUsage(input_tokens=7000, output_tokens=2000, total_tokens=9000, total_cost=cost, calls=5)


def _sim_result(persona: str, *, usage: TokenUsage, target: TokenUsage | None = None) -> SimulationResult:
    return SimulationResult(
        messages=[],
        terminated_by=TerminatedBy.judge,
        reason='done',
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=2,
        turn_metrics=[],
        token_usage=usage,
        target_token_usage=target,
        metadata={'persona': persona, 'scenario': 'refund'},
    )


def _write_pre_total_sim_run(path: Path, results: list[SimulationResult]) -> None:
    """Save a sim run the way versions before run-level usage did: no ``token_usage_total``, no ``target_token_usage``."""
    run = SimulationRun(
        run_name='refund-agent',
        created_at=datetime(2026, 11, 1, tzinfo=timezone.utc),
        mode='run',
        target_kind='orq_agent',
        evaluator_names=[],
        total_results=len(results),
        scorer_averages={},
        results=results,
    )
    data = run.model_dump(mode='json')
    data.pop('token_usage_total')
    data.pop('target_token_usage')
    path.write_text(json.dumps(data))


def _hero_cost(page: str) -> str:
    match = re.search(r'<p class="report-hero-cost">(.*?)</p>', page, re.S)
    assert match, 'hero cost line missing'
    return ' '.join(html.unescape(re.sub(r'<[^>]+>', ' ', match.group(1))).split())


def _client(root: Path) -> TestClient:
    return TestClient(build_app(roots=[root]), raise_server_exceptions=True)


def test_sim_header_shows_the_known_part_of_a_run_without_a_saved_total(tmp_path: Path) -> None:
    # Shaped like a real pre-total run: three conversations priced, one reported tokens
    # but no cost, one errored before any call and reported no usage.
    path = tmp_path / 'sim.json'
    _write_pre_total_sim_run(
        path,
        [
            _sim_result('alice', usage=_legacy_usage(0.12)),
            _sim_result('bob', usage=_legacy_usage(0.2)),
            _sim_result('carol', usage=_legacy_usage(0.1)),
            _sim_result('dave', usage=_legacy_usage(None)),
            _sim_result('erin', usage=TokenUsage()),
        ],
    )

    (row,) = metrics.sim_run_rows([tmp_path])
    text = _hero_cost(_client(tmp_path).get(f'/r/{report_id(path)}').text)

    assert row.cost is not None
    assert f'Total cost {fmt_cost(row.cost)} (3 of 5 conversations)' in text
    assert 'Total cost unknown' not in text


def test_sim_header_total_tip_names_it_a_lower_bound(tmp_path: Path) -> None:
    path = tmp_path / 'sim.json'
    _write_pre_total_sim_run(
        path, [_sim_result('alice', usage=_legacy_usage(0.12)), _sim_result('bob', usage=_legacy_usage(None))]
    )

    page = html.unescape(_client(tmp_path).get(f'/r/{report_id(path)}').text)

    assert 'Lower bound: the cost covers only 1 of 2 conversations.' in page


def test_sim_header_sums_conversation_target_cost_when_the_run_total_is_missing(tmp_path: Path) -> None:
    # Conversations recorded their target usage, but the run predates the run-level sum.
    target = TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120, total_cost=0.01, calls=2, priced_calls=2)
    path = tmp_path / 'sim.json'
    _write_pre_total_sim_run(
        path,
        [
            _sim_result('alice', usage=_legacy_usage(0.12), target=target),
            _sim_result('bob', usage=_legacy_usage(0.2), target=target),
            _sim_result('carol', usage=_legacy_usage(0.1), target=None),
        ],
    )

    text = _hero_cost(_client(tmp_path).get(f'/r/{report_id(path)}').text)

    assert 'Target cost $0.0200 (2 of 3 conversations)' in text


def test_sim_header_keeps_the_saved_total_when_there_is_one(tmp_path: Path) -> None:
    # A saved whole-run total also covers generation and summary spend, so it wins over the result sum.
    run = SimulationRun(
        run_name='refund-agent',
        created_at=datetime(2026, 11, 1, tzinfo=timezone.utc),
        mode='run',
        target_kind='orq_agent',
        evaluator_names=[],
        total_results=2,
        scorer_averages={},
        results=[_sim_result('alice', usage=_legacy_usage(0.12)), _sim_result('bob', usage=TokenUsage())],
        token_usage_total=TokenUsage(total_cost=0.5, calls=10, priced_calls=10),
    )
    path = tmp_path / 'sim.json'
    path.write_text(run.model_dump_json())

    text = _hero_cost(_client(tmp_path).get(f'/r/{report_id(path)}').text)

    assert 'Total cost $0.5000' in text
    assert 'Total cost $0.5000 (' not in text


def test_sim_header_is_unknown_when_no_conversation_has_a_cost(tmp_path: Path) -> None:
    path = tmp_path / 'sim.json'
    _write_pre_total_sim_run(path, [_sim_result('alice', usage=_legacy_usage(None)), _sim_result('bob', usage=TokenUsage())])

    (row,) = metrics.sim_run_rows([tmp_path])
    text = _hero_cost(_client(tmp_path).get(f'/r/{report_id(path)}').text)

    assert row.cost is None
    assert 'Total cost unknown' in text


def test_rt_header_shows_the_known_part_of_a_summary_without_a_total(tmp_path: Path) -> None:
    usage = TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120, total_cost=0.3, calls=2, priced_calls=2)
    results = [
        _make_rt_result(attack_id='ASI01-1', category='ASI01', execution=ExecutionDetails(turns=1, token_usage=usage)),
        _make_rt_result(attack_id='ASI01-2', category='ASI01', execution=ExecutionDetails(turns=1, token_usage=usage)),
        _make_rt_result(attack_id='ASI01-3', category='ASI01', execution=ExecutionDetails(turns=1)),
    ]
    report = RedTeamReport(
        created_at=datetime(2026, 11, 1, tzinfo=timezone.utc),
        description='rt',
        pipeline='dynamic',
        categories_tested=['ASI01'],
        tested_agents=['agent-a'],
        total_results=3,
        results=results,
        summary=compute_report_summary(results),
    )
    data = report.model_dump(mode='json')
    data['summary']['token_usage_total'] = None  # a summary saved before the field existed
    path = tmp_path / 'rt.json'
    path.write_text(json.dumps(data))

    (row,) = metrics.redteam_overview([tmp_path]).recent
    text = _hero_cost(_client(tmp_path).get(f'/r/{report_id(path)}').text)

    assert row.cost is not None
    assert f'Total cost {fmt_cost(row.cost)} (2 of 3 attacks)' in text
