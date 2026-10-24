"""Simulation overview keeps goal outcomes and recorded spend distinct."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from evaluatorq.dashboard import metrics, view
from evaluatorq.dashboard.app import build_app


def _store(tmp_path: Path, results: list[dict], *, run_fields: dict[str, object] | None = None) -> list[Path]:
    sim = tmp_path / 'sim-runs'
    sim.mkdir()
    payload: dict[str, object] = {
        'mode': 'run', 'run_name': 'Goal test', 'created_at': '2026-10-24T10:00:00',
        'total_results': len(results), 'scorer_averages': {'quality': 1.0}, 'results': results,
    }
    payload.update(run_fields or {})
    (sim / 'run.json').write_text(json.dumps(payload))
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
    assert data.recent[0].cost == pytest.approx(6.0)
    html = view.sim_overview_body(data)
    assert 'Total spend' in html and '$6.00' in html
    assert '4 of 6 calls priced' in html
    assert 'Avg cost/sim' not in html


def test_run_cost_round_trip_uses_whole_run_usage_for_spend_and_coverage(tmp_path: Path) -> None:
    from datetime import datetime

    from evaluatorq.contracts import TokenUsage
    from evaluatorq.simulation.contracts import SimulationResult, SimulationRun, TerminatedBy
    from evaluatorq.simulation.reports.sections import build_report_sections

    result = SimulationResult(
        messages=[],
        terminated_by=TerminatedBy.judge,
        reason='done',
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=TokenUsage(
            total_tokens=2,
            input_tokens=1,
            output_tokens=1,
            input_cost=0.1,
            output_cost=0.15,
            cost_usd=0.25,
            calls=1,
            priced_calls=1,
        ),
        turn_metrics=[],
    )
    run = SimulationRun(
        run_name='Generated simulation',
        created_at=datetime(2026, 10, 24),
        mode='run',
        target_kind='callback',
        evaluator_names=[],
        total_results=1,
        scorer_averages={},
        results=[result],
        token_usage_total=TokenUsage(
            total_tokens=10,
            input_tokens=5,
            output_tokens=5,
            input_cost=0.5,
            output_cost=0.75,
            cost_usd=1.25,
            calls=2,
            priced_calls=1,
        ),
    )
    saved = json.loads(run.model_dump_json())
    roots = _store(tmp_path, saved['results'], run_fields={'token_usage_total': saved['token_usage_total']})

    overview = metrics.sim_overview(roots)
    report_sections = build_report_sections(run.results, run_token_usage_total=run.token_usage_total)
    report_usage = next(section.data for section in report_sections if section.kind == 'token_usage')
    html = view.sim_overview_body(overview)
    landing = metrics.landing(roots)

    assert report_usage['run_total_cost'] == pytest.approx(1.25)
    assert (report_usage['run_total_priced_calls'], report_usage['run_total_calls']) == (1, 2)
    assert report_usage['run_total_total_tokens'] == 10
    assert report_usage['run_total_input_tokens'] == 5
    assert report_usage['run_total_output_tokens'] == 5
    assert overview.total_cost == pytest.approx(1.25)
    assert overview.recent[0].cost == pytest.approx(1.25)
    assert overview.avg_cost == pytest.approx(0.25)
    assert overview.avg_input_cost == pytest.approx(0.1)
    assert overview.avg_output_cost == pytest.approx(0.15)
    assert (overview.priced_calls, overview.cost_calls) == (1, 2)
    assert '1 of 2 calls priced' in html
    assert landing.total_cost == pytest.approx(1.25)
    assert landing.total_tokens == 10
    assert landing.total_input_cost == pytest.approx(0.5)
    assert landing.total_output_cost == pytest.approx(0.75)
    assert landing.priced_calls == 1 and landing.cost_calls == 2


def test_landing_keeps_missing_whole_run_cost_components_unknown(tmp_path: Path) -> None:
    roots = _store(
        tmp_path,
        [
            {
                'goal_achieved': True,
                'token_usage': {'total_tokens': 2, 'input_cost': 0.1, 'output_cost': 0.15, 'cost_usd': 0.25},
            }
        ],
        run_fields={
            'token_usage_total': {'total_tokens': 10, 'input_cost': None, 'output_cost': 0.75, 'cost_usd': 1.25}
        },
    )

    landing = metrics.landing(roots)

    assert landing.total_tokens == 10
    assert landing.total_input_cost is None
    assert landing.total_output_cost == pytest.approx(0.75)


def test_landing_legacy_sim_run_sums_result_tokens_and_cost_components(tmp_path: Path) -> None:
    roots = _store(
        tmp_path,
        [
            {
                'goal_achieved': True,
                'token_usage': {
                    'total_tokens': 2,
                    'input_cost': 0.1,
                    'output_cost': 0.15,
                    'cost_usd': 0.25,
                    'calls': 1,
                    'priced_calls': 1,
                },
            },
            {
                'goal_achieved': False,
                'token_usage': {
                    'total_tokens': 3,
                    'input_cost': 0.2,
                    'output_cost': 0.3,
                    'cost_usd': 0.5,
                    'calls': 2,
                    'priced_calls': 1,
                },
            },
        ],
    )

    landing = metrics.landing(roots)

    assert landing.total_tokens == 5
    assert landing.total_cost == pytest.approx(0.75)
    assert landing.total_input_cost == pytest.approx(0.3)
    assert landing.total_output_cost == pytest.approx(0.45)


@pytest.mark.parametrize(('run_cost', 'expected_total'), [(None, None), (0.0, 0.0)])
def test_run_cost_distinguishes_unknown_from_zero_when_result_cost_is_known(
    tmp_path: Path, run_cost: float | None, expected_total: float | None
) -> None:
    roots = _store(
        tmp_path,
        [{'goal_achieved': True, 'token_usage': {'cost_usd': 0.25, 'calls': 1, 'priced_calls': 1}}],
        run_fields={
            'token_usage_total': {'cost_usd': run_cost, 'calls': 1, 'priced_calls': int(run_cost is not None)}
        },
    )

    overview = metrics.sim_overview(roots)

    assert overview.total_cost == expected_total
    assert overview.recent[0].cost == expected_total
    assert overview.avg_cost == pytest.approx(0.25)


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


def test_all_error_row_label_names_missing_verdict_target_and_lifecycle() -> None:
    from datetime import datetime

    from evaluatorq.dashboard.metrics import SimRunRow

    row = SimRunRow(
        rid='sim-1',
        name='Repeated run',
        when=datetime(2026, 10, 24),
        targets=[('Agent A', 'agent')],
        status='error',
        score=None,
        cases=6,
        cost=None,
        error=False,
        achieved=0,
        errors=6,
    )

    html = view._sim_row(row)

    assert 'target Agent A, status error, no goal verdict available' in html
    assert '<span class="rg-broke-n is-none">—</span>' in html


@pytest.mark.parametrize(
    'error_metadata',
    [
        {'error': 'judge request timed out'},
        {'criteria_errors': ['criterion judge failed']},
    ],
)
def test_saved_metadata_errors_match_report_outcomes_after_round_trip(
    tmp_path: Path, error_metadata: dict[str, object]
) -> None:
    from evaluatorq.contracts import TokenUsage
    from evaluatorq.dashboard.report_tabs import _sim_outcomes_donut
    from evaluatorq.simulation.contracts import SimulationResult, TerminatedBy
    from evaluatorq.simulation.reports.sections import build_report_sections

    result = SimulationResult(
        messages=[],
        terminated_by=TerminatedBy.judge,
        reason='Judge finished the conversation.',
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        turn_metrics=[],
        metadata=error_metadata,
    )
    saved_result = json.loads(json.dumps(result.model_dump(mode='json')))
    roots = _store(tmp_path, [saved_result])

    overview = metrics.sim_overview(roots)
    summary = next(section.data for section in build_report_sections([result]) if section.kind == 'summary')
    donut = _sim_outcomes_donut([result])

    assert overview.goal_completion is None
    assert (overview.achieved, overview.not_achieved, overview.errors) == (0, 0, 1)
    assert overview.recent[0].achieved == 0
    assert overview.recent[0].errors == 1
    assert summary['goals_achieved'] == 0
    assert summary['errors'] == 1
    assert 'Errors · 1' in donut
    assert 'no goal verdict' in donut


def test_zero_cost_is_known_and_out_of_range_page_keeps_rows(tmp_path: Path) -> None:
    roots = _store(tmp_path, [{'goal_achieved': True, 'token_usage': {'cost_usd': 0.0}}])
    data = metrics.sim_overview(roots, page=999)
    assert data.page == 1 and len(data.recent) == 1
    assert data.total_cost == 0.0
    assert '$0.0000' in view.sim_overview_body(data)
