"""`repetitions` runs every simulation case several times, each as a separate fresh conversation.

A case is one persona and scenario pair. Repeating it is not generating more
cases and it is not scoring one transcript several times.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from evaluatorq.simulation import generate_and_simulate, simulate
from evaluatorq.simulation.contracts import (
    CommunicationStyle,
    Persona,
    Scenario,
    SimulationDatapoint,
    SimulationResult,
    TerminatedBy,
)
from evaluatorq.simulation.hooks import DefaultHooks, SimulationRunMeta, _conversation_plan
from evaluatorq.simulation.replay import load_simulation_replay
from evaluatorq.simulation.run_config import GenerateAndSimulateRunConfig, SimulateRunConfig
from evaluatorq.simulation.utils.run_store import get_sim_runs_dir
from tests.simulation.test_hooks import _StubJudge, _StubUserSim

if TYPE_CHECKING:
    from pathlib import Path


def _case(case_id: str) -> SimulationDatapoint:
    persona = Persona(
        name=f'persona-{case_id}',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='background',
    )
    return SimulationDatapoint(
        id=case_id,
        persona=persona,
        scenario=Scenario(name=f'scenario-{case_id}', goal='goal'),
        user_system_prompt='',
        first_message='hi',
    )


async def _fine(_messages: Any) -> str:
    return 'fine'


async def _run(**overrides: Any) -> list[SimulationResult]:
    kwargs: dict[str, Any] = {
        'target': _fine,
        'max_turns': 1,
        'evaluator_names': ['goal_achieved'],
        'user_simulator': cast('Any', _StubUserSim()),
        'judge': cast('Any', _StubJudge(terminate=True)),
        'upload_results': False,
        'executive_summary': False,
        'recommendations': False,
    }
    kwargs.update(overrides)
    return await simulate(**kwargs)


def _lineage(results: list[SimulationResult]) -> list[tuple[str, str, int]]:
    return [(r.metadata['datapoint_id'], r.metadata['base_datapoint_id'], r.metadata['trial_index']) for r in results]


@pytest.mark.asyncio
async def test_the_default_runs_each_case_once() -> None:
    results = await _run(datapoints=[_case('a'), _case('b')])

    assert _lineage(results) == [('a', 'a', 0), ('b', 'b', 0)], 'one conversation per case, under the case id'


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', [0, -1, True, False, 2.5, 2.0, '3', None])
async def test_a_count_that_is_not_a_whole_number_of_at_least_one_is_rejected(bad: Any) -> None:
    calls: list[int] = []

    async def target(messages: Any) -> str:
        calls.append(len(messages))
        return 'fine'

    with pytest.raises(ValueError, match='repetitions must be an integer of at least 1'):
        await _run(datapoints=[_case('a')], repetitions=bad, target=target)
    assert calls == [], 'nothing ran'


@pytest.mark.asyncio
async def test_a_bad_count_is_rejected_before_generation_is_paid_for() -> None:
    with (
        patch('evaluatorq.simulation.api._generate_datapoints_inner', new_callable=AsyncMock) as generate,
        pytest.raises(ValueError, match='repetitions must be an integer of at least 1'),
    ):
        await generate_and_simulate(target=_fine, agent_description='An agent.', repetitions=0, upload_results=False)
    generate.assert_not_awaited()


@pytest.mark.parametrize('model', [SimulateRunConfig, GenerateAndSimulateRunConfig])
@pytest.mark.parametrize('bad', [0, -1, True, 2.5, '3'])
def test_the_cli_config_rejects_the_same_counts(model: Any, bad: Any) -> None:
    with pytest.raises(ValueError, match='repetitions'):
        model(repetitions=bad)
    assert model().repetitions == 1


@pytest.mark.asyncio
async def test_one_case_repeated_three_times_is_three_separate_conversations() -> None:
    case = _case('a')
    before = case.model_dump()
    histories: list[int] = []

    async def target(messages: Any) -> str:
        histories.append(len(messages))
        return 'fine'

    results = await _run(datapoints=[case], repetitions=3, target=target)

    assert _lineage(results) == [('a#t0', 'a', 0), ('a#t1', 'a', 1), ('a#t2', 'a', 2)]
    assert len({r.thread_id for r in results}) == 3, 'each conversation has its own thread'
    assert len(histories) == 3, 'the target was really called once per conversation'
    assert len(set(histories)) == 1, 'every conversation started from the same fresh history'
    assert {(r.metadata['persona'], r.metadata['scenario']) for r in results} == {('persona-a', 'scenario-a')}
    assert case.model_dump() == before, 'the case passed in is not modified'


@pytest.mark.asyncio
async def test_six_generated_cases_repeated_ten_times_are_sixty_conversations_from_one_generation() -> None:
    cases = [_case(f'c{i}') for i in range(6)]
    with patch(
        'evaluatorq.simulation.api._generate_datapoints_inner',
        new_callable=AsyncMock,
        return_value=(cases, None, False, None),
    ) as generate:
        results = await generate_and_simulate(
            target=_fine,
            agent_description='Handle IT incidents and ticket updates.',
            num_personas=3,
            num_scenarios=2,
            repetitions=10,
            max_turns=1,
            evaluator_names=['goal_achieved'],
            user_simulator=cast('Any', _StubUserSim()),
            judge=cast('Any', _StubJudge(terminate=True)),
            upload_results=False,
            executive_summary=False,
            recommendations=False,
        )

    generate.assert_awaited_once()
    assert len(results) == 60
    assert len({r.metadata['datapoint_id'] for r in results}) == 60
    per_case: dict[str, list[int]] = {}
    for result in results:
        per_case.setdefault(result.metadata['base_datapoint_id'], []).append(result.metadata['trial_index'])
    assert per_case == {f'c{i}': list(range(10)) for i in range(6)}


@pytest.mark.asyncio
async def test_parallelism_bounds_all_conversations_together() -> None:
    in_flight = 0
    peak = 0

    async def target(_messages: Any) -> str:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return 'fine'

    results = await _run(datapoints=[_case('a'), _case('b')], repetitions=4, target=target, datapoint_parallelism=2)

    assert len(results) == 8
    assert peak == 2, 'two cases of four trials each never ran more than two conversations at once'


@pytest.mark.asyncio
async def test_a_retried_target_call_does_not_add_a_conversation() -> None:
    attempts: list[int] = []

    async def flaky(messages: Any) -> str:
        attempts.append(len(messages))
        if len(attempts) % 2 == 1:
            raise ConnectionError('transient')
        return 'fine'

    results = await _run(datapoints=[_case('a')], repetitions=3, target=flaky, max_target_retries=2, datapoint_parallelism=1)

    assert len(attempts) == 6, 'every conversation failed once and was retried'
    assert [r.metadata['trial_index'] for r in results] == [0, 1, 2]
    assert all(r.terminated_by is not TerminatedBy.error for r in results)


@pytest.mark.asyncio
async def test_a_failed_conversation_stays_in_the_results_as_a_failure() -> None:
    conversations = 0

    async def second_one_fails(_messages: Any) -> str:
        nonlocal conversations
        conversations += 1
        if conversations == 2:
            raise ValueError('target broke')
        return 'fine'

    results = await _run(
        datapoints=[_case('a')],
        repetitions=3,
        target=second_one_fails,
        max_target_retries=0,
        datapoint_parallelism=1,
        raise_on_execution_failure=False,
    )

    assert _lineage(results) == [('a#t0', 'a', 0), ('a#t1', 'a', 1), ('a#t2', 'a', 2)]
    assert [r.terminated_by is TerminatedBy.error for r in results] == [False, True, False]
    assert results[1].goal_achieved is False, 'a failed conversation is never counted as a success'


@pytest.mark.asyncio
async def test_the_run_plan_shows_the_total_and_hooks_see_one_id_per_conversation() -> None:
    plans: list[SimulationRunMeta] = []
    started: list[str] = []

    class Recorder(DefaultHooks):
        async def on_confirm(self, meta: SimulationRunMeta) -> bool:
            plans.append(meta)
            return True

        async def on_datapoint_start(self, datapoint: SimulationDatapoint) -> None:
            started.append(datapoint.id)

    await _run(datapoints=[_case('a'), _case('b')], repetitions=3, hooks=Recorder())

    assert plans[0]['num_datapoints'] == 6
    assert plans[0].get('repetitions') == 3
    assert _conversation_plan(plans[0]) == '2 datapoints x 3 repetitions = 6 conversations'
    assert sorted(started) == ['a#t0', 'a#t1', 'a#t2', 'b#t0', 'b#t1', 'b#t2']


def test_the_run_plan_for_one_repetition_reads_as_before() -> None:
    meta = cast('SimulationRunMeta', {'num_datapoints': 4})
    assert _conversation_plan(meta) == '4 datapoints'


@pytest.mark.asyncio
async def test_a_saved_repeated_run_stores_the_cases_once_and_replays_without_multiplying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    await _run(datapoints=[_case('a'), _case('b')], repetitions=3, save=True)

    runs_dir = get_sim_runs_dir()
    saved = json.loads(next(runs_dir.glob('*.json')).read_text(encoding='utf-8'))
    assert saved['repetitions'] == 3
    assert saved['total_results'] == 6
    assert [dp['id'] for dp in saved['datapoints']] == ['a', 'b'], 'the base cases, not the six trials'
    assert [dp.id for dp in load_simulation_replay('latest', runs_dir).datapoints] == ['a', 'b']

    replayed = await _run(previous_run='latest')
    assert _lineage(replayed) == [('a', 'a', 0), ('b', 'b', 0)], 'a replay runs each case once by default'

    repeated_again = await _run(previous_run='latest', repetitions=2)
    assert len(repeated_again) == 4, 'two cases twice, not six trials twice'


@pytest.mark.asyncio
async def test_repetitions_of_a_case_share_one_heatmap_cell_with_its_count() -> None:
    from evaluatorq.simulation.reports.sections import build_report_sections

    results = await _run(datapoints=[_case('a'), _case('b')], repetitions=3)

    heatmap = next(s for s in build_report_sections(results) if s.kind == 'persona_scenario_heatmap').data
    assert len(heatmap['cells']) == 2
    assert {cell['n'] for cell in heatmap['cells']} == {3}
    assert {cell['completed_conversations'] for cell in heatmap['cells']} == {3}


def test_the_dashboard_heatmap_shows_n_only_where_a_cell_averages_several_conversations() -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs
    from tests.dashboard.test_report_tabs import _make_sim_run

    # alice has two conversations in one cell (a pass and a fail); bob has one.
    sim_run = _make_sim_run(personas=['alice', 'alice', 'bob'], goal_achieved_flags=[True, False, True])

    for html in (sim_report_tabs('rid', sim_run), sim_report_tabs('rid', sim_run, standalone=True)):
        heatmap = html.split('class="rk-heatmap"', 1)[1].split('</table>', 1)[0]
        assert heatmap.count('>n=2<') == 1
        assert 'n=1' not in heatmap, 'a single conversation is not presented as a sample'
        assert '2 conversations · 2 completed · 0 errored' in heatmap
        assert 'goal achieved in 50%' in heatmap, 'the binary rate is named apart from the mean score'


def test_the_cli_offers_the_option_on_both_commands() -> None:
    import typer
    from typer.testing import CliRunner

    from evaluatorq.simulation.cli import app as sim_app

    # Read the option from the command definition: help text is styled differently per terminal.
    group = typer.main.get_command(sim_app)
    for command in ('run', 'simulate'):
        options = {opt for param in group.commands[command].params for opt in param.opts}  # type: ignore[attr-defined]
        assert '--repetitions' in options
        rejected = CliRunner().invoke(sim_app, [command, '--repetitions', '0'])
        assert rejected.exit_code == 2, 'zero is a usage error'
