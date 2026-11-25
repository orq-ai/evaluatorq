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

from evaluatorq.contracts import SaveMode
from evaluatorq.contracts import AgentResponse, AgentTarget
from evaluatorq.simulation import generate_and_simulate, simulate
from evaluatorq.simulation.contracts import (
    CommunicationStyle,
    Message,
    Persona,
    Scenario,
    SimulationDatapoint,
    SimulationResult,
    SimulationRun,
    TerminatedBy,
)
from evaluatorq.simulation.hooks import DefaultHooks, SimulationRunMeta, _conversation_count, _conversation_plan
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
    return [(r.metadata['datapoint_id'], r.metadata['base_datapoint_id'], r.metadata['repetition_index']) for r in results]


@pytest.mark.asyncio
async def test_the_default_runs_each_case_once() -> None:
    results = await _run(datapoints=[_case('a'), _case('b')])

    assert _lineage(results) == [('a', 'a', 0), ('b', 'b', 0)], 'one conversation per case, under the case id'


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', [0, -1, True, False, 2.5, 2.0, '3'])
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
    with pytest.raises(ValueError, match='repetitions must be an integer of at least 1'):
        model(repetitions=bad)
    assert model().repetitions is None, 'unset, so a replay can restore its own count'
    assert model.model_json_schema()['properties']['repetitions']['anyOf'][0] == {'type': 'integer', 'minimum': 1}


@pytest.mark.asyncio
async def test_one_case_repeated_three_times_is_three_separate_conversations() -> None:
    case = _case('a')
    before = case.model_dump()
    histories: list[int] = []

    async def target(messages: Any) -> str:
        histories.append(len(messages))
        return 'fine'

    results = await _run(datapoints=[case], repetitions=3, target=target)

    assert _lineage(results) == [('a#r0', 'a', 0), ('a#r1', 'a', 1), ('a#r2', 'a', 2)]
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
        per_case.setdefault(result.metadata['base_datapoint_id'], []).append(result.metadata['repetition_index'])
    assert per_case == {f'c{i}': list(range(10)) for i in range(6)}


@pytest.mark.asyncio
async def test_parallelism_bounds_all_conversations_together() -> None:
    in_flight = 0
    peak = 0
    both_slots_filled = asyncio.Event()

    async def target(_messages: Any) -> str:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        if in_flight == 2:
            both_slots_filled.set()
        # The first conversation holds its slot until a second one is running, so
        # reaching two at once does not depend on how long a sleep happens to last.
        await asyncio.wait_for(both_slots_filled.wait(), timeout=5)
        in_flight -= 1
        return 'fine'

    results = await _run(datapoints=[_case('a'), _case('b')], repetitions=4, target=target, datapoint_parallelism=2)

    assert len(results) == 8
    assert both_slots_filled.is_set(), 'two conversations did run at once'
    assert peak <= 2, 'two cases of four repetitions each never ran more than two conversations at once'


class _CountingTarget(AgentTarget):
    """Records every copy `new()` makes and which copy answered each call."""

    def __init__(self, *, registry: list[_CountingTarget], answered_by: list[int]) -> None:
        super().__init__()
        self._registry = registry
        self._answered_by = answered_by
        registry.append(self)

    async def respond(self, messages: list[Message]) -> AgentResponse:
        self._answered_by.append(id(self))
        return AgentResponse(text='fine')

    def new(self) -> _CountingTarget:
        return type(self)(registry=self._registry, answered_by=self._answered_by)


@pytest.mark.asyncio
async def test_every_repetition_talks_to_its_own_copy_of_the_target() -> None:
    registry: list[_CountingTarget] = []
    answered_by: list[int] = []
    original = _CountingTarget(registry=registry, answered_by=answered_by)

    results = await _run(datapoints=[_case('a')], repetitions=3, target=original)

    assert len(results) == 3
    assert len(answered_by) == 3
    assert len(set(answered_by)) == 3, 'each conversation was answered by a different copy'
    assert id(original) not in answered_by, 'the target passed in is never shared between conversations'


@pytest.mark.asyncio
async def test_a_case_id_that_already_looks_like_a_repetition_id_is_rejected_when_repeating() -> None:
    with pytest.raises(ValueError, match=r"cannot already end in #r<digits>.*\['a#r1'\]"):
        await _run(datapoints=[_case('a'), _case('a#r1')], repetitions=2)

    results = await _run(datapoints=[_case('a#r1')])
    assert _lineage(results) == [('a#r1', 'a#r1', 0)], 'with one repetition the id is never rewritten, so it is allowed'


@pytest.mark.asyncio
async def test_uploaded_experiment_rows_tell_the_repetitions_of_a_case_apart() -> None:
    import importlib

    evaluatorq_module = importlib.import_module('evaluatorq.evaluatorq')
    real = evaluatorq_module.evaluatorq
    rows: list[dict[str, Any]] = []

    async def recording(*args: Any, **kwargs: Any) -> Any:
        rows.extend(dp.inputs for dp in kwargs['data'])
        return await real(*args, **kwargs)

    with patch.object(evaluatorq_module, 'evaluatorq', recording):
        await _run(datapoints=[_case('a')], repetitions=2)
        await _run(datapoints=[_case('b')])

    assert [(r['datapoint']['id'], r.get('base_datapoint_id'), r.get('repetition_index')) for r in rows] == [
        ('a', 'a', 0),
        ('a', 'a', 1),
        ('b', None, None),
    ], 'a repeated case labels its repetitions; a single run keeps the row it always had'


@pytest.mark.asyncio
async def test_run_batch_stamps_the_same_lineage_as_simulate() -> None:
    from evaluatorq.simulation.runner.simulation import SimulationRunner

    async def boom(_messages: Any) -> str:
        raise ValueError('target down')

    for target in (_fine, boom):
        runner = SimulationRunner(
            target=target,
            model='gpt-4o-mini',
            max_turns=1,
            user_simulator=cast('Any', _StubUserSim()),
            judge=cast('Any', _StubJudge(terminate=True)),
        )
        results = await runner.run_batch([_case('a'), _case('b')], max_concurrency=1)
        assert _lineage(results) == [('a', 'a', 0), ('b', 'b', 0)]


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
    assert [r.metadata['repetition_index'] for r in results] == [0, 1, 2]
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

    assert _lineage(results) == [('a#r0', 'a', 0), ('a#r1', 'a', 1), ('a#r2', 'a', 2)]
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

    assert plans[0]['num_cases'] == 2, 'num_cases counts cases, not conversations'
    assert plans[0].get('num_conversations') == 6
    assert plans[0].get('repetitions') == 3
    assert _conversation_plan(plans[0]) == '2 cases x 3 repetitions = 6 conversations'
    assert sorted(started) == ['a#r0', 'a#r1', 'a#r2', 'b#r0', 'b#r1', 'b#r2']


def test_the_run_plan_for_one_repetition_reads_as_before() -> None:
    meta = cast('SimulationRunMeta', {'num_cases': 4})
    assert _conversation_plan(meta) == '4 cases'
    assert _conversation_count(meta) == 4


def test_a_payload_without_the_conversation_count_still_counts_every_conversation() -> None:
    meta = cast('SimulationRunMeta', {'num_cases': 2, 'repetitions': 3})
    assert _conversation_count(meta) == 6
    assert _conversation_plan(meta) == '2 cases x 3 repetitions = 6 conversations'


@pytest.mark.asyncio
async def test_the_run_span_counts_cases_and_conversations_apart() -> None:
    from evaluatorq.common import tracing

    stamped: dict[str, Any] = {}
    real = tracing.set_span_attrs

    def recording(*args: Any, **kwargs: Any) -> None:
        stamped.update(kwargs.get('attrs') or {})
        real(*args, **kwargs)

    with patch.object(tracing, 'set_span_attrs', recording):
        await _run(datapoints=[_case('a'), _case('b')], repetitions=3)

    assert stamped['orq.simulation.cases_count'] == 2
    assert stamped['orq.simulation.conversations_count'] == 6
    assert stamped['orq.simulation.repetitions'] == 3


@pytest.mark.asyncio
async def test_a_saved_repeated_run_stores_the_cases_once_and_replays_with_its_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    await _run(datapoints=[_case('a'), _case('b')], repetitions=3, save=SaveMode.FINAL)

    runs_dir = get_sim_runs_dir()
    saved = json.loads(next(runs_dir.glob('*.json')).read_text(encoding='utf-8'))
    assert saved['repetitions'] == 3
    assert saved['total_results'] == 6
    assert [dp['id'] for dp in saved['datapoints']] == ['a', 'b'], 'the base cases, not the six conversations'
    replay = load_simulation_replay('latest', runs_dir=runs_dir)
    assert [dp.id for dp in replay.datapoints] == ['a', 'b']
    assert replay.repetitions == 3

    replayed = await _run(previous_run='latest')
    assert len(replayed) == 6, 'a replay restores the saved count, like it restores the turn limit'
    assert {r.metadata['base_datapoint_id'] for r in replayed} == {'a', 'b'}, 'the base cases, not repetitions of repetitions'

    once = await _run(previous_run='latest', repetitions=1)
    assert _lineage(once) == [('a', 'a', 0), ('b', 'b', 0)], 'an explicit count still wins'


def test_a_run_saved_before_repetitions_replays_each_case_once(tmp_path: Path) -> None:
    payload = {'datapoints': [_case('a').model_dump(mode='json')], 'max_turns': 2}
    (tmp_path / 'old.json').write_text(json.dumps(payload), encoding='utf-8')

    assert load_simulation_replay('old.json', runs_dir=tmp_path).repetitions is None


@pytest.mark.asyncio
async def test_a_run_saved_by_1_97_0_with_trial_index_still_loads_and_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    await _run(datapoints=[_case('a')], repetitions=2, save=SaveMode.FINAL)
    path = next(get_sim_runs_dir().glob('*.json'))
    # Rewrite the saved run into the 1.97.0 shape: `trial_index` and `#t<index>` ids.
    legacy = path.read_text(encoding='utf-8').replace('"repetition_index"', '"trial_index"').replace('a#r', 'a#t')
    assert '"trial_index"' in legacy
    path.write_text(legacy, encoding='utf-8')

    run = SimulationRun.model_validate_json(legacy)
    assert [(r.metadata['datapoint_id'], r.metadata['repetition_index']) for r in run.results] == [('a#t0', 0), ('a#t1', 1)]
    assert all('trial_index' not in r.metadata for r in run.results)

    replayed = await _run(previous_run='latest')
    assert _lineage(replayed) == [('a#r0', 'a', 0), ('a#r1', 'a', 1)]


@pytest.mark.asyncio
async def test_repetitions_of_a_case_share_one_heatmap_cell_with_its_count() -> None:
    from evaluatorq.simulation.reports.sections import build_report_sections

    results = await _run(datapoints=[_case('a'), _case('b')], repetitions=3)

    heatmap = next(s for s in build_report_sections(results) if s.kind == 'persona_scenario_heatmap').data
    assert len(heatmap['cells']) == 2
    assert {cell['n'] for cell in heatmap['cells']} == {3}
    assert {cell['completed_conversations'] for cell in heatmap['cells']} == {3}
    assert {cell['goal_achieved_conversations'] for cell in heatmap['cells']} == {
        sum(r.goal_achieved for r in results) // 2
    }


def test_the_markdown_report_prints_passed_k_of_n_only_for_a_repeated_cell() -> None:
    from evaluatorq.contracts import ReportSection
    from evaluatorq.simulation.reports.export_md import _render_persona_scenario_heatmap_section
    from evaluatorq.simulation.reports.sections import achieved_of_n

    section = ReportSection(
        kind='persona_scenario_heatmap',
        title='Persona x Scenario Success',
        data={
            'personas': ['alice', 'bob'],
            'scenarios': ['refund'],
            'cells': [
                {'persona': 'alice', 'scenario': 'refund', 'success_rate': 0.7, 'n': 10, 'goal_achieved_conversations': 7},
                {'persona': 'bob', 'scenario': 'refund', 'success_rate': 1.0, 'n': 1, 'goal_achieved_conversations': 1},
            ],
        },
    )

    rendered = _render_persona_scenario_heatmap_section(section)
    assert '70% (7 of 10)' in rendered
    assert '100% |' in rendered and '(1 of 1)' not in rendered, 'one conversation is not presented as a sample'
    assert achieved_of_n({'n': 4, 'success_rate': 0.5}) == '(2 of 4)', 'a cell saved without the count still reads'


def test_the_dashboard_heatmap_shows_n_only_where_a_cell_averages_several_conversations() -> None:
    from evaluatorq.dashboard.report_tabs import sim_report_tabs
    from tests.dashboard.test_report_tabs import _make_sim_run

    # alice has two conversations in one cell (a pass and a fail); bob has one.
    sim_run = _make_sim_run(personas=['alice', 'alice', 'bob'], goal_achieved_flags=[True, False, True])

    for html in (sim_report_tabs(rid='rid', run=sim_run), sim_report_tabs(rid='rid', run=sim_run, standalone=True)):
        heatmap = html.split('class="rk-heatmap"', 1)[1].split('</table>', 1)[0]
        assert heatmap.count('>n=2<') == 1
        assert 'n=1' not in heatmap, 'a single conversation is not presented as a sample'
        assert '2 conversations · 2 completed · 0 errored' in heatmap
        assert 'goal achieved in 1 of 2 (50%)' in heatmap, 'the binary rate is named apart from the mean score'


def test_the_cli_offers_the_option_on_both_commands() -> None:
    import click
    import typer
    from typer.testing import CliRunner

    from evaluatorq.simulation.cli import app as sim_app

    # Read the option from the command definition: help text is styled differently per terminal.
    group = typer.main.get_command(sim_app)
    assert isinstance(group, click.Group)
    for command in ('run', 'simulate'):
        options = {opt for param in group.commands[command].params for opt in param.opts}
        assert '--repetitions' in options
        rejected = CliRunner().invoke(sim_app, args=[command, '--repetitions', '0'])
        assert rejected.exit_code == 2, 'zero is a usage error'
