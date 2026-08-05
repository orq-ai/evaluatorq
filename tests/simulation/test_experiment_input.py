"""Tests for Orq experiments as simulation input (direct + extension)."""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from evaluatorq.simulation.api import _resolve_or_generate_datapoints
from evaluatorq.simulation.cli import app
from evaluatorq.simulation.experiments import (
    _seed_context,
    datapoints_from_experiment,
    extend_from_experiment,
)
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Persona,
    Scenario,
    SimulationDatapoint,
)
from evaluatorq.types import DataPoint

runner = CliRunner()


def _sim_datapoint(persona_name: str = 'P', scenario_name: str = 'S') -> SimulationDatapoint:
    persona = Persona(
        name=persona_name,
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.terse,
        background='bg',
    )
    scenario = Scenario(name=scenario_name, goal='get a refund', context='webshop')
    return SimulationDatapoint(
        id=f'{persona_name}-{scenario_name}',
        persona=persona,
        scenario=scenario,
        user_system_prompt='',
        first_message='hi',
    )


def _experiment_rows() -> list[DataPoint]:
    """Rows in the two shapes an experiment realistically carries: a prior sim
    upload (``datapoint``) and a hand-built persona+scenario row. The fetcher
    also appends a ``messages`` key, which the extractor must ignore."""
    dp = _sim_datapoint()
    return [
        DataPoint(inputs={'datapoint': dp.model_dump(mode='json'), 'messages': [{'role': 'assistant', 'content': 'x'}]}),
        DataPoint(
            inputs={
                'persona': dp.persona.model_dump(mode='json'),
                'scenario': dp.scenario.model_dump(mode='json'),
                'first_message': 'hello',
                'messages': [],
            }
        ),
    ]


def _patch_fetch(monkeypatch: pytest.MonkeyPatch, rows: list[DataPoint]) -> dict[str, Any]:
    calls: dict[str, Any] = {}

    async def fake_fetch(api_key: str, experiment_id: str, run_id: str | None = None, **kwargs: Any) -> list[DataPoint]:
        calls.update(api_key=api_key, experiment_id=experiment_id, run_id=run_id)
        return rows

    monkeypatch.setattr('evaluatorq.fetch_data.fetch_experiment_datapoints', fake_fetch)
    return calls


# ---------------------------------------------------------------------------
# direct mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_direct_extracts_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_fetch(monkeypatch, _experiment_rows())
    out = await datapoints_from_experiment('ex_1', run_id='run_9', api_key='key')
    assert [dp.persona.name for dp in out] == ['P', 'P']
    assert out[1].first_message == 'hello'
    assert calls == {'api_key': 'key', 'experiment_id': 'ex_1', 'run_id': 'run_9'}


@pytest.mark.asyncio
async def test_direct_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('ORQ_API_KEY', raising=False)
    with pytest.raises(ValueError, match='ORQ_API_KEY'):
        await datapoints_from_experiment('ex_1')


@pytest.mark.asyncio
async def test_direct_bad_row_names_index(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fetch(monkeypatch, [DataPoint(inputs={'question': 'not a sim shape'})])
    with pytest.raises(ValueError, match="'ex_1' row 0"):
        await datapoints_from_experiment('ex_1', api_key='key')


@pytest.mark.asyncio
async def test_direct_zero_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fetch(monkeypatch, [])
    with pytest.raises(ValueError, match='zero simulation-compatible rows'):
        await datapoints_from_experiment('ex_1', api_key='key')


@pytest.mark.asyncio
async def test_direct_multi_element_row_error_does_not_blame_wrap_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    # A row carrying two datapoints must fail with a message about the row,
    # not the unrelated wrap_simulation_agent contract the extractor also serves.
    dp = _sim_datapoint().model_dump(mode='json')
    _patch_fetch(monkeypatch, [DataPoint(inputs={'datapoints': [dp, dp]})])
    with pytest.raises(ValueError, match="'ex_1' row 0.*row must encode exactly one datapoint, got 2") as exc:
        await datapoints_from_experiment('ex_1', api_key='key')
    assert 'wrap_simulation_agent' not in str(exc.value)


# ---------------------------------------------------------------------------
# simulate() wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolver_uses_experiment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_API_KEY', 'key')
    _patch_fetch(monkeypatch, _experiment_rows())
    out = await _resolve_or_generate_datapoints(
        caller='simulate',
        datapoints=None,
        personas=None,
        scenarios=None,
        dataset_id=None,
        experiment_id='ex_1',
        experiment_run_id=None,
        model='m',
        generation_client=None,
    )
    assert len(out) == 2


@pytest.mark.asyncio
async def test_resolver_experiment_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match='exactly one of'):
        await _resolve_or_generate_datapoints(
            caller='simulate',
            datapoints=None,
            personas=None,
            scenarios=None,
            dataset_id='ds_1',
            experiment_id='ex_1',
            experiment_run_id=None,
            model='m',
            generation_client=None,
        )


@pytest.mark.asyncio
async def test_resolver_run_id_requires_experiment_id() -> None:
    with pytest.raises(ValueError, match="'experiment_run_id' requires 'experiment_id'"):
        await _resolve_or_generate_datapoints(
            caller='simulate',
            datapoints=[_sim_datapoint()],
            personas=None,
            scenarios=None,
            dataset_id=None,
            experiment_id=None,
            experiment_run_id='run_9',
            model='m',
            generation_client=None,
        )


# ---------------------------------------------------------------------------
# extension mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extend_seeds_generators(monkeypatch: pytest.MonkeyPatch) -> None:
    # Cartesian-product seeds: 2 personas x 1 scenario, so dedup must collapse
    # the repeated scenario in the generator context.
    rows = [
        DataPoint(inputs={'datapoint': _sim_datapoint('Alice', 'Refund').model_dump(mode='json')}),
        DataPoint(inputs={'datapoint': _sim_datapoint('Bob', 'Refund').model_dump(mode='json')}),
    ]
    _patch_fetch(monkeypatch, rows)

    captured: dict[str, Any] = {}

    class FakeGenerator:
        def __init__(self, **kwargs: Any) -> None:
            captured['model'] = kwargs.get('model')

        async def generate_from_description(self, **kwargs: Any) -> list[SimulationDatapoint]:
            captured.update(kwargs)
            return [_sim_datapoint('New', 'Fresh')]

        async def close(self) -> None:
            captured['closed'] = True

    monkeypatch.setattr('evaluatorq.simulation.generators.DatapointGenerator', FakeGenerator)

    out = await extend_from_experiment('ex_1', num_personas=2, num_scenarios=3, api_key='key')

    assert [dp.id for dp in out] == ['New-Fresh']
    assert captured['num_personas'] == 2
    assert captured['num_scenarios'] == 3
    assert captured['closed'] is True
    assert 'get a refund' in captured['agent_description']
    context = captured['context']
    assert 'Alice' in context and 'Bob' in context
    assert context.count('Refund:') == 1  # deduped scenario
    assert 'NEW personas' in context


def test_describe_agent_falls_back_when_all_goals_blank() -> None:
    from evaluatorq.simulation.experiments import _describe_agent

    dp = _sim_datapoint()
    dp = dp.model_copy(update={'scenario': dp.scenario.model_copy(update={'goal': ''})})
    description = _describe_agent([dp])
    assert description  # not the bare truncated 'goals such as: ' prefix
    assert not description.rstrip().endswith(':')


def test_seed_context_dedupes() -> None:
    seeds = [_sim_datapoint('A', 'S1'), _sim_datapoint('A', 'S2')]
    context = _seed_context(seeds)
    assert context.count('- A: bg') == 1
    assert 'S1' in context and 'S2' in context


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'args',
    [
        ['-i', 'x.jsonl', '--experiment-id', 'ex'],
        ['--dataset-id', 'ds', '--experiment-id', 'ex'],
        ['--experiment-run-id', 'run', '-i', 'x.jsonl'],
    ],
)
def test_cli_experiment_flag_validation(args: list[str]) -> None:
    result = runner.invoke(app, ['simulate', *args])
    assert result.exit_code != 0
