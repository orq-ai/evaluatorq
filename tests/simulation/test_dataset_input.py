"""Tests for Orq datasets as simulation input (direct + extension).

Direct mode is the canonical loader `simulate(dataset_id=...)` resolves through; extension mode
(`extend_from_dataset`) seeds the generators from the dataset's personas/scenarios. The Orq fetch is
mocked at ``fetch_data`` so no network or key is needed.
"""

from __future__ import annotations

from typing import Any

import pytest

from evaluatorq.contracts import LLMCallConfig
from evaluatorq.simulation.api import _resolve_or_generate_datapoints
from evaluatorq.simulation.datasets import datapoints_from_dataset, extend_from_dataset
from evaluatorq.simulation.types import CommunicationStyle, Persona, Scenario, SimulationDatapoint
from evaluatorq.types import DataPoint


def _sim_datapoint(persona_name: str = 'P', scenario_name: str = 'S') -> SimulationDatapoint:
    persona = Persona(
        name=persona_name,
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='bg',
    )
    scenario = Scenario(name=scenario_name, goal='get a refund')
    return SimulationDatapoint(
        id=f'{persona_name}-{scenario_name}',
        persona=persona,
        scenario=scenario,
        user_system_prompt='',
        first_message='hi',
    )


def _dataset_rows() -> list[DataPoint]:
    """The two row shapes a dataset realistically carries: a prior sim upload (``datapoint``) and a
    hand-built persona+scenario row. A trailing ``messages`` key must be ignored by the extractor."""
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


class _Batch:
    def __init__(self, datapoints: list[DataPoint]) -> None:
        self.datapoints = datapoints


def _patch_fetch(monkeypatch: pytest.MonkeyPatch, rows: list[DataPoint]) -> dict[str, Any]:
    """Patch the dataset batch fetch; record the key and dataset id it was called with."""
    calls: dict[str, Any] = {}

    def fake_setup(api_key: str) -> object:
        calls['api_key'] = api_key
        return object()

    async def fake_batches(client: object, dataset_id: str) -> Any:
        calls['dataset_id'] = dataset_id
        yield _Batch(rows)

    monkeypatch.setattr('evaluatorq.fetch_data.setup_orq_client', fake_setup)
    monkeypatch.setattr('evaluatorq.fetch_data.fetch_dataset_batches', fake_batches)
    return calls


# ---------------------------------------------------------------------------
# direct mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_direct_extracts_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_fetch(monkeypatch, _dataset_rows())
    out = await datapoints_from_dataset('ds_1', api_key='key')
    assert [dp.persona.name for dp in out] == ['P', 'P']
    assert out[1].first_message == 'hello'
    assert calls == {'api_key': 'key', 'dataset_id': 'ds_1'}


@pytest.mark.asyncio
async def test_direct_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('ORQ_API_KEY', raising=False)
    with pytest.raises(ValueError, match='ORQ_API_KEY'):
        await datapoints_from_dataset('ds_1')


@pytest.mark.asyncio
async def test_direct_bad_row_names_index(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fetch(monkeypatch, [DataPoint(inputs={'question': 'not a sim shape'})])
    with pytest.raises(ValueError, match="'ds_1' row 0"):
        await datapoints_from_dataset('ds_1', api_key='key')


@pytest.mark.asyncio
async def test_direct_zero_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fetch(monkeypatch, [])
    with pytest.raises(ValueError, match='zero simulation-compatible datapoints'):
        await datapoints_from_dataset('ds_1', api_key='key')


# ---------------------------------------------------------------------------
# simulate() wiring — the private fetch now delegates to datapoints_from_dataset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolver_uses_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('ORQ_API_KEY', 'key')
    _patch_fetch(monkeypatch, _dataset_rows())
    out = await _resolve_or_generate_datapoints(
        caller='simulate',
        datapoints=None,
        personas=None,
        scenarios=None,
        dataset_id='ds_1',
        experiment_id=None,
        experiment_run_id=None,
        llm_config=LLMCallConfig(model='m'),
        generation_client=None,
    )
    assert len(out) == 2


# ---------------------------------------------------------------------------
# extension mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extend_seeds_generators(monkeypatch: pytest.MonkeyPatch) -> None:
    # Cartesian-product seeds: 2 personas x 1 scenario, so dedup must collapse the repeated scenario.
    rows = [
        DataPoint(inputs={'datapoint': _sim_datapoint('Alice', 'Refund').model_dump(mode='json')}),
        DataPoint(inputs={'datapoint': _sim_datapoint('Bob', 'Refund').model_dump(mode='json')}),
    ]
    _patch_fetch(monkeypatch, rows)

    captured: dict[str, Any] = {}

    class FakeGenerator:
        def __init__(self, **kwargs: Any) -> None:
            captured['config'] = kwargs.get('config')

        async def generate_from_description(self, **kwargs: Any) -> list[SimulationDatapoint]:
            captured.update(kwargs)
            return [_sim_datapoint('New', 'Fresh')]

        async def close(self) -> None:
            captured['closed'] = True

    monkeypatch.setattr('evaluatorq.simulation.generators.DatapointGenerator', FakeGenerator)

    out = await extend_from_dataset('ds_1', num_personas=2, num_scenarios=3, api_key='key')

    assert [dp.id for dp in out] == ['New-Fresh']
    assert captured['num_personas'] == 2
    assert captured['num_scenarios'] == 3
    assert captured['closed'] is True
    assert 'get a refund' in captured['agent_description']
    context = captured['context']
    assert 'Alice' in context and 'Bob' in context
    assert context.count('Refund:') == 1  # deduped scenario
    assert 'NEW personas' in context
