"""Replaying a prior simulation run (`previous_run=` / `--from-run`)."""
# ruff: noqa: S101

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from evaluatorq.common.replay import ReplayError
from evaluatorq.simulation.replay import load_simulation_replay
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Criterion,
    Persona,
    Scenario,
    SimulationDatapoint,
    StartingEmotion,
)
from evaluatorq.simulation.utils.run_store import auto_save_run, build_simulation_run, get_sim_runs_dir

if TYPE_CHECKING:
    from pathlib import Path


def _datapoint(name: str = 'dp-1') -> SimulationDatapoint:
    persona = Persona(
        name='Impatient Ien',
        patience=2,
        assertiveness=8,
        politeness=4,
        technical_level=6,
        communication_style=CommunicationStyle.terse,
        background='Long-time customer with a stalled refund.',
    )
    scenario = Scenario(
        name='Refund chase',
        goal='Get the refund released today',
        context='Third contact about the same order.',
        starting_emotion=StartingEmotion.frustrated,
        criteria=[Criterion(description='Agent states a refund date', type='must_happen')],
    )
    return SimulationDatapoint(
        id=name,
        persona=persona,
        scenario=scenario,
        user_system_prompt='',
        first_message='Where is my refund?',
    )


def test_saved_run_carries_its_datapoints(tmp_path: Path) -> None:
    dp = _datapoint()
    run = build_simulation_run(
        run_name='sim',
        mode='simulate',
        target_kind='callback',
        evaluator_names=[],
        results=[],
        datapoints=[dp],
    )
    path = auto_save_run(run=run, run_name='sim')

    stored = json.loads(path.read_text(encoding='utf-8'))
    assert stored['datapoints'][0]['id'] == 'dp-1'

    replayed = load_simulation_replay('latest', get_sim_runs_dir())
    assert replayed.datapoints == [dp]


def test_replay_resolves_by_run_name(tmp_path: Path) -> None:
    run = build_simulation_run(
        run_name='nightly',
        mode='simulate',
        target_kind='callback',
        evaluator_names=[],
        results=[],
        datapoints=[_datapoint('dp-a'), _datapoint('dp-b')],
    )
    auto_save_run(run=run, run_name='nightly')

    replayed = load_simulation_replay('nightly', get_sim_runs_dir())
    assert [dp.id for dp in replayed.datapoints] == ['dp-a', 'dp-b']


def test_run_saved_without_datapoints_cannot_be_replayed(tmp_path: Path) -> None:
    run = build_simulation_run(
        run_name='legacy',
        mode='simulate',
        target_kind='callback',
        evaluator_names=[],
        results=[],
    )
    auto_save_run(run=run, run_name='legacy')

    with pytest.raises(ReplayError, match='records no simulation datapoints'):
        load_simulation_replay('latest', get_sim_runs_dir())


def test_missing_run_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match='Could not resolve previous simulation run'):
        load_simulation_replay('does-not-exist', get_sim_runs_dir())


@pytest.mark.asyncio
async def test_previous_run_short_circuits_other_sources(tmp_path: Path) -> None:
    from evaluatorq.simulation.api import _resolve_or_generate_datapoints

    dp = _datapoint()
    run = build_simulation_run(
        run_name='sim',
        mode='simulate',
        target_kind='callback',
        evaluator_names=[],
        results=[],
        datapoints=[dp],
    )
    auto_save_run(run=run, run_name='sim')

    resolved = await _resolve_or_generate_datapoints(
        caller='simulate',
        datapoints=None,
        personas=None,
        scenarios=None,
        dataset_id=None,
        previous_run='latest',
        model='openai/gpt-5.4-mini',
        generation_client=None,
    )
    assert resolved == [dp]


@pytest.mark.asyncio
async def test_previous_run_is_mutually_exclusive(tmp_path: Path) -> None:
    from evaluatorq.simulation.api import _resolve_or_generate_datapoints

    with pytest.raises(ValueError, match='Pass exactly one of previous_run'):
        await _resolve_or_generate_datapoints(
            caller='simulate',
            datapoints=[_datapoint()],
            personas=None,
            scenarios=None,
            dataset_id=None,
            previous_run='latest',
            model='openai/gpt-5.4-mini',
            generation_client=None,
        )
