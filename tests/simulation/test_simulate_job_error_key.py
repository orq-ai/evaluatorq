"""simulate()'s job must report a runner-handled failure through the top-level
``error`` key. The built-in scorers already fail such a row, but 'Failed Jobs'
counts JobResult.error, so without the key a dead run still reads as 0 failures.
"""

from __future__ import annotations

# ruff: noqa: S101
from typing import Any

import pytest

from evaluatorq.simulation.api import _build_simulation_job_and_cache
from evaluatorq.simulation.runner.simulation import _error_result
from evaluatorq.simulation.types import (
    CommunicationStyle,
    EmotionalArc,
    Message,
    Persona,
    Scenario,
    SimulationDatapoint,
    SimulationResult,
    StartingEmotion,
    TerminatedBy,
)
from evaluatorq.types import DataPoint


async def _stub_target(messages: list[Message]) -> str:
    return 'hi'


def _datapoint() -> SimulationDatapoint:
    persona = Persona(
        name='P',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.terse,
        background='bg',
        emotional_arc=EmotionalArc.stable,
    )
    scenario = Scenario(name='S', goal='g', context='c', starting_emotion=StartingEmotion.neutral, criteria=[])
    return SimulationDatapoint(id='dp1', persona=persona, scenario=scenario, user_system_prompt='', first_message='hi')


async def _run_job(result: SimulationResult) -> dict[str, Any]:
    data = DataPoint(inputs={'text': 'hi'})
    sim_dp = _datapoint()
    job_fn, _cache, runner = _build_simulation_job_and_cache(
        job_name='sim',
        sim_dp_by_id={id(data): sim_dp},
        target=_stub_target,
        target_agent=None,
        model='gpt-5.6-luna',
        max_turns=3,
        user_simulator=None,
        judge=None,
        generation_client=None,
        hooks=None,
    )

    async def fake_run(*_args: Any, **_kwargs: Any) -> SimulationResult:
        return result

    runner._run_with_timeout = fake_run  # type: ignore[method-assign]  # noqa: SLF001
    return await job_fn(data, 0)


@pytest.mark.asyncio
async def test_simulate_job_reports_a_runner_error() -> None:
    out = await _run_job(_error_result('401 authentication_error'))
    assert out['error'] == '401 authentication_error'
    # The transcript survives the failure, as on the wrap_simulation_agent path.
    assert out['output'] is not None


@pytest.mark.asyncio
async def test_simulate_job_emits_the_key_as_none_on_a_judged_run() -> None:
    result = _error_result('unused')
    result.terminated_by = TerminatedBy.judge
    result.reason = 'goal achieved'
    out = await _run_job(result)
    # Emitted, not omitted: a missing key and a clean run must not look the same.
    assert 'error' in out
    assert out['error'] is None
