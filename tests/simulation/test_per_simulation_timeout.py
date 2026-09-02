"""simulate()'s per-row path bypasses SimulationRunner.run_batch entirely, so
its own timeout wrapper (_run_with_timeout) never fired for a caller of
simulate() -- a stalled conversation had no wall-clock bound beyond per-call
LLM/target timeouts (F4). per_simulation_timeout_s closes that gap.
"""

from __future__ import annotations

import asyncio
from typing import Any

# ruff: noqa: S101
import pytest

from evaluatorq.simulation.api import simulate
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Judgment,
    Persona,
    Scenario,
    SimulationDatapoint,
    TerminatedBy,
    TokenUsage as _TU,
)


class _StubUserSim:
    def update_context(self, *, persona_context=None, scenario_context=None) -> None:  # noqa: ANN001
        pass

    async def generate_first_message(self) -> str:
        return "hello"

    async def respond_async(self, messages, *, llm_purpose=None) -> str:  # noqa: ANN001
        return "more"

    def reset_usage(self) -> None:
        pass

    def get_usage(self) -> _TU:
        return _TU(prompt_tokens=0, completion_tokens=0, total_tokens=0)


class _StubJudge:
    async def evaluate(self, messages) -> Judgment:  # noqa: ANN001
        return Judgment(
            should_terminate=False,
            reason="stub",
            goal_achieved=False,
            rules_broken=[],
            goal_completion_score=0.0,
        )

    def reset_usage(self) -> None:
        pass

    def get_usage(self) -> _TU:
        return _TU(prompt_tokens=0, completion_tokens=0, total_tokens=0)


def _datapoint() -> SimulationDatapoint:
    return SimulationDatapoint(
        id="dp1",
        persona=Persona(
            name="p",
            patience=0.5,
            assertiveness=0.5,
            politeness=0.5,
            technical_level=0.5,
            communication_style=CommunicationStyle.casual,
            background="b",
        ),
        scenario=Scenario(name="s", goal="g"),
        user_system_prompt="You are a user.",
        first_message="Hello",
    )


async def _slow_target(messages: list[Any]) -> str:
    await asyncio.sleep(5)
    return "too slow"  # pragma: no cover


async def _fast_target(messages: list[Any]) -> str:
    return "ok"


@pytest.mark.asyncio
async def test_per_simulation_timeout_s_unset_leaves_simulate_unaffected(monkeypatch):
    """Without per_simulation_timeout_s, simulate() keeps today's behaviour
    exactly: a normal-speed conversation runs to max_turns, not to a timeout."""
    results = await simulate(
        datapoints=[_datapoint()],
        target=_fast_target,
        max_turns=1,
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(),  # pyright: ignore[reportArgumentType]
        upload_results=False,
        executive_summary=False,
    )
    assert len(results) == 1
    assert results[0].terminated_by == TerminatedBy.max_turns


@pytest.mark.asyncio
async def test_per_simulation_timeout_s_terminates_a_stalled_conversation(monkeypatch):
    """A per_simulation_timeout_s shorter than the stalled target's own timeout
    must still cut the conversation off -- proof _run_with_timeout is actually
    reached from simulate()'s per-row job_fn path."""
    results = await simulate(
        datapoints=[_datapoint()],
        target=_slow_target,
        max_turns=1,
        # Target's own timeout is generous; only the per-simulation wall clock
        # should be able to cut this off in time for the test to be fast.
        target_agent_timeout_ms=60_000,
        max_target_retries=0,
        per_simulation_timeout_s=0.2,
        user_simulator=_StubUserSim(),  # pyright: ignore[reportArgumentType]
        judge=_StubJudge(),  # pyright: ignore[reportArgumentType]
        upload_results=False,
        executive_summary=False,
        # The subject here is the wall clock, not the exit gate: a timed-out row
        # raises under the default exit_on_failure=True.
        exit_on_failure=False,
    )
    assert len(results) == 1
    assert results[0].terminated_by == TerminatedBy.timeout
