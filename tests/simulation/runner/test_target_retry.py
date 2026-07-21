"""Per-turn target retry + error tracking for SimulationRunner.

Verifies:
- A rich ``target_agent`` whose ``respond()`` returns an error marker on the
  first attempt is retried and, on eventual success, the run does NOT terminate
  as an error.
- When the target keeps failing past ``max_target_retries``, the run terminates
  with ``terminated_by == error``, the failed turn IS appended (an assistant
  message is present), and ``metadata['error']`` is populated.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.contracts import AgentResponse, AgentResponseError, Message, TokenUsage
from evaluatorq.simulation.runner.simulation import SimulationRunner
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Persona,
    Scenario,
    SimulationDatapoint,
    TerminatedBy,
)

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Shared helpers (mirror tests/simulation/test_target_model.py conventions)
# ---------------------------------------------------------------------------


def _make_datapoint() -> SimulationDatapoint:
    persona = Persona(
        name='Retry Tester',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='Testing target retry',
    )
    scenario = Scenario(name='Retry Scenario', goal='Verify retry behaviour')
    return SimulationDatapoint(
        id='dp-target-retry-001',
        persona=persona,
        scenario=scenario,
        user_system_prompt='system',
        first_message='Hello, can you help me?',
    )


def _make_mock_judgment() -> MagicMock:
    j = MagicMock()
    j.should_terminate = True
    j.goal_achieved = True
    j.goal_completion_score = 1.0
    j.rules_broken = []
    j.reason = 'Done'
    j.response_quality = 0.9
    j.hallucination_risk = 0.1
    j.tone_appropriateness = 0.9
    j.factual_accuracy = 0.9
    return j


def _make_mock_user_simulator() -> MagicMock:
    sim = MagicMock()
    sim.generate_first_message = AsyncMock(return_value='Hello')
    sim.respond_async = AsyncMock(return_value='thanks')
    sim.get_usage = MagicMock(return_value=TokenUsage())
    return sim


def _make_mock_judge() -> MagicMock:
    j = MagicMock()
    j.evaluate = AsyncMock(return_value=_make_mock_judgment())
    j.get_usage = MagicMock(return_value=TokenUsage())
    return j


class _FlakyTarget:
    """Returns an error-marker AgentResponse for the first ``fail_times`` calls."""

    def __init__(self, fail_times: int) -> None:
        self._fail = fail_times
        self.calls = 0

    async def respond(self, messages: list[Message]) -> AgentResponse:
        self.calls += 1
        if self._fail > 0:
            self._fail -= 1
            return AgentResponse(
                text='[ERROR]',
                error=AgentResponseError(message='boom', error_type='target_error'),
            )
        return AgentResponse(text='ok')


async def test_target_retries_then_succeeds() -> None:
    target = _FlakyTarget(fail_times=1)
    runner = SimulationRunner(
        target_agent=target,
        max_target_retries=2,
        target_agent_timeout_ms=5000,
        max_turns=1,
        user_simulator=_make_mock_user_simulator(),
        judge=_make_mock_judge(),
    )

    result = await runner.run(datapoint=_make_datapoint())

    assert target.calls == 2
    assert result.terminated_by != TerminatedBy.error, result.reason


async def test_target_exhausted_terminates_with_error_and_failed_turn() -> None:
    target = _FlakyTarget(fail_times=99)
    runner = SimulationRunner(
        target_agent=target,
        max_target_retries=1,
        target_agent_timeout_ms=5000,
        max_turns=3,
        user_simulator=_make_mock_user_simulator(),
        judge=_make_mock_judge(),
    )

    result = await runner.run(datapoint=_make_datapoint())

    assert result.terminated_by == TerminatedBy.error
    # The failed turn IS recorded: an assistant message is present.
    assert any(m.role == 'assistant' for m in result.messages)
    assert result.metadata.get('error')
