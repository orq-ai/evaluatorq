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

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.contracts import (
    AgentResponse,
    AgentResponseError,
    AgentTarget,
    Message,
    TokenUsage,
)
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


class _FlakyTarget(AgentTarget):
    """Returns an error-marker AgentResponse for the first ``fail_times`` calls.

    The fail budget and call counter live in a shared dict so the
    per-conversation clone the runner mints (``new()``) keeps counting against
    the same state — these tests assert retry behavior, not clone isolation.
    """

    def __init__(self, fail_times: int, *, _state: dict[str, int] | None = None) -> None:
        super().__init__()
        self._state = _state if _state is not None else {'fail': fail_times, 'calls': 0}

    @property
    def calls(self) -> int:
        return self._state['calls']

    def new(self) -> _FlakyTarget:
        return _FlakyTarget(0, _state=self._state)

    async def respond(self, messages: list[Message]) -> AgentResponse:
        self._state['calls'] += 1
        if self._state['fail'] > 0:
            self._state['fail'] -= 1
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


def _make_continue_judgment() -> MagicMock:
    """Judge verdict that never terminates -- keeps the loop going to turn 2."""
    j = _make_mock_judgment()
    j.should_terminate = False
    j.goal_achieved = False
    j.goal_completion_score = 0.0
    j.reason = 'continue'
    return j


def _make_continue_judge() -> MagicMock:
    j = MagicMock()
    j.evaluate = AsyncMock(return_value=_make_continue_judgment())
    j.get_usage = MagicMock(return_value=TokenUsage())
    return j


def _make_usage_tracking_user_simulator() -> MagicMock:
    state = {'usage': TokenUsage()}
    sim = MagicMock()

    def add_usage(usage: TokenUsage) -> None:
        state['usage'] = state['usage'] + usage

    async def generate_first_message() -> str:
        add_usage(TokenUsage(input_tokens=2, output_tokens=1, total_tokens=3))
        return 'generated first message'

    async def respond_async(messages: list[Message], *, llm_purpose: str | None = None) -> str:
        add_usage(TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2))
        return 'continue'

    sim.generate_first_message = AsyncMock(side_effect=generate_first_message)
    sim.respond_async = AsyncMock(side_effect=respond_async)
    sim.get_usage = MagicMock(side_effect=lambda: state['usage'])
    sim.reset_usage = MagicMock(side_effect=lambda: state.update(usage=TokenUsage()))
    return sim


def _make_usage_tracking_judge() -> MagicMock:
    state = {'usage': TokenUsage()}
    judge = MagicMock()

    async def evaluate(messages: list[Message]) -> MagicMock:
        state['usage'] = state['usage'] + TokenUsage(input_tokens=3, output_tokens=1, total_tokens=4)
        return _make_continue_judgment()

    judge.evaluate = AsyncMock(side_effect=evaluate)
    judge.get_usage = MagicMock(side_effect=lambda: state['usage'])
    judge.reset_usage = MagicMock(side_effect=lambda: state.update(usage=TokenUsage()))
    return judge


class _HangSecond(AgentTarget):
    """Answers the first turn quickly, then hangs forever on the second.

    The first response carries non-zero token usage so the timeout test can
    verify accrued usage is not discarded (RES bug hunt task 11).
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def new(self) -> _HangSecond:
        return _HangSecond()

    async def respond(self, messages: list[Message]) -> AgentResponse:
        self.calls += 1
        if self.calls == 1:
            return AgentResponse(
                text='first answer',
                usage=TokenUsage(input_tokens=5, output_tokens=7, total_tokens=12),
            )
        await asyncio.sleep(100)
        raise AssertionError('unreachable')  # pragma: no cover


async def test_outer_timeout_retains_partial_transcript() -> None:
    target = _HangSecond()
    runner = SimulationRunner(
        target_agent=target,
        max_target_retries=0,
        target_agent_timeout_ms=60000,
        user_simulator=_make_mock_user_simulator(),
        judge=_make_continue_judge(),
    )

    result = await runner._run_with_timeout(_make_datapoint(), max_turns=5, timeout_s=0.3)

    assert result.terminated_by == TerminatedBy.timeout
    # Partial transcript retained: the turn-1 assistant answer survived the outer
    # cancellation instead of being discarded as an empty list.
    assert result.messages != []
    assert any(m.content == 'first answer' for m in result.messages)

    # RES bug hunt task 11: a timed-out run must not zero out the usage accrued
    # before the outer cancellation, and must surface as an error rather than a
    # clean run (goals_failed).
    assert result.turn_metrics, 'expected the completed turn to be recorded'
    expected_usage = sum((tm.token_usage for tm in result.turn_metrics), start=TokenUsage())
    assert expected_usage.total_tokens > 0, 'test fixture must produce non-zero usage'
    assert result.token_usage == expected_usage
    assert result.metadata.get('error')
    assert result.metadata.get('error_type') == 'timeout'


async def test_outer_timeout_keeps_total_usage_beyond_completed_turns() -> None:
    target = _HangSecond()
    runner = SimulationRunner(
        target_agent=target,
        max_target_retries=0,
        target_agent_timeout_ms=60000,
        user_simulator=_make_usage_tracking_user_simulator(),
        judge=_make_usage_tracking_judge(),
    )

    datapoint = _make_datapoint().model_copy(update={'first_message': ''})
    result = await runner._run_with_timeout(datapoint, max_turns=5, timeout_s=0.3)

    assert result.terminated_by == TerminatedBy.timeout
    assert result.token_usage == TokenUsage(input_tokens=11, output_tokens=10, total_tokens=21)
    assert result.token_usage != sum((tm.token_usage for tm in result.turn_metrics), start=TokenUsage())


class _AlwaysHang(AgentTarget):
    def new(self) -> _AlwaysHang:
        return _AlwaysHang()

    async def respond(self, messages: list[Message]) -> AgentResponse:
        await asyncio.sleep(100)
        raise AssertionError('unreachable')  # pragma: no cover


async def test_target_timeout_uses_canonical_marker_and_warns_without_criteria(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = SimulationRunner(
        target_agent=_AlwaysHang(),
        max_target_retries=0,
        target_agent_timeout_ms=1,
        max_turns=1,
        user_simulator=_make_mock_user_simulator(),
        judge=_make_mock_judge(),
    )

    with caplog.at_level('WARNING', logger='evaluatorq.simulation.runner.simulation'):
        result = await runner.run(datapoint=_make_datapoint())

    assert result.terminated_by == TerminatedBy.timeout
    assert result.metadata.get('error_type') == 'timeout'
    assert any('Target failed (timeout)' in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# Callback target (``target=``) — wrapped internally via CallableTarget.
# ---------------------------------------------------------------------------


def _sync_callback(messages: list[Message]) -> str:
    return 'sync ok'


async def _async_callback(messages: list[Message]) -> str:
    return 'async ok'


async def _awaitable_callback_response() -> str:
    return 'awaited ok'


def _sync_callback_returning_awaitable(messages: list[Message]):
    return _awaitable_callback_response()


class _RaisingCallback:
    """Raises on every call so the retry helper exhausts and the run errors."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, messages: list[Message]) -> str:
        self.calls += 1
        raise RuntimeError('callback exploded')


async def test_sync_callback_target_succeeds() -> None:
    runner = SimulationRunner(
        target=_sync_callback,
        max_target_retries=1,
        target_agent_timeout_ms=5000,
        max_turns=1,
        user_simulator=_make_mock_user_simulator(),
        judge=_make_mock_judge(),
    )

    result = await runner.run(datapoint=_make_datapoint())

    assert result.terminated_by != TerminatedBy.error, result.reason
    assert any(m.role == 'assistant' and m.content == 'sync ok' for m in result.messages)


async def test_async_callback_target_succeeds() -> None:
    runner = SimulationRunner(
        target=_async_callback,
        max_target_retries=1,
        target_agent_timeout_ms=5000,
        max_turns=1,
        user_simulator=_make_mock_user_simulator(),
        judge=_make_mock_judge(),
    )

    result = await runner.run(datapoint=_make_datapoint())

    assert result.terminated_by != TerminatedBy.error, result.reason
    assert any(m.role == 'assistant' and m.content == 'async ok' for m in result.messages)


async def test_sync_callback_returning_awaitable_is_awaited() -> None:
    runner = SimulationRunner(
        target=_sync_callback_returning_awaitable,
        max_target_retries=1,
        target_agent_timeout_ms=5000,
        max_turns=1,
        user_simulator=_make_mock_user_simulator(),
        judge=_make_mock_judge(),
    )

    result = await runner.run(datapoint=_make_datapoint())

    assert result.terminated_by != TerminatedBy.error, result.reason
    assert any(m.role == 'assistant' and m.content == 'awaited ok' for m in result.messages)


async def test_raising_callback_target_retried_then_terminates_with_error() -> None:
    callback = _RaisingCallback()
    runner = SimulationRunner(
        target=callback,
        max_target_retries=1,
        target_agent_timeout_ms=5000,
        max_turns=3,
        user_simulator=_make_mock_user_simulator(),
        judge=_make_mock_judge(),
    )

    result = await runner.run(datapoint=_make_datapoint())

    assert callback.calls == 2  # 1 initial attempt + 1 retry
    assert result.terminated_by == TerminatedBy.error
    assert any(m.role == 'assistant' for m in result.messages)
    assert result.metadata.get('error')
