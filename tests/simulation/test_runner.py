"""Tests for SimulationRunner."""

import pytest

from evaluatorq.simulation.runner.simulation import (
    SimulationRunner,
    _invert_roles_for_simulator,
)
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Message,
    Persona,
    Scenario,
    TerminatedBy,
)


def _make_persona():
    return Persona(
        name="Test User",
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background="A test user",
    )


def _make_scenario():
    return Scenario(name="Test Scenario", goal="Get help")


class TestSimulationRunnerValidation:
    def test_target_callback_is_rejected(self):
        with pytest.raises(TypeError, match="target_callback"):
            SimulationRunner(target_callback=lambda msgs: "ok")  # pyright: ignore[reportCallIssue]

    def test_requires_target(self):
        with pytest.raises(ValueError, match="Must provide either"):
            SimulationRunner(model="test")

    def test_max_turns_validation(self):
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            SimulationRunner(
                target=lambda msgs: "ok",
                max_turns=0,
            )

    def test_model_validation(self):
        with pytest.raises(ValueError, match="model must be a non-empty"):
            SimulationRunner(
                target=lambda msgs: "ok",
                model="   ",
            )


class TestSimulationRunnerRun:
    @pytest.mark.asyncio
    async def test_run_requires_persona_or_datapoint(self):
        runner = SimulationRunner(target=lambda msgs: "ok")
        result = await runner.run(persona=_make_persona())
        assert result.terminated_by == TerminatedBy.error
        assert "Must provide either datapoint" in result.reason

    @pytest.mark.asyncio
    async def test_run_error_handling(self):
        """Runner should never throw, always return error result."""

        async def failing_callback(msgs):
            raise RuntimeError("API down")

        runner = SimulationRunner(target=failing_callback, model="test-model")
        result = await runner.run(
            persona=_make_persona(),
            scenario=_make_scenario(),
            first_message="Hello",
        )
        # Should get an error result (either from missing API key or actual error)
        assert result.terminated_by == TerminatedBy.error


class TestSimulationRunnerMisc:
    def test_accepts_valid_config(self):
        runner = SimulationRunner(
            target=lambda msgs: "ok",
            model="azure/gpt-4o-mini",
            max_turns=5,
        )
        assert runner is not None

    @pytest.mark.asyncio
    async def test_close_can_be_called_multiple_times(self):
        runner = SimulationRunner(target=lambda msgs: "ok")
        await runner.close()
        await runner.close()

    @pytest.mark.asyncio
    async def test_run_batch_empty_datapoints(self, monkeypatch):
        monkeypatch.setenv("ORQ_API_KEY", "test-key")
        runner = SimulationRunner(target=lambda msgs: "ok")
        results = await runner.run_batch([])
        assert results == []


class TestInvertRolesForSimulator:
    """Tests for _invert_roles_for_simulator helper."""

    def test_swaps_user_and_assistant(self):
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
            Message(role="user", content="Help me"),
        ]
        result = _invert_roles_for_simulator(messages)

        assert result[0] == Message(role="assistant", content="Hello")
        assert result[1] == Message(role="user", content="Hi there")
        assert result[2] == Message(role="assistant", content="Help me")

    def test_preserves_system_role(self):
        messages = [Message(role="system", content="You are helpful")]
        result = _invert_roles_for_simulator(messages)
        assert result[0] == Message(role="system", content="You are helpful")

    def test_empty_messages(self):
        assert _invert_roles_for_simulator([]) == []

    def test_preserves_tool_calls_on_assistant_message(self):
        """Regression: inversion must keep tool_calls/tool_call_id/name fields."""
        from evaluatorq.contracts import FunctionCall, StrategyToolCall

        tool_call = StrategyToolCall(
            id="call_123",
            function=FunctionCall(name="lookup", arguments='{"q": "x"}'),
        )
        messages = [
            Message(role="assistant", content="thinking", tool_calls=[tool_call]),
            Message(
                role="tool",
                content="result",
                tool_call_id="call_123",
                name="lookup",
            ),
        ]
        result = _invert_roles_for_simulator(messages)

        assert result[0].role == "user"
        assert result[0].tool_calls == [tool_call]
        assert result[1].role == "tool"
        assert result[1].tool_call_id == "call_123"
        assert result[1].name == "lookup"

    def test_message_with_tool_role_round_trips(self):
        """Tool-role messages with full superset fields survive serialize/deserialize."""
        msg = Message(
            role="tool",
            content="42",
            tool_call_id="call_abc",
            name="calculator",
        )
        dumped = msg.model_dump()
        rebuilt = Message.model_validate(dumped)
        assert rebuilt == msg
        assert rebuilt.role == "tool"
        assert rebuilt.tool_call_id == "call_abc"
        assert rebuilt.name == "calculator"


class TestSimulationRunnerBatchValidation:
    @pytest.mark.asyncio
    async def test_max_concurrency_validation(self):
        runner = SimulationRunner(target=lambda msgs: "ok")
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            await runner.run_batch([], max_concurrency=0)

    @pytest.mark.asyncio
    async def test_negative_max_concurrency_validation(self):
        runner = SimulationRunner(target=lambda msgs: "ok")
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            await runner.run_batch([], max_concurrency=-5)


class TestTargetUsageOnRetryAndFailure:
    """Billed target attempts must reach `SimulationResult.token_usage` exactly once.

    The run total is `user_simulator.get_usage() + judge.get_usage() +
    target_usage_acc`, and the target accumulator is fed from exactly one place
    in the turn loop — so asserting the total catches both a dropped attempt and
    a double-counted one.
    """

    JUDGE_USAGE = 2
    SIM_USAGE = 3

    def _runner(self, target, *, judge_turns: int):
        from unittest.mock import AsyncMock, MagicMock

        from evaluatorq.contracts import TokenUsage

        def _judgment(*, terminate: bool):
            j = MagicMock()
            j.should_terminate = terminate
            j.goal_achieved = terminate
            j.goal_completion_score = 1.0 if terminate else 0.5
            j.rules_broken = []
            j.reason = "done" if terminate else "keep going"
            j.response_quality = 0.9
            j.hallucination_risk = 0.1
            j.tone_appropriateness = 0.9
            j.factual_accuracy = 0.9
            return j

        judge = MagicMock()
        judge.evaluate = AsyncMock(
            side_effect=[_judgment(terminate=i == judge_turns - 1) for i in range(judge_turns)]
        )
        judge.get_usage = MagicMock(return_value=TokenUsage(total_tokens=self.JUDGE_USAGE, calls=1))

        user_sim = MagicMock()
        user_sim.generate_first_message = AsyncMock(return_value="Hi, I need help.")
        user_sim.respond_async = AsyncMock(return_value="ok thanks")
        user_sim.get_usage = MagicMock(return_value=TokenUsage(total_tokens=self.SIM_USAGE, calls=1))

        return SimulationRunner(
            target_agent=target,
            model="azure/gpt-4o-mini",
            max_turns=2,
            user_simulator=user_sim,
            judge=judge,
        )

    @staticmethod
    def _scripted_target(script):
        from evaluatorq.contracts import AgentTarget

        class _Scripted(AgentTarget):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            async def respond(self, messages):
                self.calls += 1
                return script[min(self.calls - 1, len(script) - 1)]

            def new(self):
                return self

        return _Scripted()

    @staticmethod
    def _billed_error(total: int):
        from evaluatorq.contracts import AgentResponse, AgentResponseError, TokenUsage

        return AgentResponse(
            text="[refused]",
            usage=TokenUsage(total_tokens=total, calls=1),
            error=AgentResponseError(message="refused", error_type="target_error", code="x"),
        )

    @pytest.mark.asyncio
    async def test_burned_retry_tokens_reach_the_run_total_once(self, monkeypatch):
        monkeypatch.setenv("ORQ_API_KEY", "test-key")
        from evaluatorq.contracts import AgentResponse, TokenUsage

        ok = AgentResponse(text="agent reply", usage=TokenUsage(total_tokens=11, calls=1))
        target = self._scripted_target([self._billed_error(7), ok])
        runner = self._runner(target, judge_turns=1)

        result = await runner.run(
            persona=_make_persona(), scenario=_make_scenario(), first_message="Hi"
        )

        assert target.calls == 2  # one refusal, one successful retry
        # 7 burned + 11 billed; adding the surviving response.usage on top of the
        # accumulator would give 29 here.
        assert result.token_usage.total_tokens == self.JUDGE_USAGE + self.SIM_USAGE + 18
        assert result.token_usage.calls == 2 + 2

    @pytest.mark.asyncio
    async def test_failed_final_turn_with_usage_is_billed(self, monkeypatch):
        """Every attempt refused, each one charged: the run total must show it
        rather than reporting the target as free."""
        monkeypatch.setenv("ORQ_API_KEY", "test-key")

        target = self._scripted_target([self._billed_error(5)])
        runner = self._runner(target, judge_turns=1)

        result = await runner.run(
            persona=_make_persona(), scenario=_make_scenario(), first_message="Hi"
        )

        assert result.terminated_by == TerminatedBy.error
        assert target.calls == 3  # 1 + 2 retries
        assert result.token_usage.total_tokens == self.JUDGE_USAGE + self.SIM_USAGE + 15
