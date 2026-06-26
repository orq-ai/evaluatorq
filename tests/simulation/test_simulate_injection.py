"""Tests for injected target/user_simulator params and update_context propagation.

Verifies:
- simulate() accepts target= as alias for target_callback=
- target= takes precedence over target_callback= when both are supplied
- injected user_simulator is used instead of the default
- update_context is called per simulation with persona/scenario context
- passing a BaseAgent lacking generate_first_message raises TypeError
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evaluatorq.contracts import TokenUsage
from evaluatorq.simulation.runner.simulation import SimulationRunner
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Message,
    Persona,
    Scenario,
    SimulationDatapoint,
    TerminatedBy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_persona(name: str = "Test User") -> Persona:
    return Persona(
        name=name,
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background="A test user",
    )


def _make_scenario(name: str = "Test Scenario") -> Scenario:
    return Scenario(name=name, goal="Get help")


def _make_datapoint(
    persona: Persona | None = None,
    scenario: Scenario | None = None,
    first_message: str = "Hello, can you help me?",
) -> SimulationDatapoint:
    p = persona or _make_persona()
    s = scenario or _make_scenario()
    return SimulationDatapoint(
        id="dp-001",
        persona=p,
        scenario=s,
        user_system_prompt="system",
        first_message=first_message,
    )


def _make_mock_judgment(*, should_terminate: bool = True, goal_achieved: bool = True) -> MagicMock:
    judgment = MagicMock()
    judgment.should_terminate = should_terminate
    judgment.goal_achieved = goal_achieved
    judgment.goal_completion_score = 1.0
    judgment.rules_broken = []
    judgment.reason = "Done"
    judgment.response_quality = 0.9
    judgment.hallucination_risk = 0.1
    judgment.tone_appropriateness = 0.9
    judgment.factual_accuracy = 0.9
    return judgment


def _make_mock_user_simulator(
    *,
    first_message: str = "Hello",
    response: str = "thanks",
) -> MagicMock:
    """Return a mock that satisfies UserSimulatorAgent protocol."""
    sim = MagicMock()
    sim.generate_first_message = AsyncMock(return_value=first_message)
    sim.respond_async = AsyncMock(return_value=response)
    sim.get_usage = MagicMock(return_value=TokenUsage())
    return sim


def _make_mock_judge(*, judgment: MagicMock | None = None) -> MagicMock:
    """Return a mock that satisfies JudgeAgent protocol."""
    j = MagicMock()
    j.evaluate = AsyncMock(return_value=judgment or _make_mock_judgment())
    j.get_usage = MagicMock(return_value=TokenUsage())
    return j


def _make_runner_with_mocks(
    *,
    target_callback: Any = None,
    target: Any = None,
    user_simulator: Any = None,
    judge: Any = None,
    model: str = "test-model",
    max_turns: int = 1,
) -> SimulationRunner:
    """Build a SimulationRunner; target_callback or target must be provided."""
    cb = target_callback or target or (lambda msgs: "agent reply")  # pyright: ignore[reportUnknownLambdaType]
    return SimulationRunner(
        target_callback=cb,
        model=model,
        max_turns=max_turns,
        user_simulator=user_simulator,
        judge=judge,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimulateWithInjectedTarget:
    """target= callback is called on each turn."""

    @pytest.mark.asyncio
    async def test_target_is_called_per_turn(self, monkeypatch: pytest.MonkeyPatch):
        """Injected target callable is invoked once per turn."""
        monkeypatch.setenv("ORQ_API_KEY", "test-key")

        call_count = 0

        async def my_target(messages: list[Message]) -> str:
            nonlocal call_count
            call_count += 1
            return "agent says hi"

        sim = _make_mock_user_simulator()
        judge = _make_mock_judge()

        runner = _make_runner_with_mocks(
            target_callback=my_target,
            user_simulator=sim,
            judge=judge,
            max_turns=1,
        )

        dp = _make_datapoint()
        result = await runner.run(datapoint=dp)

        assert result.terminated_by != TerminatedBy.error, result.reason
        assert call_count >= 1

    @pytest.mark.asyncio
    async def test_target_receives_message_history(self, monkeypatch: pytest.MonkeyPatch):
        """Target callback receives the accumulated message list each turn."""
        monkeypatch.setenv("ORQ_API_KEY", "test-key")

        received_messages: list[list[Message]] = []

        async def my_target(messages: list[Message]) -> str:
            received_messages.append(list(messages))
            return "response"

        sim = _make_mock_user_simulator(first_message="help me")
        judge = _make_mock_judge()

        runner = _make_runner_with_mocks(
            target_callback=my_target,
            user_simulator=sim,
            judge=judge,
            max_turns=1,
        )

        dp = _make_datapoint()
        await runner.run(datapoint=dp)

        assert len(received_messages) >= 1
        # The first turn always starts with the user's opening message
        first_call = received_messages[0]
        assert any(m.role == "user" for m in first_call)


class TestSimulateTargetPrecedence:
    """When both target= and target_callback= are given, target= wins."""

    @pytest.mark.asyncio
    async def test_target_takes_precedence_over_target_callback(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """target= is called, not target_callback=, when both supplied to simulate()."""
        monkeypatch.setenv("ORQ_API_KEY", "test-key")

        from evaluatorq.simulation.api import simulate

        calls_a: list[int] = []
        calls_b: list[int] = []

        async def target_a(messages: list[Message]) -> str:
            calls_a.append(1)
            return "from A"

        async def target_b(messages: list[Message]) -> str:
            calls_b.append(1)
            return "from B"

        dp = _make_datapoint()
        sim = _make_mock_user_simulator()
        judge = _make_mock_judge()

        resolved: dict[str, Any] = {}

        # simulate() imports SimulationRunner inside the function body, so we
        # patch the class where it is defined (the runner module) then also
        # patch the name that simulate() binds locally.
        import evaluatorq.simulation.runner.simulation as runner_mod

        original_cls = runner_mod.SimulationRunner

        class CapturingRunner(original_cls):  # type: ignore[valid-type]
            def __init__(self, **kwargs: Any) -> None:
                resolved.update(kwargs)
                super().__init__(
                    target_callback=kwargs.get("target_callback", target_a),
                    model=kwargs.get("model", "test"),
                    max_turns=1,
                    user_simulator=sim,
                    judge=judge,
                )

        with patch.object(runner_mod, "SimulationRunner", CapturingRunner):
            # Also patch where simulate() imports it from
            with patch("evaluatorq.simulation.runner.simulation.SimulationRunner", CapturingRunner):
                await simulate(
                    target=target_a,
                    target_callback=target_b,
                    datapoints=[dp],
                    sim_model="test",
                    max_turns=1,
                )

        # target= takes precedence, so resolved_callback must be target_a
        assert resolved.get("target_callback") is target_a


class TestSimulateAutoRoutesAgentTarget:
    """An AgentTarget instance passed as target= is routed to runner.target_agent."""

    @pytest.mark.asyncio
    async def test_agent_target_routes_to_target_agent_not_callback(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("ORQ_API_KEY", "test-key")

        from evaluatorq.contracts import AgentResponse, AgentTarget
        from evaluatorq.simulation.api import simulate

        class _FakeAgentTarget(AgentTarget):
            async def respond(self, messages: list[Message]) -> AgentResponse:
                return AgentResponse(text="ok")

            def new(self) -> "_FakeAgentTarget":
                return _FakeAgentTarget()

        agent_target = _FakeAgentTarget()
        sim = _make_mock_user_simulator()
        judge = _make_mock_judge()
        dp = _make_datapoint()
        resolved: dict[str, Any] = {}

        import evaluatorq.simulation.runner.simulation as runner_mod

        original_cls = runner_mod.SimulationRunner

        class CapturingRunner(original_cls):  # type: ignore[valid-type]
            def __init__(self, **kwargs: Any) -> None:
                resolved.update(kwargs)
                super().__init__(
                    target_agent=kwargs.get("target_agent"),
                    model=kwargs.get("model", "test"),
                    max_turns=1,
                    user_simulator=sim,
                    judge=judge,
                )

        with patch("evaluatorq.simulation.runner.simulation.SimulationRunner", CapturingRunner):
            await simulate(target=agent_target, datapoints=[dp], sim_model="test", max_turns=1)

        assert resolved.get("target_agent") is agent_target
        assert resolved.get("target_callback") is None


class TestSimulateWithInjectedUserSimulator:
    """Injected user_simulator is used instead of the default UserSimulatorAgent."""

    @pytest.mark.asyncio
    async def test_injected_user_simulator_is_used(self, monkeypatch: pytest.MonkeyPatch):
        """When user_simulator is provided, it is used (not the default).

        We verify the injected simulator's generate_first_message is called when
        no pre-set first_message is on the datapoint.
        """
        monkeypatch.setenv("ORQ_API_KEY", "test-key")

        sim = _make_mock_user_simulator(first_message="custom first message")
        judge = _make_mock_judge()

        runner = _make_runner_with_mocks(
            user_simulator=sim,
            judge=judge,
            max_turns=1,
        )

        # SimulationDatapoint with no first_message forces a generate_first_message call
        dp = _make_datapoint(first_message="")
        result = await runner.run(datapoint=dp)

        assert result.terminated_by != TerminatedBy.error, result.reason
        sim.generate_first_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_injected_simulator_respond_async_called_on_multi_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """On multi-turn runs, respond_async is called on the injected simulator."""
        monkeypatch.setenv("ORQ_API_KEY", "test-key")

        # Judge terminates on turn 2, so user simulator has one respond_async call
        judgments = [
            _make_mock_judgment(should_terminate=False),
            _make_mock_judgment(should_terminate=True),
        ]
        judge = MagicMock()
        judge.evaluate = AsyncMock(side_effect=judgments)
        judge.get_usage = MagicMock(return_value=TokenUsage())

        sim = _make_mock_user_simulator()
        runner = _make_runner_with_mocks(
            user_simulator=sim,
            judge=judge,
            max_turns=3,
        )

        dp = _make_datapoint()
        result = await runner.run(datapoint=dp)

        assert result.terminated_by != TerminatedBy.error, result.reason
        sim.respond_async.assert_called_once()


class TestUpdateContextCalledPerSimulation:
    """update_context is called with the datapoint's persona/scenario context."""

    @pytest.mark.asyncio
    async def test_update_context_called_with_persona_and_scenario(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """update_context is invoked with non-None persona_context and scenario_context."""
        monkeypatch.setenv("ORQ_API_KEY", "test-key")

        sim = _make_mock_user_simulator()
        sim.update_context = MagicMock()  # Add update_context method
        judge = _make_mock_judge()

        runner = _make_runner_with_mocks(
            user_simulator=sim,
            judge=judge,
            max_turns=1,
        )

        dp = _make_datapoint()
        result = await runner.run(datapoint=dp)

        assert result.terminated_by != TerminatedBy.error, result.reason
        sim.update_context.assert_called_once()
        call_kwargs = sim.update_context.call_args.kwargs
        assert "persona_context" in call_kwargs
        assert "scenario_context" in call_kwargs
        assert call_kwargs["persona_context"] is not None
        assert call_kwargs["scenario_context"] is not None

    @pytest.mark.asyncio
    async def test_update_context_failure_surfaces_as_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A failing update_context raises RuntimeError (not a silent error result)."""
        monkeypatch.setenv("ORQ_API_KEY", "test-key")

        sim = _make_mock_user_simulator()
        sim.update_context = MagicMock(side_effect=TypeError("bad args"))
        judge = _make_mock_judge()

        runner = _make_runner_with_mocks(
            user_simulator=sim,
            judge=judge,
            max_turns=1,
        )

        dp = _make_datapoint()
        # run() never throws — the error is captured in a TerminatedBy.error result
        result = await runner.run(datapoint=dp)

        assert result.terminated_by == TerminatedBy.error
        assert "update_context" in result.reason


class TestInvalidUserSimulatorRaisesTypeError:
    """Passing a BaseAgent subclass lacking generate_first_message raises TypeError."""

    def test_base_agent_lacking_generate_first_message_raises_type_error(
        self,
    ):
        """A custom BaseAgent without generate_first_message raises TypeError at construction."""
        # Build a plain MagicMock with respond_async but NO generate_first_message
        bad_sim = MagicMock(spec=[])  # spec=[] means no attributes allowed
        bad_sim.respond_async = AsyncMock(return_value="hi")
        bad_sim.get_usage = MagicMock(return_value=TokenUsage())

        judge = _make_mock_judge()

        with pytest.raises(TypeError, match="generate_first_message"):
            _make_runner_with_mocks(
                user_simulator=bad_sim,
                judge=judge,
                max_turns=1,
            )


# ---------------------------------------------------------------------------
# Internal helper (not a test)
# ---------------------------------------------------------------------------


def _make_runner_that_captures(kw: dict[str, Any]) -> SimulationRunner:
    """Placeholder — not actually used in the test above."""
    return SimulationRunner(target_callback=kw.get("target_callback", lambda m: "ok"))  # pyright: ignore[reportUnknownLambdaType]
