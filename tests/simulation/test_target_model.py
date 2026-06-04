"""Tests for target_model provenance in SimulationResult.metadata.

Verifies:
- A plain target_callback run does NOT set metadata["target_model"] to the runner's
  configured model (or any runner-internal value).
- An AgentTarget whose respond() returns AgentResponse(model="x") surfaces that model
  as metadata["target_model"] on the SimulationResult.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.contracts import AgentResponse, AgentTarget, Message
from evaluatorq.simulation.runner.simulation import SimulationRunner
from evaluatorq.contracts import TokenUsage
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Datapoint,
    Persona,
    Scenario,
    TerminatedBy,
)


# ---------------------------------------------------------------------------
# Shared helpers (mirror test_simulate_injection.py conventions)
# ---------------------------------------------------------------------------


def _make_persona(name: str = 'Target Model Tester') -> Persona:
    return Persona(
        name=name,
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='Testing target model provenance',
    )


def _make_scenario(name: str = 'Target Model Scenario') -> Scenario:
    return Scenario(name=name, goal='Verify model field')


def _make_datapoint(
    persona: Persona | None = None,
    scenario: Scenario | None = None,
) -> Datapoint:
    p = persona or _make_persona()
    s = scenario or _make_scenario()
    return Datapoint(
        id='dp-target-model-001',
        persona=p,
        scenario=s,
        user_system_prompt='system',
        first_message='Hello, can you help me?',
    )


def _make_mock_judgment(*, should_terminate: bool = True, goal_achieved: bool = True) -> MagicMock:
    j = MagicMock()
    j.should_terminate = should_terminate
    j.goal_achieved = goal_achieved
    j.goal_completion_score = 1.0
    j.rules_broken = []
    j.reason = 'Done'
    j.response_quality = 0.9
    j.hallucination_risk = 0.1
    j.tone_appropriateness = 0.9
    j.factual_accuracy = 0.9
    return j


def _make_mock_user_simulator(first_message: str = 'Hello') -> MagicMock:
    sim = MagicMock()
    sim.generate_first_message = AsyncMock(return_value=first_message)
    sim.respond_async = AsyncMock(return_value='thanks')
    sim.get_usage = MagicMock(return_value=TokenUsage())
    return sim


def _make_mock_judge(*, should_terminate: bool = True) -> MagicMock:
    j = MagicMock()
    j.evaluate = AsyncMock(return_value=_make_mock_judgment(should_terminate=should_terminate))
    j.get_usage = MagicMock(return_value=TokenUsage())
    return j


# ---------------------------------------------------------------------------
# A minimal AgentTarget that reports a model identity in its AgentResponse
# ---------------------------------------------------------------------------


class _ModelReportingTarget(AgentTarget):
    """Stub target that returns AgentResponse with a known model string."""

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self._model_name = model_name

    async def respond(self, messages: list[Message]) -> AgentResponse:
        return AgentResponse(text='I can help you.', model=self._model_name)

    def new(self) -> '_ModelReportingTarget':
        return _ModelReportingTarget(self._model_name)


class _SilentTarget(AgentTarget):
    """Stub target that returns AgentResponse WITHOUT a model field."""

    async def respond(self, messages: list[Message]) -> AgentResponse:
        return AgentResponse(text='I can help you.')

    def new(self) -> '_SilentTarget':
        return _SilentTarget()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTargetModelCallback:
    """plain target_callback → target_model MUST NOT be set to the runner's model."""

    @pytest.mark.asyncio
    async def test_callback_target_does_not_set_target_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """metadata["target_model"] must be absent (or None) when using target_callback.

        The WRONG behaviour is metadata["target_model"] == runner._model (the
        user-simulator/judge model). A plain callback can call any provider and we
        have no way to know the model; we must not invent a value.
        """
        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        runner_model = 'azure/gpt-4o-mini'

        async def my_callback(messages: list[Message]) -> str:
            return 'agent reply'

        sim = _make_mock_user_simulator()
        judge = _make_mock_judge(should_terminate=True)

        runner = SimulationRunner(
            target_callback=my_callback,
            model=runner_model,
            max_turns=1,
            user_simulator=sim,
            judge=judge,
        )

        result = await runner.run(datapoint=_make_datapoint())

        assert result.terminated_by != TerminatedBy.error, result.reason
        # Key MUST NOT be set to the runner's internal model.
        target_model = result.metadata.get('target_model')
        assert target_model != runner_model, (
            f'target_model incorrectly set to runner model "{runner_model}" — '
            'this is the user-simulator/judge model, not the evaluated target.'
        )

    @pytest.mark.asyncio
    async def test_callback_target_target_model_is_absent_or_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When using target_callback, metadata["target_model"] should be absent or None."""
        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        async def my_callback(messages: list[Message]) -> str:
            return 'agent reply'

        sim = _make_mock_user_simulator()
        judge = _make_mock_judge(should_terminate=True)

        runner = SimulationRunner(
            target_callback=my_callback,
            model='some-runner-model',
            max_turns=1,
            user_simulator=sim,
            judge=judge,
        )

        result = await runner.run(datapoint=_make_datapoint())

        assert result.terminated_by != TerminatedBy.error, result.reason
        # The key should be absent, or if present, be None
        assert result.metadata.get('target_model') is None


class TestTargetModelAgentTarget:
    """AgentTarget returning AgentResponse(model=...) → metadata["target_model"] set."""

    @pytest.mark.asyncio
    async def test_agent_target_with_model_surfaces_target_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AgentTarget that returns model in AgentResponse → metadata["target_model"]."""
        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        expected_model = 'gpt-4o-2024-11-20'
        target = _ModelReportingTarget(expected_model)

        sim = _make_mock_user_simulator()
        judge = _make_mock_judge(should_terminate=True)

        runner = SimulationRunner(
            target_agent=target,
            model='judge-model',
            max_turns=1,
            user_simulator=sim,
            judge=judge,
        )

        result = await runner.run(datapoint=_make_datapoint())

        assert result.terminated_by != TerminatedBy.error, result.reason
        assert result.metadata.get('target_model') == expected_model, (
            f'Expected target_model="{expected_model}", got {result.metadata.get("target_model")!r}'
        )

    @pytest.mark.asyncio
    async def test_agent_target_without_model_leaves_target_model_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AgentTarget that omits model in AgentResponse → target_model absent/None."""
        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        target = _SilentTarget()

        sim = _make_mock_user_simulator()
        judge = _make_mock_judge(should_terminate=True)

        runner = SimulationRunner(
            target_agent=target,
            model='judge-model',
            max_turns=1,
            user_simulator=sim,
            judge=judge,
        )

        result = await runner.run(datapoint=_make_datapoint())

        assert result.terminated_by != TerminatedBy.error, result.reason
        assert result.metadata.get('target_model') is None

    @pytest.mark.asyncio
    async def test_target_model_consistent_across_termination_modes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """target_model is present when judge terminates AND when max_turns is reached."""
        monkeypatch.setenv('ORQ_API_KEY', 'test-key')

        expected_model = 'claude-3-5-sonnet'
        target = _ModelReportingTarget(expected_model)

        # Max-turns path: judge never terminates
        sim = _make_mock_user_simulator()
        judge = _make_mock_judge(should_terminate=False)

        runner = SimulationRunner(
            target_agent=target,
            model='judge-model',
            max_turns=1,
            user_simulator=sim,
            judge=judge,
        )

        result = await runner.run(datapoint=_make_datapoint())

        assert result.terminated_by == TerminatedBy.max_turns, result.reason
        assert result.metadata.get('target_model') == expected_model, (
            f'Expected target_model="{expected_model}" on max_turns path, got {result.metadata.get("target_model")!r}'
        )
