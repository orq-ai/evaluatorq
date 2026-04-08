"""Tests for SimulationRunner."""

import pytest

from evaluatorq.simulation.runner.simulation import (
    SimulationRunner,
    _invert_roles_for_simulator,
)
from evaluatorq.simulation.types import (
    ChatMessage,
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
    def test_requires_target(self):
        with pytest.raises(ValueError, match="Must provide either"):
            SimulationRunner(model="test")

    def test_max_turns_validation(self):
        with pytest.raises(ValueError, match="max_turns must be >= 1"):
            SimulationRunner(
                target_callback=lambda msgs: "ok",
                max_turns=0,
            )

    def test_model_validation(self):
        with pytest.raises(ValueError, match="model must be a non-empty"):
            SimulationRunner(
                target_callback=lambda msgs: "ok",
                model="   ",
            )


class TestSimulationRunnerRun:
    @pytest.mark.asyncio
    async def test_run_requires_persona_or_datapoint(self):
        runner = SimulationRunner(target_callback=lambda msgs: "ok")
        result = await runner.run(persona=_make_persona())
        assert result.terminated_by == TerminatedBy.error
        assert "Must provide either datapoint" in result.reason

    @pytest.mark.asyncio
    async def test_run_error_handling(self):
        """Runner should never throw, always return error result."""

        async def failing_callback(msgs):
            raise RuntimeError("API down")

        runner = SimulationRunner(target_callback=failing_callback, model="test-model")
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
            target_callback=lambda msgs: "ok",
            model="azure/gpt-4o-mini",
            max_turns=5,
        )
        assert runner is not None

    @pytest.mark.asyncio
    async def test_close_can_be_called_multiple_times(self):
        runner = SimulationRunner(target_callback=lambda msgs: "ok")
        await runner.close()
        await runner.close()

    @pytest.mark.asyncio
    async def test_run_batch_empty_datapoints(self, monkeypatch):
        monkeypatch.setenv("ORQ_API_KEY", "test-key")
        runner = SimulationRunner(target_callback=lambda msgs: "ok")
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

        assert result[0] == ChatMessage(role="assistant", content="Hello")
        assert result[1] == ChatMessage(role="user", content="Hi there")
        assert result[2] == ChatMessage(role="assistant", content="Help me")

    def test_preserves_system_role(self):
        messages = [Message(role="system", content="You are helpful")]
        result = _invert_roles_for_simulator(messages)
        assert result[0] == ChatMessage(role="system", content="You are helpful")

    def test_empty_messages(self):
        assert _invert_roles_for_simulator([]) == []


class TestSimulationRunnerBatchValidation:
    @pytest.mark.asyncio
    async def test_max_concurrency_validation(self):
        runner = SimulationRunner(target_callback=lambda msgs: "ok")
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            await runner.run_batch([], max_concurrency=0)

    @pytest.mark.asyncio
    async def test_negative_max_concurrency_validation(self):
        runner = SimulationRunner(target_callback=lambda msgs: "ok")
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            await runner.run_batch([], max_concurrency=-5)
