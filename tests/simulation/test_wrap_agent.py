"""Tests for wrap_simulation_agent."""

import pytest

import evaluatorq.simulation as simulation_module
from evaluatorq.simulation.types import DEFAULT_MODEL, Message, SimulationResult, TerminatedBy, TokenUsage
from evaluatorq.simulation.wrap_agent import _validate_shape, wrap_simulation_agent
from evaluatorq.types import DataPoint


def _make_result(content: str) -> SimulationResult:
    return SimulationResult(
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content=content),
        ],
        terminated_by=TerminatedBy.judge,
        reason="Goal achieved",
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=TokenUsage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        ),
        turn_metrics=[],
    )


class TestValidateShape:
    def test_valid_shape(self):
        _validate_shape({"name": "test", "goal": "help"}, "scenario", ["name", "goal"])

    def test_missing_key(self):
        with pytest.raises(ValueError, match="missing required field 'goal'"):
            _validate_shape({"name": "test"}, "scenario", ["name", "goal"])

    def test_not_an_object(self):
        with pytest.raises(ValueError, match="Expected 'scenario' to be an object"):
            _validate_shape("not a dict", "scenario", ["name"])

    def test_none_value(self):
        with pytest.raises(ValueError, match="Expected 'scenario' to be an object"):
            _validate_shape(None, "scenario", ["name"])


_FULL_PERSONA = {
    "name": "Persona A",
    "patience": 0.5,
    "assertiveness": 0.5,
    "politeness": 0.5,
    "technical_level": 0.5,
    "communication_style": "casual",
    "background": "Test persona",
}

_FULL_SCENARIO = {
    "name": "Scenario A",
    "goal": "Help",
}


class TestWrapSimulationAgent:
    @pytest.mark.asyncio
    async def test_returns_first_output_when_multiple_results(self, monkeypatch):
        async def fake_simulate(**_kwargs):
            return [_make_result("First"), _make_result("Second")]

        monkeypatch.setattr(simulation_module, "simulate", fake_simulate)

        job = wrap_simulation_agent(
            target_callback=lambda _messages: "unused",
        )
        result = await job(
            DataPoint(
                inputs={
                    "personas": [_FULL_PERSONA],
                    "scenarios": [_FULL_SCENARIO],
                }
            ),
            0,
        )

        assert isinstance(result["output"], dict)
        assert result["output"]["output"][0]["content"][0]["text"] == "First"

    @pytest.mark.asyncio
    async def test_uses_simulation_model_by_default(self, monkeypatch):
        async def fake_simulate(**_kwargs):
            return [_make_result("Only result")]

        monkeypatch.setattr(simulation_module, "simulate", fake_simulate)

        job = wrap_simulation_agent(
            target_callback=lambda _messages: "unused",
        )
        result = await job(
            DataPoint(
                inputs={
                    "persona": _FULL_PERSONA,
                    "scenario": _FULL_SCENARIO,
                }
            ),
            0,
        )

        assert result["output"]["model"] == DEFAULT_MODEL

    @pytest.mark.asyncio
    async def test_rejects_multiple_datapoints(self):
        job = wrap_simulation_agent(
            target_callback=lambda _messages: "unused",
        )
        dp = {
            "persona": _FULL_PERSONA,
            "scenario": _FULL_SCENARIO,
            "first_message": "Hello",
        }
        with pytest.raises(ValueError, match="exactly one datapoint"):
            await job(
                DataPoint(inputs={"datapoints": [dp, dp]}),
                0,
            )

    @pytest.mark.asyncio
    async def test_rejects_multiple_personas_or_scenarios(self):
        job = wrap_simulation_agent(
            target_callback=lambda _messages: "unused",
        )
        with pytest.raises(ValueError, match="exactly one persona-scenario pair"):
            await job(
                DataPoint(
                    inputs={
                        "personas": [_FULL_PERSONA, _FULL_PERSONA],
                        "scenarios": [_FULL_SCENARIO],
                    }
                ),
                0,
            )

    @pytest.mark.asyncio
    async def test_rejects_multiple_scenarios(self):
        job = wrap_simulation_agent(
            target_callback=lambda _messages: "unused",
        )
        with pytest.raises(ValueError, match="exactly one persona-scenario pair"):
            await job(
                DataPoint(
                    inputs={
                        "personas": [_FULL_PERSONA],
                        "scenarios": [_FULL_SCENARIO, _FULL_SCENARIO],
                    }
                ),
                0,
            )
