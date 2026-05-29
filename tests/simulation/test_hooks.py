from __future__ import annotations

import pytest

from evaluatorq.simulation.hooks import (
    DefaultHooks,
    SimulationHooks,
    SimulationRunMeta,
)
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Datapoint,
    Persona,
    Scenario,
    SimulationResult,
    TerminatedBy,
    TokenUsage,
    TurnMetrics,
)


@pytest.fixture
def datapoint_factory():
    def _make(dp_id: str) -> Datapoint:
        persona = Persona(
            name=f"p-{dp_id}",
            patience=0.5,
            assertiveness=0.5,
            politeness=0.5,
            technical_level=0.5,
            communication_style=CommunicationStyle.casual,
            background="d",
        )
        scenario = Scenario(name=f"s-{dp_id}", goal="g")
        return Datapoint(
            id=dp_id,
            persona=persona,
            scenario=scenario,
            user_system_prompt="",
            first_message="hi",
        )

    return _make


def _meta() -> SimulationRunMeta:
    return SimulationRunMeta(
        num_datapoints=1,
        model="m",
        max_turns=3,
        parallelism=1,
        evaluation_name="e",
        evaluator_names=["goal_achieved"],
    )


def _result() -> SimulationResult:
    return SimulationResult(
        messages=[],
        terminated_by=TerminatedBy.judge,
        reason="ok",
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        turn_metrics=[],
        metadata={"datapoint_id": "dp1"},
    )


def _turn_metrics() -> TurnMetrics:
    return TurnMetrics(
        turn_number=1,
        token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        judge_reason="r",
    )


def test_default_hooks_satisfies_protocol():
    assert isinstance(DefaultHooks(), SimulationHooks)


def test_default_hooks_confirm_returns_true():
    # Library callers + hooks=None must never be blocked (redteam parity).
    assert DefaultHooks().on_confirm(_meta()) is True


def test_default_hooks_all_methods_silent(datapoint_factory):
    hooks = DefaultHooks()
    dp = datapoint_factory("dp1")
    # None of these raise; all return None (except on_confirm which returns bool).
    assert hooks.on_run_start(_meta()) is None
    assert hooks.on_datapoint_start(dp) is None
    assert hooks.on_turn_complete("dp1", _turn_metrics()) is None
    assert hooks.on_datapoint_complete(_result()) is None
    assert hooks.on_evaluator_complete("dp1", "goal_achieved", 1.0, _result()) is None
    assert hooks.on_datapoint_error(dp, RuntimeError("boom")) is None
    assert hooks.on_run_complete([_result()]) is None
