"""Tests for simulation evaluators."""

from typing import Any

import pytest

from evaluatorq.simulation.evaluators import (
    conversation_quality_scorer,
    criteria_met_scorer,
    get_all_evaluators,
    get_evaluator,
    goal_achieved_scorer,
    turn_efficiency_scorer,
)
from evaluatorq.simulation.types import (
    SimulationResult,
    TerminatedBy,
    TokenUsage,
)


def _make_result(**overrides: Any) -> SimulationResult:
    defaults: dict[str, Any] = dict(
        messages=[],
        terminated_by=TerminatedBy.judge,
        reason="test",
        goal_achieved=False,
        goal_completion_score=0.0,
        rules_broken=[],
        turn_count=3,
        token_usage=TokenUsage(),
        turn_metrics=[],
    )
    defaults.update(overrides)
    return SimulationResult(**defaults)


class TestGoalAchievedScorer:
    def test_goal_achieved(self):
        result = _make_result(goal_achieved=True)
        assert goal_achieved_scorer(result) == 1.0

    def test_goal_not_achieved(self):
        result = _make_result(goal_achieved=False)
        assert goal_achieved_scorer(result) == 0.0


class TestCriteriaMetScorer:
    def test_all_criteria_met(self):
        result = _make_result(criteria_results={"a": True, "b": True})
        assert criteria_met_scorer(result) == 1.0

    def test_no_criteria_met(self):
        result = _make_result(criteria_results={"a": False, "b": False})
        assert criteria_met_scorer(result) == 0.0

    def test_some_criteria_met(self):
        result = _make_result(criteria_results={"a": True, "b": False})
        assert criteria_met_scorer(result) == 0.5

    def test_no_criteria(self):
        result = _make_result(criteria_results=None)
        assert criteria_met_scorer(result) == 1.0


class TestTurnEfficiencyScorer:
    def test_goal_not_achieved(self):
        result = _make_result(goal_achieved=False, turn_count=1)
        assert turn_efficiency_scorer(result) == 0.0

    def test_quick_resolution(self):
        result = _make_result(goal_achieved=True, turn_count=2)
        assert turn_efficiency_scorer(result) == 1.0

    def test_medium_resolution(self):
        result = _make_result(goal_achieved=True, turn_count=4)
        assert turn_efficiency_scorer(result) == 0.9

    def test_slow_resolution(self):
        result = _make_result(goal_achieved=True, turn_count=6)
        assert turn_efficiency_scorer(result) == 0.7

    def test_very_slow_resolution(self):
        result = _make_result(goal_achieved=True, turn_count=10)
        assert turn_efficiency_scorer(result) == 0.6

    def test_single_turn_resolution(self):
        result = _make_result(goal_achieved=True, turn_count=1)
        assert turn_efficiency_scorer(result) == 1.0

    def test_floor_at_many_turns(self):
        result = _make_result(goal_achieved=True, turn_count=20)
        assert turn_efficiency_scorer(result) == 0.3


class TestConversationQualityScorer:
    def test_perfect_score(self):
        result = _make_result(
            goal_achieved=True,
            turn_count=2,
            criteria_results={"a": True},
        )
        assert conversation_quality_scorer(result) == 1.0

    def test_zero_score(self):
        result = _make_result(
            goal_achieved=False,
            turn_count=10,
            criteria_results={"a": False},
        )
        assert conversation_quality_scorer(result) == 0.0


class TestEvaluatorRegistry:
    def test_get_evaluator(self):
        scorer = get_evaluator("goal_achieved")
        assert scorer is goal_achieved_scorer

    def test_get_unknown_evaluator(self):
        with pytest.raises(ValueError, match="Unknown evaluator"):
            get_evaluator("nonexistent")

    def test_get_all_evaluators(self):
        evaluators = get_all_evaluators()
        assert "goal_achieved" in evaluators
        assert "criteria_met" in evaluators
        assert "turn_efficiency" in evaluators
        assert "conversation_quality" in evaluators

    def test_get_all_returns_copy(self):
        a = get_all_evaluators()
        b = get_all_evaluators()
        assert a is not b
