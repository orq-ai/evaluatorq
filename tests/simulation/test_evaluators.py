"""Tests for simulation evaluators."""

from typing import Any

import pytest

from evaluatorq.contracts import TokenUsage
from evaluatorq.simulation.evaluators import (
    SimulationScoringConfig,
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
        result = _make_result(goal_achieved=True, turn_count=8)
        assert turn_efficiency_scorer(result) == 0.5

    def test_curve_is_monotonic_across_the_decay_seam(self):
        """The decay continues from the last cliff, so one more turn never scores higher."""
        scores = [turn_efficiency_scorer(_make_result(goal_achieved=True, turn_count=n)) for n in range(1, 16)]
        assert scores == sorted(scores, reverse=True)
        assert scores[:8] == [1.0, 1.0, 0.9, 0.9, 0.7, 0.7, 0.6, 0.5]

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

    def test_breakdown_recombines_to_the_reported_score(self):
        result = _make_result(goal_achieved=True, turn_count=4, criteria_results={"a": True, "b": False})
        score = conversation_quality_scorer(result)
        breakdown = score.breakdown

        assert breakdown["components"] == {
            "goal_achieved": goal_achieved_scorer(result),
            "criteria_met": criteria_met_scorer(result),
            "turn_efficiency": turn_efficiency_scorer(result),
        }
        recombined = sum(
            component * breakdown["weights"][name] for name, component in breakdown["components"].items()
        )
        assert round(recombined * 100) / 100 == score
        assert float(score) == 0.82

    def test_breakdown_carries_the_callers_weights(self):
        goal_only = SimulationScoringConfig(
            goal_achieved_weight=1.0, criteria_met_weight=0.0, turn_efficiency_weight=0.0
        )
        result = _make_result(goal_achieved=True, turn_count=4, criteria_results={"a": True, "b": False})
        score = conversation_quality_scorer(result, goal_only)

        assert score.breakdown["weights"] == {
            "goal_achieved": 1.0,
            "criteria_met": 0.0,
            "turn_efficiency": 0.0,
        }
        # The components are still all three, so a reader can see what the weights discarded.
        assert score.breakdown["components"]["criteria_met"] == 0.5

    def test_the_score_is_still_a_plain_float_to_every_caller(self):
        """`SimulationScorer` is `Callable[..., float]`; the breakdown must not change that."""
        result = _make_result(goal_achieved=True, turn_count=2, criteria_results={"a": True})
        score = conversation_quality_scorer(result)

        assert isinstance(score, float)
        assert score == 1.0
        assert score + 1.0 == 2.0
        assert sorted([score, 0.5]) == [0.5, score]


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


class TestSimulationScoringConfig:
    """The config is the documented policy surface; these cover its bounds and threading."""

    def test_omitting_config_matches_an_explicit_default(self):
        result = _make_result(goal_achieved=True, turn_count=4, criteria_results={"a": True, "b": False})
        assert conversation_quality_scorer(result) == conversation_quality_scorer(result, SimulationScoringConfig())

    def test_default_turn_efficiency_curve_is_pinned(self):
        """Pin the whole default curve, cliffs and tail, against literals.

        Comparing the default path to `SimulationScoringConfig()` only proves the two
        agree; it passes just as happily if both are wrong. These are the numbers a
        report actually shows.
        """
        expected = {1: 1.0, 2: 1.0, 3: 0.9, 4: 0.9, 5: 0.7, 6: 0.7, 7: 0.6, 8: 0.5, 12: 0.3, 20: 0.3}
        actual = {
            turns: turn_efficiency_scorer(_make_result(goal_achieved=True, turn_count=turns))
            for turns in expected
        }
        assert actual == expected

    def test_worked_example_from_the_docstring(self):
        # goal_achieved 1.0 * 0.4 + criteria_met 0.5 * 0.3 + turn_efficiency 0.9 * 0.3 = 0.82
        result = _make_result(goal_achieved=True, turn_count=4, criteria_results={"a": True, "b": False})
        assert conversation_quality_scorer(result) == 0.82

    def test_weights_are_applied(self):
        result = _make_result(goal_achieved=True, turn_count=4, criteria_results={"a": True, "b": False})
        goal_only = SimulationScoringConfig(
            goal_achieved_weight=1.0, criteria_met_weight=0.0, turn_efficiency_weight=0.0
        )
        assert conversation_quality_scorer(result, goal_only) == 1.0

    def test_wider_cliffs_stop_penalising_a_long_task(self):
        config = SimulationScoringConfig(turn_efficiency_cliffs=((6, 1.0), (10, 0.9), (16, 0.7)))
        result = _make_result(goal_achieved=True, turn_count=9)
        assert turn_efficiency_scorer(result) == 0.4
        assert turn_efficiency_scorer(result, config) == 0.9

    def test_failed_run_scores_zero_regardless_of_config(self):
        config = SimulationScoringConfig(turn_efficiency_cliffs=((100, 1.0),))
        assert turn_efficiency_scorer(_make_result(goal_achieved=False, turn_count=1), config) == 0.0

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            SimulationScoringConfig(goal_achieved_weight=0.5)

    def test_cliff_turns_must_increase(self):
        with pytest.raises(ValueError, match="must strictly increase"):
            SimulationScoringConfig(turn_efficiency_cliffs=((4, 0.9), (2, 1.0)))

    def test_cliff_scores_must_not_increase(self):
        with pytest.raises(ValueError, match="must not increase with turns"):
            SimulationScoringConfig(turn_efficiency_cliffs=((2, 0.5), (4, 0.9)))

    def test_cliffs_must_not_be_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            SimulationScoringConfig(turn_efficiency_cliffs=())

    def test_floor_must_not_exceed_last_cliff(self):
        with pytest.raises(ValueError, match="exceeds the last cliff score"):
            SimulationScoringConfig(turn_efficiency_floor=0.9)

    def test_unknown_field_is_rejected(self):
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            SimulationScoringConfig(turn_efficiency_cliff=((2, 1.0),))  # pyright: ignore[reportCallIssue]

    def test_get_evaluator_binds_the_config(self):
        config = SimulationScoringConfig(turn_efficiency_cliffs=((12, 1.0),))
        result = _make_result(goal_achieved=True, turn_count=10)
        assert get_evaluator("turn_efficiency", config)(result) == 1.0
        assert get_evaluator("turn_efficiency")(result) == 0.3

    def test_get_all_evaluators_binds_the_config(self):
        config = SimulationScoringConfig(turn_efficiency_cliffs=((12, 1.0),))
        result = _make_result(goal_achieved=True, turn_count=10, criteria_results={"a": True})
        evaluators = get_all_evaluators(config)
        assert evaluators["conversation_quality"](result) == 1.0
        # Judge-verdict scorers have nothing to tune, so they are returned unwrapped.
        assert evaluators["goal_achieved"] is goal_achieved_scorer


class TestScoringConfigIsReachable:
    """`scoring=` must reach the scorers the same way `recommendations=` reaches its config."""

    def test_public_entry_points_accept_scoring(self):
        import inspect

        from evaluatorq.simulation import generate_and_simulate, simulate

        for fn in (simulate, generate_and_simulate):
            param = inspect.signature(fn).parameters.get("scoring")
            assert param is not None, f"{fn.__name__} has no scoring= parameter"
            assert param.default is None

    def test_internal_config_carries_it_to_the_scorers(self):
        from evaluatorq.simulation._config import SimulationConfig

        config = SimulationConfig(scoring=SimulationScoringConfig(turn_efficiency_cliffs=((12, 1.0),)))
        scorer = get_evaluator("turn_efficiency", config.scoring)
        assert scorer(_make_result(goal_achieved=True, turn_count=10)) == 1.0
