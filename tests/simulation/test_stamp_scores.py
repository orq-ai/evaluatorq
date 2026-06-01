"""Unit tests for _stamp_evaluator_scores (RES-598 review #5).

The scorer no longer mutates SimulationResult.metadata mid-run; scores are
stamped once from the final evaluatorq result. These tests lock in that the
mirror still lands the same data, keyed by DataPoint identity.
"""

from __future__ import annotations

from evaluatorq.simulation.api import _stamp_evaluator_scores
from evaluatorq.simulation.types import (
    SimulationResult,
    TerminatedBy,
    TokenUsage,
)
from evaluatorq.types import (
    DataPoint,
    DataPointResult,
    EvaluationResult,
    EvaluatorScore,
    JobResult,
)


def _sim_result() -> SimulationResult:
    return SimulationResult(
        messages=[],
        terminated_by=TerminatedBy.max_turns,
        reason="",
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=TokenUsage(),
        turn_metrics=[],
    )


def test_stamps_scores_onto_matching_result_by_identity():
    dp = DataPoint(inputs={"datapoint": {}})
    sim = _sim_result()
    cache = {id(dp): sim}
    eq_results = [
        DataPointResult(
            data_point=dp,
            job_results=[
                JobResult(
                    job_name="simulation",
                    output=None,
                    evaluator_scores=[
                        EvaluatorScore(
                            evaluator_name="goal_achieved",
                            score=EvaluationResult(value=1.0),
                        ),
                        EvaluatorScore(
                            evaluator_name="criteria_met",
                            score=EvaluationResult(value=0.0),
                        ),
                    ],
                )
            ],
        )
    ]

    _stamp_evaluator_scores(eq_results, cache, "my-run")

    assert sim.metadata["evaluator_scores"] == {
        "goal_achieved": 1.0,
        "criteria_met": 0.0,
    }
    assert sim.metadata["evaluation_name"] == "my-run"


def test_skips_rows_with_no_cached_result():
    """Error rows carry a placeholder DataPoint whose id isn't in the cache."""
    placeholder = DataPoint(inputs={})
    eq_results = [DataPointResult(data_point=placeholder, error="boom")]

    # No cache entry, no job_results — must not raise.
    _stamp_evaluator_scores(eq_results, {}, "")
