"""Unit tests for _stamp_evaluator_scores (RES-598 review #5).

The scorer no longer mutates SimulationResult.metadata mid-run; scores are
stamped once from the final evaluatorq result. These tests lock in that the
mirror still lands the same data, keyed by DataPoint identity.
"""

from __future__ import annotations

import asyncio

from evaluatorq.contracts import TokenUsage
from evaluatorq.simulation.api import _notify_evaluator_complete, _stamp_evaluator_scores
from evaluatorq.simulation.hooks import DefaultHooks
from evaluatorq.simulation.types import (
    SimulationResult,
    TerminatedBy,
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
        reason='',
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=1,
        token_usage=TokenUsage(),
        turn_metrics=[],
    )


def test_stamps_scores_onto_matching_result_by_identity():
    dp = DataPoint(inputs={'datapoint': {}})
    sim = _sim_result()
    cache = {id(dp): sim}
    eq_results = [
        DataPointResult(
            data_point=dp,
            job_results=[
                JobResult(
                    job_name='simulation',
                    output=None,
                    evaluator_scores=[
                        EvaluatorScore(
                            evaluator_name='goal_achieved',
                            score=EvaluationResult(value=1.0),
                        ),
                        EvaluatorScore(
                            evaluator_name='criteria_met',
                            score=EvaluationResult(value=0.0),
                        ),
                    ],
                )
            ],
        )
    ]

    _stamp_evaluator_scores(eq_results, cache, 'my-run')

    assert sim.metadata['evaluator_scores'] == {
        'goal_achieved': 1.0,
        'criteria_met': 0.0,
    }
    assert sim.metadata['evaluation_name'] == 'my-run'


def test_skips_rows_with_no_cached_result():
    """Error rows carry a placeholder DataPoint whose id isn't in the cache."""
    placeholder = DataPoint(inputs={})
    eq_results = [DataPointResult(data_point=placeholder, error='boom')]

    # No cache entry, no job_results — must not raise.
    _stamp_evaluator_scores(eq_results, {}, '')


def test_skips_errored_evaluator_score_and_does_not_notify_callback(caplog):
    dp = DataPoint(inputs={'datapoint': {}})
    sim = _sim_result()
    cache = {id(dp): sim}
    eq_results = [
        DataPointResult(
            data_point=dp,
            job_results=[
                JobResult(
                    job_name='simulation',
                    output=None,
                    evaluator_scores=[
                        EvaluatorScore(
                            evaluator_name='goal_achieved',
                            score=EvaluationResult(value=''),
                            error='judge died',
                        ),
                        EvaluatorScore(
                            evaluator_name='healthy_score',
                            score=EvaluationResult(value=0.5),
                        ),
                    ],
                )
            ],
        )
    ]

    with caplog.at_level('WARNING', logger='evaluatorq.simulation.api'):
        _stamp_evaluator_scores(eq_results, cache, 'my-run')

    class RecordingHooks(DefaultHooks):
        def __init__(self) -> None:
            self.callback_scores: list[tuple[str, float]] = []

        async def on_evaluator_complete(self, datapoint_id, name, score, result) -> None:
            self.callback_scores.append((name, score))

    hooks = RecordingHooks()
    asyncio.run(_notify_evaluator_complete([sim], hooks))

    assert hooks.callback_scores == [('healthy_score', 0.5)]
    assert 'goal_achieved' not in sim.metadata.get('evaluator_scores', {})
    assert 'Skipping evaluator goal_achieved score: judge died' in caplog.text


def test_skips_non_numeric_evaluator_score_without_error(caplog):
    dp = DataPoint(inputs={'datapoint': {}})
    sim = _sim_result()
    cache = {id(dp): sim}
    eq_results = [
        DataPointResult(
            data_point=dp,
            job_results=[
                JobResult(
                    job_name='simulation',
                    output=None,
                    evaluator_scores=[
                        EvaluatorScore(
                            evaluator_name='criteria_met',
                            score=EvaluationResult(value='not numeric'),
                        )
                    ],
                )
            ],
        )
    ]

    with caplog.at_level('WARNING', logger='evaluatorq.simulation.api'):
        _stamp_evaluator_scores(eq_results, cache, 'my-run')

    assert 'evaluator_scores' not in sim.metadata
    assert "Skipping evaluator criteria_met score: non-numeric value 'not numeric'" in caplog.text


def test_skips_bool_evaluator_score_as_non_numeric(caplog):
    dp = DataPoint(inputs={'datapoint': {}})
    sim = _sim_result()
    cache = {id(dp): sim}
    eq_results = [
        DataPointResult(
            data_point=dp,
            job_results=[
                JobResult(
                    job_name='simulation',
                    output=None,
                    evaluator_scores=[
                        EvaluatorScore(
                            evaluator_name='goal_achieved',
                            score=EvaluationResult(value=True),
                        )
                    ],
                )
            ],
        )
    ]

    with caplog.at_level('WARNING', logger='evaluatorq.simulation.api'):
        _stamp_evaluator_scores(eq_results, cache, 'my-run')

    assert 'evaluator_scores' not in sim.metadata
    assert 'Skipping evaluator goal_achieved score: non-numeric value True' in caplog.text
