"""Unit tests for _stamp_evaluator_scores (RES-598 review #5).

The scorer no longer mutates SimulationResult.metadata mid-run; scores are
stamped once from the final evaluatorq result. These tests lock in that the
mirror still lands the same data, keyed by DataPoint identity, and that an
evaluator that errored or returned a non-numeric value is *recorded* under
``metadata['evaluator_errors']`` rather than dropped on the floor.
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


def _eq_results(dp: DataPoint, scores: list[EvaluatorScore]) -> list[DataPointResult]:
    return [
        DataPointResult(
            data_point=dp,
            job_results=[JobResult(job_name='simulation', output=None, evaluator_scores=scores)],
        )
    ]


def test_records_errored_evaluator_and_still_notifies_the_hook(caplog):
    """An evaluator that died is recorded under evaluator_errors, kept out of the
    numeric evaluator_scores, and still reaches on_evaluator_complete."""
    dp = DataPoint(inputs={'datapoint': {}})
    sim = _sim_result()
    cache = {id(dp): sim}
    eq_results = _eq_results(
        dp,
        [
            EvaluatorScore(evaluator_name='goal_achieved', score=EvaluationResult(value=''), error='judge died'),
            EvaluatorScore(evaluator_name='healthy_score', score=EvaluationResult(value=0.5)),
        ],
    )

    events: list[tuple[SimulationResult, EvaluatorScore]] = []
    with caplog.at_level('WARNING', logger='evaluatorq.simulation.api'):
        _stamp_evaluator_scores(eq_results, cache, 'my-run', events_out=events)

    assert sim.metadata['evaluator_scores'] == {'healthy_score': 0.5}
    assert sim.metadata['evaluator_errors'] == {'goal_achieved': 'judge died'}
    assert 'Evaluator goal_achieved produced no usable score (judge died)' in caplog.text

    class RecordingHooks(DefaultHooks):
        def __init__(self) -> None:
            self.seen: list[tuple[str, object, str | None]] = []

        async def on_evaluator_complete(self, datapoint_id, name, score, result) -> None:
            self.seen.append((name, score.score.value, score.error))

    hooks = RecordingHooks()
    asyncio.run(_notify_evaluator_complete(events, hooks))

    # The failed evaluator reaches the hook too — that is the whole point of
    # handing it the EvaluatorScore instead of a bare float it cannot produce.
    assert hooks.seen == [('goal_achieved', '', 'judge died'), ('healthy_score', 0.5, None)]


def test_warns_when_evaluator_errors_metadata_is_not_a_dict(caplog):
    dp = DataPoint(inputs={'datapoint': {}})
    sim = _sim_result()
    sim.metadata['datapoint_id'] = 'dp-1'
    sim.metadata['evaluator_errors'] = ['already broken']
    eq_results = _eq_results(
        dp,
        [EvaluatorScore(evaluator_name='goal_achieved', score=EvaluationResult(value=''), error='judge died')],
    )

    with caplog.at_level('WARNING', logger='evaluatorq.simulation.api'):
        _stamp_evaluator_scores(eq_results, {id(dp): sim}, 'my-run')

    assert sim.metadata['evaluator_errors'] == ['already broken']
    assert "Datapoint 'dp-1' has evaluator_errors with unexpected type list" in caplog.text


def test_records_non_numeric_evaluator_score_without_error(caplog):
    dp = DataPoint(inputs={'datapoint': {}})
    sim = _sim_result()
    cache = {id(dp): sim}
    eq_results = _eq_results(
        dp, [EvaluatorScore(evaluator_name='criteria_met', score=EvaluationResult(value='not numeric'))]
    )

    events = []
    with caplog.at_level('WARNING', logger='evaluatorq.simulation.api'):
        _stamp_evaluator_scores(eq_results, cache, 'my-run', events_out=events)

    assert 'evaluator_scores' not in sim.metadata
    assert sim.metadata['evaluator_errors'] == {'criteria_met': "non-numeric value 'not numeric'"}
    assert "Evaluator criteria_met produced no usable score (non-numeric value 'not numeric')" in caplog.text
    assert events[0][1].error == "non-numeric value 'not numeric'"


def test_records_bool_evaluator_score_as_non_numeric(caplog):
    dp = DataPoint(inputs={'datapoint': {}})
    sim = _sim_result()
    cache = {id(dp): sim}
    eq_results = _eq_results(dp, [EvaluatorScore(evaluator_name='goal_achieved', score=EvaluationResult(value=True))])

    with caplog.at_level('WARNING', logger='evaluatorq.simulation.api'):
        _stamp_evaluator_scores(eq_results, cache, 'my-run')

    assert 'evaluator_scores' not in sim.metadata
    assert sim.metadata['evaluator_errors'] == {'goal_achieved': 'non-numeric value True'}
    assert 'Evaluator goal_achieved produced no usable score (non-numeric value True)' in caplog.text


def test_events_out_collects_every_score_in_order():
    dp = DataPoint(inputs={'datapoint': {}})
    sim = _sim_result()
    eq_results = _eq_results(
        dp,
        [
            EvaluatorScore(evaluator_name='a', score=EvaluationResult(value=1.0)),
            EvaluatorScore(evaluator_name='b', score=EvaluationResult(value=''), error='boom'),
        ],
    )
    events: list[tuple[SimulationResult, EvaluatorScore]] = []

    _stamp_evaluator_scores(eq_results, {id(dp): sim}, '', events_out=events)

    assert [(s is sim, score.evaluator_name) for s, score in events] == [(True, 'a'), (True, 'b')]
