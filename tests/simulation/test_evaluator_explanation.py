"""Judge reasoning must reach the evaluator EvaluationResult.

Covers the sim scorer adapter (explanation + pass_ derived from the judge's
reasoning, numeric value preserved).
"""

# ruff: noqa: S101

from __future__ import annotations

from typing import Any

import pytest

from evaluatorq.contracts import TokenUsage
from evaluatorq.simulation.api import _adapt_simulation_scorer
from evaluatorq.simulation.evaluators import get_evaluator
from evaluatorq.simulation.types import SimulationResult, TerminatedBy
from evaluatorq.types import DataPoint, EvaluationResult


def _make_result(**overrides: Any) -> SimulationResult:
    defaults: dict[str, Any] = dict(
        messages=[],
        terminated_by=TerminatedBy.judge,
        reason='judge said the goal was met',
        goal_achieved=True,
        goal_completion_score=1.0,
        rules_broken=[],
        turn_count=3,
        token_usage=TokenUsage(),
        turn_metrics=[],
    )
    defaults.update(overrides)
    return SimulationResult(**defaults)


@pytest.mark.asyncio
async def test_goal_achieved_scorer_carries_reason_and_pass() -> None:
    result = _make_result(goal_achieved=True, reason='order was placed successfully')
    dp = DataPoint(inputs={})
    evaluator = _adapt_simulation_scorer('goal_achieved', get_evaluator('goal_achieved'), {id(dp): result})

    assert evaluator.get('evaluator_type') == 'python_eval'
    score = await evaluator['scorer']({'data': dp, 'output': {}})
    assert isinstance(score, EvaluationResult)

    assert score.value == 1.0  # numeric preserved
    assert score.pass_ is True
    assert score.explanation == 'order was placed successfully'


@pytest.mark.asyncio
async def test_criteria_met_scorer_summarizes_criteria() -> None:
    result = _make_result(
        goal_achieved=False,
        criteria_results={'greeted user': True, 'did not leak PII': False},
        metadata={
            'criteria_meta': [
                {'id': 'criteria_0', 'description': 'greeted user', 'type': 'must_happen', 'passed': True},
                {'id': 'criteria_1', 'description': 'leaked PII', 'type': 'must_not_happen', 'passed': False},
            ]
        },
    )
    dp = DataPoint(inputs={})
    evaluator = _adapt_simulation_scorer('criteria_met', get_evaluator('criteria_met'), {id(dp): result})
    score = await evaluator['scorer']({'data': dp, 'output': {}})
    assert isinstance(score, EvaluationResult)

    assert score.value == 0.5  # 1 of 2 criteria — average preserved
    assert score.pass_ is False
    assert score.explanation is not None
    assert 'PASS [required]: greeted user' in score.explanation
    assert 'FAIL [prohibited]: leaked PII' in score.explanation


# ---------------------------------------------------------------------------
# The reported pass/fail must agree with the score the scorer computed.
#
# RES-1308's reporting half: `criteria_met_scorer` returns 0.0 for an unaudited
# or errored run, but `_sim_evaluation_details` derived pass_ from criteria_meta
# alone, so the evaluator span and the uploaded Orq experiment showed those runs
# green. Each test below asserts value and pass_ together — the pair is the
# contract, either half alone proves nothing.
# ---------------------------------------------------------------------------


async def _score_criteria_met(result: SimulationResult) -> EvaluationResult:
    dp = DataPoint(inputs={})
    evaluator = _adapt_simulation_scorer('criteria_met', get_evaluator('criteria_met'), {id(dp): result})
    score = await evaluator['scorer']({'data': dp, 'output': {}})
    assert isinstance(score, EvaluationResult)
    return score


@pytest.mark.asyncio
async def test_criteria_met_reports_fail_when_criteria_are_unverified() -> None:
    result = _make_result(
        goal_achieved=True,
        criteria_verified=False,
        criteria_results={'greeted user': True, 'did not leak PII': True},
        metadata={
            'criteria_meta': [
                {'id': 'criteria_0', 'description': 'greeted user', 'type': 'must_happen', 'passed': True},
                {'id': 'criteria_1', 'description': 'leaked PII', 'type': 'must_not_happen', 'passed': True},
            ]
        },
    )
    score = await _score_criteria_met(result)

    assert score.value == 0.0  # the scorer already calls this unknown
    assert score.pass_ is False  # ... and the reported verdict must not say PASS
    assert score.explanation is not None
    assert 'unverified' in score.explanation
    assert 'PASS [required]' not in score.explanation


@pytest.mark.asyncio
@pytest.mark.parametrize('terminated_by', [TerminatedBy.error, TerminatedBy.timeout])
async def test_criteria_met_reports_fail_for_an_errored_run(terminated_by: TerminatedBy) -> None:
    # No criteria_meta at all: the run died before the judge audited anything.
    # This used to report 'No criteria defined for this scenario.' + pass=True
    # for a scenario that does have criteria.
    result = _make_result(goal_achieved=False, terminated_by=terminated_by, reason='boom')
    score = await _score_criteria_met(result)

    assert score.value == 0.0
    assert score.pass_ is False
    assert score.explanation is not None
    assert terminated_by.value in score.explanation
    assert 'No criteria defined' not in score.explanation


@pytest.mark.asyncio
async def test_criteria_met_still_passes_a_verified_clean_run() -> None:
    """The guard must not swallow the honest green case."""
    result = _make_result(
        goal_achieved=True,
        criteria_verified=True,
        criteria_results={'greeted user': True},
        metadata={
            'criteria_meta': [
                {'id': 'criteria_0', 'description': 'greeted user', 'type': 'must_happen', 'passed': True},
            ]
        },
    )
    score = await _score_criteria_met(result)

    assert score.value == 1.0
    assert score.pass_ is True


@pytest.mark.asyncio
async def test_criteria_met_reports_no_criteria_only_when_there_are_none() -> None:
    result = _make_result(goal_achieved=True, criteria_verified=True)
    score = await _score_criteria_met(result)

    assert score.value == 1.0
    assert score.pass_ is True
    assert score.explanation == 'No criteria defined for this scenario.'
