"""Judge reasoning must reach the evaluator EvaluationResult.

Covers the sim scorer adapter (explanation + pass_ derived from the judge's
reasoning, numeric value preserved).
"""

# ruff: noqa: S101

from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# The structured half: `raw_output` carries what the explanation flattens to prose.
# ---------------------------------------------------------------------------


async def _score(name: str, result: SimulationResult) -> EvaluationResult:
    dp = DataPoint(inputs={})
    evaluator = _adapt_simulation_scorer(name, get_evaluator(name), {id(dp): result})
    score = await evaluator['scorer']({'data': dp, 'output': {}})
    assert isinstance(score, EvaluationResult)
    return score


def _result_with_criteria() -> SimulationResult:
    return _make_result(
        goal_achieved=False,
        criteria_verified=True,
        criteria_results={'greeted user': True, 'did not leak PII': False},
        metadata={
            'criteria_meta': [
                {
                    'id': 'criteria_0',
                    'description': 'greeted user',
                    'type': 'must_happen',
                    'passed': True,
                    'audited': True,
                    'evidence': 'turn 1: "hi there"',
                },
                {'id': 'criteria_1', 'description': 'leaked PII', 'type': 'must_not_happen', 'passed': False},
            ]
        },
    )


@pytest.mark.asyncio
async def test_criteria_met_raw_output_validates_back_into_criteria_meta() -> None:
    from evaluatorq.simulation.types import CriteriaMeta

    score = await _score('criteria_met', _result_with_criteria())

    assert score.raw_output is not None
    records = score.raw_output['criteria']
    # JSON-safe on the wire, and lossless: it round-trips back into the model it came from.
    assert json.loads(json.dumps(records)) == records
    parsed = [CriteriaMeta.model_validate(record) for record in records]
    assert [c.id for c in parsed] == ['criteria_0', 'criteria_1']
    assert [c.passed for c in parsed] == [True, False]
    assert parsed[0].evidence == 'turn 1: "hi there"'
    assert parsed[1].type == 'must_not_happen'


@pytest.mark.asyncio
async def test_criteria_met_raw_output_reports_invalid_entries() -> None:
    result = _make_result(
        goal_achieved=True,
        criteria_verified=True,
        metadata={
            'criteria_meta': [
                {'id': 'criteria_0', 'description': 'greeted user', 'type': 'must_happen', 'passed': True},
                {'nonsense': True},
            ]
        },
    )
    score = await _score('criteria_met', result)

    assert score.raw_output is not None
    assert len(score.raw_output['criteria']) == 1
    assert score.raw_output['invalid'] == [repr({'nonsense': True})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'overrides',
    [
        pytest.param({'terminated_by': TerminatedBy.error}, id='errored-run'),
        pytest.param({'criteria_verified': False}, id='unverified-run'),
    ],
)
async def test_criteria_met_publishes_no_records_for_an_unaudited_run(overrides: dict[str, Any]) -> None:
    """Those records describe verdicts the judge never reached — publishing them beside pass=False would invite trust."""
    result = _result_with_criteria()
    for key, value in overrides.items():
        setattr(result, key, value)
    score = await _score('criteria_met', result)

    assert score.pass_ is False
    assert score.raw_output is None


@pytest.mark.asyncio
async def test_criteria_met_raw_output_is_none_when_there_are_no_criteria() -> None:
    score = await _score('criteria_met', _make_result(goal_achieved=True, criteria_verified=True))

    assert score.explanation == 'No criteria defined for this scenario.'
    assert score.raw_output is None


@pytest.mark.asyncio
async def test_conversation_quality_raw_output_recombines_to_the_score() -> None:
    result = _make_result(goal_achieved=True, turn_count=4, criteria_results={'a': True, 'b': False})
    score = await _score('conversation_quality', result)

    assert score.value == 0.82
    assert score.raw_output is not None
    components = score.raw_output['components']
    weights = score.raw_output['weights']
    assert components == {'goal_achieved': 1.0, 'criteria_met': 0.5, 'turn_efficiency': 0.9}
    recombined = sum(value * weights[name] for name, value in components.items())
    assert round(recombined * 100) / 100 == score.value
    assert json.loads(json.dumps(score.raw_output)) == score.raw_output


@pytest.mark.asyncio
async def test_a_user_supplied_conversation_quality_scorer_degrades_to_no_raw_output() -> None:
    """A plain float carries no breakdown; the adapter must report none rather than invent one."""
    result = _make_result(goal_achieved=True)
    dp = DataPoint(inputs={})
    evaluator = _adapt_simulation_scorer('conversation_quality', lambda _result: 0.5, {id(dp): result})
    score = await evaluator['scorer']({'data': dp, 'output': {}})
    assert isinstance(score, EvaluationResult)

    assert score.value == 0.5
    assert score.raw_output is None


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['goal_achieved', 'turn_efficiency'])
async def test_the_other_built_in_scorers_report_no_raw_output(name: str) -> None:
    score = await _score(name, _result_with_criteria())

    assert score.raw_output is None
