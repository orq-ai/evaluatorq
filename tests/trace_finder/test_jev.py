"""Public contracts for adapting projected traces to EvaluatorQ."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, cast

import pytest
from loguru import logger

from evaluatorq import DataPoint, DataPointResult, EvaluationResult, EvaluatorScore, JobResult
from evaluatorq.common.judge import EvaluatorResponsePayload, JudgeOutcome
from evaluatorq.trace_finder import jev
from evaluatorq.trace_finder.jev import (
    build_datapoint,
    build_jev_evaluator,
    matches_selection,
    parse_datapoint_result,
    run_jev,
)
from evaluatorq.trace_finder.models import CompiledQuery, JevProjection, TraceRecord


def _compiled(kind: str = 'choice') -> CompiledQuery:
    documents: dict[str, dict[str, Any]] = {
        'choice': {
            'task': {
                'kind': 'choice',
                'instructions': 'Classify the prevailing customer sentiment.',
                'criteria': {'frustrated': 'The customer is frustrated.', 'neutral': 'Neither positive nor negative.'},
                'state': {},
            },
            'selection': {'kind': 'values', 'values': ['frustrated']},
        },
        'noul': {
            'task': {
                'kind': 'noul',
                'instructions': 'Does the customer ask to cancel?',
                'state': {},
                'noul_threshold': 0.7,
            },
            'selection': {'kind': 'values', 'values': [True]},
        },
        'score': {
            'task': {
                'kind': 'score',
                'instructions': 'Score refund-request strength.',
                'criteria': ['No request.', 'Explicit request.'],
                'state': {},
            },
            'selection': {'kind': 'threshold', 'operator': 'gte', 'value': 0.75},
        },
    }
    return CompiledQuery.model_validate(documents[kind])


def _trace() -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id='trace-1',
        span_id='span-1',
        timestamp=datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc),
        messages=({'role': 'user', 'content': 'I want my money back.'},),
        project='alpha',
        model='gpt-5',
        provider='openai',
        status='completed',
        product='chat',
        trace_type='conversation',
    )


def _projection() -> JevProjection:
    return JevProjection(
        payload={'trace_status': 'completed', 'messages': [{'role': 'user', 'content': 'I want my money back.'}]},
        serialized='{"messages":[],"trace_status":"completed"}',
        estimated_tokens=16,
        omitted_messages=0,
        omitted_bytes=0,
    )


def test_build_datapoint_preserves_identifiers_projection_and_replay_marker() -> None:
    assert build_datapoint(_trace(), _projection()) == DataPoint(
        inputs={
            'trace_id': 'trace-1',
            'span_id': 'span-1',
            'jev_state': _projection().payload,
            'messages': [{'role': 'assistant', 'content': 'JEV state prepared'}],
        }
    )


@pytest.mark.parametrize('kind', ['choice', 'noul', 'score'])
@pytest.mark.asyncio
async def test_build_jev_evaluator_passes_exact_question_and_state(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_run_judge(**kwargs: Any) -> JudgeOutcome:
        calls.append(kwargs)
        value: str | bool | float = {'choice': 'frustrated', 'noul': True, 'score': 1.0}[kind]
        return JudgeOutcome(
            payload=EvaluatorResponsePayload(value=value, explanation='classified', abstain=False),
            raw_output={'type': kind, kind: value},
            endpoint='classify',
        )

    monkeypatch.setattr(jev, 'run_judge', fake_run_judge)
    compiled = _compiled(kind)
    evaluator = build_jev_evaluator(compiled, model='typesafe/jev-latest', client=cast(Any, 'client'))

    scorer = cast(Any, evaluator['scorer'])
    result = await scorer({'data': build_datapoint(_trace(), _projection()), 'output': 'JEV state prepared'})

    assert evaluator['name'] == 'jev'
    assert result.value in {'frustrated', True, 1.0}
    assert calls[0]['model'] == 'typesafe/jev-latest'
    assert calls[0]['cfg'].timeout_ms == 90_000
    assert calls[0]['prompt_template'] == ''
    assert calls[0]['replacements'] == {}
    assert calls[0]['classify'] == compiled.task.model_copy(update={'state': _projection().payload})


@pytest.mark.parametrize(
    ('kind', 'value'),
    [
        ('choice', 'unknown'),
        ('choice', True),
        ('noul', 'true'),
        ('score', True),
        ('score', 1.1),
        ('score', float('nan')),
        ('score', -0.1),
    ],
)
def test_invalid_verdict_is_a_terminal_error(kind: str, value: str | bool | float) -> None:
    classification = parse_datapoint_result(_result(value=value), _compiled(kind))
    assert classification.error is not None
    assert classification.value is None
    assert not classification.matched


def _result(
    *,
    value: str | bool | float = 'frustrated',
    raw_answer: dict[str, Any] | None = None,
    datapoint_error: str | None = None,
    job_error: str | None = None,
    evaluator_error: str | None = None,
    job_results: list[JobResult] | None = None,
) -> DataPointResult:
    if job_results is None:
        raw_output = None if raw_answer is None else {'jury': {'votes': [{'repetitions': [{'raw_output': raw_answer}]}]}}
        score = EvaluatorScore(
            evaluator_name='jev',
            error=evaluator_error,
            score=EvaluationResult(value=value, explanation='The customer is frustrated.', raw_output=raw_output),
        )
        job_results = [
            JobResult(job_name='replay', output='JEV state prepared', error=job_error, evaluator_scores=[score])
        ]
    return DataPointResult(
        data_point=build_datapoint(_trace(), _projection()), error=datapoint_error, job_results=job_results
    )


def _result_from_evaluation(evaluation: EvaluationResult) -> DataPointResult:
    return DataPointResult(
        data_point=build_datapoint(_trace(), _projection()),
        job_results=[
            JobResult(
                job_name='replay',
                output='JEV state prepared',
                evaluator_scores=[EvaluatorScore(evaluator_name='jev', score=evaluation)],
            )
        ],
    )


def test_outcome_result_maps_clean_abstention_to_an_inconclusive_classification() -> None:
    evaluation = jev._outcome_result(
        JudgeOutcome(
            payload=EvaluatorResponsePayload(value=None, explanation='Unable to classify.', abstain=True),
            raw_output={'abstain': True},
            endpoint='classify',
        ),
        model='typesafe/jev-latest',
    )

    jury = cast(dict[str, Any], evaluation.raw_output)['jury']
    vote = jury['votes'][0]
    assert vote['success'] is True
    assert vote['abstained'] is True
    assert vote['value'] is None
    assert vote['error'] is None
    assert vote['repetitions_failed'] == 0
    assert jury['judges_succeeded'] == 0
    assert jury['judges_failed'] == 0
    assert jury['inconclusive'] is True

    classification = parse_datapoint_result(_result_from_evaluation(evaluation), _compiled())
    assert classification.matched is False
    assert classification.error == 'judge abstained'


def test_parse_datapoint_result_extracts_label_confidence_probabilities_and_raw_result() -> None:
    result = _result(
        raw_answer={'choice': 'frustrated', 'confidence': 0.86, 'probabilities': {'frustrated': 0.86, 'neutral': 0.14}}
    )

    classification = parse_datapoint_result(result, _compiled())

    assert classification.value == 'frustrated'
    assert classification.confidence == 0.86
    assert classification.probabilities == {'frustrated': 0.86, 'neutral': 0.14}
    assert classification.matched is True
    assert classification.error is None
    assert classification.summary == 'The customer is frustrated.'
    assert classification.raw_result == result.model_dump(mode='json', by_alias=True)


def test_parse_datapoint_result_keeps_absent_classification_details_absent() -> None:
    classification = parse_datapoint_result(_result(raw_answer={'choice': 'frustrated'}), _compiled())
    assert classification.confidence is None
    assert classification.probabilities is None


def test_parse_datapoint_result_returns_terminal_classification_for_malformed_tree() -> None:
    result = _result()
    assert result.job_results is not None
    assert result.job_results[0].evaluator_scores is not None
    result.job_results[0].evaluator_scores[0].score.raw_output = {'jury': {'votes': []}}

    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(message.record['message']), level='WARNING')
    try:
        classification = parse_datapoint_result(result, _compiled())
    finally:
        logger.remove(sink_id)

    assert classification.value is None
    assert not classification.matched
    assert classification.error == 'malformed JEV result tree'
    assert classification.raw_result == result.model_dump(mode='json', by_alias=True)
    assert any(
        'trace-1' in message and 'span-1' in message and 'malformed JEV result tree' in message for message in messages
    )


@pytest.mark.parametrize(
    ('compiled', 'value', 'matched'),
    [
        (_compiled(), 'frustrated', True),
        (_compiled(), 'neutral', False),
        (_compiled('noul'), True, True),
        (_compiled('noul'), False, False),
        (_compiled('score'), 0.75, True),
        (_compiled('score'), 0.749, False),
    ],
)
def test_matches_selection_is_inclusive_at_threshold_and_exact_for_values(
    compiled: CompiledQuery, value: object, matched: bool
) -> None:
    assert matches_selection(value, compiled) is matched


@pytest.mark.parametrize(
    ('result', 'message'),
    [
        (_result(datapoint_error='input unavailable'), 'input unavailable'),
        (_result(job_error='synthetic replay failed'), 'synthetic replay failed'),
        (_result(evaluator_error='judge timed out'), 'judge timed out'),
        (_result(job_results=[]), 'expected exactly one job result'),
        (
            _result(job_results=[JobResult(job_name='one', output='x'), JobResult(job_name='two', output='y')]),
            'expected exactly one job result',
        ),
    ],
)
def test_parse_datapoint_result_returns_terminal_classification_for_error_tree(
    result: DataPointResult, message: str
) -> None:
    classification = parse_datapoint_result(result, _compiled())
    assert classification.trace_id == 'trace-1'
    assert classification.span_id == 'span-1'
    assert classification.value is None
    assert not classification.matched
    assert classification.error is not None
    assert message in classification.error


@pytest.mark.asyncio
async def test_run_jev_uses_matching_parallelism_and_calls_terminal_callback_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result(raw_answer={'choice': 'frustrated'})
    received: list[tuple[str, dict[str, Any]]] = []
    completed: list[object] = []

    def fake_build_jev_evaluator(*_: object, **__: object) -> dict[str, str]:
        return {'name': 'jev'}

    monkeypatch.setattr(jev, 'build_jev_evaluator', fake_build_jev_evaluator)

    async def fake_evaluatorq(name: str, **kwargs: Any) -> list[DataPointResult]:
        received.append((name, kwargs))
        await kwargs['on_datapoint_complete'](result)
        return [result]

    async def on_complete(classification: object) -> None:
        completed.append(classification)

    monkeypatch.setattr(jev, 'evaluatorq', fake_evaluatorq)
    projections = {'trace-1': _projection()}

    classifications = await run_jev(
        (_trace(),),
        projections,
        _compiled(),
        model='typesafe/jev-latest',
        client=cast(Any, 'client'),
        parallelism=3,
        on_complete=on_complete,
    )

    assert len(completed) == 1
    assert classifications[0].trace_id == 'trace-1'
    assert received[0][0] == 'jev-trace-finder'
    kwargs = received[0][1]
    assert kwargs['data'] == [build_datapoint(_trace(), _projection())]
    assert kwargs['evaluators'] == [{'name': 'jev'}]
    assert kwargs['inference'] is False
    assert kwargs['datapoint_parallelism'] == 3
    assert kwargs['llm_parallelism'] == 3
    assert kwargs['print_results'] is False
    assert kwargs['_send_results'] is False
