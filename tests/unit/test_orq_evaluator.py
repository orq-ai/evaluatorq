"""Orq-hosted evaluators adapt to evaluatorq's scorer contract."""

from __future__ import annotations

from typing import Any, cast

import pytest

from evaluatorq import DataPoint, EvaluationResult, Trace, orq_evaluator
from evaluatorq.contracts import Message


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        return self.payload


class _Evals:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def invoke_async(self, **kwargs: Any) -> _Response:
        self.calls.append(kwargs)
        return _Response(self.response)


class _Client:
    def __init__(self, response: dict[str, Any]) -> None:
        self.evals = _Evals(response)


@pytest.mark.asyncio
async def test_orq_evaluator_maps_trace_row_to_invoke_request() -> None:
    client = _Client({
        'type': 'llm_evaluator',
        'value': {
            'value': 0.75,
            'explanation': 'mostly grounded',
            'trace_id': 'evaluator-trace',
            'span_id': 'evaluator-span',
        },
    })
    evaluator = orq_evaluator('eval-1', client=client)
    trace = Trace(
        trace_id='trace-1',
        query='latest question',
        input_messages=[
            Message(role='system', content='Be accurate.'),
            Message(role='user', content='earlier question'),
            Message(role='assistant', content='earlier answer'),
            Message(role='user', content='latest question'),
        ],
        output_messages=[Message(role='assistant', content='recorded answer')],
        retrievals=[],
    )
    datapoint = trace.to_datapoint()

    result = cast(
        EvaluationResult,
        await evaluator['scorer']({'data': datapoint, 'output': trace.output_messages}),
    )

    assert evaluator['name'] == 'orq:eval-1'
    assert result.value == 0.75
    assert result.explanation == 'mostly grounded'
    assert result.pass_ is None
    assert result.raw_output is not None
    assert result.raw_output['value']['trace_id'] == 'evaluator-trace'
    assert client.evals.calls[0] == {
        'id': 'eval-1',
        'query': 'latest question',
        'output': 'recorded answer',
        'reference': None,
        'retrievals': [],
        'messages': [
            {'role': 'system', 'content': 'Be accurate.'},
            {'role': 'user', 'content': 'earlier question'},
            {'role': 'assistant', 'content': 'earlier answer'},
        ],
    }


@pytest.mark.asyncio
async def test_orq_evaluator_forwards_explicit_model_override() -> None:
    client = _Client({'type': 'number', 'value': 0.9})
    evaluator = orq_evaluator('eval-llm', model='openai/gpt-5.4-mini', client=client)
    datapoint = DataPoint(inputs={'messages': [{'role': 'user', 'content': 'question'}]})

    await evaluator['scorer']({'data': datapoint, 'output': 'answer'})

    assert client.evals.calls[0]['model'] == 'openai/gpt-5.4-mini'


@pytest.mark.asyncio
async def test_orq_boolean_evaluator_maps_value_to_pass() -> None:
    client = _Client({'type': 'boolean', 'value': True})
    evaluator = orq_evaluator('eval-boolean', client=client)
    datapoint = DataPoint(
        inputs={
            'messages': [
                {'role': 'user', 'content': 'question'},
                {'role': 'assistant', 'content': 'answer'},
            ]
        }
    )

    result = cast(EvaluationResult, await evaluator['scorer']({'data': datapoint, 'output': 'answer'}))

    assert result.value is True
    assert result.pass_ is True


@pytest.mark.parametrize(
    ('payload', 'expected_value'),
    [
        ({'type': 'string', 'value': 'clear'}, 'clear'),
        ({'type': 'number', 'value': 0.85}, 0.85),
        ({'type': 'string_array', 'values': ['grounded', 'concise']}, {'values': ['grounded', 'concise']}),
        (
            {'type': 'rouge_n', 'value': {'rouge_1': {'fmeasure': 0.8}}},
            {'rouge_1': {'fmeasure': 0.8}},
        ),
        (
            {'type': 'bert_score', 'value': {'f1': 0.7, 'precision': 0.8, 'recall': 0.6}},
            {'f1': 0.7, 'precision': 0.8, 'recall': 0.6},
        ),
        ({'type': 'structured', 'value': {'correct': True, 'style': 'clear'}}, {'correct': True, 'style': 'clear'}),
    ],
)
@pytest.mark.asyncio
async def test_orq_evaluator_maps_supported_response_types(
    payload: dict[str, Any], expected_value: Any,
) -> None:
    client = _Client(payload)
    evaluator = orq_evaluator('eval-supported', client=client)
    datapoint = DataPoint(inputs={'messages': [{'role': 'user', 'content': 'question'}]})

    result = cast(EvaluationResult, await evaluator['scorer']({'data': datapoint, 'output': 'answer'}))

    assert result.value == expected_value
    assert result.raw_output == payload


@pytest.mark.asyncio
async def test_orq_llm_evaluator_maps_nested_value_and_explanation() -> None:
    client = _Client({'type': 'llm_evaluator', 'value': {'value': False, 'explanation': 'not grounded'}})
    evaluator = orq_evaluator('eval-llm', client=client)
    datapoint = DataPoint(inputs={'messages': [{'role': 'user', 'content': 'question'}]})

    result = cast(EvaluationResult, await evaluator['scorer']({'data': datapoint, 'output': 'answer'}))

    assert result.value is False
    assert result.pass_ is False
    assert result.explanation == 'not grounded'


@pytest.mark.asyncio
async def test_orq_http_evaluator_maps_nested_value_and_explanation() -> None:
    client = _Client({'type': 'http_eval', 'value': {'value': 0.4, 'explanation': 'partial'}})
    evaluator = orq_evaluator('eval-http', client=client)
    datapoint = DataPoint(inputs={'messages': [{'role': 'user', 'content': 'question'}]})

    result = cast(EvaluationResult, await evaluator['scorer']({'data': datapoint, 'output': 'answer'}))

    assert result.value == 0.4
    assert result.explanation == 'partial'


@pytest.mark.asyncio
async def test_orq_evaluator_rejects_unknown_response_shape() -> None:
    client = _Client({'type': 'future_type', 'unexpected': 'value'})
    evaluator = orq_evaluator('eval-future', client=client)
    datapoint = DataPoint(
        inputs={'messages': [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}]}
    )

    with pytest.raises(ValueError, match='unsupported response shape'):
        await evaluator['scorer']({'data': datapoint, 'output': 'answer'})
