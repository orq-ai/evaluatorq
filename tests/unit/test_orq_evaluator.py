"""Orq-hosted evaluators adapt to evaluatorq's scorer contract."""

from __future__ import annotations

from inspect import Parameter, signature
from typing import Any, cast

import pytest

from evaluatorq import DataPoint, EvaluationResult, Trace, orq_evaluator
from evaluatorq.contracts import FunctionCall, Message, StrategyToolCall


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
    evaluator = orq_evaluator(evaluator_id='eval-1', client=client)
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
async def test_orq_evaluator_history_stops_before_user_turn_with_tool_output() -> None:
    client = _Client({'type': 'number', 'value': 1.0})
    evaluator = orq_evaluator(evaluator_id='eval-history', client=client)
    tool_call = StrategyToolCall(
        id='call-1',
        function=FunctionCall(name='lookup', arguments='{}'),
    )
    trace = Trace(
        trace_id='trace-tools',
        input_messages=[
            Message(role='system', content='Be accurate.'),
            Message(role='user', content='earlier question'),
            Message(role='assistant', content='earlier answer'),
            Message(role='user', content='latest question'),
        ],
        output_messages=[
            Message(role='assistant', tool_calls=[tool_call]),
            Message(role='tool', tool_call_id='call-1', name='lookup', content='tool result'),
            Message(role='assistant', content='recorded answer'),
        ],
    )

    await evaluator['scorer']({'data': trace.to_datapoint(), 'output': trace.output_messages})

    assert client.evals.calls[0]['query'] == 'latest question'
    assert client.evals.calls[0]['messages'] == [
        {'role': 'system', 'content': 'Be accurate.'},
        {'role': 'user', 'content': 'earlier question'},
        {'role': 'assistant', 'content': 'earlier answer'},
    ]


@pytest.mark.asyncio
async def test_orq_evaluator_derives_query_from_multimodal_latest_user_turn() -> None:
    client = _Client({'type': 'number', 'value': 1.0})
    evaluator = orq_evaluator(evaluator_id='eval-multimodal', client=client)
    datapoint = DataPoint(
        inputs={
            'messages': [
                {'role': 'user', 'content': 'earlier question'},
                {'role': 'assistant', 'content': 'earlier answer'},
                {
                    'role': 'user',
                    'content': [{'type': 'input_text', 'text': 'latest question'}],
                },
            ]
        }
    )

    await evaluator['scorer']({'data': datapoint, 'output': 'recorded answer'})

    assert client.evals.calls[0]['query'] == 'latest question'
    assert client.evals.calls[0]['messages'] == [
        {'role': 'user', 'content': 'earlier question'},
        {'role': 'assistant', 'content': 'earlier answer'},
    ]


@pytest.mark.asyncio
async def test_orq_evaluator_forwards_empty_reference() -> None:
    client = _Client({'type': 'number', 'value': 1.0})
    evaluator = orq_evaluator(evaluator_id='eval-empty-reference', client=client)
    datapoint = DataPoint(inputs={'messages': [{'role': 'user', 'content': 'question'}]}, expected_output='')

    await evaluator['scorer']({'data': datapoint, 'output': 'answer'})

    assert client.evals.calls[0]['reference'] == ''


@pytest.mark.asyncio
async def test_orq_evaluator_forwards_explicit_model_override() -> None:
    client = _Client({'type': 'number', 'value': 0.9})
    evaluator = orq_evaluator(evaluator_id='eval-llm', model='openai/gpt-5.4-mini', client=client)
    datapoint = DataPoint(inputs={'messages': [{'role': 'user', 'content': 'question'}]})

    await evaluator['scorer']({'data': datapoint, 'output': 'answer'})

    assert client.evals.calls[0]['model'] == 'openai/gpt-5.4-mini'


def test_orq_evaluator_name_includes_model_override() -> None:
    plain = orq_evaluator(evaluator_id='faithfulness', client=_Client({'type': 'number', 'value': 1.0}))
    judged_by_gpt = orq_evaluator(
        evaluator_id='faithfulness', model='openai/gpt-5.6-luna', client=_Client({'type': 'number', 'value': 1.0})
    )
    judged_by_claude = orq_evaluator(
        evaluator_id='faithfulness',
        model='anthropic/claude-sonnet-4-5',
        client=_Client({'type': 'number', 'value': 1.0}),
    )

    assert plain['name'] == 'orq:faithfulness'
    assert judged_by_gpt['name'] == 'orq:faithfulness@openai/gpt-5.6-luna'
    assert judged_by_claude['name'] == 'orq:faithfulness@anthropic/claude-sonnet-4-5'
    assert len({plain['name'], judged_by_gpt['name'], judged_by_claude['name']}) == 3


@pytest.mark.asyncio
async def test_orq_evaluator_resolves_client_lazily_once_across_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    resolve_calls: list[tuple[str | None, str | None]] = []
    shared_client = _Client({'type': 'number', 'value': 1.0})

    def fake_resolve_orq_client(api_key: str | None = None, base_url: str | None = None) -> _Client:
        resolve_calls.append((api_key, base_url))
        return shared_client

    monkeypatch.setattr('evaluatorq.evaluators.resolve_orq_client', fake_resolve_orq_client)

    evaluator = orq_evaluator(evaluator_id='eval-lazy', api_key='sk-test', base_url='https://self.hosted')
    datapoint = DataPoint(inputs={'messages': [{'role': 'user', 'content': 'question'}]})

    assert resolve_calls == []  # constructing the evaluator must not build a client

    for _ in range(5):
        await evaluator['scorer']({'data': datapoint, 'output': 'answer'})

    # Resolved once, reused for every later row — with both credentials forwarded.
    assert resolve_calls == [('sk-test', 'https://self.hosted')]
    assert len(shared_client.evals.calls) == 5


@pytest.mark.asyncio
async def test_orq_boolean_evaluator_maps_value_to_pass() -> None:
    client = _Client({'type': 'boolean', 'value': True})
    evaluator = orq_evaluator(evaluator_id='eval-boolean', client=client)
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
    evaluator = orq_evaluator(evaluator_id='eval-supported', client=client)
    datapoint = DataPoint(inputs={'messages': [{'role': 'user', 'content': 'question'}]})

    result = cast(EvaluationResult, await evaluator['scorer']({'data': datapoint, 'output': 'answer'}))

    assert result.value == expected_value
    assert result.raw_output == payload


@pytest.mark.asyncio
async def test_orq_llm_evaluator_maps_nested_value_and_explanation() -> None:
    client = _Client({'type': 'llm_evaluator', 'value': {'value': False, 'explanation': 'not grounded'}})
    evaluator = orq_evaluator(evaluator_id='eval-llm', client=client)
    datapoint = DataPoint(inputs={'messages': [{'role': 'user', 'content': 'question'}]})

    result = cast(EvaluationResult, await evaluator['scorer']({'data': datapoint, 'output': 'answer'}))

    assert result.value is False
    assert result.pass_ is False
    assert result.explanation == 'not grounded'


@pytest.mark.asyncio
async def test_orq_http_evaluator_maps_nested_value_and_explanation() -> None:
    client = _Client({'type': 'http_eval', 'value': {'value': 0.4, 'explanation': 'partial'}})
    evaluator = orq_evaluator(evaluator_id='eval-http', client=client)
    datapoint = DataPoint(inputs={'messages': [{'role': 'user', 'content': 'question'}]})

    result = cast(EvaluationResult, await evaluator['scorer']({'data': datapoint, 'output': 'answer'}))

    assert result.value == 0.4
    assert result.explanation == 'partial'


@pytest.mark.asyncio
async def test_orq_evaluator_rejects_unknown_response_shape() -> None:
    client = _Client({'type': 'future_type', 'unexpected': 'value'})
    evaluator = orq_evaluator(evaluator_id='eval-future', client=client)
    datapoint = DataPoint(
        inputs={'messages': [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}]}
    )

    with pytest.raises(ValueError, match='unsupported response shape'):
        await evaluator['scorer']({'data': datapoint, 'output': 'answer'})


def test_orq_evaluator_uses_one_keyword_identifier() -> None:
    parameters = signature(orq_evaluator).parameters

    assert parameters['evaluator_id'].kind is Parameter.KEYWORD_ONLY
    assert 'name' not in parameters
