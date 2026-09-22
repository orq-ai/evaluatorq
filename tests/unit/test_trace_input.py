"""Trace imports become ordinary evaluatorq datapoints."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Literal

import httpx
import pytest

from evaluatorq import Trace, TraceInput, fetch_traces
from evaluatorq.common.trace_input import _message_candidates, _parse_messages, _trace_from_spans
from evaluatorq.contracts import Message


def _span(
    span_id: str,
    *,
    parent_id: str | None,
    started_at: str,
    span_type: str = 'span.agent',
    attributes: dict[str, Any] | None = None,
    input_: Any = None,
    output: Any = None,
) -> dict[str, Any]:
    return {
        'span_id': span_id,
        'parent_span_id': parent_id,
        'started_at': started_at,
        'type': span_type,
        'attributes': attributes or {},
        'input': input_,
        'output': output,
    }


def test_latest_non_evaluator_leaf_walks_to_conversation_span() -> None:
    spans = [
        _span(
            'agent',
            parent_id='root',
            started_at='2026-01-01T00:00:01Z',
            input_={
                'messages': [
                    {'role': 'user', 'content': 'look up the order'},
                    {
                        'role': 'assistant',
                        'content': None,
                        'tool_calls': [
                            {
                                'id': 'call_1',
                                'type': 'function',
                                'function': {'name': 'lookup_order', 'arguments': '{"id":"123"}'},
                            }
                        ],
                    },
                    {'role': 'tool', 'tool_call_id': 'call_1', 'content': '{"status":"sent"}'},
                ]
            },
            output={'choices': [{'message': {'role': 'assistant', 'content': 'It was sent.'}}]},
        ),
        _span('tool', parent_id='agent', started_at='2026-01-01T00:00:02Z', span_type='span.tool'),
        _span(
            'evaluator',
            parent_id='root',
            started_at='2026-01-01T00:00:03Z',
            span_type='span.chat',
            attributes={'orq.span_type': 'span.evaluator'},
            output={'role': 'assistant', 'content': 'judge output'},
        ),
        _span(
            'evaluator-model',
            parent_id='evaluator',
            started_at='2026-01-01T00:00:04Z',
            span_type='span.chat',
            output={'role': 'assistant', 'content': 'nested judge output'},
        ),
        _span('root', parent_id=None, started_at='2026-01-01T00:00:00Z', span_type='trace'),
    ]

    imported = _trace_from_spans('trace-1', spans)

    assert imported.requested_span_id is None
    assert imported.message_span_id == 'agent'
    assert imported.message_format == 'chat_completions'
    assert [message.role for message in imported.messages] == ['user', 'assistant', 'tool', 'assistant']
    assert imported.tools_called == ['lookup_order']
    assert imported.query == 'look up the order'
    assert imported.import_error is None


def test_trace_messages_remove_only_largest_exact_boundary_overlap() -> None:
    trace = Trace(
        trace_id='trace-overlap',
        input_messages=[
            Message(role='user', content='question'),
            Message(role='assistant', content='answer'),
        ],
        output_messages=[
            Message(role='assistant', content='answer'),
            Message(role='tool', content='result'),
            Message(role='assistant', content='done'),
        ],
    )

    assert [message.content for message in trace.messages] == ['question', 'answer', 'result', 'done']


def test_exact_span_with_no_messages_does_not_walk_to_parent() -> None:
    spans = [
        _span('parent-with-messages', parent_id=None, started_at='2026-01-01T00:00:00Z', input_={'messages': [
            {'role': 'user', 'content': 'question'},
        ]}, output={'role': 'assistant', 'content': 'answer'}),
        _span('child-empty', parent_id='parent-with-messages', started_at='2026-01-01T00:00:01Z'),
    ]

    imported = _trace_from_spans('trace-1', spans, requested_span_id='child-empty')

    assert imported.message_span_id is None
    assert 'child-empty' in (imported.import_error or '')


def test_trace_only_walks_from_latest_span_to_message_parent() -> None:
    spans = [
        _span('parent-with-messages', parent_id=None, started_at='2026-01-01T00:00:00Z', input_={'messages': [
            {'role': 'user', 'content': 'question'},
        ]}, output={'role': 'assistant', 'content': 'answer'}),
        _span('child-empty', parent_id='parent-with-messages', started_at='2026-01-01T00:00:01Z'),
    ]

    imported = _trace_from_spans('trace-1', spans)

    assert imported.requested_span_id is None
    assert imported.message_span_id == 'parent-with-messages'


def test_responses_items_preserve_function_call_trajectory() -> None:
    spans = [
        _span(
            'responses',
            parent_id=None,
            started_at='2026-01-01T00:00:00Z',
            attributes={
                'openresponses.input': [
                    {'role': 'user', 'content': 'check weather'},
                    {
                        'type': 'function_call',
                        'id': 'fc_1',
                        'call_id': 'call_1',
                        'name': 'weather',
                        'arguments': '{"city":"Oslo"}',
                    },
                    {'type': 'function_call_output', 'call_id': 'call_1', 'output': '{"c":4}'},
                ],
                'openresponses.output': [
                    {
                        'type': 'message',
                        'role': 'assistant',
                        'content': [{'type': 'output_text', 'text': 'It is 4 C.'}],
                    }
                ],
            },
        )
    ]

    imported = _trace_from_spans('trace-2', spans)

    assert imported.message_format == 'responses'
    assert [message.role for message in imported.messages] == ['user', 'assistant', 'tool', 'assistant']
    assert imported.messages[1].tool_calls is not None
    assert imported.messages[1].tool_calls[0].item_id == 'fc_1'
    assert imported.tools_called == ['weather']


def test_top_level_responses_input_object_is_detected() -> None:
    messages, detected = _parse_messages(
        {
            'input': [
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [{'type': 'input_text', 'text': 'check weather'}],
                }
            ]
        },
        default_role='user',
    )

    assert detected == 'responses'
    assert [message.content for message in messages] == ['check weather']


@pytest.mark.parametrize(
    ('payload', 'default_role', 'expected_content'),
    [
        ({'input': 'question'}, 'user', 'question'),
        ({'output': 'answer'}, 'assistant', 'answer'),
        ({'input': [{'role': 'user', 'content': 'question'}]}, 'user', 'question'),
        ({'output': [{'role': 'assistant', 'content': 'answer'}]}, 'assistant', 'answer'),
    ],
)
def test_responses_envelopes_preserve_scalar_and_untyped_messages(
    payload: object, default_role: Literal['user', 'assistant'], expected_content: str
) -> None:
    messages, detected = _parse_messages(payload, default_role=default_role)

    assert detected == 'responses'
    assert [message.content for message in messages] == [expected_content]


def test_responses_input_is_not_shadowed_by_scalar_output() -> None:
    messages, detected = _parse_messages(
        {'output': 'answer', 'input': [{'role': 'user', 'content': 'question'}]},
        default_role='user',
    )

    assert detected == 'responses'
    assert [message.content for message in messages] == ['question']


@pytest.mark.parametrize(
    ('payload', 'expected_format'),
    [
        ({'messages': [{'role': 'user', 'content': 'hi'}]}, 'chat_completions'),
        (
            {
                'input': [
                    {
                        'type': 'message',
                        'role': 'user',
                        'content': [{'type': 'input_text', 'text': 'hi'}],
                    }
                ]
            },
            'responses',
        ),
        ([{'role': 'user', 'parts': [{'type': 'text', 'content': 'hi'}]}], 'otel_genai'),
    ],
)
def test_message_formats_normalize(payload: object, expected_format: str) -> None:
    messages, message_format = _parse_messages(payload, default_role='user')
    assert messages[0].content == 'hi'
    assert message_format == expected_format


def test_message_candidates_follow_format_precedence_and_include_events() -> None:
    span = {
        'attributes': {
            'gen_ai.input.messages': '[{"role":"user","content":"flat"}]',
            'gen_ai': {'input': {'messages': [{'role': 'user', 'content': 'nested'}]}},
            'openresponses.input': [{'type': 'message', 'role': 'user', 'content': 'responses'}],
            'input': {'messages': [{'role': 'user', 'content': 'generic'}]},
        },
        'events': [
            {
                'attributes': {
                    'gen_ai.input.messages': [{'role': 'user', 'content': 'event'}],
                }
            }
        ],
        'input': {'messages': [{'role': 'user', 'content': 'top-level'}]},
    }

    candidates = _message_candidates(span, 'input')

    assert candidates[0] == ('otel_genai', '[{"role":"user","content":"flat"}]')
    assert candidates[1] == ('otel_genai', [{'role': 'user', 'content': 'nested'}])
    assert candidates[-2:] == [
        ('otel_genai', [{'role': 'user', 'content': 'event'}]),
        (None, {'messages': [{'role': 'user', 'content': 'top-level'}]}),
    ]


def test_responses_full_envelopes_and_top_level_strings_are_normalized() -> None:
    span = {
        'attributes': {
            'orq.openresponses.request': json.dumps({'input': 'question'}),
            'orq.openresponses.response': json.dumps({'output': 'answer'}),
        }
    }

    imported = _trace_from_spans('trace-envelopes', [{**_span('s', parent_id=None, started_at='2026-01-01T00:00:00Z'), **span}])

    assert [message.content for message in imported.input_messages] == ['question']
    assert [message.content for message in imported.output_messages] == ['answer']
    assert imported.message_format == 'responses'


def test_nested_responses_full_envelopes_are_normalized() -> None:
    span = {
        'attributes': {
            'openresponses': {
                'request': json.dumps({'input': 'nested question'}),
                'response': json.dumps({'output': 'nested answer'}),
            }
        }
    }

    imported = _trace_from_spans(
        'trace-nested-envelopes',
        [{**_span('s', parent_id=None, started_at='2026-01-01T00:00:00Z'), **span}],
    )

    assert [message.content for message in imported.input_messages] == ['nested question']
    assert [message.content for message in imported.output_messages] == ['nested answer']
    assert imported.message_format == 'responses'


@pytest.mark.parametrize(
    ('item', 'expected_name'),
    [
        ({'type': 'function_call', 'call_id': 'call_1', 'id': 'fc_1', 'name': 'lookup', 'arguments': '{}'}, 'lookup'),
        ({'type': 'mcp_call', 'call_id': 'call_2', 'tool_name': 'search', 'arguments': '{}'}, 'search'),
        ({'type': 'custom_tool_call', 'call_id': 'call_3', 'name': 'custom', 'input': '{}'}, 'custom'),
    ],
)
def test_direct_responses_tool_items_are_preserved(item: dict[str, object], expected_name: str) -> None:
    messages, detected = _parse_messages(item, default_role='assistant')

    assert detected == 'responses'
    assert messages[0].tool_calls is not None
    assert messages[0].tool_calls[0].id == item['call_id']
    assert messages[0].tool_calls[0].function.name == expected_name


def test_malformed_chat_identifiers_are_dropped_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        messages, detected = _parse_messages(
            {'role': 'tool', 'content': 'result', 'tool_call_id': 123, 'name': {'bad': 'shape'}},
            default_role='tool',
        )

    assert detected == 'chat_completions'
    assert messages[0].content == 'result'
    assert messages[0].tool_call_id is None
    assert messages[0].name is None
    assert any('tool_call_id must be a string' in record.message for record in caplog.records)
    assert any('name must be a string' in record.message for record in caplog.records)


def test_otel_unknown_and_multimodal_parts_remain_visible() -> None:
    messages, detected = _parse_messages(
        [
            {
                'role': 'user',
                'parts': [
                    {'type': 'text', 'content': 'look'},
                    {'type': 'image', 'url': 'https://example.test/a.png'},
                    {'type': 'unrecognized', 'value': 3},
                    'bad-part',
                ],
            }
        ],
        default_role='user',
    )

    assert detected == 'otel_genai'
    assert messages[0].content == 'look\n[image]\n{"type":"unrecognized","value":3}'


def test_flat_otel_prompt_and_untyped_parts_are_supported() -> None:
    messages, detected = _parse_messages(
        [{'role': 'user', 'parts': [{'content': 'prompt'}]}],
        default_role='user',
    )
    assert detected == 'otel_genai'
    assert messages[0].content == 'prompt'

    spans = [
        _span(
            'flat-prompt',
            parent_id=None,
            started_at='2026-01-01T00:00:00Z',
            attributes={'gen_ai.input.prompt': 'flat prompt', 'gen_ai.output.completion': 'flat completion'},
        )
    ]
    imported = _trace_from_spans('trace-flat-prompt', spans)
    assert [message.content for message in imported.messages] == ['flat prompt', 'flat completion']
    assert imported.message_format == 'otel_genai'


@pytest.mark.parametrize("tool_result_key", ["response", "result"])
def test_flat_otel_genai_attributes_preserve_tool_parts(tool_result_key: str) -> None:
    tool_result = json.dumps(
        [
            {"role": "user", "parts": [{"type": "text", "content": "find it"}]},
            {
                "role": "assistant",
                "parts": [
                    {"type": "tool_call", "id": "call_1", "name": "search", "arguments": {"q": "x"}}
                ],
            },
            {
                "role": "tool",
                "parts": [{"type": "tool_call_response", "id": "call_1", tool_result_key: {"found": True}}],
            },
        ]
    )
    spans = [
        _span(
            'otel',
            parent_id=None,
            started_at='2026-01-01T00:00:00Z',
            attributes={
                'gen_ai.input.messages': tool_result,
                'gen_ai.output.messages': '[{"role":"assistant","parts":[{"type":"text","content":"Found it."}]}]',
            },
        )
    ]

    imported = _trace_from_spans('trace-3', spans)

    assert imported.message_format == 'otel_genai'
    assert [message.role for message in imported.messages] == ['user', 'assistant', 'tool', 'assistant']
    assert imported.tools_called == ['search']
    assert imported.messages[2].content == '{"found":true}'


def test_span_with_only_recorded_input_is_a_valid_trace() -> None:
    imported = _trace_from_spans(
        'trace-broken',
        [
            _span(
                'root',
                parent_id=None,
                started_at='2026-01-01T00:00:00Z',
                input_={'messages': [{'role': 'user', 'content': 'hello'}]},
            )
        ],
    )

    assert [message.content for message in imported.messages] == ['hello']
    assert imported.import_error is None
    datapoint = imported.to_datapoint()
    assert datapoint.inputs['source_trace_id'] == 'trace-broken'
    assert datapoint.inputs['recorded_output'] == []


def test_otel_messages_can_live_on_span_events() -> None:
    spans = [
        {
            'span_id': 'otel-event',
            'started_at': '2026-01-01T00:00:00Z',
            'events': [
                {
                    'name': 'gen_ai.client.inference.operation.details',
                    'attributes': {
                        'gen_ai.input.messages': [
                            {'role': 'user', 'parts': [{'type': 'text', 'content': 'event question'}]}
                        ],
                        'gen_ai.output.messages': [
                            {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'event answer'}]}
                        ],
                    },
                }
            ],
        }
    ]

    imported = _trace_from_spans('trace-event', spans)

    assert imported.message_format == 'otel_genai'
    assert [message.content for message in imported.messages] == ['event question', 'event answer']
    assert imported.import_error is None


def test_trace_becomes_native_datapoint_with_linkage() -> None:
    imported = Trace(
        trace_id='trace-1',
        requested_span_id='requested',
        message_span_id='agent',
        message_format='chat_completions',
        input_messages=[Message(role='user', content='question')],
        output_messages=[Message(role='assistant', content='answer')],
        query='question',
        retrievals=['document excerpt'],
        tools_called=['search'],
        session_id='session-1',
        actor_id='actor-1',
        expected_output='human reference',
    )

    datapoint = imported.to_datapoint()

    assert datapoint.inputs['messages'][-1] == {'role': 'assistant', 'content': 'answer'}
    assert datapoint.inputs['query'] == 'question'
    assert datapoint.inputs['source_trace_id'] == 'trace-1'
    assert datapoint.inputs['source_requested_span_id'] == 'requested'
    assert datapoint.inputs['source_span_id'] == 'agent'
    assert datapoint.inputs['recorded_output'] == [{'role': 'assistant', 'content': 'answer'}]
    assert datapoint.inputs['retrievals'] == ['document excerpt']
    assert datapoint.inputs['tools_called'] == ['search']
    assert datapoint.expected_output == 'human reference'


def test_human_correction_becomes_expected_output_but_rating_does_not() -> None:
    spans = [
        _span(
            'agent',
            parent_id=None,
            started_at='2026-01-01T00:00:00Z',
            input_={'messages': [{'role': 'user', 'content': 'question'}]},
            output={'role': 'assistant', 'content': 'recorded answer'},
        )
    ]

    imported = _trace_from_spans(
        'trace-feedback',
        spans,
        trace_metadata={
            'metadata': {'channel': 'support'},
            'feedback': [
                {'property': 'rating', 'value': ['good']},
                {'property': 'correction', 'value': 'human corrected answer'},
            ]
        },
    )

    assert imported.expected_output == 'human corrected answer'
    assert imported.metadata['channel'] == 'support'
    rating_only = _trace_from_spans(
        'trace-rating',
        spans,
        trace_metadata={'feedback': [{'property': 'rating', 'value': ['good']}]},
    )
    assert rating_only.expected_output is None


def test_trace_input_accepts_one_exact_span() -> None:
    source = TraceInput(trace_id='trace-1', span_id='span-2')
    assert source.trace_id == 'trace-1'
    assert source.span_id == 'span-2'


def test_trace_input_requires_trace_for_span() -> None:
    with pytest.raises(ValueError, match='span_id requires trace_id'):
        TraceInput(span_id='span-2')


def test_trace_input_rejects_explicit_trace_with_query() -> None:
    with pytest.raises(ValueError, match='cannot be combined'):
        TraceInput(trace_id='trace-1', search='checkout')


@pytest.mark.asyncio
async def test_fetch_supports_explicit_trace_and_search() -> None:
    seen_search_body: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == '/v2/traces/v3oql':
            seen_search_body.update(json.loads(request.content))
            return httpx.Response(200, json={'data': [{'trace_id': 'searched'}], 'has_more': False})
        trace_id = request.url.path.split('/')[3]
        return httpx.Response(
            200,
            json=[
                _span(
                    f'span-{trace_id}',
                    parent_id=None,
                    started_at='2026-01-01T00:00:00Z',
                    input_={'messages': [{'role': 'user', 'content': 'q'}]},
                    output={'role': 'assistant', 'content': 'a'},
                )
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        explicit = await fetch_traces(
            TraceInput(trace_id='explicit'),
            api_key='test-key',
            http_client=client,
        )
        searched = await fetch_traces(
            TraceInput(
                search='agent',
                start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                end_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
            ),
            api_key='test-key',
            http_client=client,
        )

    assert [row.trace_id for row in explicit] == ['explicit']
    assert [row.trace_id for row in searched] == ['searched']
    assert seen_search_body['filters']['search'] == 'agent'
    assert seen_search_body['start_date'] == 1767225600000
