"""Bounded trace projection tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from evaluatorq.trace_finder.models import TraceRecord
from evaluatorq.trace_finder.orq_source import _conversation_messages
from evaluatorq.trace_finder.projection import (
    estimate_tokens,
    project_trace,
)


def test_token_budget_uses_a_conservative_bound_for_dense_punctuation() -> None:
    serialized = ''.join('{}[],:;' for _ in range(100))
    assert estimate_tokens(serialized) == len(serialized.encode('utf-8'))


def test_parts_text_block_with_content_key_is_kept() -> None:
    text = 'x' * 2000
    trace = _trace(messages=({'role': 'user', 'content': [{'type': 'text', 'content': text}]},))
    assert text in project_trace(trace).serialized


def test_chat_shaped_parts_text_is_normalized_into_message_content() -> None:
    messages = _conversation_messages({
        'input': [{'role': 'user', 'parts': [{'type': 'text', 'text': 'Keep this request.'}]}],
        'output': [{'role': 'assistant', 'parts': [{'type': 'text', 'text': 'Keep this answer.'}]}],
    })

    assert [message['content'] for message in messages] == ['Keep this request.', 'Keep this answer.']


def test_dict_content_text_slot_is_truncated_to_fit_the_projection_budget() -> None:
    text = 'prefix-' + 'x' * 2000 + '-tail'
    trace = _trace(messages=({'role': 'user', 'content': {'type': 'text', 'content': text}},))

    projection = project_trace(trace, token_budget=300)

    block = projection.payload['messages'][0]['content'][0]
    assert block['text'].startswith('[... earlier bytes omitted ...]')
    assert block['text'].endswith('-tail')
    assert projection.estimated_tokens <= 300
    assert projection.omitted_bytes > 0


@pytest.mark.parametrize('block_type', ['text', 'input_text', 'output_text'])
@pytest.mark.parametrize('nested_text', [False, True])
def test_truncates_structured_text_blocks_preserving_shape_and_byte_accounting(
    block_type: str, nested_text: bool
) -> None:
    text = 'prefix-' + '😀' * 1000 + '-tail'
    block = {'type': block_type, 'text': {'value': text, 'annotations': []} if nested_text else text}
    image = {'type': 'image_url', 'image_url': {'url': 'https://example.test/picture'}}
    trace = _trace(messages=({'role': 'user', 'content': [image, block]},))

    projection = project_trace(trace, token_budget=450)

    content = projection.payload['messages'][0]['content']
    assert content[0] == image
    assert content[1]['type'] == block_type
    shortened = content[1]['text']['value'] if nested_text else content[1]['text']
    assert shortened.startswith('[... earlier bytes omitted ...]')
    assert shortened.endswith('-tail')
    tail = shortened.removeprefix('[... earlier bytes omitted ...]')
    assert projection.omitted_bytes == len(text.encode()) - len(tail.encode())
    assert projection.omitted_messages == 0
    assert projection.estimated_tokens <= 450
    assert json.loads(projection.serialized) == projection.payload
    assert trace.messages[0]['content'][1] == block


def test_project_trace_keeps_tool_result_categories_but_drops_orphans() -> None:
    trace = _trace(
        status='failed',
        messages=(
            {'role': 'user', 'content': 'Find my order.', 'reasoning': 'recorded private thought'},
            {
                'role': 'assistant',
                'content': 'I will look it up.',
                'thinking': 'recorded private thought',
                'tool_calls': [
                    {
                        'id': 'call_123',
                        'type': 'function',
                        'function': {
                            'name': 'lookup_customer',
                            'arguments': '{"customer_id":"cust_1","labels":["vip"],"reasoning":"discard me"}',
                        },
                    },
                    {
                        'id': 'call_456',
                        'type': 'function',
                        'function': {'name': 'charge_card', 'arguments': 'not valid json'},
                    },
                ],
            },
            {
                'role': 'tool',
                'tool_call_id': 'call_123',
                'content': 'secret tool body: account details',
                'status': 'completed',
            },
            {
                'role': 'tool',
                'tool_call_id': 'call_456',
                'content': 'secret tool body: payment declined',
                'status': 'failed',
            },
            {'role': 'tool', 'tool_call_id': 'orphan', 'content': 'secret orphan tool body'},
            {'role': 'assistant', 'content': 'Your payment could not be processed.', 'reasoning_content': 'discard me'},
        ),
    )

    projection = project_trace(trace)

    assert projection.payload == {
        'trace_status': 'failed',
        'messages': [
            {'role': 'user', 'content': 'Find my order.'},
            {
                'role': 'assistant',
                'content': 'I will look it up.',
                'tool_calls': [
                    {
                        'id': 'call_123',
                        'name': 'lookup_customer',
                        'arguments': {'customer_id': 'cust_1', 'labels': ['vip']},
                        'status': 'completed',
                        'result_category': None,
                    },
                    {
                        'id': 'call_456',
                        'name': 'charge_card',
                        'arguments': 'not valid json',
                        'status': 'error',
                        'result_category': 'provider_error',
                    },
                ],
            },
            {'role': 'assistant', 'content': 'Your payment could not be processed.'},
        ],
    }
    assert projection.omitted_messages == 0
    assert projection.omitted_bytes == 0
    assert 'secret tool body' not in projection.serialized
    assert 'result_excerpt' not in projection.serialized
    assert 'secret orphan tool body' not in projection.serialized
    assert 'lookup_customer' in projection.serialized
    assert 'error' in projection.serialized
    assert json.loads(projection.serialized) == projection.payload


def test_idless_tool_call_does_not_attach_an_idless_result() -> None:
    trace = _trace(messages=(
        {'role': 'assistant', 'tool_calls': [{'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}]},
        {'role': 'tool', 'content': 'unrelated private result', 'status': 'failed'},
    ))

    projection = project_trace(trace)
    call = projection.payload['messages'][0]['tool_calls'][0]

    assert call['status'] == 'pending'
    assert call['result_category'] is None
    assert 'unrelated private result' not in projection.serialized


@pytest.mark.parametrize('failure_signal', [{'status': 'failed'}, {'is_error': True}, {'error': 'unavailable'}])
def test_otel_tool_error_is_projected_with_error_status(failure_signal: dict[str, Any]) -> None:
    messages = _conversation_messages(
        {
            'attributes': {
                'gen_ai': {
                    'output': [
                        {
                            'role': 'assistant',
                            'parts': [
                                {
                                    'type': 'tool_call',
                                    'id': 'c1',
                                    'name': 'lookup_invoice',
                                    'arguments': {},
                                }
                            ],
                        },
                        {
                            'role': 'tool',
                            'parts': [
                                {
                                    'type': 'tool_call_response',
                                    'id': 'c1',
                                    'response': 'invoice service unavailable',
                                    **failure_signal,
                                }
                            ],
                        },
                    ]
                }
            }
        }
    )
    trace = _trace(messages=tuple(messages))

    projection = project_trace(trace)

    projected_call = projection.payload['messages'][0]['tool_calls'][0]
    assert projected_call['status'] == 'error'
    assert projected_call['result_category'] == 'unavailable'


def test_otel_explicit_success_status_precedes_error_prefix_heuristic() -> None:
    messages = _conversation_messages(
        {
            'attributes': {
                'gen_ai': {
                    'output': [
                        {
                            'role': 'assistant',
                            'parts': [
                                {'type': 'tool_call', 'id': 'c1', 'name': 'lookup_invoice', 'arguments': {}}
                            ],
                        },
                        {
                            'role': 'tool',
                            'parts': [
                                {
                                    'type': 'tool_call_response',
                                    'id': 'c1',
                                    'response': 'Error: no previous error was found',
                                    'status': 'completed',
                                }
                            ],
                        },
                    ]
                }
            }
        }
    )
    trace = _trace(messages=tuple(messages))

    projection = project_trace(trace)

    projected_call = projection.payload['messages'][0]['tool_calls'][0]
    assert projected_call['status'] == 'completed'


def test_tool_result_text_mentioning_error_is_not_marked_as_failure() -> None:
    trace = _trace(
        messages=(
            {
                'role': 'assistant',
                'tool_calls': [
                    {'id': 'c1', 'type': 'function', 'function': {'name': 'lookup_invoice', 'arguments': '{}'}}
                ],
            },
            {
                'role': 'tool',
                'tool_call_id': 'c1',
                'content': 'The error shown above was resolved successfully.',
            },
        )
    )

    projection = project_trace(trace)

    assert projection.payload['messages'][0]['tool_calls'][0]['status'] == 'completed'


@pytest.mark.parametrize(('body', 'category'), [
    ('Error: invoice not found. secret=never persist', 'not_found'),
    ('request failed: permission denied; token=opaque', 'permission_denied'),
    ('provider rate limit exceeded', 'rate_limited'),
    ('operation timed out', 'timeout'),
    ('service unavailable', 'unavailable'),
    ('invalid request payload', 'invalid_request'),
    ('Error: opaque provider detail', 'provider_error'),
])
def test_tool_result_persists_only_allowlisted_error_category(body: str, category: str) -> None:
    result = {'role': 'tool', 'tool_call_id': 'c1', 'content': body}
    if not body.startswith('Error:'):
        result['status'] = 'failed'
    trace = _trace(messages=(
        {'role': 'assistant', 'tool_calls': [{'id': 'c1', 'function': {'name': 'lookup', 'arguments': '{}'}}]},
        result,
    ))

    projection = project_trace(trace)
    call = projection.payload['messages'][0]['tool_calls'][0]

    assert call['status'] == 'error'
    assert call['result_category'] == category
    assert category in _allowed_result_categories()
    assert body not in projection.serialized
    assert 'result_excerpt' not in projection.serialized


def test_successful_tool_text_mentioning_failure_category_stays_completed() -> None:
    trace = _trace(messages=(
        {'role': 'assistant', 'tool_calls': [{'id': 'c1', 'function': {'name': 'lookup', 'arguments': '{}'}}]},
        {
            'role': 'tool',
            'tool_call_id': 'c1',
            'content': 'No matching item was not found in cache; the fallback service is available.',
        },
    ))

    call = project_trace(trace).payload['messages'][0]['tool_calls'][0]

    assert call['status'] == 'completed'
    assert call['result_category'] is None


def test_nested_unknown_and_base64_secrets_never_enter_projection() -> None:
    import base64

    secret = 'undocumented-provider-secret-7b91'
    encoded = base64.b64encode(secret.encode()).decode()
    body = json.dumps({
        'error': {
            'provider_payload': {
                'opaque': secret,
                'binary': encoded,
                'unclassified': {'credential_blob': 'nested-live-value'},
            }
        }
    })
    trace = _trace(messages=(
        {'role': 'assistant', 'tool_calls': [{'id': 'c1', 'function': {'name': 'lookup', 'arguments': '{}'}}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': body},
    ))

    projection = project_trace(trace)

    assert projection.payload['messages'][0]['tool_calls'][0]['status'] == 'error'
    assert projection.payload['messages'][0]['tool_calls'][0]['result_category'] == 'provider_error'
    for value in (secret, encoded, 'nested-live-value', body):
        assert value not in projection.serialized


def _allowed_result_categories() -> set[str]:
    return {
        'not_found', 'permission_denied', 'rate_limited', 'timeout', 'unavailable',
        'invalid_request', 'provider_error',
    }


def test_project_trace_keeps_a_complete_newest_suffix_and_reports_discarded_units() -> None:
    old_user = {'role': 'user', 'content': 'u' * 500}
    old_assistant = {'role': 'assistant', 'content': 'a' * 500}
    newest = {'role': 'user', 'content': 'Newest request'}

    projection = project_trace(_trace(messages=(old_user, old_assistant, newest)), token_budget=300)

    assert projection.payload == {'trace_status': 'completed', 'messages': [newest]}
    assert projection.omitted_messages == 2
    assert projection.omitted_bytes == sum(_serialized_bytes(message) for message in (old_user, old_assistant))


def test_project_trace_tail_truncates_an_oversized_newest_message() -> None:
    oversized_content = 'prefix-' + ('😀' * 1_000) + '-tail'

    projection = project_trace(_trace(messages=({'role': 'user', 'content': oversized_content},)), token_budget=240)

    content = projection.payload['messages'][0]['content']
    assert projection.estimated_tokens <= 240
    assert projection.omitted_messages == 0
    assert projection.omitted_bytes > 0
    assert content.startswith('[... earlier bytes omitted ...]')
    assert content.endswith('-tail')
    assert json.loads(projection.serialized) == projection.payload


def test_project_trace_truncates_parts_text_and_keeps_unknown_text_block() -> None:
    trace = _trace(messages=({
        'role': 'user',
        'parts': [{'type': 'custom_text', 'text': 'prefix-' + 'x' * 2000 + '-tail'}],
    },))

    projection = project_trace(trace, token_budget=300)

    text = projection.payload['messages'][0]['parts'][0]['text']
    assert text.startswith('[... earlier bytes omitted ...]')
    assert text.endswith('-tail')
    assert projection.estimated_tokens <= 300


def test_project_trace_omits_large_non_text_media_and_unrelated_metadata() -> None:
    trace = _trace(messages=({
        'role': 'user',
        'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + 'A' * 30_000}}],
        'request_metadata': 'B' * 30_000,
    },))

    projection = project_trace(trace, token_budget=300)

    message = projection.payload['messages'][0]
    assert message['content'] == [{'type': 'image_url', 'omitted': 'non-text content'}]
    assert 'request_metadata' not in message
    assert projection.estimated_tokens <= 300


def test_project_trace_tail_truncates_oversized_parsed_tool_arguments() -> None:
    arguments = json.dumps(
        {'filters': {'labels': ['priority', 'enterprise'], 'query': 'customer-' + ('x' * 2_000)}}
    )
    trace = _trace(
        messages=(
            {
                'role': 'assistant',
                'content': None,
                'tool_calls': [
                    {
                        'id': 'call_789',
                        'type': 'function',
                        'function': {'name': 'search_orders', 'arguments': arguments},
                    }
                ],
            },
            {'role': 'tool', 'tool_call_id': 'call_789', 'content': 'secret tool body', 'status': 'completed'},
        )
    )

    projection = project_trace(trace, token_budget=280)

    projected_call = projection.payload['messages'][0]['tool_calls'][0]
    assert projection.estimated_tokens <= 280
    assert projection.omitted_bytes > 0
    assert projected_call['id'] == 'call_789'
    assert projected_call['name'] == 'search_orders'
    assert projected_call['status'] == 'completed'
    assert projected_call['arguments'].startswith('[... earlier bytes omitted ...]')
    assert json.loads(projection.serialized) == projection.payload


def test_tool_argument_that_fits_retains_its_structure_when_peer_is_truncated() -> None:
    trace = _trace(messages=({
        'role': 'assistant',
        'content': None,
        'tool_calls': [
            {'id': 'one', 'function': {'name': 'small', 'arguments': '{"id":1}'}},
            {'id': 'two', 'function': {'name': 'large', 'arguments': '{"text":"' + 'x' * 2000 + '"}'}},
        ],
    },))

    projection = project_trace(trace, token_budget=390)

    calls = projection.payload['messages'][0]['tool_calls']
    assert calls[0]['arguments'] == {'id': 1}
    assert isinstance(calls[1]['arguments'], str)


def test_projection_bounds_oversized_tool_identifiers() -> None:
    trace = _trace(messages=({
        'role': 'assistant',
        'content': 'Checking the order.',
        'tool_calls': [{'id': 'call-' + 'x' * 20_000, 'function': {'name': 'lookup_' + 'y' * 20_000, 'arguments': '{}'}}],
    },))

    projection = project_trace(trace, token_budget=900)

    call = projection.payload['messages'][0]['tool_calls'][0]
    assert call['id'].startswith('[... earlier bytes omitted ...]')
    assert call['name'].startswith('[... earlier bytes omitted ...]')
    assert projection.estimated_tokens <= 900


def test_projection_marks_a_newest_unit_whose_tool_structure_cannot_fit() -> None:
    trace = _trace(messages=({
        'role': 'assistant',
        'content': 'Checking the order.',
        'tool_calls': [
            {'id': f'call-{index}', 'function': {'name': 'lookup', 'arguments': '{}'}}
            for index in range(100)
        ],
    },))

    projection = project_trace(trace, token_budget=256)

    assert projection.payload['messages'] == [{'role': 'assistant', 'content': '[... earlier bytes omitted ...]'}]
    assert projection.omitted_bytes == _serialized_bytes(trace.messages[0])
    assert projection.estimated_tokens <= 256


def _trace(*, messages: tuple[dict[str, Any], ...], status: str = 'completed') -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id='trace-1',
        span_id='span-1',
        timestamp=datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc),
        messages=messages,
        project='alpha',
        model='gpt-5',
        provider='openai',
        status=status,
        product='chat',
        trace_type='conversation',
    )


def _serialized_bytes(message: dict[str, Any]) -> int:
    return len(json.dumps(message, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8'))
