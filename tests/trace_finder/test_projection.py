"""Bounded trace projection tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pytest

from evaluatorq.trace_finder.models import TraceRecord
from evaluatorq.trace_finder.projection import project_trace


@pytest.mark.parametrize('block_type', ['text', 'input_text', 'output_text'])
@pytest.mark.parametrize('nested_text', [False, True])
def test_truncates_structured_text_blocks_preserving_shape_and_byte_accounting(
    block_type: str, nested_text: bool
) -> None:
    text = 'prefix-' + '😀' * 1000 + '-tail'
    block = {'type': block_type, 'text': {'value': text, 'annotations': []} if nested_text else text}
    image = {'type': 'image_url', 'image_url': {'url': 'https://example.test/picture'}}
    trace = _trace(messages=({'role': 'user', 'content': [image, block]},))

    projection = project_trace(trace, token_budget=150)

    content = projection.payload['messages'][0]['content']
    assert content[0] == image
    assert content[1]['type'] == block_type
    shortened = content[1]['text']['value'] if nested_text else content[1]['text']
    assert shortened.startswith('[... earlier bytes omitted ...]')
    assert shortened.endswith('-tail')
    tail = shortened.removeprefix('[... earlier bytes omitted ...]')
    assert projection.omitted_bytes == len(text.encode()) - len(tail.encode())
    assert projection.omitted_messages == 0
    assert projection.estimated_tokens <= 150
    assert json.loads(projection.serialized) == projection.payload
    assert trace.messages[0]['content'][1] == block


def test_project_trace_preserves_visible_tool_structure_without_tool_bodies() -> None:
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
                    },
                    {
                        'id': 'call_456',
                        'name': 'charge_card',
                        'arguments': 'not valid json',
                        'status': 'error',
                    },
                ],
            },
            {'role': 'assistant', 'content': 'Your payment could not be processed.'},
        ],
    }
    assert projection.omitted_messages == 0
    assert projection.omitted_bytes == 0
    assert 'secret tool body' not in projection.serialized
    assert 'lookup_customer' in projection.serialized
    assert 'error' in projection.serialized
    assert json.loads(projection.serialized) == projection.payload


def test_project_trace_keeps_a_complete_newest_suffix_and_reports_discarded_units() -> None:
    old_user = {'role': 'user', 'content': 'u' * 500}
    old_assistant = {'role': 'assistant', 'content': 'a' * 500}
    newest = {'role': 'user', 'content': 'Newest request'}

    projection = project_trace(_trace(messages=(old_user, old_assistant, newest)), token_budget=100)

    assert projection.payload == {'trace_status': 'completed', 'messages': [newest]}
    assert projection.omitted_messages == 2
    assert projection.omitted_bytes == sum(_serialized_bytes(message) for message in (old_user, old_assistant))


def test_project_trace_tail_truncates_an_oversized_newest_message() -> None:
    oversized_content = 'prefix-' + ('😀' * 1_000) + '-tail'

    projection = project_trace(_trace(messages=({'role': 'user', 'content': oversized_content},)), token_budget=80)

    content = projection.payload['messages'][0]['content']
    assert projection.estimated_tokens <= 80
    assert projection.omitted_messages == 0
    assert projection.omitted_bytes > 0
    assert content.startswith('[... earlier bytes omitted ...]')
    assert content.endswith('-tail')
    assert json.loads(projection.serialized) == projection.payload


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

    projection = project_trace(trace, token_budget=80)

    projected_call = projection.payload['messages'][0]['tool_calls'][0]
    assert projection.estimated_tokens <= 80
    assert projection.omitted_bytes > 0
    assert projected_call['id'] == 'call_789'
    assert projected_call['name'] == 'search_orders'
    assert projected_call['status'] == 'completed'
    assert projected_call['arguments'].startswith('[... earlier bytes omitted ...]')
    assert json.loads(projection.serialized) == projection.payload


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
