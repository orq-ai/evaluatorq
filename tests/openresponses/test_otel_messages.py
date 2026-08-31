"""Responses items -> OTel parts messages.

The defect these guard: writing raw Responses items into gen_ai.input.messages
loses every role-less item (function_call, function_call_output, reasoning),
because consumers key on `role`. The parts shape keeps them.
"""

# ruff: noqa: S101

from __future__ import annotations

import json
from unittest.mock import MagicMock

from evaluatorq.openresponses.otel_messages import items_to_input_messages, items_to_output_messages
from evaluatorq.openresponses.tracing import record_openresponses_request

_TOOL_TRANSCRIPT = [
    {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': 'weather in Berlin?'}]},
    {'type': 'function_call', 'call_id': 'c1', 'name': 'get_weather', 'arguments': '{"city": "Berlin"}'},
    {'type': 'function_call_output', 'call_id': 'c1', 'output': '12C'},
]


def test_tool_call_survives_as_an_assistant_part() -> None:
    messages = items_to_input_messages(_TOOL_TRANSCRIPT)

    assert [m['role'] for m in messages] == ['user', 'assistant', 'tool']
    call = messages[1]['parts'][0]
    assert call['type'] == 'tool_call'
    assert call['id'] == 'c1'
    assert call['name'] == 'get_weather'
    # Arguments arrive as a JSON string and are parsed, as the Go side does.
    assert call['arguments'] == {'city': 'Berlin'}
    result = messages[2]['parts'][0]
    assert result == {'type': 'tool_call_response', 'id': 'c1', 'response': '12C'}


def test_plain_message_content_becomes_text_parts() -> None:
    messages = items_to_input_messages(_TOOL_TRANSCRIPT[:1])
    assert messages[0]['parts'] == [{'type': 'text', 'content': 'weather in Berlin?'}]


def test_bare_role_dict_is_treated_as_a_message() -> None:
    # No 'type' key: the Responses API accepts this shorthand and so do callers.
    assert items_to_input_messages([{'role': 'user', 'content': 'hi'}]) == [
        {'role': 'user', 'parts': [{'type': 'text', 'content': 'hi'}]}
    ]


def test_string_input_becomes_one_user_message() -> None:
    assert items_to_input_messages('hi') == [{'role': 'user', 'parts': [{'type': 'text', 'content': 'hi'}]}]


def test_reasoning_item_keeps_its_own_part_type() -> None:
    messages = items_to_input_messages([
        {'type': 'reasoning', 'content': [{'type': 'reasoning_text', 'text': 'thinking'}], 'summary': []}
    ])
    assert messages == [{'role': 'assistant', 'parts': [{'type': 'reasoning', 'content': 'thinking'}]}]


def test_reasoning_summary_is_not_duplicated() -> None:
    messages = items_to_input_messages([
        {
            'type': 'reasoning',
            'content': [{'type': 'summary_text', 'text': 'same'}],
            'summary': [{'type': 'summary_text', 'text': 'same'}],
        }
    ])
    assert messages[0]['parts'] == [{'type': 'reasoning', 'content': 'same'}]


def test_encrypted_reasoning_is_marked_rather_than_dropped() -> None:
    messages = items_to_input_messages([{'type': 'reasoning', 'encrypted_content': 'x', 'content': []}])
    assert messages[0]['parts'] == [{'type': 'reasoning', 'content': '[encrypted]'}]


def test_unknown_item_type_is_kept_as_data() -> None:
    # An item type neither side maps (web search, image generation) must not
    # vanish; it lands in a generic part carrying the whole item.
    item = {'type': 'web_search_call', 'id': 'ws1', 'status': 'completed'}
    messages = items_to_input_messages([item])
    assert messages == [{'role': 'assistant', 'parts': [{'type': 'data', 'content': item}]}]


def test_orq_builtin_tool_call_becomes_a_tool_call_and_its_result() -> None:
    # Orq's own tools are typed `orq:<name>`, not `function_call` — the live
    # agent run emitted these and they were landing in the generic bucket.
    messages = items_to_input_messages([
        {'type': 'orq:google_search', 'call_id': 'c7', 'name': 'google_search', 'arguments': '{"q": "x"}', 'result': {'hits': 1}}
    ])
    assert messages[0] == {
        'role': 'assistant',
        'parts': [{'type': 'tool_call', 'id': 'c7', 'name': 'google_search', 'arguments': {'q': 'x'}}],
    }
    assert messages[1] == {'role': 'tool', 'parts': [{'type': 'tool_call_response', 'id': 'c7', 'response': {'hits': 1}}]}


def test_mcp_and_custom_tool_calls_map_to_tool_calls() -> None:
    messages = items_to_input_messages([
        {'type': 'mcp_call', 'call_id': 'm1', 'tool_name': 'search', 'arguments': '{}', 'error': 'boom'},
        {'type': 'custom_tool_call', 'call_id': 'k1', 'name': 'grep', 'input': 'pattern'},
        {'type': 'custom_tool_call_output', 'call_id': 'k1', 'output': 'match'},
    ])
    assert messages[0]['parts'][0] == {'type': 'tool_call', 'id': 'm1', 'name': 'search', 'arguments': {}}
    assert messages[1] == {'role': 'tool', 'parts': [{'type': 'tool_call_response', 'id': 'm1', 'response': {'error': 'boom'}}]}
    assert messages[2]['parts'][0] == {'type': 'tool_call', 'id': 'k1', 'name': 'grep', 'arguments': 'pattern'}
    assert messages[3] == {'role': 'tool', 'parts': [{'type': 'tool_call_response', 'id': 'k1', 'response': 'match'}]}


def test_output_messages_carry_finish_reason_and_drop_tool_results() -> None:
    messages = items_to_output_messages(
        [
            {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'sunny'}]},
            {'type': 'function_call', 'call_id': 'c1', 'name': 'get_weather', 'arguments': '{}'},
            {'type': 'function_call_output', 'call_id': 'c1', 'output': '12C'},
        ],
        'completed',
    )

    assert len(messages) == 2
    assert all(m['finish_reason'] == 'completed' for m in messages)
    assert messages[0]['parts'] == [{'type': 'text', 'content': 'sunny'}]
    assert messages[1]['parts'][0]['type'] == 'tool_call'


def test_request_span_records_parts_in_gen_ai_and_raw_items_in_openresponses() -> None:
    span = MagicMock()
    attrs: dict[str, object] = {}
    span.set_attribute.side_effect = lambda k, v: attrs.__setitem__(k, v)

    record_openresponses_request(
        span,
        {'model': 'openai/gpt-4o-mini', 'input': _TOOL_TRANSCRIPT, 'instructions': 'be brief'},
    )

    recorded = json.loads(str(attrs['gen_ai.input.messages']))
    assert [m['role'] for m in recorded] == ['user', 'assistant', 'tool']
    # The raw items stay verbatim under the key Orq's gateway uses for them.
    assert json.loads(str(attrs['openresponses.input'])) == _TOOL_TRANSCRIPT
    assert attrs['gen_ai.system_instructions'] == 'be brief'
