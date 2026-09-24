"""MCP and custom tool calls reach AgentResponse, and agree with the OTel trace view.

Fixtures are built from the openai SDK's own Responses models, so a field rename
upstream fails here instead of confirming a hand-written guess at the shape.
"""

from __future__ import annotations

import json

from openai.types.responses import (
    ResponseCustomToolCall,
    ResponseCustomToolCallOutput,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from openai.types.responses.response_output_item import McpCall

from evaluatorq.common.output_adapters import output_to_text
from evaluatorq.contracts import AgentResponse
from evaluatorq.openresponses.otel_messages import items_to_input_messages

_MCP = McpCall(
    id='mcp_1',
    type='mcp_call',
    name='search_docs',
    server_label='docs',
    arguments='{"q": "refunds"}',
    output='3 hits',
)
_CUSTOM = ResponseCustomToolCall(type='custom_tool_call', call_id='ct_1', name='run_sql', input='SELECT 1')
_TEXT = ResponseOutputMessage(
    id='msg_1',
    type='message',
    role='assistant',
    status='completed',
    content=[ResponseOutputText(type='output_text', text='done', annotations=[])],
)


def test_mcp_and_custom_tool_calls_are_tool_calls_on_agent_response() -> None:
    response = AgentResponse.from_output_items([_MCP, _CUSTOM, _TEXT])

    assert [(c.name, c.call_id, c.arguments, c.result) for c in response.tool_calls] == [
        ('search_docs', 'mcp_1', '{"q": "refunds"}', '3 hits'),
        ('run_sql', 'ct_1', 'SELECT 1', None),
    ]
    assert response.text == 'done'


def test_mcp_error_is_the_call_result() -> None:
    failed = _MCP.model_copy(update={'output': None, 'error': 'server down'})

    (call,) = AgentResponse.from_output_items([failed]).tool_calls

    assert call.result is not None
    assert 'server down' in call.result


def test_tool_output_item_attaches_to_its_call() -> None:
    items = [
        _CUSTOM,
        ResponseCustomToolCallOutput(type='custom_tool_call_output', call_id='ct_1', output='1'),
        ResponseFunctionToolCall(type='function_call', call_id='fc_1', name='f', arguments='{}'),
    ]

    calls = AgentResponse.from_output_items(items).tool_calls

    assert [(c.call_id, c.result) for c in calls] == [('ct_1', '1'), ('fc_1', None)]


def test_agent_response_and_trace_view_see_the_same_calls_and_results() -> None:
    # SDK objects straight into both views: dumping them first would hide a view that only reads dicts.
    items = [
        _MCP,
        _CUSTOM,
        ResponseCustomToolCallOutput(type='custom_tool_call_output', call_id='ct_1', output='1'),
        _TEXT,
    ]

    response = AgentResponse.from_output_items(items)
    trace = items_to_input_messages(items, default_role='assistant')
    calls = [p for m in trace for p in m['parts'] if p['type'] == 'tool_call']
    results = {p['id']: p['response'] for m in trace for p in m['parts'] if p['type'] == 'tool_call_response'}
    texts = [p['content'] for m in trace for p in m['parts'] if p['type'] == 'text']

    assert [(c.call_id, c.name, c.arguments) for c in response.tool_calls] == [
        (p['id'], p['name'], json.dumps(p['arguments']) if isinstance(p['arguments'], dict) else p['arguments'])
        for p in calls
    ]
    assert {c.call_id: c.result for c in response.tool_calls} == {k: str(v) for k, v in results.items()}
    assert [response.text] == texts


def test_repeated_call_ids_each_get_their_own_result() -> None:
    calls = AgentResponse.from_output_items([
        {'type': 'function_call', 'call_id': 'a', 'name': 'f', 'arguments': '{}'},
        {'type': 'function_call_output', 'call_id': 'a', 'output': 'r1'},
        {'type': 'function_call', 'call_id': 'a', 'name': 'f', 'arguments': '{}'},
        {'type': 'function_call_output', 'call_id': 'a', 'output': 'r2'},
    ]).tool_calls

    assert [c.result for c in calls] == ['r1', 'r2']


def test_result_never_overwrites_a_recorded_one(caplog) -> None:
    calls = AgentResponse.from_output_items([
        {'type': 'mcp_call', 'id': 'm', 'name': 'f', 'arguments': '{}', 'output': 'own'},
        {'type': 'function_call_output', 'call_id': 'm', 'output': 'other'},
    ]).tool_calls

    assert calls[0].result == 'own'
    assert 'no matching call awaiting a result' in caplog.text


def test_result_repeating_the_recorded_one_is_silent(caplog) -> None:
    calls = AgentResponse.from_output_items([
        {'type': 'function_call', 'call_id': 'a', 'name': 'f', 'arguments': '{}', 'output': 'r'},
        {'type': 'function_call_output', 'call_id': 'a', 'output': 'r'},
    ]).tool_calls

    assert [c.result for c in calls] == ['r']
    assert 'awaiting a result' not in caplog.text


def test_result_before_its_call_is_skipped_with_a_warning(caplog) -> None:
    calls = AgentResponse.from_output_items([
        {'type': 'function_call_output', 'call_id': 'a', 'output': 'early'},
        {'type': 'function_call', 'call_id': 'a', 'name': 'f', 'arguments': '{}'},
    ]).tool_calls

    assert calls[0].result is None
    assert 'comes later' in caplog.text


def test_result_item_without_output_is_not_null() -> None:
    calls = AgentResponse.from_output_items([
        {'type': 'function_call', 'call_id': 'a', 'name': 'f', 'arguments': '{}'},
        {'type': 'function_call_output', 'call_id': 'a'},
    ]).tool_calls

    assert calls[0].result is None


def test_string_message_content_is_text() -> None:
    response = AgentResponse.from_output_items([{'type': 'message', 'role': 'assistant', 'content': 'hi'}])

    assert response.text == 'hi'


def test_fc_item_id_is_kept_on_the_call() -> None:
    (call,) = AgentResponse.from_output_items([
        ResponseFunctionToolCall(type='function_call', id='fc_1', call_id='call_1', name='f', arguments='{}')
    ]).tool_calls

    assert (call.call_id, call.id) == ('call_1', 'fc_1')


def test_failed_mcp_call_shows_its_error_to_the_judge() -> None:
    failed = _MCP.model_copy(update={'output': None, 'error': 'server down'})

    assert 'server down' in output_to_text(AgentResponse.from_output_items([failed]))


def test_tool_output_without_call_id_does_not_attach_to_an_idless_call(caplog) -> None:
    response = AgentResponse.from_output_items([
        {'type': 'custom_tool_call', 'name': 'run_sql', 'input': 'SELECT 1'},
        {'type': 'custom_tool_call_output', 'output': 'secret'},
    ])

    assert response.tool_calls[0].result is None
    assert 'no matching call' in caplog.text


def test_sdk_items_render_as_judge_visible_text() -> None:
    text = output_to_text([_MCP, _CUSTOM, _TEXT])

    assert 'search_docs' in text
    assert 'run_sql' in text
    assert 'done' in text
    assert 'McpCall(' not in text
