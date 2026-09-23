"""MCP and custom tool calls reach AgentResponse, and agree with the OTel trace view.

Fixtures are built from the openai SDK's own Responses models, so a field rename
upstream fails here instead of confirming a hand-written guess at the shape.
"""

from __future__ import annotations

from openai.types.responses import (
    ResponseCustomToolCall,
    ResponseCustomToolCallOutput,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from openai.types.responses.response_output_item import McpCall

from evaluatorq.contracts import AgentResponse
from evaluatorq.openresponses.otel_messages import items_to_output_messages

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


def test_agent_response_and_trace_view_see_the_same_tool_calls() -> None:
    items = [_MCP, _CUSTOM, _TEXT]
    dicts = [item.model_dump() for item in items]

    from_response = [(c.call_id, c.name) for c in AgentResponse.from_output_items(items).tool_calls]
    from_trace = [
        (part['id'], part['name'])
        for message in items_to_output_messages(dicts)
        for part in message['parts']
        if part['type'] == 'tool_call'
    ]

    assert from_response == from_trace
