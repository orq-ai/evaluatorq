"""Split Orq tool records reach the transcript as one assistant tool call plus one tool message."""

from __future__ import annotations

from evaluatorq.contracts import AgentResponse
from evaluatorq.simulation.runner.simulation import build_assistant_message


CALL = {'type': 'function_call', 'id': 'fc_1', 'call_id': 'c1', 'name': 'refund', 'arguments': '{"order_id":"a2"}'}
RESULT = {'type': 'orq:refund', 'id': 'orq_9', 'call_id': 'c1', 'name': 'refund', 'result': {'ok': False}}


def test_split_records_render_one_tool_call_and_one_tool_message():
    assistant, *tool_messages = build_assistant_message(AgentResponse.from_output_items([CALL, RESULT]))
    assert assistant.role == 'assistant'
    assert assistant.tool_calls is not None and len(assistant.tool_calls) == 1
    assert assistant.tool_calls[0].function.arguments == '{"order_id":"a2"}'
    assert len(tool_messages) == 1
    assert tool_messages[0].role == 'tool'
    assert tool_messages[0].content == '{"ok": false}'


def test_placeholder_arguments_on_the_result_record_do_not_replace_the_calls():
    items = [CALL, {**RESULT, 'arguments': '{}'}]
    assistant, *tool_messages = build_assistant_message(AgentResponse.from_output_items(items))
    assert assistant.tool_calls is not None and len(assistant.tool_calls) == 1
    assert assistant.tool_calls[0].function.arguments == '{"order_id":"a2"}'
    assert len(tool_messages) == 1


def test_conflicting_results_render_unique_call_ids_that_keep_the_calls_arguments():
    items = [CALL, RESULT, {**RESULT, 'result': 'other'}]
    assistant, *tool_messages = build_assistant_message(AgentResponse.from_output_items(items))
    assert assistant.tool_calls is not None
    assert [c.id for c in assistant.tool_calls] == ['c1', 'c1#2']
    assert {c.function.arguments for c in assistant.tool_calls} == {'{"order_id":"a2"}'}
    assert [m.tool_call_id for m in tool_messages] == ['c1', 'c1#2']
