"""Split Orq tool records reach the transcript as one assistant tool call plus one tool message."""

from __future__ import annotations

from evaluatorq.common.messages import messages_to_text
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


def test_conflicting_results_render_one_call_with_the_calls_arguments_and_result():
    items = [CALL, RESULT, {**RESULT, 'result': 'other'}]
    assistant, *tool_messages = build_assistant_message(AgentResponse.from_output_items(items))
    assert assistant.tool_calls is not None
    assert [c.id for c in assistant.tool_calls] == ['c1']
    assert [c.function.arguments for c in assistant.tool_calls] == ['{"order_id":"a2"}']
    assert [m.tool_call_id for m in tool_messages] == ['c1']


def test_lone_result_renders_a_tool_call_with_unavailable_arguments(caplog):
    response = AgentResponse.from_output_items([RESULT])
    assert [c.arguments for c in response.tool_calls] == ['']
    assistant, *tool_messages = build_assistant_message(response)
    assert assistant.tool_calls is not None and len(assistant.tool_calls) == 1
    assert assistant.tool_calls[0].function.arguments == ''
    assert [m.tool_call_id for m in tool_messages] == ['c1']
    assert messages_to_text([assistant, *tool_messages]).splitlines()[0] == '[tool_call: refund(arguments unavailable)]'
    assert sum('has no matching call record' in r.getMessage() for r in caplog.records) == 1


def test_orphan_native_output_is_dropped_with_a_call_input_missing_warning(caplog):
    response = AgentResponse.from_output_items([{'type': 'function_call_output', 'call_id': 'zz', 'output': 'x'}])
    assert response.tool_calls == []
    assert any('call input missing' in r.getMessage() and "'zz'" in r.getMessage() for r in caplog.records)
