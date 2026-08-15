import logging

from evaluatorq.contracts import AgentResponse, TextOutputItem, ToolCallOutputItem
from evaluatorq.simulation.runner.simulation import build_assistant_message


def test_assistant_message_keeps_tool_calls():
    response = AgentResponse(output=[ToolCallOutputItem(name='refund', arguments='{}')])
    msg = build_assistant_message(response)
    assert msg.tool_calls is not None
    assert msg.tool_calls[0].function.name == 'refund'
    assert msg.tool_calls[0].function.arguments == '{}'


def test_tool_only_turn_warns(caplog):
    response = AgentResponse(output=[ToolCallOutputItem(name='refund', arguments='{}')])
    with caplog.at_level(logging.WARNING, logger='evaluatorq.simulation.runner.simulation'):
        build_assistant_message(response)
    assert any('tool call' in r.message.lower() for r in caplog.records)


def test_assistant_message_with_text_and_no_tool_calls_does_not_warn(caplog):
    response = AgentResponse(output=[TextOutputItem(text='hello', annotations=[])])
    with caplog.at_level(logging.WARNING, logger='evaluatorq.simulation.runner.simulation'):
        msg = build_assistant_message(response)
    assert msg.content == 'hello'
    assert msg.tool_calls is None
    assert not any('tool call' in r.message.lower() for r in caplog.records)
