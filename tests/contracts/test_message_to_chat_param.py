"""message_to_chat_param: Message -> OpenAI chat dict, tool-call preserving — RES-877 review."""

from __future__ import annotations

from evaluatorq.contracts import FunctionCall, Message, StrategyToolCall, message_to_chat_param


def test_plain_user_message():
    assert message_to_chat_param(Message(role="user", content="hi")) == {"role": "user", "content": "hi"}


def test_plain_message_none_content_becomes_empty():
    assert message_to_chat_param(Message(role="user", content=None)) == {"role": "user", "content": ""}


def test_assistant_tool_calls_preserved_with_none_content():
    m = Message(
        role="assistant",
        content=None,
        tool_calls=[StrategyToolCall(id="c1", function=FunctionCall(name="lookup", arguments='{"q":"x"}'))],
    )
    param = message_to_chat_param(m)
    # content stays None (OpenAI accepts null alongside tool_calls)
    assert param["content"] is None
    assert param["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"q":"x"}'}}
    ]


def test_tool_role_shape():
    param = message_to_chat_param(Message(role="tool", tool_call_id="c1", name="lookup", content="result"))
    assert param == {"role": "tool", "tool_call_id": "c1", "name": "lookup", "content": "result"}


def test_tool_role_without_name_omits_name_key():
    param = message_to_chat_param(Message(role="tool", tool_call_id="c1", content="result"))
    assert "name" not in param
    assert param == {"role": "tool", "tool_call_id": "c1", "content": "result"}


def test_tool_role_ignores_stray_tool_calls():
    """tool_calls belong on assistant messages; on a tool row they are malformed and dropped."""
    m = Message(
        role="tool",
        tool_call_id="c1",
        content="result",
        tool_calls=[StrategyToolCall(id="c1", function=FunctionCall(name="x", arguments="{}"))],
    )
    param = message_to_chat_param(m)
    assert "tool_calls" not in param
    assert param["role"] == "tool"
