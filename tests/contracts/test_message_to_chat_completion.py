"""Message.to_chat_completion: Message -> OpenAI chat dict, tool-call preserving — RES-877 review."""

from __future__ import annotations

import pytest

from evaluatorq.contracts import (
    FunctionCall,
    InputImageContent,
    InputTextContent,
    Message,
    StrategyToolCall,
)


def test_plain_user_message():
    assert Message(role="user", content="hi").to_chat_completion() == {"role": "user", "content": "hi"}


def test_plain_message_none_content_becomes_empty():
    assert Message(role="user", content=None).to_chat_completion() == {"role": "user", "content": ""}


def test_assistant_tool_calls_preserved_with_none_content():
    m = Message(
        role="assistant",
        content=None,
        tool_calls=[StrategyToolCall(id="c1", function=FunctionCall(name="lookup", arguments='{"q":"x"}'))],
    )
    param = m.to_chat_completion()
    # content stays None (OpenAI accepts null alongside tool_calls)
    assert param["content"] is None
    assert param["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"q":"x"}'}}
    ]


def test_tool_role_shape():
    param = Message(role="tool", tool_call_id="c1", name="lookup", content="result").to_chat_completion()
    assert param == {"role": "tool", "tool_call_id": "c1", "name": "lookup", "content": "result"}


def test_tool_role_without_name_omits_name_key():
    param = Message(role="tool", tool_call_id="c1", content="result").to_chat_completion()
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
    param = m.to_chat_completion()
    assert "tool_calls" not in param
    assert param["role"] == "tool"


# --- RES-1018: multi-modal content at the tool / assistant-tool_calls sinks ----


def test_tool_role_flattens_multipart_text_content():
    """A tool result carrying multi-part text content is flattened to a string
    rather than emitting raw ContentPart objects into the payload."""
    m = Message(
        role="tool",
        tool_call_id="c1",
        content=[InputTextContent(type="input_text", text="the result")],
    )
    assert m.to_chat_completion()["content"] == "the result"


def test_tool_role_raises_on_non_text_multipart_content():
    """Tool messages are text-only; an image part fails loud instead of leaking a repr."""
    m = Message(
        role="tool",
        tool_call_id="c1",
        content=[InputImageContent(type="input_image", image_url="https://x/y.png")],
    )
    with pytest.raises(NotImplementedError):
        m.to_chat_completion()


def test_assistant_tool_calls_renders_multipart_content_blocks():
    """When an assistant-with-tool_calls turn carries multi-part content, it renders
    to chat content blocks instead of raw Pydantic objects."""
    m = Message(
        role="assistant",
        content=[InputTextContent(type="input_text", text="calling a tool")],
        tool_calls=[StrategyToolCall(id="c1", function=FunctionCall(name="lookup", arguments="{}"))],
    )
    param = m.to_chat_completion()
    assert param["content"] == [{"type": "text", "text": "calling a tool"}]
    assert param["tool_calls"][0]["id"] == "c1"
