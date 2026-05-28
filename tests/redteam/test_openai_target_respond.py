"""Tests for OpenAIModelTarget.respond (stateless) — RES-808 PR3.

OpenAIModelTarget keeps a stateful ``send_prompt`` (accumulates ``_history``)
for redteam orchestrator/runner callers, AND a stateless ``respond(messages)``
for callers that own the transcript (sim). ``respond`` must not read or write
``_history`` and must prepend exactly one system prompt.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("openai")

from evaluatorq.contracts import AgentResponse, Message
from evaluatorq.redteam.backends.openai import OpenAIModelTarget


def _make_openai_response(content: str = "reply", model: str = "gpt-4o-mini") -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = "stop"
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model
    response.id = "chatcmpl-test"
    return response


def _make_target(client: MagicMock) -> OpenAIModelTarget:
    return OpenAIModelTarget(model="gpt-4o-mini", system_prompt="SYS", client=client)


@pytest.mark.asyncio
async def test_respond_returns_agent_response():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_make_openai_response("hi"))
    target = _make_target(client)

    with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
        result = await target.respond([Message(role="user", content="hello")])

    assert isinstance(result, AgentResponse)
    assert result.text == "hi"


@pytest.mark.asyncio
async def test_respond_prepends_single_system_prompt_and_strips_input_system():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_make_openai_response())
    target = _make_target(client)

    with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
        await target.respond(
            [
                Message(role="system", content="caller system (should be stripped)"),
                Message(role="user", content="q1"),
                Message(role="assistant", content="a1"),
                Message(role="user", content="q2"),
            ]
        )

    sent = client.chat.completions.create.call_args.kwargs["messages"]
    assert sent == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]


@pytest.mark.asyncio
async def test_respond_extracts_tool_calls_into_output():
    """respond must surface tool calls from the completion as ToolCallOutputItems."""
    client = MagicMock()
    tc = MagicMock()
    tc.id = "call_abc"
    tc.function = MagicMock()
    tc.function.name = "lookup"
    tc.function.arguments = '{"q": "x"}'
    response = _make_openai_response(content="")
    response.choices[0].message.tool_calls = [tc]
    client.chat.completions.create = AsyncMock(return_value=response)
    target = _make_target(client)

    with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
        result = await target.respond([Message(role="user", content="go")])

    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.name == "lookup"
    assert call.arguments == '{"q": "x"}'
    assert call.id == "call_abc"


@pytest.mark.asyncio
async def test_send_prompt_remains_stateful_accumulates_history():
    """The stateful send_prompt path must still thread _history across turns.

    Guards the intentional respond(stateless)/send_prompt(stateful) divergence —
    a refactor collapsing both onto the stateless path would break multi-turn
    redteam and must be caught here (until PR4/RES-877 removes the split).
    """
    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[_make_openai_response("a1"), _make_openai_response("a2")]
    )
    target = _make_target(client)

    with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
        await target.send_prompt("q1")
        await target.send_prompt("q2")

    # _history accumulated both turns (unlike respond, which leaves it empty).
    assert target._history != []
    # The second call saw the first turn (system + q1 + a1 + q2), not just q2.
    second_sent = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    contents = [m["content"] for m in second_sent]
    assert "q1" in contents
    assert "q2" in contents


@pytest.mark.asyncio
async def test_respond_is_stateless_does_not_touch_history():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_make_openai_response())
    target = _make_target(client)

    with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
        await target.respond([Message(role="user", content="one")])
        await target.respond([Message(role="user", content="two")])

    # respond never appends to _history; send_prompt is the stateful path.
    assert target._history == []
    # Second call did not carry the first call's turn.
    second_sent = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert second_sent == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "two"},
    ]
