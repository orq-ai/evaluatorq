"""Tests for OpenAIModelTarget.respond (stateless) — RES-877 PR6.

OpenAIModelTarget is fully stateless: it exposes only ``respond(messages)``.
The orchestrator owns multi-turn conversation state. Tests verify that
``respond`` prepends exactly one system prompt and does not accumulate state
across calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("openai")

from evaluatorq.common import model_catalogue
from evaluatorq.contracts import AgentResponse, FunctionCall, Message, StrategyToolCall
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
async def test_respond_preserves_tool_calls_in_replayed_transcript():
    """A replayed assistant tool_calls message + tool result must reach the API
    intact, not flattened to {role, content}."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_make_openai_response())
    target = _make_target(client)

    transcript = [
        Message(role="user", content="q1"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[StrategyToolCall(id="call_1", function=FunctionCall(name="lookup", arguments='{"q":"x"}'))],
        ),
        Message(role="tool", tool_call_id="call_1", name="lookup", content="result-text"),
        Message(role="user", content="q2"),
    ]

    with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
        await target.respond(transcript)

    sent = client.chat.completions.create.call_args.kwargs["messages"]
    assistant = next(m for m in sent if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert assistant["tool_calls"][0]["function"]["name"] == "lookup"
    tool_row = next(m for m in sent if m["role"] == "tool")
    assert tool_row["tool_call_id"] == "call_1"
    assert tool_row["content"] == "result-text"


@pytest.mark.asyncio
async def test_respond_prices_unpriced_usage_from_catalogue(monkeypatch: pytest.MonkeyPatch):
    """RES-1295: an unpriced Chat Completions usage comes back priced, with
    priced_calls == calls, once a catalogue entry exists for the model."""
    model_catalogue.reset_catalogue_cache()

    async def fake_load(client=None):  # noqa: ANN001, ARG001
        return {"gpt-4o-mini": model_catalogue.ModelInfo(0.00025, 0.002, "openai", supports_responses=True)}

    monkeypatch.setattr(model_catalogue, "_load_catalogue", fake_load)

    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_make_openai_response())
    target = _make_target(client)

    with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
        result = await target.respond([Message(role="user", content="hello")])

    assert result.usage is not None
    assert result.usage.calls == result.usage.priced_calls == 1
    assert result.usage.total_cost is not None
    assert result.usage.total_cost > 0

    model_catalogue.reset_catalogue_cache()


@pytest.mark.asyncio
async def test_respond_is_stateless_across_calls():
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_make_openai_response())
    target = _make_target(client)

    with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
        await target.respond([Message(role="user", content="one")])
        await target.respond([Message(role="user", content="two")])

    second_sent = client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert second_sent == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "two"},
    ]


@pytest.mark.asyncio
async def test_reasoning_effort_switches_the_token_budget_field():
    """A reasoning model rejects `max_tokens`, so asking for an effort must switch it.

    Sending both is a 400 before the target ever answers, which surfaces as a dead
    target rather than as a config error.
    """
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_make_openai_response())
    target = OpenAIModelTarget(
        model="gpt-4o-mini", system_prompt="SYS", client=client, reasoning_effort="high"
    )

    with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
        await target.respond([Message(role="user", content="hello")])

    sent = client.chat.completions.create.call_args.kwargs
    assert sent["reasoning_effort"] == "high"
    assert "max_tokens" not in sent
    assert sent["max_completion_tokens"] == target.max_tokens


@pytest.mark.asyncio
async def test_without_reasoning_effort_the_token_budget_stays_max_tokens():
    """The non-reasoning path is unchanged — older models only accept `max_tokens`."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=_make_openai_response())
    target = _make_target(client)

    with patch("evaluatorq.redteam.tracing.get_tracer", return_value=None):
        await target.respond([Message(role="user", content="hello")])

    sent = client.chat.completions.create.call_args.kwargs
    assert sent["max_tokens"] == target.max_tokens
    assert "max_completion_tokens" not in sent
    assert "reasoning_effort" not in sent
