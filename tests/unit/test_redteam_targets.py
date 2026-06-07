"""Unit tests for red teaming target integrations (CallableTarget — no optional deps)."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from evaluatorq.contracts import Message
from evaluatorq.integrations.callable_integration import CallableTarget
from evaluatorq.redteam.contracts import AgentResponse, OutputMessage, TextOutputItem, TokenUsage, ToolCallOutputItem


def _last(messages: list[Message]) -> str:
    """Read the last turn's content off a Message list."""
    return messages[-1].content or ""


class TestCallableTarget:
    @pytest.mark.asyncio
    async def test_sync_function(self) -> None:
        target = CallableTarget(lambda messages: f"echo: {_last(messages)}")
        result = await target.respond([Message(role="user", content="hello")])
        assert result.text == "echo: hello"
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_async_function(self) -> None:
        async def my_agent(messages: list[Message]) -> str:
            return f"async: {_last(messages)}"

        target = CallableTarget(my_agent)
        result = await target.respond([Message(role="user", content="hello")])
        assert result.text == "async: hello"
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_async_callable_object_is_detected(self) -> None:
        """A callable *object* with an async __call__ must be awaited, not run in a thread."""

        class AsyncAgent:
            async def __call__(self, messages: list[Message]) -> str:
                return f"obj: {_last(messages)}"

        target = CallableTarget(AsyncAgent())
        result = await target.respond([Message(role="user", content="hi")])
        assert result.text == "obj: hi"

    @pytest.mark.asyncio
    async def test_agent_response_return_is_passed_through(self) -> None:
        out: list[OutputMessage] = []
        out.append(ToolCallOutputItem(name="search", arguments=json.dumps({"query": "hello"})))
        out.append(TextOutputItem(text="done", annotations=[]))
        response = AgentResponse(output=out)

        target = CallableTarget(lambda messages: response)
        result = await target.respond([Message(role="user", content="hello")])

        assert result is response
        assert result.tool_calls[0].name == "search"

    @pytest.mark.asyncio
    async def test_timeout_is_not_wrapped(self) -> None:
        async def my_agent(messages: list[Message]) -> str:
            raise asyncio.TimeoutError

        target = CallableTarget(my_agent)
        with pytest.raises(asyncio.TimeoutError):
            await target.respond([Message(role="user", content="hello")])

    @pytest.mark.asyncio
    async def test_reset_calls_reset_fn(self) -> None:
        reset = MagicMock()
        target = CallableTarget(lambda messages: _last(messages), reset_fn=reset)
        target.new()
        reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_without_reset_fn_is_noop(self) -> None:
        target = CallableTarget(lambda messages: _last(messages))
        target.new()  # should not raise

    def test_clone_returns_independent_instance(self) -> None:
        def fn(messages: list[Message]) -> str:
            return _last(messages)

        reset = MagicMock()
        target = CallableTarget(fn, reset_fn=reset)
        cloned = target.new()
        assert cloned is not target
        assert cloned._fn is fn
        assert cloned._reset_fn is reset

    def test_clone_preserves_usage_fn(self) -> None:
        def fn(messages: list[Message]) -> str:
            return _last(messages)

        def usage_fn(messages: list[Message], response: str) -> TokenUsage:
            return TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, calls=1)

        target = CallableTarget(fn, usage_fn=usage_fn)
        cloned = target.new()
        assert cloned._usage_fn is usage_fn

    @pytest.mark.asyncio
    async def test_get_agent_context_default_is_minimal(self) -> None:
        def my_fn(messages: list[Message]) -> str:
            return _last(messages)

        target = CallableTarget(my_fn)
        ctx = await target.get_agent_context()
        assert ctx.key == "my_fn"
        assert ctx.tools == []
        assert ctx.memory_stores == []
        assert ctx.description == "opaque callable target"

    @pytest.mark.asyncio
    async def test_get_agent_context_lambda_uses_stable_key(self) -> None:
        """Lambdas all share __name__ == '<lambda>'; fall back to a stable key."""
        target = CallableTarget(lambda messages: _last(messages))
        ctx = await target.get_agent_context()
        assert ctx.key == "callable_target"

    @pytest.mark.asyncio
    async def test_get_agent_context_returns_override(self) -> None:
        from evaluatorq.redteam.contracts import AgentContext, ToolInfo

        override = AgentContext(
            key="wrapped-agent",
            tools=[ToolInfo(name="send_email")],
            description="wraps a real agent",
        )
        target = CallableTarget(lambda messages: _last(messages), agent_context=override)
        ctx = await target.get_agent_context()
        assert ctx is override

    def test_clone_preserves_agent_context(self) -> None:
        from evaluatorq.redteam.contracts import AgentContext

        override = AgentContext(key="k")
        target = CallableTarget(lambda messages: _last(messages), agent_context=override)
        cloned = target.new()
        assert cloned._agent_context is override


class TestCallableTargetConversation:
    @pytest.mark.asyncio
    async def test_single_message_is_a_list_of_one(self) -> None:
        """Opening turn: the callable receives a one-element Message list."""
        seen: list[list[Message]] = []

        def fn(messages: list[Message]) -> str:
            seen.append(messages)
            return "ok"

        target = CallableTarget(fn)
        await target.respond([Message(role="user", content="hello")])
        assert len(seen) == 1
        assert [(m.role, m.content) for m in seen[0]] == [("user", "hello")]

    @pytest.mark.asyncio
    async def test_full_transcript_is_forwarded(self) -> None:
        """Later turns: the callable receives the whole conversation as typed Messages."""
        seen: list[list[Message]] = []

        def fn(messages: list[Message]) -> str:
            seen.append(messages)
            return "ok"

        target = CallableTarget(fn)
        convo = [
            Message(role="user", content="first"),
            Message(role="assistant", content="reply"),
            Message(role="user", content="second"),
        ]
        await target.respond(convo)
        # Forwarded verbatim — same typed objects, no conversion.
        assert seen[0] == convo

    @pytest.mark.asyncio
    async def test_async_callable_sees_full_count(self) -> None:
        async def fn(messages: list[Message]) -> str:
            return f"saw {len(messages)} messages"

        target = CallableTarget(fn)
        result = await target.respond([
            Message(role="user", content="a"),
            Message(role="assistant", content="b"),
            Message(role="user", content="c"),
        ])
        assert result.text == "saw 3 messages"

    @pytest.mark.asyncio
    async def test_tool_terminated_transcript_is_accepted(self) -> None:
        """No last-turn-must-be-user guard: a transcript ending in a tool result is forwarded."""
        from evaluatorq.contracts import FunctionCall, StrategyToolCall

        seen: list[list[Message]] = []

        def fn(messages: list[Message]) -> str:
            seen.append(messages)
            return "ok"

        target = CallableTarget(fn)
        convo = [
            Message(role="user", content="search please"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    StrategyToolCall(id="call_1", function=FunctionCall(name="search", arguments='{"q": "x"}'))
                ],
            ),
            Message(role="tool", tool_call_id="call_1", name="search", content="result"),
        ]
        result = await target.respond(convo)
        assert result.text == "ok"
        # Tool turns reach the callable intact; it can render them via to_chat_completion.
        assert seen[0] == convo
        assistant_turn = seen[0][1]
        assert assistant_turn.tool_calls is not None
        assert assistant_turn.tool_calls[0].function.name == "search"
        assert assistant_turn.to_chat_completion()["tool_calls"][0]["function"]["name"] == "search"


class TestCallableTargetUsage:
    @pytest.mark.asyncio
    async def test_usage_fn_called_with_transcript_and_response(self) -> None:
        """usage_fn receives the full forwarded transcript and the response text."""
        captured: list[tuple[list[Message], str]] = []

        def usage_fn(messages: list[Message], response: str) -> TokenUsage:
            captured.append((messages, response))
            return TokenUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5, calls=1)

        target = CallableTarget(lambda messages: f"echo:{_last(messages)}", usage_fn=usage_fn)
        result = await target.respond([Message(role="user", content="hello")])

        assert result.text == "echo:hello"
        assert isinstance(result.usage, TokenUsage)
        assert result.usage.total_tokens == 5
        assert len(captured) == 1
        msgs, resp = captured[0]
        assert [m.content for m in msgs] == ["hello"]
        assert resp == "echo:hello"

    @pytest.mark.asyncio
    async def test_usage_fn_sees_whole_multi_turn_transcript(self) -> None:
        """The full transcript reaches usage_fn so multi-turn token accounting is correct."""
        captured: dict[str, list[str]] = {}

        def usage_fn(messages: list[Message], response: str) -> TokenUsage:
            captured["contents"] = [m.content or "" for m in messages]
            return TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, calls=1)

        target = CallableTarget(lambda messages: "response", usage_fn=usage_fn)
        await target.respond([
            Message(role="user", content="first"),
            Message(role="assistant", content="reply"),
            Message(role="user", content="last"),
        ])
        assert captured["contents"] == ["first", "reply", "last"]

    @pytest.mark.asyncio
    async def test_usage_none_when_no_usage_fn(self) -> None:
        """Without usage_fn, AgentResponse.usage is None."""
        target = CallableTarget(lambda messages: "response")
        result = await target.respond([Message(role="user", content="hi")])
        assert result.text == "response"
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_usage_fn_exception_yields_none_and_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """When usage_fn raises, usage is None and a warning is logged."""
        import logging

        def bad_usage_fn(messages: list[Message], response: str) -> TokenUsage:
            raise ValueError("boom")

        target = CallableTarget(lambda messages: "reply", usage_fn=bad_usage_fn)
        with caplog.at_level(logging.WARNING):
            result = await target.respond([Message(role="user", content="hi")])

        assert result.text == "reply"
        assert result.usage is None
        assert any("usage_fn raised" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_usage_fn_returning_none_propagates(self) -> None:
        """If usage_fn returns None explicitly, AgentResponse.usage is None."""
        target = CallableTarget(lambda messages: "ok", usage_fn=lambda m, r: None)
        result = await target.respond([Message(role="user", content="hi")])
        assert result.usage is None

    @pytest.mark.asyncio
    async def test_async_callable_with_usage_fn(self) -> None:
        """usage_fn works with async callables too."""
        async def my_agent(messages: list[Message]) -> str:
            return f"async:{_last(messages)}"

        def usage_fn(messages: list[Message], response: str) -> TokenUsage:
            return TokenUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15, calls=1)

        target = CallableTarget(my_agent, usage_fn=usage_fn)
        result = await target.respond([Message(role="user", content="test")])

        assert result.text == "async:test"
        assert isinstance(result.usage, TokenUsage)
        assert result.usage.prompt_tokens == 5
