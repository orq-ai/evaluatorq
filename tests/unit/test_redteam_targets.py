"""Unit tests for red teaming target integrations (CallableTarget — no optional deps)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from evaluatorq.integrations.callable_integration import CallableTarget
import json

from evaluatorq.redteam.contracts import AgentResponse, OutputMessage, TextOutputItem, ToolCallOutputItem


class TestCallableTarget:
    @pytest.mark.asyncio
    async def test_sync_function(self) -> None:
        target = CallableTarget(lambda prompt: f"echo: {prompt}")
        result = await target.send_prompt("hello")
        assert result.text == "echo: hello"
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_async_function(self) -> None:
        async def my_agent(prompt: str) -> str:
            return f"async: {prompt}"

        target = CallableTarget(my_agent)
        result = await target.send_prompt("hello")
        assert result.text == "async: hello"
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_agent_response_return_is_passed_through(self) -> None:
        out: list[OutputMessage] = []
        out.append(ToolCallOutputItem(name="search", arguments=json.dumps({"query": "hello"})))
        out.append(TextOutputItem(text="done", annotations=[]))
        response = AgentResponse(output=out)

        target = CallableTarget(lambda prompt: response)
        result = await target.send_prompt("hello")

        assert result is response
        assert result.tool_calls[0].name == "search"

    @pytest.mark.asyncio
    async def test_timeout_is_not_wrapped(self) -> None:
        async def my_agent(prompt: str) -> str:
            raise asyncio.TimeoutError

        target = CallableTarget(my_agent)
        with pytest.raises(asyncio.TimeoutError):
            await target.send_prompt("hello")

    @pytest.mark.asyncio
    async def test_reset_calls_reset_fn(self) -> None:
        reset = MagicMock()
        target = CallableTarget(lambda p: p, reset_fn=reset)
        target.new()
        reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_without_reset_fn_is_noop(self) -> None:
        target = CallableTarget(lambda p: p)
        target.new()  # should not raise

    def test_clone_returns_independent_instance(self) -> None:
        def fn(p: str) -> str:
            return p

        reset = MagicMock()
        target = CallableTarget(fn, reset_fn=reset)
        cloned = target.new()
        assert cloned is not target
        assert cloned._fn is fn
        assert cloned._reset_fn is reset

    @pytest.mark.asyncio
    async def test_get_agent_context_default_is_minimal(self) -> None:
        def my_fn(prompt: str) -> str:
            return prompt

        target = CallableTarget(my_fn)
        ctx = await target.get_agent_context()
        assert ctx.key == "my_fn"
        assert ctx.tools == []
        assert ctx.memory_stores == []
        assert ctx.description == "opaque callable target"

    @pytest.mark.asyncio
    async def test_get_agent_context_returns_override(self) -> None:
        from evaluatorq.redteam.contracts import AgentContext, ToolInfo

        override = AgentContext(
            key="wrapped-agent",
            tools=[ToolInfo(name="send_email")],
            description="wraps a real agent",
        )
        target = CallableTarget(lambda p: p, agent_context=override)
        ctx = await target.get_agent_context()
        assert ctx is override

    def test_clone_preserves_agent_context(self) -> None:
        from evaluatorq.redteam.contracts import AgentContext

        override = AgentContext(key="k")
        target = CallableTarget(lambda p: p, agent_context=override)
        cloned = target.new()
        assert cloned._agent_context is override
