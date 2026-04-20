"""Unit tests for red teaming target integrations (CallableTarget — no optional deps)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from evaluatorq.integrations.callable_integration import CallableTarget


class TestCallableTarget:
    @pytest.mark.asyncio
    async def test_sync_function(self) -> None:
        target = CallableTarget(lambda prompt: f"echo: {prompt}")
        result = await target.send_prompt("hello")
        assert result == "echo: hello"

    @pytest.mark.asyncio
    async def test_async_function(self) -> None:
        async def my_agent(prompt: str) -> str:
            return f"async: {prompt}"

        target = CallableTarget(my_agent)
        result = await target.send_prompt("hello")
        assert result == "async: hello"

    @pytest.mark.asyncio
    async def test_reset_calls_reset_fn(self) -> None:
        reset = MagicMock()
        target = CallableTarget(lambda p: p, reset_fn=reset)
        target.reset_conversation()
        reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_without_reset_fn_is_noop(self) -> None:
        target = CallableTarget(lambda p: p)
        target.reset_conversation()  # should not raise

    def test_clone_returns_independent_instance(self) -> None:
        def fn(p: str) -> str:
            return p

        reset = MagicMock()
        target = CallableTarget(fn, reset_fn=reset)
        cloned = target.clone()
        assert cloned is not target
        assert cloned._fn is fn
        assert cloned._reset_fn is reset
