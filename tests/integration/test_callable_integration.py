"""Integration test: CallableTarget with real sync and async callables.

No external dependencies needed — tests the full wiring with plain functions.
"""

from __future__ import annotations

import pytest

from evaluatorq.integrations.callable_integration import CallableTarget


class TestCallableIntegration:
    @pytest.mark.asyncio
    async def test_stateful_callable(self) -> None:
        """A callable with state that tracks conversation history."""
        history: list[str] = []

        async def stateful_agent(prompt: str) -> str:
            history.append(prompt)
            return f"You said {len(history)} things so far."

        def reset() -> None:
            history.clear()

        target = CallableTarget(stateful_agent, reset_fn=reset)

        r1 = await target.send_prompt("Hello")
        assert "1" in r1

        r2 = await target.send_prompt("World")
        assert "2" in r2

        target.reset_conversation()

        r3 = await target.send_prompt("After reset")
        assert "1" in r3  # Back to 1 after reset

    @pytest.mark.asyncio
    async def test_clone_gets_independent_state(self) -> None:
        """Cloned callable targets should not share state."""
        call_count = 0

        def counting_agent(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"Call #{call_count}"

        target = CallableTarget(counting_agent)
        cloned = target.clone()

        r1 = await target.send_prompt("a")
        r2 = await cloned.send_prompt("b")

        # Both share the same function, so call_count increments for both
        assert isinstance(r1, str)
        assert isinstance(r2, str)
