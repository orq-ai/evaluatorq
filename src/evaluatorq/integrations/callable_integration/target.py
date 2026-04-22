"""Red teaming target wrapper for arbitrary callables."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Union

from evaluatorq.redteam.backends.base import AgentTarget

# Accepted callable signatures
AgentCallable = Union[Callable[[str], Awaitable[str]], Callable[[str], str]]


class CallableTarget(AgentTarget):
    """Wraps any sync or async function as a red teaming target.

    Use this as an escape hatch for frameworks that don't have a dedicated
    integration. You provide a function that takes a prompt string and
    returns a response string — the wrapper handles the rest.

    Usage::

        from evaluatorq.integrations.callable_integration import CallableTarget

        # Async function
        async def my_agent(prompt: str) -> str:
            result = await some_framework.run(prompt)
            return result.text

        target = CallableTarget(my_agent)

        # Or a simple sync function
        target = CallableTarget(lambda prompt: "I can't help with that.")

        # Pass to red teaming
        config = DynamicRunConfig(targets=[target])
    """

    memory_entity_id: str | None = None
    """Callables are opaque — the wrapper cannot manage memory isolation.
    If the wrapped callable holds state, use ``reset_fn`` to clear it."""

    def __init__(
        self,
        fn: AgentCallable,
        *,
        reset_fn: Callable[[], None] | None = None,
    ) -> None:
        """Create a callable red teaming target.

        Args:
            fn: A sync or async function that takes a prompt and returns a response.
            reset_fn: Optional callback invoked on ``reset_conversation()``.
                Use this if your callable has state that needs clearing between attacks.
        """
        self._fn = fn
        self._is_async = asyncio.iscoroutinefunction(fn)
        self._reset_fn = reset_fn

    async def send_prompt(self, prompt: str) -> str:
        """Send a prompt to the callable and return its response."""
        try:
            if self._is_async:
                return await self._fn(prompt)  # type: ignore[misc]  # pyright: ignore[reportReturnType, reportGeneralTypeIssues]
            return await asyncio.to_thread(self._fn, prompt)  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
        except Exception as exc:
            raise RuntimeError(f"CallableTarget: callable raised {exc!r}") from exc

    def reset_conversation(self) -> None:
        """Reset conversation state via the optional reset callback."""
        if self._reset_fn is not None:
            self._reset_fn()

    def clone(self) -> CallableTarget:
        """Create a copy sharing the same callable."""
        return CallableTarget(self._fn, reset_fn=self._reset_fn)
