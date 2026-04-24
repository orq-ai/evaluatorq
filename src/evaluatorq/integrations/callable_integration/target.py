"""Red teaming target wrapper for arbitrary callables."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from evaluatorq.redteam.backends.base import AgentTarget
from evaluatorq.redteam.contracts import AgentContext

# Accepted callable signatures
AgentCallable = Callable[[str], Awaitable[str]] | Callable[[str], str]


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
        agent_context: AgentContext | None = None,
    ) -> None:
        """Create a callable red teaming target.

        Args:
            fn: A sync or async function that takes a prompt and returns a response.
            reset_fn: Optional callback invoked on ``new()`` to clear shared callable state between attacks.
            agent_context: Optional :class:`AgentContext` describing the wrapped
                callable's tools, memory, system prompt, etc. The red teaming
                pipeline uses this for capability-aware strategy filtering —
                without it, all strategies (including nonsensical ones) will be
                applied. If not provided, a minimal context is returned.
        """
        self._fn = fn
        self._is_async = asyncio.iscoroutinefunction(fn)
        self._reset_fn = reset_fn
        self._agent_context = agent_context

    async def send_prompt(self, prompt: str) -> str:
        """Send a prompt to the callable and return its response."""
        try:
            if self._is_async:
                return await self._fn(prompt)  # type: ignore[misc]  # pyright: ignore[reportReturnType, reportGeneralTypeIssues]
            return await asyncio.to_thread(self._fn, prompt)  # type: ignore[return-value]  # pyright: ignore[reportReturnType]
        except Exception as exc:
            raise RuntimeError(f"CallableTarget: callable raised {exc!r}") from exc

    async def get_agent_context(self) -> AgentContext:
        """Return the user-provided agent context, or a minimal placeholder."""
        if self._agent_context is not None:
            return self._agent_context
        key = getattr(self._fn, "__name__", None) or "callable_target"
        return AgentContext(key=str(key), description="opaque callable target")

    def new(self) -> CallableTarget:
        """Return a fresh copy sharing the same callable, with state reset via reset_fn."""
        if self._reset_fn is not None:
            self._reset_fn()
        return CallableTarget(self._fn, reset_fn=self._reset_fn, agent_context=self._agent_context)
