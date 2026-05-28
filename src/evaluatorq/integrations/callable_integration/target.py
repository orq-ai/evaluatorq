"""Red teaming target wrapper for arbitrary callables."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from evaluatorq.redteam.backends.base import AgentTarget
from evaluatorq.redteam.contracts import AgentContext, AgentResponse, OutputMessage, TextOutputItem, TokenUsage

# Accepted callable signatures — may return str (backward-compat) or AgentResponse
AgentCallable = (
    Callable[[str], Awaitable[AgentResponse]]
    | Callable[[str], Awaitable[str]]
    | Callable[[str], AgentResponse]
    | Callable[[str], str]
)


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

        # Plumb token counts via usage_fn
        def get_usage(prompt: str, response: str) -> TokenUsage:
            return TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15, calls=1)

        target = CallableTarget(my_agent, usage_fn=get_usage)

        # Pass to red teaming
        config = DynamicRunConfig(targets=[target])
    """

    def __init__(
        self,
        fn: AgentCallable,
        *,
        reset_fn: Callable[[], None] | None = None,
        usage_fn: Callable[[str, str], TokenUsage | None] | None = None,
        agent_context: AgentContext | None = None,
    ) -> None:
        """Create a callable red teaming target.

        Args:
            fn: A sync or async function that takes a prompt and returns a response.
            reset_fn: Optional callback invoked on ``new()`` to clear shared callable state between attacks.
            usage_fn: Optional callable taking ``(prompt, response) -> TokenUsage | None``.
                Use this to plumb token counts from your underlying framework when the
                callable itself only returns a string. The function must be synchronous;
                async usage extraction is not supported. Exceptions raised by ``usage_fn``
                are logged as warnings and result in ``usage=None``.
            agent_context: Optional :class:`AgentContext` describing the wrapped
                callable's tools, memory, system prompt, etc. The red teaming
                pipeline uses this for capability-aware strategy filtering —
                without it, all strategies (including nonsensical ones) will be
                applied. If not provided, a minimal context is returned.
        Callables are opaque — the wrapper cannot manage memory isolation.
        If the wrapped callable holds state, use ``reset_fn`` to clear it.
        """
        super().__init__(memory_entity_id=None)
        self._fn = fn
        self._is_async = asyncio.iscoroutinefunction(fn)
        self._reset_fn = reset_fn
        self._usage_fn = usage_fn
        self._agent_context = agent_context

    async def send_prompt(self, prompt: str) -> AgentResponse:
        """Send a prompt to the callable and return a structured response.

        Callables that return a plain ``str`` are wrapped in an
        :class:`AgentResponse`. Callables that already return :class:`AgentResponse`
        are passed through. Token usage from ``usage_fn`` (if provided) is
        attached to the returned ``AgentResponse``.
        """
        try:
            if self._is_async:
                coro: Any = self._fn(prompt)
                result = await coro  # pyright: ignore[reportGeneralTypeIssues]
            else:
                result = await asyncio.to_thread(self._fn, prompt)  # type: ignore[arg-type]
        except (asyncio.CancelledError, asyncio.TimeoutError):
            raise
        except Exception as exc:
            raise RuntimeError(f"CallableTarget: callable raised {exc!r}") from exc

        if isinstance(result, AgentResponse):
            return result
        text = str(result) if result is not None else ""
        usage: TokenUsage | None = None
        if self._usage_fn is not None:
            try:
                usage = self._usage_fn(prompt, text)
            except Exception:
                logger.exception("usage_fn raised an exception; using usage=None")
        text_item: OutputMessage = TextOutputItem(text=text, annotations=[])
        return AgentResponse(output=[text_item], usage=usage)

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
        return CallableTarget(self._fn, reset_fn=self._reset_fn, usage_fn=self._usage_fn, agent_context=self._agent_context)
