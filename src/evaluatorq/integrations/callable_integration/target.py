"""Red teaming target wrapper for arbitrary callables."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from evaluatorq.contracts import AgentTarget, Message
from evaluatorq.redteam.contracts import AgentContext, AgentResponse, OutputMessage, TextOutputItem, TokenUsage

# Accepted callable signatures — receive the conversation as a list of typed
# :class:`~evaluatorq.contracts.Message` objects. The list holds one message for
# the opening turn and grows with every turn, so the same callable handles
# single- and multi-turn red teaming. May return ``str`` or :class:`AgentResponse`.
AgentCallable = (
    Callable[[list[Message]], Awaitable[AgentResponse]]
    | Callable[[list[Message]], Awaitable[str]]
    | Callable[[list[Message]], AgentResponse]
    | Callable[[list[Message]], str]
)

# usage_fn receives the full forwarded transcript and the response text, so
# token accounting stays correct across multi-turn conversations.
UsageFn = Callable[[list[Message], str], "TokenUsage | None"]


def _is_async_callable(fn: object) -> bool:
    """True if ``fn`` is a coroutine function or a callable object with an async ``__call__``."""
    if asyncio.iscoroutinefunction(fn):
        return True
    call = getattr(fn, "__call__", None)  # noqa: B004 — intentional: detect async callable objects
    return call is not None and asyncio.iscoroutinefunction(call)


class CallableTarget(AgentTarget):
    """Wraps any sync or async function as a red teaming target.

    Use this as an escape hatch for frameworks that don't have a dedicated
    integration. You provide a function that takes the conversation (a list of
    typed :class:`~evaluatorq.contracts.Message` objects) and returns a response
    — the wrapper handles the rest. The list contains one message on the opening
    turn and every prior turn on later turns, so a stateless callable still sees
    full context, matching the stateless OpenAI / Vercel / OpenAI-Agents targets
    (which likewise consume the typed ``Message`` list at the boundary).

    Usage::

        from evaluatorq.contracts import Message
        from evaluatorq.integrations.callable_integration import CallableTarget

        # Async function — receives the whole conversation as Message objects
        async def my_agent(messages: list[Message]) -> str:
            result = await some_framework.run(messages[-1].content)
            return result.text

        target = CallableTarget(my_agent)

        # Need OpenAI chat-completion dicts? Convert at the boundary yourself:
        async def openai_agent(messages: list[Message]) -> str:
            chat = [m.to_chat_completion() for m in messages]
            return (await client.chat.completions.create(model="gpt-4o", messages=chat)).choices[0].message.content

        target = CallableTarget(openai_agent)

        # Plumb token counts via usage_fn — it sees the full transcript
        def get_usage(messages: list[Message], response: str) -> TokenUsage:
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
        usage_fn: UsageFn | None = None,
        agent_context: AgentContext | None = None,
    ) -> None:
        """Create a callable red teaming target.

        Args:
            fn: A sync or async function taking the conversation as a
                ``list[Message]`` and returning a ``str`` or an
                :class:`AgentResponse`. The list grows by one turn each round, so
                the same callable serves single- and multi-turn runs. Callables
                that want OpenAI chat-completion dicts can call
                :meth:`Message.to_chat_completion` on each element themselves.
            reset_fn: Optional callback invoked on ``new()`` to clear shared callable state between attacks.
            usage_fn: Optional callable taking ``(messages, response) -> TokenUsage | None``,
                where ``messages`` is the full transcript forwarded to ``fn`` and
                ``response`` is the response text. Use this to plumb token counts
                from your underlying framework when the callable itself only
                returns a string. The function must be synchronous; async usage
                extraction is not supported. Exceptions raised by ``usage_fn`` are
                logged as warnings and result in ``usage=None``.
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
        self._is_async = _is_async_callable(fn)
        self._reset_fn = reset_fn
        self._usage_fn = usage_fn
        self._agent_context = agent_context

    async def respond(self, messages: list[Message]) -> AgentResponse:
        """Send the full conversation to the wrapped callable; return a structured response.

        Forwards the entire transcript as typed :class:`Message` objects — tool
        turns included — so a stateless callable sees prior context. The list
        holds a single message on the opening turn and grows each round. Like the
        stateless OpenAI / Vercel targets, no constraint is placed on the last
        turn's role. Callables that return a plain ``str`` are wrapped in an
        :class:`AgentResponse`; those returning :class:`AgentResponse` pass
        through. Token usage from ``usage_fn`` (if provided) is attached to the
        returned ``AgentResponse``.
        """
        fn: Callable[[list[Message]], Any] = self._fn
        try:
            if self._is_async:
                result = await fn(messages)
            else:
                result = await asyncio.to_thread(fn, messages)
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
                usage = self._usage_fn(messages, text)
            except Exception:
                logger.exception("usage_fn raised an exception; using usage=None")
        text_item: OutputMessage = TextOutputItem(text=text, annotations=[])
        return AgentResponse(output=[text_item], usage=usage)

    async def get_agent_context(self) -> AgentContext:
        """Return the user-provided agent context, or a minimal placeholder."""
        if self._agent_context is not None:
            return self._agent_context
        name = getattr(self._fn, "__name__", "")
        # Lambdas share the useless name "<lambda>"; fall back to a stable key.
        key = name if name and name != "<lambda>" else "callable_target"
        return AgentContext(key=key, description="opaque callable target")

    def new(self) -> CallableTarget:
        """Return a fresh copy sharing the same callable, with state reset via reset_fn."""
        if self._reset_fn is not None:
            self._reset_fn()
        return CallableTarget(
            self._fn,
            reset_fn=self._reset_fn,
            usage_fn=self._usage_fn,
            agent_context=self._agent_context,
        )
