"""Simulation/AgentTarget wrapper for Pydantic AI agents."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from evaluatorq.contracts import (
    AgentContext,
    AgentResponse,
    AgentTarget,
    Message,
    OutputMessage,
    TextOutputItem,
    TokenUsage,
    ToolCallOutputItem,
    content_to_text,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic_ai import Agent

logger = logging.getLogger(__name__)


class PydanticAITarget(AgentTarget):
    """Wraps a Pydantic AI ``Agent`` as a unified ``AgentTarget``.

    Pydantic AI threads multi-turn context through its own typed message objects
    (``message_history=``), not a role/content list. So, like ``LangGraphTarget``,
    this target owns conversation state internally: each :meth:`respond` forwards
    only the latest user turn and re-feeds the accumulated history. Use
    :meth:`new` to get an independent instance for parallel jobs.

    Quirks handled here:
    - Message format: simulation passes ``list[Message]``; Pydantic AI wants a
      single ``user_prompt`` plus typed ``message_history``. We send the last
      user message and thread ``result.all_messages()`` across turns.
    - Async: ``agent.run`` is awaitable, so no thread offload is needed.
    - Tokens: ``RunUsage`` exposes ``input_tokens`` / ``output_tokens`` but no
      total; total is derived. ``usage`` is a property on recent versions and a
      method on older ones, so both are tried.

    Usage::

        from pydantic_ai import Agent
        from evaluatorq.integrations.pydantic_ai_integration import PydanticAITarget

        agent = Agent(model, system_prompt="You are a support agent.")
        target = PydanticAITarget(agent)
        results = await simulate(target=target, ...)
    """

    def __init__(
        self,
        agent: Agent[Any, Any],
        *,
        run_kwargs: dict[str, Any] | None = None,
        agent_context: AgentContext | None = None,
    ) -> None:
        """Create a Pydantic AI target.

        Args:
            agent: A Pydantic AI ``Agent`` instance.
            run_kwargs: Optional extra keyword arguments forwarded to
                ``agent.run()`` (e.g. ``{"model_settings": {...}}``).
            agent_context: Optional :class:`AgentContext` override returned
                verbatim from :meth:`get_agent_context`.
        """
        super().__init__(memory_entity_id=uuid4().hex)
        self._agent = agent
        self._run_kwargs = run_kwargs or {}
        self._agent_context = agent_context
        # Accumulated Pydantic AI message history (typed ModelMessage objects).
        # Not safe for concurrent respond() on one instance — use .new().
        self._history: list[Any] = []

    async def respond(self, messages: list[Message]) -> AgentResponse:
        """Send the latest user turn to the agent; thread history internally."""
        if not messages or messages[-1].role != "user":
            raise ValueError("PydanticAITarget.respond requires messages[-1].role == 'user'")
        prompt = content_to_text(messages[-1].content)

        result = await self._agent.run(
            prompt,
            message_history=self._history or None,
            **self._run_kwargs,
        )
        # Persist the full typed history so the next turn continues this thread.
        # If this fails, history is now STALE (missing this turn's exchange) and
        # later turns will see a truncated conversation — surface it loudly rather
        # than silently degrading.
        try:
            self._history = list(result.all_messages())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "PydanticAITarget: result.all_messages() failed (%s); conversation "
                "history is now stale and subsequent turns may be incoherent",
                exc,
            )

        # Build output items from this turn's messages, preserving interleaved
        # text/tool ordering so tool-usage criteria are visible to the judge —
        # parity with LangGraphTarget / OpenAIAgentTarget. Falls back to plain
        # text if the message objects don't expose the expected parts.
        output: list[OutputMessage] = _build_output(result)
        if not output:
            text = "" if result.output is None else str(result.output)
            output = [TextOutputItem(text=text, annotations=[])]
        return AgentResponse(output=output, usage=_extract_usage(result))

    async def get_agent_context(self) -> AgentContext:
        """Return agent context introspected from the wrapped agent."""
        if self._agent_context is not None:
            return self._agent_context
        agent = self._agent
        raw_key = str(getattr(agent, "name", None) or "pydantic_ai_agent").lower()
        key = re.sub(r"[^a-z0-9_]+", "_", raw_key).strip("_") or "pydantic_ai_agent"
        model_attr = getattr(agent, "model", None)
        model = getattr(model_attr, "model_name", None) or (
            str(model_attr) if model_attr is not None else None
        )
        return AgentContext(
            key=key,
            display_name=raw_key,
            description="Pydantic AI agent target",
            model=model,
        )

    def new(self) -> PydanticAITarget:
        """Return an independent instance with fresh conversation state."""
        # The Agent is shared by reference: Pydantic AI agents are stateless, and
        # each clone owns its own _history, so parallel runs don't interfere.
        return PydanticAITarget(
            self._agent,
            run_kwargs=dict(self._run_kwargs),
            agent_context=self._agent_context,
        )


def _make_part_classifier() -> Callable[[Any], str]:
    """Return a function mapping a Pydantic AI message part to a kind string.

    Prefers ``isinstance`` against the real part classes (so subclassed variants
    are still recognised); falls back to ``type(part).__name__`` if the message
    module can't be imported. Returns one of ``"TextPart"``, ``"ToolCallPart"``,
    ``"ToolReturnPart"`` or ``""``.
    """
    try:
        from pydantic_ai.messages import TextPart, ToolCallPart, ToolReturnPart
    except Exception:  # pragma: no cover - defensive
        def by_name(part: Any) -> str:
            name = type(part).__name__
            return name if name in {"TextPart", "ToolCallPart", "ToolReturnPart"} else ""

        return by_name

    def by_isinstance(part: Any) -> str:
        if isinstance(part, TextPart):
            return "TextPart"
        if isinstance(part, ToolCallPart):
            return "ToolCallPart"
        if isinstance(part, ToolReturnPart):
            return "ToolReturnPart"
        return ""

    return by_isinstance


# Built once: the underlying import is cached in sys.modules, but the classifier
# closure itself need only be constructed a single time.
_classify_part = _make_part_classifier()


def _build_output(result: Any) -> list[OutputMessage]:
    """Convert a run's new messages into ordered AgentResponse output items.

    Pydantic AI message parts map as: ``TextPart`` -> :class:`TextOutputItem`,
    ``ToolCallPart`` -> :class:`ToolCallOutputItem` (with its matching
    ``ToolReturnPart`` content merged in by ``tool_call_id``). Order is preserved
    so ReAct-style text/tool interleaving round-trips. Defensive: any unexpected
    shape simply yields no items and the caller falls back to ``result.output``.
    """
    try:
        new_messages = list(result.new_messages())
    except Exception:  # pragma: no cover - defensive
        return []

    classify = _classify_part
    items: list[OutputMessage] = []
    tool_index: dict[str, int] = {}
    for msg in new_messages:
        for part in getattr(msg, "parts", []) or []:
            kind = classify(part)
            if kind == "TextPart":
                text = getattr(part, "content", "")
                if isinstance(text, str) and text:
                    items.append(TextOutputItem(text=text, annotations=[]))
            elif kind == "ToolCallPart":
                name = str(getattr(part, "tool_name", "") or "")
                args = getattr(part, "args", "{}")
                args_str = args if isinstance(args, str) else json.dumps(args, default=str)
                call_id = str(getattr(part, "tool_call_id", "") or "")
                tc = (
                    ToolCallOutputItem(name=name, arguments=args_str, id=call_id, call_id=call_id)
                    if call_id
                    else ToolCallOutputItem(name=name, arguments=args_str)
                )
                items.append(tc)
                if call_id:
                    tool_index[call_id] = len(items) - 1
            elif kind == "ToolReturnPart":
                call_id = str(getattr(part, "tool_call_id", "") or "")
                idx = tool_index.get(call_id)
                if idx is not None and isinstance(items[idx], ToolCallOutputItem):
                    out = getattr(part, "content", "")
                    out_str = out if isinstance(out, str) else str(out)
                    items[idx] = items[idx].model_copy(update={"result": out_str})
    return items


def _extract_usage(result: Any) -> TokenUsage | None:
    """Pull a TokenUsage out of a Pydantic AI run result, tolerant of version drift.

    ``usage`` is a property on recent Pydantic AI and a method on older releases;
    ``RunUsage`` exposes ``input_tokens``/``output_tokens`` but no total, so the
    total is derived. Returns ``None`` when usage cannot be read so a run never
    fails just because token accounting is unavailable.
    """
    raw = getattr(result, "usage", None)
    # On recent versions ``usage`` is a property returning a RunUsage; on older
    # ones it is a method. Only call it when it isn't already a usage object, so
    # the property path doesn't emit a deprecation warning.
    if raw is not None and not hasattr(raw, "input_tokens") and callable(raw):
        try:
            raw = raw()
        except Exception:  # pragma: no cover - defensive
            return None
    if raw is None:
        return None
    prompt = int(getattr(raw, "input_tokens", 0) or 0)
    completion = int(getattr(raw, "output_tokens", 0) or 0)
    total = getattr(raw, "total_tokens", None)
    total_tokens = int(total) if total else prompt + completion
    calls = int(getattr(raw, "requests", 0) or 0) or 1
    if prompt == 0 and completion == 0 and total_tokens == 0:
        return None
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total_tokens,
        calls=calls,
    )
