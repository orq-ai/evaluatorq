"""Simulation/red-teaming target wrapper for CrewAI crews.

CrewAI is the biggest format divergence among the supported frameworks, which
makes it a good generality stress test for the unified ``AgentTarget`` protocol.
Three quirks are handled here and called out explicitly:

1. **Sync API.** ``Crew.kickoff`` is synchronous and blocking, so it is run in a
   worker thread via ``asyncio.to_thread`` to avoid stalling the event loop that
   drives the simulation's other concurrent datapoints.
2. **No message-list interface.** A crew is driven by ``inputs`` interpolated
   into task descriptions, not a role/content transcript. The full conversation
   is flattened into a single string and injected under one input key (default
   ``"conversation"``), which the crew's task description must reference as
   ``{conversation}``.
3. **Multi-agent.** A crew may contain several agents; "the response" is the
   crew's final output (``CrewOutput.raw``). Intermediate agent/tool steps are
   not surfaced as tool-call items.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from evaluatorq.contracts import AgentTarget, Message
from evaluatorq.redteam.contracts import (
    AgentContext,
    AgentResponse,
    OutputMessage,
    TextOutputItem,
    TokenUsage,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from crewai import Crew

logger = logging.getLogger(__name__)

DEFAULT_INPUT_KEY = "conversation"

_ROLE_LABELS = {
    "user": "Customer",
    "assistant": "Agent",
    "system": "System",
    "tool": "Tool",
}


class CrewAITarget(AgentTarget):
    """Wraps a CrewAI ``Crew`` as a unified ``AgentTarget``.

    Each :meth:`respond` flattens the conversation into a single string and runs
    a fresh ``kickoff`` (a crew has no built-in turn memory). The crew's task
    description must reference the flattened transcript via ``{conversation}``
    (or whatever ``input_key`` you pass).

    For parallel jobs pass a ``crew_factory`` so :meth:`new` can build an
    independent crew; otherwise the same crew instance is reused.

    Usage::

        from crewai import Agent, Task, Crew
        from evaluatorq.integrations.crewai_integration import CrewAITarget

        def make_crew() -> Crew:
            agent = Agent(role=..., goal=..., backstory=..., llm=llm)
            task = Task(description="Conversation so far:\\n{conversation}",
                        expected_output="The agent's next reply.", agent=agent)
            return Crew(agents=[agent], tasks=[task])

        target = CrewAITarget(make_crew(), crew_factory=make_crew)
        results = await simulate(target=target, ...)
    """

    def __init__(
        self,
        crew: Crew,
        *,
        crew_factory: Callable[[], Crew] | None = None,
        input_key: str = DEFAULT_INPUT_KEY,
        extra_inputs: dict[str, Any] | None = None,
        agent_context: AgentContext | None = None,
    ) -> None:
        """Create a CrewAI target.

        Args:
            crew: A constructed CrewAI ``Crew``.
            crew_factory: Optional zero-arg callable returning a fresh ``Crew``;
                used by :meth:`new` for parallel-safe instances. When omitted,
                :meth:`new` reuses the same crew instance.
            input_key: The ``inputs`` key the flattened transcript is passed
                under; the crew's task description must reference ``{input_key}``.
            extra_inputs: Optional static inputs merged into every ``kickoff``
                (e.g. interpolated constants the task description references).
            agent_context: Optional :class:`AgentContext` override.
        """
        super().__init__(memory_entity_id=uuid4().hex)
        self._crew = crew
        self._crew_factory = crew_factory
        self._input_key = input_key
        self._extra_inputs = extra_inputs or {}
        self._agent_context = agent_context

    async def respond(self, messages: list[Message]) -> AgentResponse:
        """Flatten the transcript and run one synchronous crew kickoff off-thread."""
        if not messages or messages[-1].role != "user":
            raise ValueError("CrewAITarget.respond requires messages[-1].role == 'user'")
        conversation = _flatten(messages)
        inputs = {**self._extra_inputs, self._input_key: conversation}

        try:
            output = await asyncio.to_thread(self._crew.kickoff, inputs=inputs)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            raise
        except Exception as exc:
            raise RuntimeError(f"CrewAITarget: crew.kickoff() raised an error: {exc}") from exc

        text = str(getattr(output, "raw", "") or "")
        out: list[OutputMessage] = [TextOutputItem(text=text, annotations=[])]
        return AgentResponse(output=out, usage=_extract_usage(output))

    async def get_agent_context(self) -> AgentContext:
        """Return agent context introspected from the crew's agents."""
        if self._agent_context is not None:
            return self._agent_context
        agents = getattr(self._crew, "agents", None) or []
        roles = [str(getattr(a, "role", "")) for a in agents if getattr(a, "role", None)]
        raw_key = (roles[0] if roles else "crewai_crew").lower()
        # Real CrewAI role strings contain spaces, slashes, punctuation; collapse
        # anything outside [a-z0-9_] so the key is always well-formed.
        key = re.sub(r"[^a-z0-9_]+", "_", raw_key).strip("_") or "crewai_crew"
        description = (
            f"CrewAI crew target ({len(agents)} agent(s): {', '.join(roles)})"
            if roles
            else "CrewAI crew target"
        )
        return AgentContext(key=key, display_name=key, description=description)

    def new(self) -> CrewAITarget:
        """Return an independent instance for parallel jobs.

        Uses ``crew_factory`` to build a fresh crew when provided; otherwise
        reuses the same crew instance (acceptable for sequential runs, but pass a
        factory if you run datapoints in parallel against a stateful crew).
        """
        crew = self._crew_factory() if self._crew_factory is not None else self._crew
        return CrewAITarget(
            crew,
            crew_factory=self._crew_factory,
            input_key=self._input_key,
            extra_inputs=dict(self._extra_inputs),
            agent_context=self._agent_context,
        )


def _flatten(messages: list[Message]) -> str:
    """Render a role/content transcript as a single labelled string.

    CrewAI has no native multi-turn message interface, so the whole conversation
    collapses into one block the task description interpolates. Empty-content
    messages (e.g. pure tool-call turns) are skipped.
    """
    lines: list[str] = []
    for m in messages:
        if not m.content:
            continue
        label = _ROLE_LABELS.get(m.role, m.role.capitalize())
        # Escape braces: CrewAI fills {input_key} via str.format on the task
        # description, so any literal { } in user content would otherwise raise
        # KeyError or collide with other task variables (template injection).
        safe = m.content.replace("{", "{{").replace("}", "}}")
        lines.append(f"{label}: {safe}")
    return "\n".join(lines)


def _extract_usage(output: Any) -> TokenUsage | None:
    """Map ``CrewOutput.token_usage`` onto :class:`TokenUsage`.

    Returns ``None`` when usage is absent or all-zero so a run never fails over
    token accounting. CrewAI exposes ``successful_requests`` rather than a call
    count, which maps onto ``calls``.
    """
    usage = getattr(output, "token_usage", None)
    if usage is None:
        return None
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = getattr(usage, "total_tokens", None)
    total_tokens = int(total) if total else prompt + completion
    calls = int(getattr(usage, "successful_requests", 0) or 0) or 1
    if prompt == 0 and completion == 0 and total_tokens == 0:
        return None
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total_tokens,
        calls=calls,
    )
