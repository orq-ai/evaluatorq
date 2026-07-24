"""Convenience adapters for creating simulation targets.

These helpers create callables for the ``target=`` parameter from common agent sources,
so users don't need to wire the plumbing themselves.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from evaluatorq.contracts import AgentResponse, content_to_text

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from evaluatorq.simulation.types import Message


def from_orq_deployment(
    agent_key: str,
) -> Callable[[list[Message]], Awaitable[AgentResponse]]:
    """Create a simulation ``target`` callable from an Orq deployment key."""
    if not agent_key.strip():
        raise ValueError('agent_key must be a non-empty string')

    async def callback(messages: list[Message]) -> AgentResponse:
        from evaluatorq.common.thread_context import current_thread_id, pipeline_metadata
        from evaluatorq.deployment import ThreadConfig, deployment

        metadata: dict[str, object] | None = dict(pipeline_metadata()) or None
        tid = current_thread_id()
        thread: ThreadConfig | None = {'id': tid} if tid else None

        resp = await deployment(
            agent_key,
            messages=[{'role': m.role, 'content': content_to_text(m.content)} for m in messages],
            metadata=metadata,
            thread=thread,
        )
        return AgentResponse(text=resp.content, usage=resp.usage)

    # Carry the key so the run-metadata label can render "deployment:<key>",
    # symmetric to how an AgentTarget exposes `agent_key`.
    callback.deployment_key = agent_key  # pyright: ignore[reportFunctionMemberAccess]
    return callback


def from_chat_completions(
    fn: Callable[[list[dict[str, str]]], Any],
) -> Callable[[list[Message]], Awaitable[str]]:
    """Create a simulation ``target`` callable from a chat completions function.

    Useful for raw OpenAI SDK, Azure OpenAI, or any OpenAI-compatible provider.
    """

    async def callback(messages: list[Message]) -> str:
        result = fn([{'role': m.role, 'content': content_to_text(m.content)} for m in messages])
        if inspect.isawaitable(result):
            return await result
        return result

    return callback
