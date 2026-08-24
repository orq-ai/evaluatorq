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
        from evaluatorq.common.tracing import record_llm_response, record_token_usage, with_llm_span
        from evaluatorq.deployment import MessageDict, ThreadConfig, deployment

        metadata: dict[str, object] | None = dict(pipeline_metadata()) or None
        tid = current_thread_id()
        thread: ThreadConfig | None = {'id': tid} if tid else None

        payload: list[MessageDict] = [{'role': m.role, 'content': content_to_text(m.content)} for m in messages]
        # Mirrors the red-team deployment leg (redteam/runtime/jobs.py): without a
        # span here the deployment invocation is the only target call that leaves
        # no LLM span under orq.simulation.target_call.
        async with with_llm_span(
            model=f'deployment:{agent_key}',
            operation='invoke',
            provider='orq',
            input_messages=payload,
            attributes={'orq.llm.purpose': 'target'},
        ) as span:
            resp = await deployment(
                agent_key,
                messages=payload,
                metadata=metadata,
                thread=thread,
            )
            record_llm_response(span, resp.raw, output_content=resp.content)
            if resp.usage is not None:
                # resp.raw is None on some deployment paths, so record_llm_response
                # finds no usage there; resp.usage is already normalised. Guarded:
                # an unconditional call writes 0/0/0, which reads as a free call.
                record_token_usage(span, usage=resp.usage)
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
