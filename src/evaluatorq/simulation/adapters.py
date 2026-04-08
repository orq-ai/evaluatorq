"""Convenience adapters for creating simulation targetCallbacks.

These helpers create ``target_callback`` functions from common agent sources,
so users don't need to wire the plumbing themselves.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from evaluatorq.simulation.types import ChatMessage


def from_orq_deployment(
    agent_key: str,
) -> Callable[[list[ChatMessage]], Awaitable[str]]:
    """Create a simulation ``target_callback`` from an Orq deployment key."""
    if not agent_key.strip():
        raise ValueError("agent_key must be a non-empty string")

    async def callback(messages: list[ChatMessage]) -> str:
        from evaluatorq.deployment import invoke

        return await invoke(
            agent_key,
            messages=[{"role": m.role, "content": m.content} for m in messages],
        )

    return callback


def from_chat_completions(
    fn: Callable[[list[dict[str, str]]], Any],
) -> Callable[[list[ChatMessage]], Awaitable[str]]:
    """Create a simulation ``target_callback`` from a chat completions function.

    Useful for raw OpenAI SDK, Azure OpenAI, or any OpenAI-compatible provider.
    """

    async def callback(messages: list[ChatMessage]) -> str:
        result = fn([{"role": m.role, "content": m.content} for m in messages])
        if inspect.isawaitable(result):
            return await result
        return result

    return callback
