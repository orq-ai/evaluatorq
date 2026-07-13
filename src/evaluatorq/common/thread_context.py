"""Per-conversation Orq thread id, carried via a ContextVar.

Multi-turn calls on the Orq router / Responses API are grouped in Orq
observability by sending a ``thread: {id: ...}`` body param on every turn
(see https://docs.orq.ai/docs/ai-gateway/thread-management). The id must be
stable for one conversation but distinct across conversations.

Targets are stateless and shared across concurrently-running conversations
(one ``SimulationRunner`` / target instance serves every datapoint), so the
thread id can't live on the target instance. A ContextVar carries it instead:
each asyncio Task gets an isolated copy of the context, so concurrent
conversations don't leak thread ids into each other.

Callers that own a conversation (the sim runner, the redteam orchestrator)
wrap it in ``conversation_thread()``; the target reads ``current_thread_id()``
and attaches the thread param. When unset, no thread param is sent.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_thread_id: ContextVar[str | None] = ContextVar('orq_thread_id', default=None)


def current_thread_id() -> str | None:
    """Return the thread id for the active conversation, or None if unset."""
    return _thread_id.get()


def thread_body_param() -> dict[str, dict[str, str]]:
    """Return ``{'thread': {'id': ...}}`` for the active conversation, or ``{}``.

    Ready to splat into an Orq router request body / SDK ``extra_body``.
    """
    tid = _thread_id.get()
    return {'thread': {'id': tid}} if tid else {}


@contextmanager
def conversation_thread(thread_id: str | None = None) -> Iterator[str]:
    """Bind a thread id for the duration of one conversation.

    Generates a uuid when none is given. Restores the previous value on exit so
    nested/sequential conversations don't bleed into each other.

    Yields:
        The bound thread id.
    """
    tid = thread_id or str(uuid.uuid4())
    token = _thread_id.set(tid)
    try:
        yield tid
    finally:
        _thread_id.reset(token)


__all__ = ['conversation_thread', 'current_thread_id', 'thread_body_param']
