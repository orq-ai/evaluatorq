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

# The evaluatorq surface driving the current run ('agent_simulation' /
# 'red_teaming'), sent as request metadata so Orq traces are attributable to the
# pipeline that produced them. Carried via a ContextVar for the same reason as
# the thread id: targets are stateless and shared across concurrent runs.
_pipeline: ContextVar[str | None] = ContextVar('evaluatorq_pipeline', default=None)

# Separator joining the run id and the per-conversation discriminators into a
# thread id. Kept out of the OQL query grammar's way: the run id is uuid hex and
# the parts are ints / sanitized keys, so ':' never appears inside a segment.
_THREAD_ID_SEP = ':'


def build_thread_id(run_id: str | None, *parts: object) -> str | None:
    """Compose a run-scoped thread id, or None when there is no run to link.

    Single source of truth for the ``{run_id}:{...}`` format that the dashboard's
    run-level deep link (``thread_id:contains:{run_id}``) depends on: sim passes
    ``build_thread_id(run_id, index)`` → ``{run_id}:{index}``; red teaming passes
    ``build_thread_id(run_id, agent_key, index)`` → ``{run_id}:{agent_key}:{index}``.
    The ``run_id`` prefix is what makes the run-level ``contains`` query match
    every conversation in the run.
    """
    if not run_id:
        return None
    return _THREAD_ID_SEP.join([run_id, *(str(p) for p in parts)])


def build_static_thread_id(run_id: str | None, target_id: str, row: int) -> str:
    """Build a deterministic thread id for one static red-team datapoint.

    Normal runs use the caller's run id, matching dynamic red-team targets. The
    fallback keeps direct job invocations correlated to a stable target/row
    pair instead of minting a random context id.
    """
    stable_run_id = run_id or f'static-{target_id}'
    thread_id = build_thread_id(stable_run_id, target_id, row)
    if thread_id is None:  # Defensive: ``stable_run_id`` is always non-empty.
        raise RuntimeError('Static thread id requires a non-empty stable run id')
    return thread_id


def build_static_thread_id(run_id: str | None, target_id: str, row: int) -> str:
    """Build a deterministic thread id for one static red-team datapoint.

    Normal runs use the caller's run id, matching dynamic red-team targets. The
    fallback keeps direct job invocations (including isolated tests) correlated
    to a stable target/row pair instead of minting a random context id.
    """
    stable_run_id = run_id or f'static-{target_id}'
    thread_id = build_thread_id(stable_run_id, target_id, row)
    if thread_id is None:  # Defensive: ``stable_run_id`` is always non-empty.
        raise RuntimeError('Static thread id requires a non-empty stable run id')
    return thread_id


def current_thread_id() -> str | None:
    """Return the thread id for the active conversation, or None if unset."""
    return _thread_id.get()


def thread_body_param() -> dict[str, dict[str, str]]:
    """Return ``{'thread': {'id': ...}}`` for the active conversation, or ``{}``.

    Ready to splat into an Orq router request body / SDK ``extra_body``.
    """
    tid = _thread_id.get()
    return {'thread': {'id': tid}} if tid else {}


def pipeline_metadata_param() -> dict[str, dict[str, str]]:
    """Return ``{'metadata': {'evaluatorq_pipeline': ...}}`` for the active run, or ``{}``.

    Ready to splat into an Orq request (``metadata=`` kwarg on the agents SDK, or
    merged into ``extra_body`` on the Responses client).
    """
    label = _pipeline.get()
    return {'metadata': {'evaluatorq_pipeline': label}} if label else {}


@contextmanager
def evaluatorq_pipeline(label: str) -> Iterator[str]:
    """Bind the pipeline label ('agent_simulation' / 'red_teaming') for a run.

    Restores the previous value on exit so nested/sequential runs don't bleed.

    Yields:
        The bound pipeline label.
    """
    token = _pipeline.set(label)
    try:
        yield label
    finally:
        _pipeline.reset(token)


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


__all__ = [
    'build_static_thread_id',
    'build_thread_id',
    'conversation_thread',
    'current_thread_id',
    'evaluatorq_pipeline',
    'pipeline_metadata_param',
    'thread_body_param',
]
