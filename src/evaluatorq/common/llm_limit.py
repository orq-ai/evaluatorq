"""Run-scoped ceiling on concurrent LLM requests.

Every call through ``common.llm_call`` takes a slot, so one number bounds the
whole run regardless of how the datapoint/job/evaluator/jury fan-out nests
above it. The ``parallelism`` knobs bound *tasks*, and a task can make any
number of requests; this bounds requests.

A job body that calls a provider SDK directly is invisible here — wrap it in
``llm_slot()`` to have it counted.

Concurrency, not rate: N slots is ``N / latency`` requests per second, so a
provider that speeds up raises the request rate at a fixed N.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

# Holds the semaphore itself, not the limit: it is created inside the running
# loop by `llm_concurrency_limit` and inherited by child tasks through the
# context copy, so it can never be shared across event loops.
_llm_semaphore: ContextVar[asyncio.Semaphore | None] = ContextVar('evaluatorq_llm_semaphore', default=None)


@asynccontextmanager
async def llm_concurrency_limit(max_concurrent: int | None) -> AsyncIterator[None]:  # noqa: RUF029 - async so the semaphore is built on the running loop
    """Bound concurrent LLM requests within this block. ``None`` leaves them unbounded.

    Set once per run. Tasks created inside inherit the limit; tasks created before
    it do not, so enter this before fanning out.
    """
    if max_concurrent is None:
        yield
        return
    if max_concurrent < 1:
        raise ValueError(f'max_concurrent must be >= 1, got {max_concurrent}')
    token = _llm_semaphore.set(asyncio.Semaphore(max_concurrent))
    try:
        yield
    finally:
        _llm_semaphore.reset(token)


@asynccontextmanager
async def llm_slot() -> AsyncIterator[None]:
    """Hold one slot of the run's LLM budget; a no-op when no limit is set.

    Wrap only the request itself — holding a slot across parsing or judging
    shrinks the budget without reducing load on the provider.
    """
    semaphore = _llm_semaphore.get()
    if semaphore is None:
        yield
        return
    async with semaphore:
        yield
