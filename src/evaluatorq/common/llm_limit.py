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
    from contextvars import Token

# Holds the semaphore itself, not the limit: it is created inside the running
# loop by `llm_concurrency_limit` and inherited by child tasks through the
# context copy, so it can never be shared across event loops.
_llm_semaphore: ContextVar[asyncio.Semaphore | None] = ContextVar('evaluatorq_llm_semaphore', default=None)


class llm_concurrency_limit:  # noqa: N801 - used as a context manager, named like one
    """Bound concurrent LLM requests within this block. ``None`` leaves them unbounded.

    Set once per run, before fanning out: tasks created inside inherit the limit,
    tasks created before it do not.

    Enterable with ``with`` or ``async with`` — the entry points wrap at whichever
    block they already open, and reindenting theirs to match would rewrite hundreds
    of untouched lines. The semaphore binds to a loop on first acquire, which is
    inside the run on either path.

    An enclosing limit is left alone when ``max_concurrent`` is ``None``, so a nested
    ``evaluatorq()`` cannot widen the budget its caller set.
    """

    def __init__(self, max_concurrent: int | None) -> None:
        if max_concurrent is not None and max_concurrent < 1:
            raise ValueError(f'max_concurrent must be >= 1, got {max_concurrent}')
        self._max_concurrent = max_concurrent
        self._token: Token[asyncio.Semaphore | None] | None = None

    def __enter__(self) -> None:
        if self._max_concurrent is not None:
            self._token = _llm_semaphore.set(asyncio.Semaphore(self._max_concurrent))

    def __exit__(self, *_exc: object) -> None:
        if self._token is not None:
            _llm_semaphore.reset(self._token)
            self._token = None

    async def __aenter__(self) -> None:
        self.__enter__()

    async def __aexit__(self, *exc: object) -> None:
        self.__exit__(*exc)


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
