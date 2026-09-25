"""Run-scoped ceiling on concurrent LLM requests.

Every call through ``common.llm_call`` takes a slot, so one number bounds the
whole run regardless of how the datapoint/job/evaluator/jury fan-out nests
above it. The ``parallelism`` knobs bound *tasks*, and a task can make any
number of requests; this bounds requests.

With no limit set, calls share a default ceiling of ``DEFAULT_LLM_PARALLELISM``
so an unconfigured run cannot flood a provider; ``-1`` disables the ceiling.

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
from weakref import WeakKeyDictionary

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from contextvars import Token

DEFAULT_LLM_PARALLELISM = 10
"""Ceiling applied when no ``llm_concurrency_limit`` is active."""

UNBOUNDED = -1
"""Pass as the limit to disable the ceiling entirely."""


def check_llm_parallelism(value: int | None) -> int | None:
    """Accept ``None``, ``UNBOUNDED`` (-1) or a positive count; reject anything else."""
    if value is not None and value != UNBOUNDED and value < 1:
        raise ValueError(f'llm_parallelism must be >= 1, or -1 for no limit, got {value}')
    return value


def check_llm_parallelism_option(value: int | None) -> int | None:
    """Typer callback: ``check_llm_parallelism`` with a CLI-shaped error."""
    import typer

    try:
        return check_llm_parallelism(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


class _Budget:
    """One limit, holding a semaphore per event loop.

    A semaphore binds to the loop that first blocks on it, so a single one shared
    by a ``with`` block spanning two ``asyncio.run`` calls breaks the second.
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._semaphores: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = WeakKeyDictionary()

    def semaphore(self) -> asyncio.Semaphore | None:
        if self._limit == UNBOUNDED:
            return None
        loop = asyncio.get_running_loop()
        semaphore = self._semaphores.get(loop)
        if semaphore is None:
            semaphore = self._semaphores[loop] = asyncio.Semaphore(self._limit)
        return semaphore


# Inherited by child tasks through the context copy. The default budget is shared
# by every run that sets no limit, per loop, so an unconfigured run cannot flood
# a provider.
_DEFAULT_BUDGET = _Budget(DEFAULT_LLM_PARALLELISM)
_llm_budget: ContextVar[_Budget | None] = ContextVar('evaluatorq_llm_budget', default=None)


class llm_concurrency_limit:  # noqa: N801 - used as a context manager, named like one
    """Bound concurrent LLM requests within this block.

    ``None`` sets nothing: an enclosing limit applies, else the default of
    ``DEFAULT_LLM_PARALLELISM``. ``-1`` (``UNBOUNDED``) disables the ceiling.

    Set once per run, before fanning out: tasks created inside inherit the limit,
    tasks created before it do not. Each entry is its own budget, so two runs
    entered concurrently do not share one.

    Enterable with ``with`` or ``async with`` — the entry points wrap at whichever
    block they already open, and reindenting theirs to match would rewrite hundreds
    of untouched lines. A sync ``with`` may span several ``asyncio.run`` calls.

    An enclosing limit is left alone when ``max_concurrent`` is ``None``, so a nested
    ``evaluatorq()`` cannot widen the budget its caller set.
    """

    def __init__(self, max_concurrent: int | None) -> None:
        self._max_concurrent = check_llm_parallelism(max_concurrent)
        self._token: Token[_Budget | None] | None = None

    def __enter__(self) -> None:
        if self._max_concurrent is not None:
            self._token = _llm_budget.set(_Budget(self._max_concurrent))

    def __exit__(self, *_exc: object) -> None:
        if self._token is not None:
            _llm_budget.reset(self._token)
            self._token = None

    async def __aenter__(self) -> None:
        self.__enter__()

    async def __aexit__(self, *exc: object) -> None:
        self.__exit__(*exc)


@asynccontextmanager
async def llm_slot() -> AsyncIterator[None]:
    """Hold one slot of the run's LLM budget; a no-op when the limit is ``UNBOUNDED``.

    Wrap only the request itself — holding a slot across parsing or judging
    shrinks the budget without reducing load on the provider.
    """
    semaphore = (_llm_budget.get() or _DEFAULT_BUDGET).semaphore()
    if semaphore is None:
        yield
        return
    async with semaphore:
        yield
