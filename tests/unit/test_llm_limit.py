"""The LLM concurrency ceiling holds across the datapoint/job/evaluator fan-out."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from evaluatorq.common.llm_limit import llm_concurrency_limit, llm_slot


class _Peak:
    """Track the high-water mark of concurrent slot holders."""

    def __init__(self) -> None:
        self.live = 0
        self.peak = 0

    async def call(self) -> None:
        async with llm_slot():
            self.live += 1
            self.peak = max(self.peak, self.live)
            await asyncio.sleep(0.01)
            self.live -= 1


@pytest.mark.asyncio
async def test_slot_bounds_concurrent_calls() -> None:
    peak = _Peak()
    async with llm_concurrency_limit(3):
        await asyncio.gather(*(peak.call() for _ in range(20)))
    assert peak.peak == 3


@pytest.mark.asyncio
async def test_bound_survives_nested_fan_out() -> None:
    """A nested fan-out is where a per-task semaphore would multiply instead of hold."""
    peak = _Peak()

    async def inner() -> None:
        await asyncio.gather(*(peak.call() for _ in range(5)))

    async def outer() -> None:
        await asyncio.gather(*(inner() for _ in range(5)))

    async with llm_concurrency_limit(4):
        await asyncio.gather(*(outer() for _ in range(3)))
    assert peak.peak == 4


@pytest.mark.asyncio
async def test_no_limit_leaves_calls_unbounded() -> None:
    peak = _Peak()
    await asyncio.gather(*(peak.call() for _ in range(8)))
    assert peak.peak == 8


@pytest.mark.asyncio
async def test_limit_does_not_leak_out_of_its_block() -> None:
    peak = _Peak()
    async with llm_concurrency_limit(2):
        await peak.call()
    await asyncio.gather(*(peak.call() for _ in range(6)))
    assert peak.peak == 6


@pytest.mark.asyncio
async def test_rejects_a_limit_below_one() -> None:
    with pytest.raises(ValueError, match='must be >= 1'):
        async with llm_concurrency_limit(0):
            pass


@pytest.mark.asyncio
async def test_execute_chat_completion_takes_a_slot() -> None:
    """The module test proves the semaphore works; this proves llm_call uses it."""
    from evaluatorq.common.llm_call import execute_chat_completion

    peak = _Peak()

    class _Completions:
        async def create(self, **_kwargs: object) -> object:
            peak.live += 1
            peak.peak = max(peak.peak, peak.live)
            await asyncio.sleep(0.01)
            peak.live -= 1
            return SimpleNamespace(choices=[], usage=None, model='fake')

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))

    async def one() -> None:
        await execute_chat_completion(
            client=cast('Any', client),
            model='fake',
            messages=[{'role': 'user', 'content': 'hi'}],
            span=None,
            timeout_s=5.0,
            inject_trace_headers=False,
        )

    async with llm_concurrency_limit(2):
        await asyncio.gather(*(one() for _ in range(10)))
    assert peak.peak == 2


@pytest.mark.asyncio
async def test_enterable_both_ways() -> None:
    """red_team wraps at an `async with`, simulate at a `with`; both must work."""
    sync_peak, async_peak = _Peak(), _Peak()

    with llm_concurrency_limit(2):
        await asyncio.gather(*(sync_peak.call() for _ in range(6)))
    async with llm_concurrency_limit(2):
        await asyncio.gather(*(async_peak.call() for _ in range(6)))

    assert (sync_peak.peak, async_peak.peak) == (2, 2)


@pytest.mark.asyncio
async def test_none_leaves_an_enclosing_limit_alone() -> None:
    """A nested evaluatorq() defaults to None; it must not widen its caller's budget."""
    peak = _Peak()
    with llm_concurrency_limit(2), llm_concurrency_limit(None):
        await asyncio.gather(*(peak.call() for _ in range(6)))
    assert peak.peak == 2


@pytest.mark.parametrize(
    ('module', 'func'),
    [
        ('evaluatorq.evaluatorq', 'evaluatorq'),
        ('evaluatorq.redteam.runner', 'red_team'),
        ('evaluatorq.simulation.api', 'simulate'),
        ('evaluatorq.simulation.api', 'generate_and_simulate'),
        ('evaluatorq.simulation.api', 'generate'),
    ],
)
def test_entry_points_expose_the_knob(module: str, func: str) -> None:
    import importlib
    import inspect

    parameter = inspect.signature(getattr(importlib.import_module(module), func)).parameters
    assert 'llm_parallelism' in parameter, f'{func} has no LLM concurrency knob'
    assert parameter['llm_parallelism'].default is None


@pytest.mark.asyncio
async def test_each_run_gets_its_own_semaphore() -> None:
    """Two concurrent runs must not share a budget, nor carry one across loops."""
    first, second = _Peak(), _Peak()

    async def run(peak: _Peak, limit: int) -> None:
        async with llm_concurrency_limit(limit):
            await asyncio.gather(*(peak.call() for _ in range(10)))

    await asyncio.gather(run(first, 2), run(second, 5))
    assert (first.peak, second.peak) == (2, 5)
