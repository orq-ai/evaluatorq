"""The LLM concurrency ceiling holds across the datapoint/job/evaluator fan-out."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, cast

import pytest

from evaluatorq.common.llm_limit import (
    DEFAULT_LLM_PARALLELISM,
    UNBOUNDED,
    active_llm_parallelism,
    check_llm_parallelism_option,
    llm_concurrency_limit,
    llm_slot,
)


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
async def test_no_limit_applies_the_default_ceiling() -> None:
    peak = _Peak()
    await asyncio.gather(*(peak.call() for _ in range(DEFAULT_LLM_PARALLELISM * 2)))
    assert peak.peak == DEFAULT_LLM_PARALLELISM


def test_default_ceiling_survives_one_asyncio_run_per_call() -> None:
    """The default semaphore is per loop; a shared one breaks the second asyncio.run."""
    for _ in range(2):
        peak = _Peak()

        async def batch(peak: _Peak = peak) -> None:
            await asyncio.gather(*(peak.call() for _ in range(DEFAULT_LLM_PARALLELISM + 2)))

        asyncio.run(batch())
        assert peak.peak == DEFAULT_LLM_PARALLELISM


def test_default_budget_works_across_event_loop_threads() -> None:
    def batch() -> int:
        peak = _Peak()

        async def run() -> None:
            await asyncio.gather(*(peak.call() for _ in range(DEFAULT_LLM_PARALLELISM + 2)))

        asyncio.run(run())
        return peak.peak

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: batch(), range(2))) == [DEFAULT_LLM_PARALLELISM] * 2


@pytest.mark.asyncio
async def test_limit_does_not_leak_out_of_its_block() -> None:
    peak = _Peak()
    async with llm_concurrency_limit(2):
        await peak.call()
    await asyncio.gather(*(peak.call() for _ in range(6)))
    assert peak.peak == 6  # below the default ceiling, so no longer bounded by 2


@pytest.mark.parametrize('bad', [0, -2])
def test_rejects_a_limit_below_one_other_than_unbounded(bad: int) -> None:
    with pytest.raises(ValueError, match='must be >= 1'):
        llm_concurrency_limit(bad)


@pytest.mark.asyncio
async def test_minus_one_disables_the_ceiling() -> None:
    peak = _Peak()
    async with llm_concurrency_limit(UNBOUNDED):
        assert active_llm_parallelism() is None
        await asyncio.gather(*(peak.call() for _ in range(DEFAULT_LLM_PARALLELISM * 3)))
    assert peak.peak == DEFAULT_LLM_PARALLELISM * 3


def test_explicit_limit_survives_one_asyncio_run_per_call() -> None:
    """A sync `with` spanning two asyncio.run calls must not reuse a loop-bound semaphore."""
    peaks = [_Peak(), _Peak()]
    with llm_concurrency_limit(2):
        for peak in peaks:

            async def batch(peak: _Peak = peak) -> None:
                await asyncio.gather(*(peak.call() for _ in range(6)))

            asyncio.run(batch())
    assert [p.peak for p in peaks] == [2, 2]


def test_evaluatorq_params_accept_minus_one_and_reject_zero() -> None:
    from pydantic import ValidationError

    from evaluatorq.types import EvaluatorParams

    assert EvaluatorParams(data=[], inference=False, llm_parallelism=-1).llm_parallelism == -1
    with pytest.raises(ValidationError, match='must be >= 1'):
        EvaluatorParams(data=[], inference=False, llm_parallelism=0)


def test_cli_option_accepts_minus_one_and_rejects_zero() -> None:
    import typer

    assert check_llm_parallelism_option(-1) == -1
    assert check_llm_parallelism_option(None) is None
    with pytest.raises(typer.BadParameter, match='must be >= 1'):
        check_llm_parallelism_option(0)


@pytest.mark.parametrize(
    ('module', 'func', 'removed'),
    [
        ('evaluatorq.pairwise', 'run_pairwise', 'max_concurrency'),
        ('evaluatorq.llm_jury', 'llm_jury_pairwise', 'max_concurrency'),
        ('evaluatorq.llm_jury', 'PairwiseComparator', 'max_concurrency'),
        ('evaluatorq.common.jury', 'run_jury', 'max_concurrency'),
        ('evaluatorq.simulation.generators.datapoint_generator', 'DatapointGenerator', 'rate_limit_delay'),
        ('evaluatorq.simulation.generators.datapoint_generator', 'DatapointGenerator', 'max_concurrent_calls'),
        ('evaluatorq.redteam.adaptive.strategy_planner', 'plan_strategies_for_categories', 'generation_parallelism'),
        ('evaluatorq.redteam.adaptive.strategy_planner', 'plan_strategies_for_vulnerabilities', 'generation_parallelism'),
        ('evaluatorq.redteam.adaptive.pipeline', 'generate_dynamic_datapoints', 'datapoint_parallelism'),
        (
            'evaluatorq.redteam.adaptive.pipeline',
            'generate_dynamic_datapoints_for_vulnerabilities',
            'datapoint_parallelism',
        ),
    ],
)
def test_per_call_site_knobs_are_gone(module: str, func: str, removed: str) -> None:
    """The llm_parallelism ceiling replaced these; a second knob would nest and multiply again."""
    import importlib
    import inspect

    assert removed not in inspect.signature(getattr(importlib.import_module(module), func)).parameters


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('outer', 'inner', 'expected'),
    [
        (2, 5, 2),
        (5, 2, 2),
        (2, UNBOUNDED, 2),
        (UNBOUNDED, 3, 3),
    ],
)
async def test_nested_limits_honor_every_finite_budget(outer: int, inner: int, expected: int) -> None:
    peak = _Peak()
    async with llm_concurrency_limit(outer):
        async with llm_concurrency_limit(inner):
            assert active_llm_parallelism() == expected
            await asyncio.gather(*(peak.call() for _ in range(12)))
    assert peak.peak == expected


@pytest.mark.asyncio
async def test_nested_groups_share_the_outer_budget() -> None:
    peak = _Peak()

    async def group(inner: int) -> None:
        async with llm_concurrency_limit(inner):
            await asyncio.gather(*(peak.call() for _ in range(8)))

    async with llm_concurrency_limit(2):
        await asyncio.gather(group(5), group(UNBOUNDED))
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


def test_every_provider_call_in_llm_call_goes_through_bounded_call() -> None:
    """Only `test_execute_chat_completion_takes_a_slot` runs an executor end to end.

    This covers the rest: a `client.*(...)` call in `llm_call.py` that is not the
    direct argument of `_bounded_call` skips the slot, and the ceiling stops holding
    for every judge, generator and trace step behind that executor.
    """
    import ast
    import inspect

    from evaluatorq.common import llm_call

    tree = ast.parse(inspect.getsource(llm_call))
    wrapped: set[int] = set()
    provider_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == '_bounded_call':
            wrapped.update(id(arg) for arg in node.args)
        root = node.func
        while isinstance(root, ast.Attribute):
            root = root.value
        if isinstance(node.func, ast.Attribute) and isinstance(root, ast.Name) and root.id == 'client':
            provider_calls.append(node)

    assert provider_calls, 'found no client calls; the scan no longer matches llm_call.py'
    unbounded = [f'line {call.lineno}' for call in provider_calls if id(call) not in wrapped]
    assert not unbounded, f'provider calls outside _bounded_call (no LLM slot): {unbounded}'
