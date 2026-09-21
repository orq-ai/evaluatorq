"""Lifecycle tests for the generation-owned trace finder run store."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, cast

import pytest

from evaluatorq.trace_finder.compiler import CompiledPlan
from evaluatorq.trace_finder.models import (
    CompiledQuery,
    FacetSelection,
    JevProjection,
    NumericFilters,
    PopulationRequest,
    RunRequest,
    Snapshot,
    TraceClassification,
    TraceRecord,
)
from evaluatorq.trace_finder.run_store import RunStore


def compiled_query() -> CompiledQuery:
    return CompiledQuery.model_validate({
        'task': {'kind': 'noul', 'instructions': 'Is this a help request?', 'state': {}},
        'selection': {'kind': 'values', 'values': [True]},
    })


def trace(index: int) -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id=f'trace-{index}',
        span_id=f'span-{index}',
        timestamp=datetime(2026, 9, 20, 12, index, tzinfo=timezone.utc),
        messages=({'role': 'user', 'content': f'hello {index}'},),
        project='alpha',
        model='gpt-5',
        provider='openai',
        status='completed',
        product='chat',
        trace_type='conversation',
        agent_name='support',
        tool_names=('lookup',),
        total_tokens=100 + index,
        duration_ms=10 + index,
        capture_metadata={'source': 'test'},
    )


def request(*, mode: str = 'immediate', population: PopulationRequest | None = None) -> RunRequest:
    return RunRequest(
        query='Find help requests',
        mode=cast(Literal['immediate', 'review'], mode),
        population=population or PopulationRequest(),
        parallelism=2,
    )


def classification(index: int, *, error: str | None = None) -> TraceClassification:
    return TraceClassification(
        trace_id=f'trace-{index}',
        span_id=f'span-{index}',
        value=None if error else True,
        matched=error is None,
        error=error,
        raw_result={'private': 'must not be exported'},
    )


class Clock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def monotonic(self) -> float:
        return self.seconds

    def now(self) -> datetime:
        return datetime(2026, 9, 20, tzinfo=timezone.utc) + timedelta(seconds=self.seconds)


class Runner:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.on_complete: Any = None
        self.calls = 0

    async def __call__(
        self,
        traces: tuple[TraceRecord, ...],
        projections: dict[str, JevProjection],
        compiled: CompiledQuery,
        *,
        parallelism: int,
        on_complete: Any,
    ) -> list[TraceClassification]:
        del projections, compiled, parallelism
        self.calls += 1
        self.on_complete = on_complete
        self.entered.set()
        try:
            await self.release.wait()
            for selected in traces:
                await on_complete(classification(int(selected.trace_id.removeprefix('trace-'))))
            return []
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class Planner:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.plan = CompiledPlan(
            compiled=compiled_query(),
            numeric=NumericFilters(tokens_min=100, duration_ms_min=20),
        )

    async def __call__(self, query: str) -> CompiledPlan:
        self.calls.append(query)
        return self.plan


class Loader:
    def __init__(self) -> None:
        self.calls: list[PopulationRequest] = []
        self.snapshot = Snapshot(traces=tuple(trace(index) for index in range(3)), capture_metadata={'source': 'test'})

    async def __call__(self, request: PopulationRequest) -> Snapshot:
        self.calls.append(request)
        return self.snapshot


def make_store(
    *,
    planner: Planner | None = None,
    loader: Loader | None = None,
    runner: Runner | None = None,
    filters: FacetSelection | None = None,
    clock: Clock | None = None,
) -> tuple[RunStore, Planner, Loader, Runner, Clock]:
    planner = planner or Planner()
    loader = loader or Loader()
    runner = runner or Runner()
    clock = clock or Clock()

    async def select_filters(query: str) -> FacetSelection:
        del query
        return filters or FacetSelection(provider=frozenset({'openai'}))

    store = RunStore(
        compiler=planner,
        population_loader=loader,
        run_jev=runner,
        filter_selector=select_filters,
        monotonic=clock.monotonic,
        now_utc=clock.now,
    )
    return store, planner, loader, runner, clock


@pytest.mark.asyncio
async def test_compile_merges_generated_filters_and_numeric_constraints_with_manual_values_winning() -> None:
    manual = PopulationRequest(
        facets=FacetSelection(provider=frozenset({'anthropic'}), status=frozenset({'failed'})),
        numeric=NumericFilters(tokens_max=900, duration_ms_min=50),
    )
    store, _, loader, runner, _ = make_store()

    started = await store.compile(request(population=manual))

    assert started.state == 'classifying'
    assert started.generated_filters == FacetSelection(provider=frozenset({'openai'}))
    assert started.generated_numeric == NumericFilters(tokens_min=100, duration_ms_min=20)
    started_request = started.request
    assert started_request is not None
    assert started_request.population.facets == FacetSelection(
        provider=frozenset({'anthropic'}), status=frozenset({'failed'})
    )
    assert started_request.population.numeric == NumericFilters(
        tokens_min=100, tokens_max=900, duration_ms_min=50
    )
    assert loader.calls[0] == started_request.population
    await runner.entered.wait()
    runner.release.set()
    assert store._task is not None
    await asyncio.wait_for(store._task, timeout=1)
    assert (await store.snapshot()).state == 'completed'


@pytest.mark.asyncio
async def test_review_start_reuses_loaded_population_without_replanning() -> None:
    store, planner, loader, runner, _ = make_store()

    review = await store.compile(request(mode='review'))
    review_request = review.request
    review_compiled = review.compiled
    assert review_request is not None
    assert review_compiled is not None
    started = await store.start(review_request, review_compiled)

    assert review.state == 'awaiting_review'
    assert started.state == 'classifying'
    assert len(loader.calls) == 1
    assert planner.calls == ['Find help requests']
    await runner.entered.wait()
    runner.release.set()
    assert store._task is not None
    await asyncio.wait_for(store._task, timeout=1)


@pytest.mark.asyncio
async def test_progress_is_counted_once_and_trace_detail_is_detached() -> None:
    store, _, _, runner, clock = make_store()
    started = await store.compile(request())
    await runner.entered.wait()
    await runner.on_complete(classification(0))
    await runner.on_complete(classification(0))
    clock.seconds = 4
    current = await store.snapshot()

    assert current.completed == 1
    assert current.matched == 1
    assert current.active == 2
    assert current.queued == 0
    assert current.elapsed == 4
    assert current.rate == pytest.approx(0.25)
    current.results['trace-0'].raw_result['private'] = 'changed'
    detail = await store.trace_detail('trace-0')
    assert detail is not None
    assert detail.classification is not None
    assert detail.classification.raw_result['private'] == 'must not be exported'
    runner.release.set()
    assert store._task is not None
    await asyncio.wait_for(store._task, timeout=1)
    assert (await store.snapshot()).state == 'completed'
    assert started.generation == (await store.snapshot()).generation


@pytest.mark.asyncio
async def test_runner_failure_and_invalid_callback_fail_the_current_generation() -> None:
    store, _, _, runner, _ = make_store()
    await store.compile(request())
    await runner.entered.wait()
    with pytest.raises(ValueError, match='selected population'):
        await store.complete_one(1, classification(99))

    failed = await store.snapshot()
    assert failed.state == 'failed'
    assert failed.completed == 0
    await store.cancel()


@pytest.mark.asyncio
async def test_cancel_waits_for_runner_and_ignores_late_callbacks() -> None:
    store, _, _, runner, _ = make_store()
    started = await store.compile(request())
    await runner.entered.wait()
    await runner.on_complete(classification(0))

    cancelled = await store.cancel()

    assert cancelled.state == 'cancelled'
    assert cancelled.completed == 1
    await store.complete_one(started.generation, classification(1))
    assert (await store.snapshot()).completed == 1


@pytest.mark.asyncio
async def test_reset_returns_a_new_empty_generation() -> None:
    store, _, _, runner, _ = make_store()
    started = await store.compile(request())
    await runner.entered.wait()

    reset = await store.reset()

    assert runner.cancelled.is_set()
    assert reset.generation == started.generation + 1
    assert reset.state == 'idle'
    assert reset.request is None
    assert reset.trace_ids == ()
    assert not reset.results
