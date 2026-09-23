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


class RaisingRunner(Runner):
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
        await on_complete(classification(int(traces[0].trace_id.removeprefix('trace-'))))
        raise RuntimeError('JEV runner exploded')


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


class RaisingPlanner(Planner):
    async def __call__(self, query: str) -> CompiledPlan:
        del query
        raise RuntimeError('compiler exploded')


class SwallowingCancellationPlanner(Planner):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()
        self.swallowed = asyncio.Event()

    async def __call__(self, query: str) -> CompiledPlan:
        del query
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.swallowed.set()
            return self.plan
        raise AssertionError('compiler cancellation was not swallowed')


class Loader:
    def __init__(self) -> None:
        self.calls: list[PopulationRequest] = []
        self.snapshot = Snapshot(traces=tuple(trace(index) for index in range(3)), capture_metadata={'source': 'test'})

    async def __call__(self, request: PopulationRequest) -> Snapshot:
        self.calls.append(request)
        return self.snapshot


class RaisingLoader(Loader):
    async def __call__(self, request: PopulationRequest) -> Snapshot:
        del request
        raise RuntimeError('population loader exploded')


class PendingLoader(Loader):
    def __init__(self) -> None:
        super().__init__()
        self.entered = asyncio.Event()

    async def __call__(self, request: PopulationRequest) -> Snapshot:
        self.calls.append(request)
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError('pending population loader unexpectedly completed')


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

    async def select_filters(query: str, population: PopulationRequest) -> FacetSelection:
        del query, population
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
    assert started.explicit_filters == manual.facets
    assert started.explicit_numeric == manual.numeric
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
async def test_filter_selector_receives_the_run_population_bounds() -> None:
    planner = Planner()
    loader = Loader()
    runner = Runner()
    clock = Clock()
    seen: list[PopulationRequest] = []

    async def select_filters(query: str, population: PopulationRequest) -> FacetSelection:
        del query
        seen.append(population)
        return FacetSelection()

    store = RunStore(
        compiler=planner,
        population_loader=loader,
        run_jev=runner,
        filter_selector=select_filters,
        monotonic=clock.monotonic,
        now_utc=clock.now,
    )
    population = PopulationRequest(
        start=datetime(2026, 9, 1, tzinfo=timezone.utc),
        end=datetime(2026, 9, 20, tzinfo=timezone.utc),
    )

    started = await store.compile(request(population=population))
    assert started.state == 'classifying'
    assert seen == [population]
    runner.release.set()
    assert store._task is not None
    await asyncio.wait_for(store._task, timeout=1)


@pytest.mark.asyncio
async def test_compiler_failure_sets_failed_state_with_error_text(caplog: pytest.LogCaptureFixture) -> None:
    store, _, loader, runner, _ = make_store(planner=RaisingPlanner())

    failed = await store.compile(request())

    assert failed.state == 'failed'
    assert failed.error == 'compiler exploded'
    assert 'compiler exploded' in caplog.text
    assert not loader.calls
    assert runner.calls == 0


@pytest.mark.asyncio
async def test_close_cancels_and_releases_the_builder_resources() -> None:
    released: list[bool] = []
    store = RunStore(
        compiler=Planner(),
        population_loader=Loader(),
        run_jev=Runner(),
        filter_selector=make_store()[0]._filter_selector,
        close=lambda: released.append(True),
    )

    await store.close()
    await store.close()

    assert released == [True]
    with pytest.raises(ValueError, match='closed'):
        await store.compile(request())


@pytest.mark.asyncio
async def test_close_awaits_async_resource_cleanup() -> None:
    released: list[str] = []

    async def cleanup() -> None:
        await asyncio.sleep(0)
        released.append('closed')

    store = RunStore(
        compiler=Planner(),
        population_loader=Loader(),
        run_jev=Runner(),
        filter_selector=make_store()[0]._filter_selector,
        close=cleanup,
    )

    await store.close()
    assert released == ['closed']


@pytest.mark.asyncio
async def test_population_loader_failure_sets_failed_state_with_error_text() -> None:
    store, _, _, runner, _ = make_store(loader=RaisingLoader())

    failed = await store.compile(request())

    assert failed.state == 'failed'
    assert failed.error == 'population loader exploded'
    assert runner.calls == 0


@pytest.mark.asyncio
async def test_jev_runner_failure_mid_run_sets_failed_state_with_error_text() -> None:
    runner = RaisingRunner()
    store, _, _, _, _ = make_store(runner=runner)

    started = await store.compile(request())
    assert started.state == 'classifying'
    await runner.entered.wait()
    assert store._task is not None
    await asyncio.wait_for(store._task, timeout=1)

    failed = await store.snapshot()
    assert failed.state == 'failed'
    assert failed.error == 'JEV runner exploded'
    assert failed.completed == 1


@pytest.mark.asyncio
async def test_compile_without_waiting_returns_the_compiling_snapshot_and_finishes_in_the_background() -> None:
    planner = SwallowingCancellationPlanner()
    store, _, loader, runner, _ = make_store(planner=planner)

    compiling = await store.compile(request(), wait=False)

    assert compiling.state == 'compiling'
    await planner.entered.wait()
    assert (await store.snapshot()).state == 'compiling'
    assert not loader.calls

    cancelled = await store.cancel()

    assert cancelled.state == 'cancelled'
    assert runner.calls == 0


@pytest.mark.asyncio
async def test_compile_without_waiting_reaches_classifying_once_planning_completes() -> None:
    store, _, loader, runner, _ = make_store()

    compiling = await store.compile(request(), wait=False)

    assert compiling.state == 'compiling'
    assert store._task is not None
    await asyncio.wait_for(store._task, timeout=1)
    assert (await store.snapshot()).state == 'classifying'
    assert len(loader.calls) == 1
    await runner.entered.wait()
    runner.release.set()
    assert store._task is not None
    await asyncio.wait_for(store._task, timeout=1)
    assert (await store.snapshot()).state == 'completed'


@pytest.mark.asyncio
async def test_cancel_finishes_when_compiler_swallows_cancellation() -> None:
    planner = SwallowingCancellationPlanner()
    store, _, loader, runner, _ = make_store(planner=planner)
    compiling = asyncio.create_task(store.compile(request()))
    await planner.entered.wait()

    cancelled = await store.cancel()

    assert cancelled.state == 'cancelled'
    assert planner.swallowed.is_set()
    assert not loader.calls
    assert runner.calls == 0
    assert (await compiling).state == 'cancelled'


@pytest.mark.asyncio
async def test_replacement_returns_a_snapshot_to_first_compile_caller() -> None:
    planner = SwallowingCancellationPlanner()
    store, _, _, runner, _ = make_store(planner=planner)
    first = asyncio.create_task(store.compile(request()))
    await planner.entered.wait()

    second = await store.start(request(), compiled_query())

    assert (await first).state in {'cancelled', 'compiling', 'classifying'}
    assert second.state == 'classifying'
    runner.release.set()
    assert store._task is not None
    await asyncio.wait_for(store._task, timeout=1)


@pytest.mark.asyncio
async def test_cancelling_start_propagates_and_sets_cancelled_state() -> None:
    loader = PendingLoader()
    store, _, _, runner, _ = make_store(loader=loader)
    starting = asyncio.create_task(store.start(request(), compiled_query()))
    await loader.entered.wait()

    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting

    cancelled = await store.snapshot()
    assert cancelled.state == 'cancelled'
    assert runner.calls == 0


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
    assert started.explicit_filters == FacetSelection()
    assert started.explicit_numeric == NumericFilters()
    assert len(loader.calls) == 1
    assert planner.calls == ['Find help requests']
    await runner.entered.wait()
    runner.release.set()
    assert store._task is not None
    await asyncio.wait_for(store._task, timeout=1)


@pytest.mark.asyncio
async def test_review_start_tracks_only_changed_filters_as_explicit() -> None:
    store, _, _, runner, _ = make_store()
    original = request(mode='review', population=PopulationRequest(facets=FacetSelection(status=frozenset({'failed'}))))
    review = await store.compile(original)
    assert review.request is not None and review.compiled is not None
    revised_population = review.request.population.model_copy(update={
        'numeric': review.request.population.numeric.model_copy(update={'tokens_min': 200}),
    })
    revised_request = review.request.model_copy(update={'population': revised_population})

    started = await store.start(revised_request, review.compiled)

    assert started.explicit_filters == original.population.facets
    assert started.explicit_numeric == NumericFilters(tokens_min=200)
    assert started.generated_filters == FacetSelection(provider=frozenset({'openai'}))
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
