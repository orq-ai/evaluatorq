"""Lifecycle tests for the generation-owned trace finder run store."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal, cast

import pytest

from evaluatorq.common.judge import ClassifyAnswer, ClassifyResponse
from evaluatorq.trace_finder.compiler import CompiledPlan
from evaluatorq.trace_finder.filter_selector import FilterSelectionResult
from evaluatorq.trace_finder.models import (
    CompiledQuery,
    DimensionAnswer,
    FacetSelection,
    TraceProjection,
    NumericFilters,
    PopulationRequest,
    RunRequest,
    Snapshot,
    TraceClassification,
    TraceRecord,
)
from evaluatorq.trace_finder.run_store import RunStore
from evaluatorq.trace_finder.explorer import ExplorerStore
from evaluatorq.trace_finder.rows import TraceRow

if TYPE_CHECKING:
    from collections.abc import Sequence


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
        answers=() if error else (DimensionAnswer(value=True, matched=True),),
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
        projections: dict[str, TraceProjection],
        dimensions: 'Sequence[CompiledQuery]',
        *,
        parallelism: int,
        on_complete: Any,
    ) -> list[TraceClassification]:
        del projections, dimensions, parallelism
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
        projections: dict[str, TraceProjection],
        dimensions: 'Sequence[CompiledQuery]',
        *,
        parallelism: int,
        on_complete: Any,
    ) -> list[TraceClassification]:
        del projections, dimensions, parallelism
        self.calls += 1
        self.on_complete = on_complete
        self.entered.set()
        await on_complete(classification(int(traces[0].trace_id.removeprefix('trace-'))))
        raise RuntimeError('classifier runner exploded')


class Planner:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.plan = CompiledPlan(
            dimensions=(compiled_query(),),
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
        return filters if filters is not None else FacetSelection(provider=frozenset({'openai'}))

    store = RunStore(
        compiler=planner,
        population_loader=loader,
        run_classifier=runner,
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
async def test_filter_response_survives_review_start_and_is_detached() -> None:
    runner = Runner()
    response = ClassifyResponse(
        model='jev-latest', answers={'provider': ClassifyAnswer(type='choice', choice='provider_0')}
    )

    async def select_filters(query: str, population: PopulationRequest) -> FilterSelectionResult:
        del query, population
        return FilterSelectionResult(FacetSelection(provider=frozenset({'openai'})), response)

    store = RunStore(
        compiler=Planner(), population_loader=Loader(), run_classifier=runner, filter_selector=select_filters
    )
    review = await store.compile(request(mode='review'))
    assert review.state == 'awaiting_review'
    assert review.filter_response is not None
    review.filter_response.answers['provider'].choice = 'changed'
    current = await store.snapshot()
    assert current.filter_response is not None
    assert current.filter_response.answers['provider'].choice == 'provider_0'

    assert current.request is not None
    started = await store.start(current.request, (compiled_query(),))
    assert started.filter_response is not None
    assert started.filter_response.answers['provider'].choice == 'provider_0'
    runner.release.set()
    assert store._task is not None
    await asyncio.wait_for(store._task, timeout=1)


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
        run_classifier=runner,
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
        run_classifier=Runner(),
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
        run_classifier=Runner(),
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
async def test_classifier_runner_failure_mid_run_sets_failed_state_with_error_text() -> None:
    runner = RaisingRunner()
    store, _, _, _, _ = make_store(runner=runner)

    started = await store.compile(request())
    assert started.state == 'classifying'
    await runner.entered.wait()
    assert store._task is not None
    await asyncio.wait_for(store._task, timeout=1)

    failed = await store.snapshot()
    assert failed.state == 'failed'
    assert failed.error == 'classifier runner exploded'
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

    second = await store.start(request(), (compiled_query(),))

    assert (await first).state in {'cancelled', 'compiling', 'classifying'}
    assert second.state == 'classifying'
    runner.release.set()
    assert store._task is not None
    await asyncio.wait_for(store._task, timeout=1)


@pytest.mark.asyncio
async def test_cancelling_start_propagates_and_sets_cancelled_state() -> None:
    loader = PendingLoader()
    store, _, _, runner, _ = make_store(loader=loader)
    starting = asyncio.create_task(store.start(request(), (compiled_query(),)))
    await loader.entered.wait()

    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting

    cancelled = await store.snapshot()
    assert cancelled.state == 'cancelled'
    assert runner.calls == 0


@pytest.mark.asyncio
async def test_review_start_returns_working_snapshot_while_changed_population_loads() -> None:
    class ReviewLoader(Loader):
        def __init__(self) -> None:
            super().__init__()
            self.block_next = False
            self.entered = asyncio.Event()

        async def __call__(self, population: PopulationRequest) -> Snapshot:
            if self.block_next:
                self.calls.append(population)
                self.entered.set()
                await asyncio.Event().wait()
                raise AssertionError('blocked population unexpectedly completed')
            return await super().__call__(population)

    loader = ReviewLoader()
    store, _, _, runner, _ = make_store(loader=loader)
    review = await store.compile(request(mode='review'))
    assert review.request is not None and review.dimensions is not None
    loader.block_next = True
    changed_population = review.request.population.model_copy(
        update={'numeric': NumericFilters(tokens_min=200)}
    )
    changed_request = review.request.model_copy(update={'population': changed_population})

    starting = await asyncio.wait_for(store.start(changed_request, review.dimensions, wait=False), timeout=1)
    assert starting.state == 'compiling'
    assert starting.phase == 'starting_classification'
    await asyncio.wait_for(loader.entered.wait(), timeout=1)
    loading = await store.snapshot()
    assert loading.state == 'compiling'
    assert loading.phase == 'loading_traces'

    cancelled = await store.cancel()
    assert cancelled.state == 'cancelled'
    assert cancelled.phase is None
    assert runner.calls == 0


@pytest.mark.asyncio
async def test_review_start_reuses_loaded_population_without_replanning() -> None:
    store, planner, loader, runner, _ = make_store()

    review = await store.compile(request(mode='review'))
    review_request = review.request
    review_dimensions = review.dimensions
    assert review_request is not None
    assert review_dimensions is not None
    started = await store.start(review_request, review_dimensions)

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
    assert review.request is not None and review.dimensions is not None
    revised_population = review.request.population.model_copy(update={
        'numeric': review.request.population.numeric.model_copy(update={'tokens_min': 200}),
    })
    revised_request = review.request.model_copy(update={'population': revised_population})

    started = await store.start(revised_request, review.dimensions)

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
    render = await store.snapshot_for_render()

    assert render.traces[0] is store._snapshot.traces[0]
    assert current.traces[0] is not store._snapshot.traces[0]

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


@pytest.mark.asyncio
async def test_compile_with_traces_runs_filter_selection_locally_but_skips_population_load() -> None:
    """The population load is skipped for already-loaded rows, but filter selection still runs.

    Its output narrows those rows locally instead of shaping an Orq query.
    """

    selector_calls: list[str] = []
    loader_calls: list[object] = []
    staged = (trace(1), trace(2))

    async def select_filters(query: str, population: PopulationRequest) -> FacetSelection:
        del population
        selector_calls.append(query)
        return FacetSelection(provider=frozenset({'openai'}))

    async def population_loader(request: PopulationRequest) -> Snapshot:
        loader_calls.append(request)
        return Snapshot(traces=())

    async def traces() -> tuple[TraceRecord, ...]:
        return staged

    planner = Planner()
    planner.plan = CompiledPlan(dimensions=(compiled_query(),), numeric=NumericFilters())
    runner = Runner()
    store = RunStore(
        compiler=planner,
        population_loader=population_loader,
        run_classifier=runner,
        filter_selector=select_filters,
    )
    snapshot = await store.compile(request(mode='review'), traces=traces)

    assert snapshot.state == 'awaiting_review'
    assert snapshot.trace_ids == ('trace-1', 'trace-2')
    assert selector_calls == ['Find help requests']
    assert loader_calls == []
    assert snapshot.generated_filters == FacetSelection(provider=frozenset({'openai'}))
    assert snapshot.request is not None
    assert snapshot.request.population == PopulationRequest()


@pytest.mark.asyncio
async def test_compile_with_traces_applies_question_numeric_bounds_to_loaded_rows() -> None:
    store, _, _, _, _ = make_store()

    snapshot = await store.compile(request(mode='review'), traces=lambda: _loaded_traces())

    assert snapshot.generated_numeric == NumericFilters(tokens_min=100, duration_ms_min=20)
    assert snapshot.trace_ids == ('trace-2',)


@pytest.mark.asyncio
async def test_compile_with_traces_completes_with_warning_when_no_loaded_row_meets_the_bounds() -> None:
    store, _, _, _, _ = make_store()

    snapshot = await store.compile(request(), traces=lambda: _small_traces())

    assert snapshot.state == 'completed'
    assert snapshot.error is None
    assert snapshot.plan_warning == 'None of the 1 loaded traces have at least 100 tokens (the largest has 50). Try New search to look beyond the loaded rows.'


def _unbounded_planner() -> Planner:
    """A planner whose plan carries no numeric bounds, isolating facet narrowing from bounds narrowing."""

    planner = Planner()
    planner.plan = CompiledPlan(dimensions=(compiled_query(),), numeric=NumericFilters())
    return planner


@pytest.mark.asyncio
async def test_compile_with_traces_narrows_loaded_rows_to_the_generated_facet_selection() -> None:
    matching = trace(1).model_copy(update={'model': 'gpt-4'})
    other = trace(2).model_copy(update={'model': 'gpt-3.5'})
    store, _, _, _, _ = make_store(planner=_unbounded_planner(), filters=FacetSelection(model=frozenset({'gpt-4'})))

    async def loader() -> tuple[TraceRecord, ...]:
        return (matching, other)

    snapshot = await store.compile(request(mode='review'), traces=loader)

    assert snapshot.state == 'awaiting_review'
    assert snapshot.generated_filters == FacetSelection(model=frozenset({'gpt-4'}))
    assert snapshot.trace_ids == ('trace-1',)
    assert tuple(item.trace_id for item in snapshot.traces) == ('trace-1',)


@pytest.mark.asyncio
async def test_unfiltered_within_run_uses_limit_loader() -> None:
    planner = Planner()
    planner.plan = CompiledPlan(dimensions=(compiled_query(),), numeric=NumericFilters())
    store, _, _, _, _ = make_store(planner=planner, filters=FacetSelection())
    asked: list[int] = []

    async def full_loader() -> tuple[TraceRecord, ...]:
        raise AssertionError('unfiltered limit runs must not hydrate the whole result set')

    async def limited_loader(limit: int) -> tuple[TraceRecord, ...]:
        asked.append(limit)
        return (trace(27),)

    snapshot = await store.compile(
        request(population=PopulationRequest(limit=1)),
        traces=full_loader,
        traces_limited=limited_loader,
    )

    assert asked == [1]
    assert snapshot.trace_ids == ('trace-27',)


@pytest.mark.asyncio
async def test_compile_with_traces_completes_with_warning_when_no_loaded_row_matches_the_generated_facet_selection() -> None:
    store, _, _, _, _ = make_store(planner=_unbounded_planner(), filters=FacetSelection(model=frozenset({'gpt-4'})))

    async def loader() -> tuple[TraceRecord, ...]:
        return (trace(1).model_copy(update={'model': 'gpt-3.5'}),)

    snapshot = await store.compile(request(), traces=loader)

    assert snapshot.state == 'completed'
    assert snapshot.error is None
    assert snapshot.plan_warning == 'None of the 1 loaded traces have model gpt-4. Try New search to look beyond the loaded rows.'


@pytest.mark.asyncio
async def test_within_results_compile_narrows_explorer_for_kept_and_empty_results() -> None:
    class ExplorerSource:
        def __init__(self) -> None:
            self.rows = (TraceRow(trace_id='trace-1'), TraceRow(trace_id='trace-2'))

        async def search(self, *args: Any, **kwargs: Any) -> tuple[TraceRow, ...]:
            del args
            kwargs['on_page'](self.rows)
            return self.rows

        async def hydrate_rows(self, rows: Any) -> dict[str, TraceRecord | None]:
            del rows
            return {}

    source = ExplorerSource()
    explorer = ExplorerStore(search=source.search, hydrate=source.hydrate_rows)
    await explorer.load(
        datetime(2026, 9, 20, tzinfo=timezone.utc),
        datetime(2026, 9, 21, tzinfo=timezone.utc),
        2,
        facets=FacetSelection(),
        numeric=NumericFilters(),
        wait=True,
    )
    planner = Planner()
    planner.plan = CompiledPlan(dimensions=(compiled_query(),), numeric=NumericFilters(tokens_min=102))
    store, _, _, _, _ = make_store(planner=planner)
    store.explorer = explorer

    async def traces() -> tuple[TraceRecord, ...]:
        return (trace(1), trace(2))

    kept = await store.compile(request(mode='review'), traces=traces)
    view = await explorer.view()
    assert kept.trace_ids == ('trace-2',)
    assert [row.trace_id for row in view.rows] == ['trace-2']
    assert view.numeric.tokens_min == 102

    planner.plan = CompiledPlan(dimensions=(compiled_query(),), numeric=NumericFilters(tokens_min=1_000))
    empty = await store.compile(request(mode='review'), traces=traces)
    view = await explorer.view()
    assert empty.state == 'completed'
    assert empty.plan_warning is not None
    assert empty.trace_ids == ()
    assert view.rows == ()
    assert view.numeric.tokens_min == 1_000


@pytest.mark.asyncio
async def test_within_results_filters_past_ai_limit_and_keeps_all_table_matches() -> None:
    rows = tuple(TraceRow(trace_id=f'trace-{index}') for index in range(1, 31))

    async def search(*args: Any, **kwargs: Any) -> tuple[TraceRow, ...]:
        del args
        kwargs['on_page'](rows)
        return rows

    async def hydrate(rows: Any) -> dict[str, TraceRecord | None]:
        return {
            row.trace_id: (
                trace(int(row.trace_id.removeprefix('trace-'))).model_copy(update={'model': 'gpt-4'})
                if row.trace_id in {'trace-29', 'trace-30'}
                else trace(int(row.trace_id.removeprefix('trace-')))
            )
            for row in rows
        }

    explorer = ExplorerStore(search=search, hydrate=hydrate)
    await explorer.load(
        datetime(2026, 9, 20, tzinfo=timezone.utc),
        datetime(2026, 9, 21, tzinfo=timezone.utc),
        len(rows),
        facets=FacetSelection(),
        numeric=NumericFilters(),
        wait=True,
    )
    loaded = await explorer.records([row.trace_id for row in rows])
    planner = _unbounded_planner()
    store, _, _, _, _ = make_store(planner=planner, filters=FacetSelection(model=frozenset({'gpt-4'})))
    store.explorer = explorer

    snapshot = await store.compile(
        request(
            mode='review',
            population=PopulationRequest(limit=1),
        ),
        traces=lambda: _records_from_loaded(loaded),
    )

    view = await explorer.view()
    assert snapshot.trace_ids == ('trace-29',)
    assert [row.trace_id for row in view.rows] == ['trace-29', 'trace-30']
    assert set(explorer.cached_records([row.trace_id for row in rows])) == {'trace-29'}


async def _records_from_loaded(records: dict[str, TraceRecord | None]) -> tuple[TraceRecord, ...]:
    return tuple(record for record in records.values() if record is not None)


async def _loaded_traces() -> tuple[TraceRecord, ...]:
    return (
        trace(1).model_copy(update={'total_tokens': 50, 'duration_ms': 500}),
        trace(2).model_copy(update={'total_tokens': 50_001, 'duration_ms': 500}),
        trace(3),
    )


async def _small_traces() -> tuple[TraceRecord, ...]:
    return (trace(1).model_copy(update={'total_tokens': 50, 'duration_ms': 500}),)


@pytest.mark.asyncio
async def test_empty_loaded_traces_explain_that_no_rows_are_usable() -> None:
    store, _, _, _, _ = make_store()

    snapshot = await store.compile(request(), traces=lambda: _empty_traces())

    assert snapshot.state == 'failed'
    assert snapshot.error == 'None of the loaded traces have usable messages to classify.'


async def _empty_traces() -> tuple[TraceRecord, ...]:
    return ()


@pytest.mark.asyncio
async def test_reset_cancels_owned_traces_hydration() -> None:
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def traces() -> tuple[TraceRecord, ...]:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return ()

    store, _, _, _, _ = make_store()
    compiling = asyncio.create_task(store.compile(request(), traces=traces))
    await entered.wait()

    reset = await store.reset()

    assert cancelled.is_set()
    assert reset.state == 'idle'
    assert (await store.snapshot()).state == 'idle'
    await compiling


@pytest.mark.asyncio
async def test_traces_error_after_cancel_preserves_cancelled_state(caplog: pytest.LogCaptureFixture) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def traces() -> tuple[TraceRecord, ...]:
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await release.wait()
            raise RuntimeError('hydration failed after cancellation')
        return ()

    store, _, _, _, _ = make_store()
    compiling = asyncio.create_task(store.compile(request(), traces=traces))
    await entered.wait()
    cancelling = asyncio.create_task(store.cancel())
    await asyncio.sleep(0)
    release.set()

    cancelled = await cancelling
    await compiling

    assert cancelled.state == 'cancelled'
    assert (await store.snapshot()).state == 'cancelled'
    assert 'hydration failed after cancellation' not in caplog.text


@pytest.mark.asyncio
async def test_load_resets_running_classification() -> None:
    store, _, _, _, _ = make_store()
    await store.compile(request(), wait=True)
    assert (await store.snapshot()).state == 'classifying'

    snapshot = await store.reset()

    assert snapshot.state == 'idle'
    assert snapshot.results == {}


@pytest.mark.asyncio
async def test_numeric_only_plan_warns_when_the_question_says_more_than_numbers() -> None:
    planner = Planner()
    planner.plan = CompiledPlan(dimensions=(), numeric=NumericFilters(tokens_min=50_001))
    store, _, _, _, _ = make_store(planner=planner, filters=FacetSelection())
    coding = request(mode='review').model_copy(update={'query': 'all coding agents with more than 50k tokens'})
    plain = request(mode='review').model_copy(update={'query': 'traces with more than 50k tokens'})

    warned = await store.compile(coding, traces=lambda: _loaded_traces())
    assert warned.plan_warning is not None
    assert '"coding agents"' in warned.plan_warning

    quiet = await store.compile(plain, traces=lambda: _loaded_traces())
    assert quiet.plan_warning is None


@pytest.mark.asyncio
async def test_aggregate_plan_with_a_filter_keeps_running_and_warns() -> None:
    planner = Planner()
    planner.plan = CompiledPlan(dimensions=(), numeric=NumericFilters(), unsupported_reason='Cost per model is a total.')
    store, _, _, _, _ = make_store(planner=planner)
    aggregate = request(mode='review').model_copy(update={'query': 'which openai model is costing us the most?'})

    snapshot = await store.compile(aggregate, traces=lambda: _loaded_traces())

    assert snapshot.state != 'cancelled'
    assert snapshot.plan_warning == (
        "Ask AI finds traces; it can't compute totals, averages or rankings. Cost per model is a total."
    )


@pytest.mark.asyncio
async def test_aggregate_plan_with_nothing_to_search_stops_before_loading_the_population() -> None:
    # "which model is costing us the most?" once completed as a green "Done" with every trace a match.
    planner = Planner()
    planner.plan = CompiledPlan(dimensions=(), numeric=NumericFilters(), unsupported_reason='Cost per model is a total.')
    store, _, loader, _, _ = make_store(planner=planner, filters=FacetSelection())
    loaded: list[str] = []

    async def load_rows() -> tuple[TraceRecord, ...]:
        loaded.append('called')
        return await _loaded_traces()

    aggregate = request(mode='review').model_copy(update={'query': 'which model is costing us the most?'})

    snapshot = await store.compile(aggregate, traces=load_rows)

    assert snapshot.state == 'cancelled'
    assert snapshot.error is None
    assert snapshot.plan_warning == (
        "Ask AI finds traces; it can't compute totals, averages or rankings. Cost per model is a total."
    )
    assert loaded == []
    assert loader.calls == []
    assert snapshot.trace_ids == ()


@pytest.mark.asyncio
async def test_compile_with_traces_records_how_many_rows_were_loaded() -> None:
    store, _, _, _, _ = make_store(planner=_unbounded_planner())

    snapshot = await store.compile(request(), traces=lambda: _small_traces())

    assert snapshot.state == 'classifying'
    assert (snapshot.loaded, snapshot.to_load) == (1, 1)


@pytest.mark.asyncio
async def test_new_search_through_the_table_queries_it_with_the_merged_filters() -> None:
    store, _, loader, _, _ = make_store(filters=FacetSelection(model=frozenset({'jev-latest'})))
    asked: list[PopulationRequest] = []

    async def table(population: PopulationRequest) -> tuple[TraceRecord, ...]:
        asked.append(population)
        return (trace(1).model_copy(update={'total_tokens': 500, 'duration_ms': 500}),)

    snapshot = await store.compile(request(mode='review'), table=table)

    assert asked[0].facets.model == frozenset({'jev-latest'})
    assert asked[0].numeric == NumericFilters(tokens_min=100, duration_ms_min=20)
    assert snapshot.within_results
    assert snapshot.trace_ids == ('trace-1',)
    assert loader.calls == []
