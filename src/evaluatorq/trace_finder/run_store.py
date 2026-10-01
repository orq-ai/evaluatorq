"""Generation-checked ownership of trace-finder compilation and classification."""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol, TypeVar

from loguru import logger

from .compiler import CompiledPlan
from .filter_selector import FilterSelectionResult
from .models import (
    FACET_NAMES,
    CompiledQuery,
    FacetSelection,
    NumericFilters,
    PopulationRequest,
    RunRequest,
    RunSnapshot,
    RunState,
    Snapshot,
    TraceClassification,
    TraceDetail,
    TraceProjection,
    TraceRecord,
)
from .progress import set_load_reporter
from .projection import project_trace as default_project_trace

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from .explorer import ExplorerStore

TReviewFilters = TypeVar('TReviewFilters', FacetSelection, NumericFilters)
T = TypeVar('T')


class PopulationLoader(Protocol):
    """Load the bounded trace population after planning has completed."""

    def __call__(self, request: PopulationRequest) -> Awaitable[Snapshot]: ...


class TableLoader(Protocol):
    """Load the records that the dashboard explorer should display."""

    def __call__(self, request: PopulationRequest) -> Awaitable[tuple[TraceRecord, ...]]: ...


class ClassifierRunner(Protocol):
    """A run_classifier callable with its model and client already bound."""

    def __call__(
        self,
        traces: tuple[TraceRecord, ...],
        projections: dict[str, TraceProjection],
        dimensions: Sequence[CompiledQuery],
        *,
        parallelism: int,
        on_complete: Callable[[TraceClassification], Awaitable[None]],
    ) -> Awaitable[list[TraceClassification]]: ...


class RunStore:
    """Own one replaceable run while keeping polling and callbacks independent.

    Every asynchronous phase belongs to a generation. A replacement increments the
    generation, and stale work can only return without changing the current view.
    This deliberately uses that generation check instead of Python 3.11-only
    cancellation-counter APIs, so the store remains Python 3.10 compatible. A
    cancellation received by the awaiting caller is cleaned up and
    then re-raised to preserve the caller's cancellation contract.
    """

    def __init__(
        self,
        *,
        compiler: Callable[[str], Awaitable[CompiledPlan]],
        population_loader: PopulationLoader,
        run_classifier: ClassifierRunner,
        filter_selector: Callable[[str, PopulationRequest], Awaitable[FacetSelection | FilterSelectionResult]],
        project_trace: Callable[[TraceRecord], TraceProjection] = default_project_trace,
        monotonic: Callable[[], float] = time.monotonic,
        now_utc: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        close: Callable[[], Awaitable[None] | None] | None = None,
        explorer: ExplorerStore | None = None,
    ) -> None:
        self._compiler = compiler
        self._close = close
        self.explorer = explorer
        self._closed = False
        self._population_loader = population_loader
        self._run_classifier = run_classifier
        self._filter_selector = filter_selector
        self._project_trace = project_trace
        self._monotonic = monotonic
        self._now_utc = now_utc
        self._lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._snapshot = RunSnapshot()
        self._task: asyncio.Task[object] | None = None
        self._stopped_generation: int | None = None
        self._generation = 0
        self._started_monotonic: float | None = None

    async def compile(
        self,
        request: RunRequest,
        *,
        wait: bool = True,
        traces: Callable[[], Awaitable[tuple[TraceRecord, ...]]] | None = None,
        traces_limited: Callable[[int], Awaitable[tuple[TraceRecord, ...]]] | None = None,
        table: Callable[[PopulationRequest], Awaitable[tuple[TraceRecord, ...]]] | None = None,
        source_generation: Callable[[], int | None] | None = None,
    ) -> RunSnapshot:
        """Compile, select the population, and stage or start the requested run.

        With ``wait=False`` the planning runs as an owned background task and the
        ``compiling`` snapshot is returned at once, so a caller that polls can show
        progress while the compiler and facet selector are still working. With
        ``traces`` the run classifies those already-loaded traces: the population load is
        skipped, and the generated facets and numeric bounds narrow the loaded rows locally.
        """

        return await self._prepare(
            request,
            None,
            compile_query=True,
            wait=wait,
            traces=traces,
            traces_limited=traces_limited,
            table=table,
            source_generation=source_generation,
        )

    async def start(
        self, request: RunRequest, dimensions: Sequence[CompiledQuery], *, wait: bool = True
    ) -> RunSnapshot:
        """Start reviewed dimensions; return the working snapshot immediately when ``wait=False``."""

        return await self._prepare(request, tuple(dimensions), compile_query=False, wait=wait)

    async def _prepare(
        self,
        request: RunRequest,
        dimensions: tuple[CompiledQuery, ...] | None,
        *,
        compile_query: bool,
        wait: bool = True,
        traces: Callable[[], Awaitable[tuple[TraceRecord, ...]]] | None = None,
        traces_limited: Callable[[int], Awaitable[tuple[TraceRecord, ...]]] | None = None,
        table: Callable[[PopulationRequest], Awaitable[tuple[TraceRecord, ...]]] | None = None,
        source_generation: Callable[[], int | None] | None = None,
    ) -> RunSnapshot:
        generation, staged_traces = await self._begin(
            request,
            compile_query=compile_query,
            within_results=traces is not None or traces_limited is not None or table is not None,
        )
        work = self._plan_and_start(
            generation,
            staged_traces,
            request,
            dimensions,
            compile_query=compile_query,
            traces=traces,
            traces_limited=traces_limited,
            table=table,
            source_generation=source_generation,
        )
        if wait:
            return await work
        task = asyncio.create_task(work)
        async with self._lock:
            if generation == self._generation and self._snapshot.state == 'compiling':
                self._task = task
                return self._view()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return await self.snapshot()

    async def _begin(
        self, request: RunRequest, *, compile_query: bool, within_results: bool
    ) -> tuple[int, tuple[TraceRecord, ...]]:
        """Replace the current generation with a fresh ``compiling`` one."""

        staged_traces: tuple[TraceRecord, ...] = ()
        async with self._lifecycle_lock:
            if self._closed:
                raise ValueError('Trace finder store is closed.')
            await self._stop_current()
            async with self._lock:
                if (
                    not compile_query
                    and self._snapshot.state == 'awaiting_review'
                    and self._snapshot.request is not None
                    and self._snapshot.request.population == request.population
                ):
                    staged_traces = self._snapshot.traces
                explicit_filters = request.population.facets
                explicit_numeric = request.population.numeric
                if (
                    not compile_query
                    and self._snapshot.state == 'awaiting_review'
                    and self._snapshot.request is not None
                ):
                    previous = self._snapshot.request.population
                    explicit_filters = _review_explicit(
                        previous.facets, self._snapshot.explicit_filters, explicit_filters, (*FACET_NAMES, 'project_id')
                    )
                    explicit_numeric = _review_explicit(
                        previous.numeric,
                        self._snapshot.explicit_numeric,
                        explicit_numeric,
                        tuple(NumericFilters.model_fields),
                    )
                self._generation += 1
                generation = self._generation
                generated_filters = FacetSelection() if compile_query else self._snapshot.generated_filters
                within_results = within_results if compile_query else self._snapshot.within_results
                generated_numeric = NumericFilters() if compile_query else self._snapshot.generated_numeric
                filter_response = None if compile_query else self._snapshot.filter_response
                filter_selection_error = None if compile_query else self._snapshot.filter_selection_error
                plan_warning = None if compile_query else self._snapshot.plan_warning
                self._snapshot = RunSnapshot(
                    generation=generation,
                    state='compiling',
                    phase='planning' if compile_query else 'starting_classification',
                    request=request.model_copy(deep=True),
                    within_results=within_results,
                    explicit_filters=explicit_filters.model_copy(deep=True),
                    explicit_numeric=explicit_numeric.model_copy(deep=True),
                    generated_filters=generated_filters.model_copy(deep=True),
                    generated_numeric=generated_numeric.model_copy(deep=True),
                    filter_response=filter_response.model_copy(deep=True) if filter_response is not None else None,
                    filter_selection_error=filter_selection_error,
                    plan_warning=plan_warning,
                    created_at=self._now_utc(),
                )
                self._started_monotonic = None
                self._task = None
                return generation, staged_traces

    async def _plan_and_start(  # noqa: C901
        self,
        generation: int,
        staged_traces: tuple[TraceRecord, ...],
        request: RunRequest,
        dimensions: tuple[CompiledQuery, ...] | None,
        *,
        compile_query: bool,
        traces: Callable[[], Awaitable[tuple[TraceRecord, ...]]] | None = None,
        traces_limited: Callable[[int], Awaitable[tuple[TraceRecord, ...]]] | None = None,
        table: Callable[[PopulationRequest], Awaitable[tuple[TraceRecord, ...]]] | None = None,
        source_generation: Callable[[], int | None] | None = None,
    ) -> RunSnapshot:
        # The table narrows with the run only while it still shows the rows this run was asked about.
        explorer_generation = (
            (await self.explorer.view()).generation
            if (traces is not None or traces_limited is not None) and self.explorer is not None
            else None
        )
        try:
            unsupported_reason: str | None = None
            if compile_query:
                plan_task = asyncio.create_task(self._plan(request.query, request.population))
                async with self._lock:
                    if generation != self._generation or self._snapshot.state != 'compiling':
                        plan_task.cancel()
                        await asyncio.gather(plan_task, return_exceptions=True)
                        return self._view()
                    self._task = plan_task
                plan, filter_result = await plan_task
                unsupported_reason = plan.unsupported_reason
                generated_filters = filter_result.selection
                generated_numeric = plan.numeric.model_copy(deep=True)
                dimensions = _revalidated(plan.dimensions)
                plan_warning = _unsupported_warning(plan.unsupported_reason) or _uncovered_words_warning(
                    request.query, dimensions, generated_filters
                )
                # Already-loaded rows are narrowed locally instead; a fresh population, from Orq or the
                # table, is queried with the question's filters.
                if traces is None:
                    population = request.population.model_copy(
                        update={
                            'facets': _merge_facets(request.population.facets, generated_filters),
                            'numeric': _merge_numeric(request.population.numeric, plan.numeric),
                        }
                    )
                    request = request.model_copy(update={'population': population})
            else:
                if dimensions is None:
                    raise ValueError('reviewed classifier dimensions are required to start a run')
                dimensions = _revalidated(dimensions)
                generated_filters = self._snapshot.generated_filters
                generated_numeric = self._snapshot.generated_numeric
                filter_result = FilterSelectionResult(
                    generated_filters, self._snapshot.filter_response, self._snapshot.filter_selection_error
                )
                plan_warning = self._snapshot.plan_warning

            if (
                compile_query
                and unsupported_reason is not None
                and not dimensions
                and generated_filters == FacetSelection()
                and generated_numeric == NumericFilters()
            ):
                # Nothing to search for: loading the population would only show every trace as a match.
                async with self._lock:
                    if generation != self._generation or self._snapshot.state != 'compiling':
                        return self._view()
                    self._snapshot = replace(
                        self._snapshot,
                        dimensions=dimensions,
                        filter_response=filter_result.response.model_copy(deep=True)
                        if filter_result.response
                        else None,
                        filter_selection_error=filter_result.error,
                        plan_warning=plan_warning,
                    )
                    # 'cancelled' is the existing terminal state that is neither success nor a system error.
                    self._finish('cancelled')
                    self._task = None
                    return self._view()

            async with self._lock:
                if generation != self._generation or self._snapshot.state != 'compiling':
                    return self._view()
                self._snapshot = replace(
                    self._snapshot,
                    phase='loading_traces' if not staged_traces else 'starting_classification',
                    dimensions=dimensions,
                    generated_filters=generated_filters.model_copy(deep=True),
                    generated_numeric=generated_numeric.model_copy(deep=True),
                    filter_response=filter_result.response.model_copy(deep=True) if filter_result.response else None,
                    filter_selection_error=filter_result.error,
                    plan_warning=plan_warning,
                    request=request.model_copy(deep=True),
                )

            if not staged_traces:
                if traces is not None or traces_limited is not None or table is not None:
                    population = request.population
                    use_limited_loader = (
                        traces_limited is not None
                        and generated_filters == FacetSelection()
                        and generated_numeric == NumericFilters()
                    )
                    loader = (
                        (lambda: traces_limited(population.limit))
                        if use_limited_loader and traces_limited is not None
                        else traces or (lambda: table(population))  # pyright: ignore[reportOptionalCall]
                    )
                    loaded_traces = await self._load_owned(generation, loader)
                    exact_source_generation = explorer_generation
                    # A fresh table load records its exact ExplorerView generation in
                    # the route callback; a within-results load uses the generation
                    # captured before its trace IDs were selected.
                    if source_generation is not None:
                        exact_source_generation = source_generation()
                    async with self._lock:
                        if generation == self._generation and self._snapshot.state == 'compiling':
                            self._snapshot = replace(self._snapshot, explorer_generation=exact_source_generation)
                    self._report_loaded(generation, len(loaded_traces or ()), len(loaded_traces or ()))
                    # Loaded rows skip the Orq query, so apply the question's generated filters here
                    # and to the table only for within-results runs.
                    staged_traces = (
                        _narrowed(loaded_traces or (), generated_filters, generated_numeric)
                        if traces is not None and not use_limited_loader
                        else tuple(loaded_traces or ())
                    )
                    if (
                        traces is not None
                        and not use_limited_loader
                        and explorer_generation is not None
                        and self.explorer is not None
                        and staged_traces != loaded_traces
                    ):
                        await self.explorer.narrow(
                            explorer_generation,
                            {trace.trace_id for trace in staged_traces},
                            facets=_merge_facets(request.population.facets, generated_filters),
                            numeric=_merge_numeric(request.population.numeric, generated_numeric),
                        )
                    staged_traces = staged_traces[: request.population.limit]
                    if (
                        (traces is not None or traces_limited is not None)
                        and self.explorer is not None
                        and explorer_generation is not None
                    ):
                        await self.explorer.retain_records(
                            explorer_generation, [trace.trace_id for trace in staged_traces]
                        )
                    if loaded_traces and not staged_traces:
                        async with self._lock:
                            if generation == self._generation and self._snapshot.state == 'compiling':
                                self._snapshot = replace(
                                    self._snapshot,
                                    plan_warning=_nothing_kept_message(
                                        loaded_traces, generated_filters, generated_numeric
                                    ),
                                )
                                self._finish('completed')
                                self._task = None
                            return self._view()
                else:
                    loaded = await self._load_population(generation, request.population)
                    staged_traces = tuple(loaded.traces[: request.population.limit])

            async with self._lock:
                if generation != self._generation or self._snapshot.state != 'compiling':
                    return self._view()
                if not staged_traces:
                    raise _empty_traces_error(loaded_rows=traces is not None or traces_limited is not None)
                trace_ids = tuple(trace.trace_id for trace in staged_traces)
                if len(set(trace_ids)) != len(trace_ids):
                    raise ValueError('The selected population contains duplicate trace IDs.')
                self._snapshot = replace(
                    self._snapshot,
                    trace_ids=trace_ids,
                    traces=tuple(trace.model_copy(deep=True) for trace in staged_traces),
                    total=len(staged_traces),
                )
                if compile_query and request.mode == 'review':
                    self._snapshot = replace(self._snapshot, state='awaiting_review', phase=None)
                    self._task = None
                    return self._view()
                projections = {trace.trace_id: self._project_trace(trace) for trace in staged_traces}
                self._snapshot = replace(
                    self._snapshot,
                    projections=projections,
                    state='classifying',
                    phase=None,
                    started_at=self._now_utc(),
                )
                self._started_monotonic = self._monotonic()
                task = asyncio.create_task(
                    self._execute(generation, staged_traces, projections, dimensions, request.parallelism)
                )
                self._task = task
                return self._view()
        except asyncio.CancelledError:
            return await self._cancelled_plan(generation)
        except Exception as error:  # noqa: BLE001 - every run error becomes a terminal failed snapshot
            async with self._lock:
                if generation == self._generation:
                    self._finish('failed', str(error))
                    self._task = None
                return self._view()

    async def _cancelled_plan(self, generation: int) -> RunSnapshot:
        """Return a superseded view, or preserve actual caller cancellation."""

        async with self._lock:
            if self._stopped_generation == generation:
                return self._view()
            if generation == self._generation:
                if self._snapshot.state == 'compiling':
                    self._finish('cancelled')
                self._task = None
        raise asyncio.CancelledError

    async def _load_population(self, generation: int, request: PopulationRequest) -> Snapshot:
        """Load one population and let replacement generations make it stale."""

        loaded = await self._load_owned(generation, lambda: self._population_loader(request))
        return loaded if loaded is not None else Snapshot(traces=())

    async def _load_owned(self, generation: int, factory: Callable[[], Awaitable[T]]) -> T | None:
        """Run one load as cancellable generation-owned work."""

        async def load() -> T:
            set_load_reporter(lambda done, total: self._report_loaded(generation, done, total))
            return await factory()

        task = asyncio.create_task(load())
        stale = False
        async with self._lock:
            stale = generation != self._generation or self._snapshot.state != 'compiling'
            if not stale:
                self._task = task
        if stale:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return None
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        except Exception:
            async with self._lock:
                if generation != self._generation or self._stopped_generation == generation:
                    return None
            raise

    def _report_loaded(self, generation: int, done: int, total: int) -> None:
        """Record loading progress; called synchronously on the loop, so no lock is needed."""
        if generation == self._generation and self._snapshot.state == 'compiling':
            self._snapshot = replace(self._snapshot, loaded=done, to_load=total)

    async def _plan(self, query: str, population: PopulationRequest) -> tuple[CompiledPlan, FilterSelectionResult]:
        """Run semantic and facet planning concurrently and clean up both."""

        async def compile_plan() -> object:
            return await self._compiler(query)

        async def select_filters() -> object:
            return await self._filter_selector(query, population)

        compiler = asyncio.create_task(compile_plan())
        filters = asyncio.create_task(select_filters())
        tasks: tuple[asyncio.Task[object], ...] = (compiler, filters)
        try:
            values = await asyncio.gather(*tasks)
            plan = values[0]
            if not isinstance(plan, CompiledPlan):
                plan = CompiledPlan.model_validate(plan)
            selected = values[1]
            if not isinstance(selected, FilterSelectionResult):
                if not isinstance(selected, FacetSelection):
                    selected = FacetSelection.model_validate(selected)
                selected = FilterSelectionResult(selected)
            return plan, selected
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            cleanup = asyncio.gather(*tasks, return_exceptions=True)
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise

    async def _execute(
        self,
        generation: int,
        traces: tuple[TraceRecord, ...],
        projections: dict[str, TraceProjection],
        dimensions: tuple[CompiledQuery, ...],
        parallelism: int,
    ) -> None:
        async def complete(classification: TraceClassification) -> None:
            await self.complete_one(generation, classification)

        try:
            await self._run_classifier(
                traces,
                projections,
                dimensions,
                parallelism=parallelism,
                on_complete=complete,
            )
            async with self._lock:
                if generation == self._generation and self._snapshot.state == 'classifying':
                    if self._snapshot.completed != self._snapshot.total:
                        raise RuntimeError('The classifier returned before all traces completed.')
                    self._finish('completed')
        except asyncio.CancelledError:
            async with self._lock:
                if generation == self._generation and self._snapshot.state == 'classifying':
                    self._finish('cancelled')
            raise
        except Exception as error:  # noqa: BLE001 - runner failures are terminal run state
            async with self._lock:
                if generation == self._generation and self._snapshot.state == 'classifying':
                    self._finish('failed', str(error))

    async def complete_one(self, generation: int, classification: TraceClassification) -> None:
        """Count one current terminal classification exactly once."""

        async with self._lock:
            current = self._snapshot
            if generation != self._generation or current.state != 'classifying':
                return
            if classification.trace_id in current.results:
                return
            try:
                if current.completed >= current.total:
                    raise ValueError('Terminal callbacks would exceed the run total.')
                if classification.trace_id not in current.trace_ids:
                    raise ValueError('Terminal callback trace ID is outside the selected population.')
                results = dict(current.results)
                results[classification.trace_id] = classification.model_copy(deep=True)
                self._snapshot = replace(
                    current,
                    results=results,
                    completed=len(results),
                    failed=current.failed + (classification.error is not None),
                    matched=current.matched + int(classification.matched and classification.error is None),
                )
            except Exception as error:
                self._finish('failed', str(error))
                raise

    async def cancel(self) -> RunSnapshot:
        """Cancel the current generation and await its owned work."""

        async with self._lifecycle_lock:
            await self._stop_current()
            async with self._lock:
                if self._snapshot.state in {'compiling', 'awaiting_review', 'classifying'}:
                    self._finish('cancelled')
                self._task = None
                return self._view()

    async def close(self) -> None:
        """Cancel owned work and release the resources the builder handed over (``close=``)."""

        async with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            try:
                await self._stop_current()
                async with self._lock:
                    if self._snapshot.state in {'compiling', 'awaiting_review', 'classifying'}:
                        self._finish('cancelled')
                    self._task = None
            finally:
                try:
                    if self.explorer is not None:
                        await self.explorer.close()
                finally:
                    if self._close is not None:
                        close, self._close = self._close, None
                        result = close()
                        if inspect.isawaitable(result):
                            await result

    async def reset(self) -> RunSnapshot:
        """Cancel owned work and return a new empty idle generation."""

        async with self._lifecycle_lock:
            if self._closed:
                raise ValueError('Trace finder store is closed.')
            await self._stop_current()
            async with self._lock:
                self._generation += 1
                self._snapshot = RunSnapshot(generation=self._generation)
                self._task = None
                self._started_monotonic = None
                return self._view()

    async def snapshot(self) -> RunSnapshot:
        """Return a detached view with current progress arithmetic."""

        async with self._lock:
            return self._view()

    async def snapshot_for_render(self) -> RunSnapshot:
        """Return a shallow view for trusted renderers that never mutate nested values."""
        async with self._lock:
            return self._view(detach=False)

    async def trace_detail(self, trace_id: str) -> TraceDetail | None:
        """Return a detached selected trace, projection, and classification."""

        async with self._lock:
            current = self._snapshot
            if trace_id not in current.trace_ids:
                return None
            trace = next((item for item in current.traces if item.trace_id == trace_id), None)
            if trace is None:
                return None
            projection = current.projections.get(trace_id)
            classification = current.results.get(trace_id)
            return TraceDetail(
                trace=trace.model_copy(deep=True),
                projection=projection.model_copy(deep=True) if projection is not None else None,
                classification=classification.model_copy(deep=True) if classification is not None else None,
                dimensions=_revalidated(current.dimensions) if current.dimensions is not None else None,
            )

    async def _stop_current(self) -> None:
        task = self._task
        if task is None or task.done():
            return
        async with self._lock:
            self._stopped_generation = self._generation
            if self._snapshot.state in {'compiling', 'classifying'}:
                self._finish('cancelled')
        task.cancel()
        cleanup = asyncio.gather(task, return_exceptions=True)
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            # The caller's own cancellation is re-raised only after the owned task finishes cleanup.
            await cleanup
            raise

    def _finish(self, state: RunState, error: str | None = None) -> None:
        if state == 'failed':
            logger.warning('Trace finder run {} failed: {}', self._generation, error)
        self._snapshot = replace(
            self._snapshot,
            state=state,
            phase=None,
            error=error,
            finished_at=self._now_utc(),
            elapsed=self._elapsed(),
        )

    def _elapsed(self) -> float:
        if self._started_monotonic is None:
            return 0.0
        if self._snapshot.state != 'classifying':
            return self._snapshot.elapsed
        return max(0.0, self._monotonic() - self._started_monotonic)

    def _view(self, *, detach: bool = True) -> RunSnapshot:
        current = self._snapshot
        elapsed = self._elapsed()
        remaining = current.total - current.completed
        parallelism = current.request.parallelism if current.request is not None else 0
        active = min(parallelism, remaining) if current.state == 'classifying' else 0
        view = replace(
            current,
            active=active,
            queued=remaining - active if current.state == 'classifying' else 0,
            percent=current.completed / current.total * 100 if current.total else 0.0,
            elapsed=elapsed,
            rate=current.completed / elapsed if elapsed else 0.0,
        )
        return _detach(view) if detach else view


def _merge_facets(caller: FacetSelection, generated: FacetSelection) -> FacetSelection:
    """Merge facets with caller-supplied non-empty values taking precedence."""

    return FacetSelection(
        project_id=caller.project_id,
        **{name: getattr(caller, name) or getattr(generated, name) for name in FACET_NAMES},
    )


def _empty_traces_error(*, loaded_rows: bool) -> ValueError:
    """Describe an empty selection using the path that supplied its rows."""

    if loaded_rows:
        return ValueError('None of the loaded traces have usable messages to classify.')
    return ValueError('No traces match the classifier-selected filters.')


def _review_explicit(
    previous: TReviewFilters,
    explicit: TReviewFilters,
    requested: TReviewFilters,
    names: tuple[str, ...],
) -> TReviewFilters:
    """Preserve user provenance for unchanged review fields and adopt edits as explicit."""

    return type(requested)(**{
        name: getattr(requested, name)
        if getattr(requested, name) != getattr(previous, name)
        else getattr(explicit, name)
        for name in names
    })


def _revalidated(dimensions: Sequence[CompiledQuery]) -> tuple[CompiledQuery, ...]:
    """Detach dimensions from caller-owned instances by validating fresh copies."""

    return tuple(CompiledQuery.model_validate(dimension.model_dump()) for dimension in dimensions)


# ponytail: a word list, not a parser; it only has to spot a question that says more than its numeric bounds.
_NUMERIC_QUERY_WORDS = frozenset([
    'all',
    'any',
    'and',
    'are',
    'between',
    'calls',
    'conversations',
    'duration',
    'faster',
    'fewer',
    'find',
    'for',
    'get',
    'has',
    'have',
    'least',
    'less',
    'list',
    'longer',
    'milliseconds',
    'minutes',
    'more',
    'most',
    'over',
    'requests',
    'runs',
    'second',
    'seconds',
    'show',
    'shorter',
    'slower',
    'than',
    'that',
    'the',
    'them',
    'those',
    'total',
    'trace',
    'traces',
    'token',
    'tokens',
    'under',
    'used',
    'using',
    'which',
    'with',
    'above',
    'below',
    'took',
    'taking',
])


def _unsupported_warning(reason: str | None) -> str | None:
    """Say plainly that an aggregate question was not answered, instead of reporting a quiet success."""

    if reason is None:
        return None
    return f"Ask AI finds traces; it can't compute totals, averages or rankings. {reason}"


def _uncovered_words_warning(query: str, dimensions: tuple[CompiledQuery, ...], filters: FacetSelection) -> str | None:
    """Warn when nothing but numeric bounds survived planning, yet the question says more than numbers."""

    if dimensions or any(getattr(filters, name) for name in FACET_NAMES):
        return None
    words = tuple(dict.fromkeys(w for w in re.findall(r'[a-z]{3,}', query.casefold()) if w not in _NUMERIC_QUERY_WORDS))
    if not words:
        return None
    logger.warning('Trace finder plan applies only numeric bounds; the question also says {}', ', '.join(words))
    return (
        f'Only the numeric bounds were applied. No AI question or filter covers "{" ".join(words)}", '
        'so rephrase the question or use Review to add one.'
    )


def _narrowed(
    traces: tuple[TraceRecord, ...], facets: FacetSelection, bounds: NumericFilters
) -> tuple[TraceRecord, ...]:
    """Keep loaded traces inside the generated filters."""

    return tuple(trace for trace in traces if _within_facets(trace, facets) and _within_numeric(trace, bounds))


def _nothing_kept_message(traces: tuple[TraceRecord, ...], facets: FacetSelection, bounds: NumericFilters) -> str:
    """Name each filter or bound that dropped loaded rows, with the nearest loaded value, so the user knows what to change."""

    reasons: list[str] = []
    for name in FACET_NAMES:
        selected: frozenset[str] = getattr(facets, name)
        if selected and not all(
            _within_facets(trace, FacetSelection.model_validate({name: selected})) for trace in traces
        ):
            reasons.append(f'{name.replace("_", " ")} {" or ".join(sorted(selected))}')
    tokens = [trace.total_tokens for trace in traces if trace.total_tokens is not None]
    durations = [trace.duration_ms for trace in traces if trace.duration_ms is not None]
    for field, values, wanted, nearest in (
        ('tokens_min', tokens, 'at least {:,} tokens', lambda: f'the largest has {max(tokens):,}'),
        ('tokens_max', tokens, 'at most {:,} tokens', lambda: f'the smallest has {min(tokens):,}'),
        (
            'duration_ms_min',
            durations,
            'a duration of at least {:,} ms',
            lambda: f'the longest took {max(durations):,} ms',
        ),
        (
            'duration_ms_max',
            durations,
            'a duration of at most {:,} ms',
            lambda: f'the shortest took {min(durations):,} ms',
        ),
    ):
        bound: int | None = getattr(bounds, field)
        if bound is not None and not all(_within_numeric(trace, NumericFilters(**{field: bound})) for trace in traces):
            reasons.append(wanted.format(bound) + (f' ({nearest()})' if values else ''))
    detail = ' and '.join(reasons) or 'the filters in the question'
    return f'None of the {len(traces):,} loaded traces have {detail}. Try New search to look beyond the loaded rows.'


def _within_facets(trace: TraceRecord, facets: FacetSelection) -> bool:
    """Apply selected facet values to one loaded trace; an empty facet places no constraint."""

    for name in FACET_NAMES:
        selected: frozenset[str] = getattr(facets, name)
        if not selected:
            continue
        values = trace.tool_names if name == 'tool_name' else (getattr(trace, name),)
        if not selected.intersection(values):
            return False
    return True


def _within_numeric(trace: TraceRecord, bounds: NumericFilters) -> bool:
    """Apply inclusive numeric bounds to one loaded trace; a trace missing a bounded value is excluded."""

    for value, low, high in (
        (trace.total_tokens, bounds.tokens_min, bounds.tokens_max),
        (trace.duration_ms, bounds.duration_ms_min, bounds.duration_ms_max),
    ):
        if low is None and high is None:
            continue
        if value is None or (low is not None and value < low) or (high is not None and value > high):
            return False
    return True


def _merge_numeric(caller: NumericFilters, generated: NumericFilters) -> NumericFilters:
    """Merge numeric bounds with each caller-supplied bound taking precedence."""

    return NumericFilters(**{
        name: getattr(caller, name) if getattr(caller, name) is not None else getattr(generated, name)
        for name in NumericFilters.model_fields
    })


def _detach(snapshot: RunSnapshot) -> RunSnapshot:
    """Copy nested mutable payloads at the public read boundary."""

    return replace(
        snapshot,
        request=snapshot.request.model_copy(deep=True) if snapshot.request is not None else None,
        dimensions=_revalidated(snapshot.dimensions) if snapshot.dimensions is not None else None,
        explicit_filters=snapshot.explicit_filters.model_copy(deep=True),
        explicit_numeric=snapshot.explicit_numeric.model_copy(deep=True),
        generated_filters=snapshot.generated_filters.model_copy(deep=True),
        generated_numeric=snapshot.generated_numeric.model_copy(deep=True),
        filter_response=snapshot.filter_response.model_copy(deep=True)
        if snapshot.filter_response is not None
        else None,
        traces=tuple(trace.model_copy(deep=True) for trace in snapshot.traces),
        results={key: value.model_copy(deep=True) for key, value in snapshot.results.items()},
        projections={key: value.model_copy(deep=True) for key, value in snapshot.projections.items()},
    )
