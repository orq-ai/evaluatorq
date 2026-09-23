"""Generation-checked ownership of trace-finder compilation and classification."""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol, TypeVar

from loguru import logger

from .compiler import CompiledPlan
from .models import (
    FACET_NAMES,
    CompiledQuery,
    FacetSelection,
    JevProjection,
    NumericFilters,
    PopulationRequest,
    RunRequest,
    RunSnapshot,
    RunState,
    Snapshot,
    TraceClassification,
    TraceDetail,
    TraceRecord,
)
from .projection import project_trace as default_project_trace

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

TReviewFilters = TypeVar('TReviewFilters', FacetSelection, NumericFilters)


class PopulationLoader(Protocol):
    """Load the bounded trace population after planning has completed."""

    def __call__(self, request: PopulationRequest) -> Awaitable[Snapshot]: ...


class JevRunner(Protocol):
    """A run_jev callable with its model and client already bound."""

    def __call__(
        self,
        traces: tuple[TraceRecord, ...],
        projections: dict[str, JevProjection],
        compiled: CompiledQuery,
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
        run_jev: JevRunner,
        filter_selector: Callable[[str, PopulationRequest], Awaitable[FacetSelection]],
        project_trace: Callable[[TraceRecord], JevProjection] = default_project_trace,
        monotonic: Callable[[], float] = time.monotonic,
        now_utc: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        close: Callable[[], Awaitable[None] | None] | None = None,
    ) -> None:
        self._compiler = compiler
        self._close = close
        self._closed = False
        self._population_loader = population_loader
        self._run_jev = run_jev
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

    async def compile(self, request: RunRequest, *, wait: bool = True) -> RunSnapshot:
        """Compile, select the population, and stage or start the requested run.

        With ``wait=False`` the planning runs as an owned background task and the
        ``compiling`` snapshot is returned at once, so a caller that polls can show
        progress while the compiler and facet selector are still working.
        """

        return await self._prepare(request, None, compile_query=True, wait=wait)

    async def start(self, request: RunRequest, compiled: CompiledQuery) -> RunSnapshot:
        """Start a reviewed task without running the compiler or facet selector."""

        return await self._prepare(request, compiled, compile_query=False)

    async def _prepare(
        self,
        request: RunRequest,
        compiled: CompiledQuery | None,
        *,
        compile_query: bool,
        wait: bool = True,
    ) -> RunSnapshot:
        generation, staged_traces = await self._begin(request, compile_query=compile_query)
        work = self._plan_and_start(generation, staged_traces, request, compiled, compile_query=compile_query)
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

    async def _begin(self, request: RunRequest, *, compile_query: bool) -> tuple[int, tuple[TraceRecord, ...]]:
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
                        previous.facets, self._snapshot.explicit_filters, explicit_filters, FACET_NAMES
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
                generated_numeric = NumericFilters() if compile_query else self._snapshot.generated_numeric
                self._snapshot = RunSnapshot(
                    generation=generation,
                    state='compiling',
                    request=request.model_copy(deep=True),
                    explicit_filters=explicit_filters.model_copy(deep=True),
                    explicit_numeric=explicit_numeric.model_copy(deep=True),
                    generated_filters=generated_filters.model_copy(deep=True),
                    generated_numeric=generated_numeric.model_copy(deep=True),
                    created_at=self._now_utc(),
                )
                self._started_monotonic = None
                self._task = None
                return generation, staged_traces

    async def _plan_and_start(
        self,
        generation: int,
        staged_traces: tuple[TraceRecord, ...],
        request: RunRequest,
        compiled: CompiledQuery | None,
        *,
        compile_query: bool,
    ) -> RunSnapshot:
        try:
            if compile_query:
                plan_task = asyncio.create_task(self._plan(request.query, request.population))
                async with self._lock:
                    if generation != self._generation or self._snapshot.state != 'compiling':
                        plan_task.cancel()
                        await asyncio.gather(plan_task, return_exceptions=True)
                        return self._view()
                    self._task = plan_task
                plan, generated_filters = await plan_task
                generated_numeric = plan.numeric.model_copy(deep=True)
                compiled = CompiledQuery.model_validate(plan.compiled.model_dump())
                population = request.population.model_copy(
                    update={
                        'facets': _merge_facets(request.population.facets, generated_filters),
                        'numeric': _merge_numeric(request.population.numeric, plan.numeric),
                    }
                )
                request = request.model_copy(update={'population': population})
            else:
                if compiled is None:
                    raise ValueError('a compiled task is required to start a run')
                compiled = CompiledQuery.model_validate(compiled.model_dump())
                generated_filters = self._snapshot.generated_filters
                generated_numeric = self._snapshot.generated_numeric

            async with self._lock:
                if generation != self._generation or self._snapshot.state != 'compiling':
                    return self._view()
                self._snapshot = replace(
                    self._snapshot,
                    compiled=compiled,
                    generated_filters=generated_filters.model_copy(deep=True),
                    generated_numeric=generated_numeric.model_copy(deep=True),
                    request=request.model_copy(deep=True),
                )

            if not staged_traces:
                loaded = await self._load_population(generation, request.population)
                staged_traces = tuple(loaded.traces[: request.population.limit])

            async with self._lock:
                if generation != self._generation or self._snapshot.state != 'compiling':
                    return self._view()
                if not staged_traces:
                    raise ValueError('No traces match the JEV-selected filters.')
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
                    self._snapshot = replace(self._snapshot, state='awaiting_review')
                    self._task = None
                    return self._view()
                projections = {trace.trace_id: self._project_trace(trace) for trace in staged_traces}
                self._snapshot = replace(
                    self._snapshot,
                    projections=projections,
                    state='classifying',
                    started_at=self._now_utc(),
                )
                self._started_monotonic = self._monotonic()
                task = asyncio.create_task(
                    self._execute(generation, staged_traces, projections, compiled, request.parallelism)
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

        async def load() -> Snapshot:
            return await self._population_loader(request)

        task = asyncio.create_task(load())
        stale = False
        async with self._lock:
            stale = generation != self._generation or self._snapshot.state != 'compiling'
            if not stale:
                self._task = task
        if stale:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return Snapshot(traces=())
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def _plan(self, query: str, population: PopulationRequest) -> tuple[CompiledPlan, FacetSelection]:
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
            if not isinstance(selected, FacetSelection):
                selected = FacetSelection.model_validate(selected)
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
        projections: dict[str, JevProjection],
        compiled: CompiledQuery,
        parallelism: int,
    ) -> None:
        async def complete(classification: TraceClassification) -> None:
            await self.complete_one(generation, classification)

        try:
            await self._run_jev(
                traces,
                projections,
                compiled,
                parallelism=parallelism,
                on_complete=complete,
            )
            async with self._lock:
                if generation == self._generation and self._snapshot.state == 'classifying':
                    if self._snapshot.completed != self._snapshot.total:
                        raise RuntimeError('JEV returned before all traces completed.')
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
                compiled=current.compiled.model_copy(deep=True) if current.compiled is not None else None,
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

    def _view(self) -> RunSnapshot:
        current = self._snapshot
        elapsed = self._elapsed()
        remaining = current.total - current.completed
        parallelism = current.request.parallelism if current.request is not None else 0
        active = min(parallelism, remaining) if current.state == 'classifying' else 0
        return _detach(
            replace(
                current,
                active=active,
                queued=remaining - active if current.state == 'classifying' else 0,
                percent=current.completed / current.total * 100 if current.total else 0.0,
                elapsed=elapsed,
                rate=current.completed / elapsed if elapsed else 0.0,
            )
        )


def _merge_facets(caller: FacetSelection, generated: FacetSelection) -> FacetSelection:
    """Merge facets with caller-supplied non-empty values taking precedence."""

    return FacetSelection(**{name: getattr(caller, name) or getattr(generated, name) for name in FACET_NAMES})


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
        compiled=snapshot.compiled.model_copy(deep=True) if snapshot.compiled is not None else None,
        explicit_filters=snapshot.explicit_filters.model_copy(deep=True),
        explicit_numeric=snapshot.explicit_numeric.model_copy(deep=True),
        generated_filters=snapshot.generated_filters.model_copy(deep=True),
        generated_numeric=snapshot.generated_numeric.model_copy(deep=True),
        traces=tuple(trace.model_copy(deep=True) for trace in snapshot.traces),
        results={key: value.model_copy(deep=True) for key, value in snapshot.results.items()},
        projections={key: value.model_copy(deep=True) for key, value in snapshot.projections.items()},
    )
