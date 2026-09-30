"""Loaded explorer rows, their UI view state and a per-trace message cache."""

from __future__ import annotations

import asyncio
import itertools
import math
import operator
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args

from loguru import logger

from .columns import COLUMNS, MATCH, sort_rows
from .models import FacetSelection, NumericFilters

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
    from datetime import datetime

    from .models import RunSnapshot, TraceClassification, TraceRecord
    from .rows import TraceRow

ViewMode = Literal['table', 'trajectories']
QuickView = Literal[
    'all', 'errors', 'matches', 'slow', 'costly', 'tokens', 'context', 'conv_costly', 'conv_tokens', 'conv_longest'
]
QUICK_VIEWS: frozenset[str] = frozenset(get_args(QuickView))
TOP_SHARE = 0.10


def _total_tokens(row: TraceRow) -> int | None:
    if row.tokens_in is None and row.tokens_out is None:
        return None
    return (row.tokens_in or 0) + (row.tokens_out or 0)


# Built-in views (BOPS-1243): the top 10% of loaded rows by one metric, heaviest first.
TOP_METRICS: Mapping[str, tuple[str, Callable[[TraceRow], float | None]]] = {
    'slow': ('duration', lambda row: row.duration_ms),
    'costly': ('cost', lambda row: row.cost_total),
    'tokens': ('total tokens', _total_tokens),
    'context': ('largest prompt', lambda row: row.context_tokens),
}
CONVERSATION_METRICS: Mapping[str, tuple[str, Callable[[Sequence[TraceRow], Mapping[str, int]], float | None]]] = {
    'conv_costly': (
        'cost',
        lambda rows, _message_counts: (
            sum(row.cost_total for row in rows if row.cost_total is not None)
            if any(row.cost_total is not None for row in rows)
            else None
        ),
    ),
    'conv_tokens': (
        'total tokens',
        lambda rows, _message_counts: (
            sum(tokens for row in rows if (tokens := _total_tokens(row)) is not None)
            if any(_total_tokens(row) is not None for row in rows)
            else None
        ),
    ),
    'conv_longest': (
        'messages',
        lambda rows, message_counts: max(
            (message_counts[row.trace_id] for row in rows if row.trace_id in message_counts), default=None
        ),
    ),
}


def ai_matched(result: TraceClassification | None) -> bool:
    """True only for a trace the classifier judged a match; a filter-only run's kept rows have no answers."""
    return result is not None and result.matched and bool(result.answers)


def conversation_key(row: TraceRow) -> str | None:
    return row.thread_id or row.session_id


def _conversation_groups(
    rows: Sequence[TraceRow],
    metric: Callable[[Sequence[TraceRow], Mapping[str, int]], float | None],
    message_counts: Mapping[str, int],
) -> list[tuple[float, tuple[TraceRow, ...]]]:
    grouped: dict[str, list[TraceRow]] = {}
    for row in rows:
        if (key := conversation_key(row)) is not None:
            grouped.setdefault(key, []).append(row)
    valued: list[tuple[float, tuple[TraceRow, ...]]] = []
    for conversation in grouped.values():
        value = metric(conversation, message_counts)
        if value is not None:
            ordered = tuple(sorted(conversation, key=lambda row: (row.started_at is None, row.started_at)))
            valued.append((value, ordered))
    return sorted(valued, key=operator.itemgetter(0), reverse=True)


def top_conversations(
    rows: Sequence[TraceRow],
    metric: Callable[[Sequence[TraceRow], Mapping[str, int]], float | None],
    message_counts: Mapping[str, int] | None = None,
) -> tuple[TraceRow, ...]:
    """Return rows from the top 10% of conversations reporting a metric."""
    groups = _conversation_groups(rows, metric, message_counts or {})
    kept = groups[: math.ceil(len(groups) * TOP_SHARE)]
    return tuple(row for _, conversation in kept for row in conversation)


def top_share(rows: Sequence[TraceRow], metric: Callable[[TraceRow], float | None]) -> tuple[TraceRow, ...]:
    """The top 10% (at least one) of rows that report the metric, largest first; rows without it are left out."""
    valued = sorted((row for row in rows if metric(row) is not None), key=lambda row: metric(row) or 0, reverse=True)
    return tuple(valued[: math.ceil(len(valued) * TOP_SHARE)])


@dataclass(frozen=True)
class Totals:
    """Sums over one set of rows; ``None`` means no row reported the figure."""

    traces: int
    errors: int
    cost: float | None
    currency: str | None
    tokens_in: int | None
    tokens_out: int | None
    cache_share: float | None
    p50_ms: int | None
    p95_ms: int | None


def _sum_known(values: Sequence[int | float | None]) -> int | float | None:
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _nearest_rank(sorted_values: Sequence[int], quantile: float) -> int | None:
    if not sorted_values:
        return None
    return sorted_values[max(0, math.ceil(quantile * len(sorted_values)) - 1)]


def totals(rows: Sequence[TraceRow]) -> Totals:
    """Total the *rows* the table currently shows.

    Cache share is cache-read over input tokens, taken only from rows that report both: Orq's
    prompt tokens already include the cached ones. Cost is left unset when currencies are mixed.
    """
    currencies = {row.currency for row in rows if row.cost_total is not None}
    cost = _sum_known([row.cost_total for row in rows]) if len(currencies) <= 1 else None
    cached_rows = [row for row in rows if row.tokens_in and row.cached_tokens is not None]
    cached_input = sum(row.tokens_in or 0 for row in cached_rows)
    durations = sorted(row.duration_ms for row in rows if row.duration_ms is not None)
    return Totals(
        traces=len(rows),
        errors=sum(row.is_error for row in rows),
        cost=float(cost) if cost is not None else None,
        currency=next(iter(currencies)) if len(currencies) == 1 else None,
        tokens_in=int(v) if (v := _sum_known([row.tokens_in for row in rows])) is not None else None,
        tokens_out=int(v) if (v := _sum_known([row.tokens_out for row in rows])) is not None else None,
        cache_share=sum(row.cached_tokens or 0 for row in cached_rows) / cached_input if cached_input else None,
        p50_ms=_nearest_rank(durations, 0.5),
        p95_ms=_nearest_rank(durations, 0.95),
    )


ExplorerState = Literal['idle', 'loading', 'loaded', 'failed']
PAGE_ROWS = 100
# One process-wide sequence, so a load's generation is unique across sessions and a view version
# only ever grows. Each load takes a new generation; each visible view change takes a new version.
_SEQUENCE = itertools.count(1)


class RowSearch(Protocol):
    def __call__(
        self,
        start: datetime,
        end: datetime,
        limit: int,
        *,
        facets: FacetSelection,
        numeric: NumericFilters,
        on_page: Callable[[tuple[TraceRow, ...]], None] | None = None,
    ) -> Awaitable[tuple[TraceRow, ...]]: ...


class RowHydrator(Protocol):
    def __call__(self, rows: Sequence[TraceRow]) -> Awaitable[dict[str, TraceRecord | None]]: ...


@dataclass(frozen=True)
class ExplorerView:
    generation: int = 0
    version: int = 0
    state: ExplorerState = 'idle'
    initial_load: bool = False
    trajectory_warming: bool = False
    rows: tuple[TraceRow, ...] = ()
    limit: int = 0
    start: datetime | None = None
    end: datetime | None = None
    facets: FacetSelection = field(default_factory=FacetSelection)
    message_counts: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    message_counting: bool = False
    message_count_error: str | None = None
    numeric: NumericFilters = field(default_factory=NumericFilters)
    error: str | None = None
    sort: str | None = None
    descending: bool = True
    page: int = 0
    view: ViewMode = 'table'
    show_tool_definitions: bool = False
    matched_only: bool = False
    quick_view: QuickView = 'all'
    narrowed_from: int | None = None

    def visible_rows(self, results: Mapping[str, TraceClassification] | None = None) -> tuple[TraceRow, ...]:
        rows = self.rows
        if self.quick_view == 'errors':
            rows = tuple(row for row in rows if row.is_error)
        elif self.quick_view == 'matches':
            rows = tuple(row for row in rows if results is not None and ai_matched(results.get(row.trace_id)))
        elif self.quick_view in TOP_METRICS:
            rows = top_share(rows, TOP_METRICS[self.quick_view][1])
        elif self.quick_view in CONVERSATION_METRICS:
            rows = top_conversations(rows, CONVERSATION_METRICS[self.quick_view][1], self.message_counts)
        if self.matched_only and results:
            rows = tuple(row for row in rows if (result := results.get(row.trace_id)) is not None and result.matched)
        if self.sort is None:
            return rows
        return sort_rows(rows, self.sort, descending=self.descending, results=results)

    def page_count(self, results: Mapping[str, TraceClassification] | None = None) -> int:
        return max(1, -(-len(self.visible_rows(results)) // PAGE_ROWS))

    def page_rows(self, results: Mapping[str, TraceClassification] | None = None) -> tuple[TraceRow, ...]:
        visible = self.visible_rows(results)
        start = min(self.page, max(1, -(-len(visible) // PAGE_ROWS)) - 1) * PAGE_ROWS
        return visible[start : start + PAGE_ROWS]


def matches_first_view(view: ExplorerView, snapshot: RunSnapshot | None) -> ExplorerView:
    """Put completed AI matches first when All has no explicit sort, for both table and export."""
    if (
        snapshot is not None
        and snapshot.within_results
        and snapshot.state == 'completed'
        and snapshot.dimensions
        and view.quick_view == 'all'
        and view.sort is None
    ):
        matched = tuple(row for row in view.rows if ai_matched(snapshot.results.get(row.trace_id)))
        unmatched = tuple(row for row in view.rows if not ai_matched(snapshot.results.get(row.trace_id)))
        return replace(view, rows=(*matched, *unmatched))
    return view


class ExplorerStore:
    """One replaceable load; a newer load's generation makes an older one's callbacks no-ops."""

    def __init__(
        self,
        *,
        search: RowSearch,
        hydrate: RowHydrator,
        load_spans: Callable[[str], Awaitable[list[Any]]] | None = None,
        load_first_error_message: Callable[[str, Sequence[Any]], Awaitable[str | None]] | None = None,
    ) -> None:
        self._search = search
        self._hydrate = hydrate
        self._load_spans = load_spans
        self._load_first_error_message = load_first_error_message
        self._view = ExplorerView()
        self._task: asyncio.Task[None] | None = None
        self._prewarm_task: asyncio.Task[None] | None = None
        self._message_count_task: asyncio.Task[None] | None = None
        self._records: dict[str, TraceRecord | None] = {}
        self._records_lock = asyncio.Lock()
        self._hydrations: dict[tuple[str, ...], asyncio.Task[dict[str, TraceRecord | None]]] = {}
        self._pre_narrow_view: ExplorerView | None = None

    async def spans(self, trace_id: str) -> list[Any]:
        """Fetch the trace's bounded span summaries on demand."""
        if self._load_spans is None:
            raise RuntimeError('Span loading is unavailable.')
        return await self._load_spans(trace_id)

    async def first_error_message(self, trace_id: str, spans: Sequence[Any]) -> str | None:
        """Load status text for the first errored span, if the source supports it."""
        if self._load_first_error_message is None:
            return None
        return await self._load_first_error_message(trace_id, spans)

    async def load(  # noqa: C901 — one task owns search, first-page prewarm, and generation state
        self,
        start: datetime,
        end: datetime,
        limit: int,
        *,
        facets: FacetSelection,
        numeric: NumericFilters,
        wait: bool = False,
        warm_trajectories: bool = True,
        initial_load: bool = False,
    ) -> ExplorerView:
        await self._cancel()
        self._hydrations.clear()
        self._records.clear()
        generation = next(_SEQUENCE)
        self._pre_narrow_view = None
        previous = self._view
        # The tab and sort survive a Load, including Match; classification may still be useful
        # while a new search loads and the user explicitly chose this ordering.
        kept_sort = previous.sort if previous.sort in COLUMNS else None
        self._view = ExplorerView(
            generation=generation,
            version=generation,
            state='loading',
            initial_load=initial_load,
            limit=limit,
            start=start,
            end=end,
            facets=facets,
            numeric=numeric,
            view=previous.view,
            quick_view=previous.quick_view,
            sort=kept_sort,
            descending=previous.descending if kept_sort is not None else True,
            show_tool_definitions=previous.show_tool_definitions,
        )

        async def prewarm(rows: Sequence[TraceRow]) -> None:
            try:
                await self.records([row.trace_id for row in rows])
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 — trajectory data must not fail the summary load
                logger.warning('Explorer trajectory prewarm failed: {}', error)
            finally:
                if self._view.generation == generation:
                    self._view = replace(self._view, trajectory_warming=False)

        def start_prewarm(rows: Sequence[TraceRow]) -> None:
            if self._prewarm_task is None and warm_trajectories:
                self._view = replace(self._view, trajectory_warming=True)
                self._prewarm_task = asyncio.create_task(prewarm(rows[:PAGE_ROWS]))

        def on_page(rows: tuple[TraceRow, ...]) -> None:
            if self._view.generation == generation:
                self._view = replace(self._view, rows=rows)
                if len(rows) >= PAGE_ROWS:
                    start_prewarm(rows)

        async def run() -> None:
            try:
                rows = await self._search(start, end, limit, facets=facets, numeric=numeric, on_page=on_page)
            except asyncio.CancelledError:
                if self._view.generation == generation and self._view.state == 'loading':
                    self._view = replace(self._view, state='failed', error='Loading was cancelled.')
                raise
            except Exception as error:  # noqa: BLE001 — keep rows loaded so far and name the failure
                if self._prewarm_task is not None and not self._prewarm_task.done():
                    self._prewarm_task.cancel()
                    await asyncio.gather(self._prewarm_task, return_exceptions=True)
                if self._view.generation == generation:
                    logger.warning('Explorer load failed after {} row(s): {}', len(self._view.rows), error)
                    self._view = replace(self._view, state='failed', error=str(error), trajectory_warming=False)
                return
            if self._view.generation == generation:
                self._view = replace(self._view, rows=rows)
                start_prewarm(rows)
                self._view = replace(self._view, state='loaded')

        self._task = asyncio.create_task(run())
        if wait:
            await self._task
        return self._view

    async def view(self) -> ExplorerView:
        return self._view

    async def set_view(
        self,
        *,
        sort: str | None = None,
        descending: bool | None = None,
        page: int | None = None,
        view: ViewMode | None = None,
        show_tool_definitions: bool | None = None,
        matched_only: bool | None = None,
        quick_view: QuickView | None = None,
    ) -> ExplorerView:
        """Change display state; ``page`` is clamped when rendered because matches affect row count.

        Switching ``quick_view`` keeps the sort and goes back to the first page unless ``page`` is given.
        """
        current = self._view
        switching_quick_view = quick_view is not None and quick_view != current.quick_view
        updated = replace(
            current,
            sort=sort if sort is not None else current.sort,
            descending=descending if descending is not None else current.descending,
            page=max(0, page) if page is not None else 0 if switching_quick_view else current.page,
            view=view if view is not None else current.view,
            show_tool_definitions=show_tool_definitions
            if show_tool_definitions is not None
            else current.show_tool_definitions,
            matched_only=matched_only if matched_only is not None else current.matched_only,
            quick_view=quick_view if quick_view is not None else current.quick_view,
            message_count_error=None
            if switching_quick_view and quick_view == 'conv_longest'
            else current.message_count_error,
        )
        display = ('sort', 'descending', 'page', 'view', 'matched_only', 'quick_view')
        if any(getattr(updated, name) != getattr(current, name) for name in display):
            updated = replace(updated, version=next(_SEQUENCE))
        self._view = updated
        return self._view

    async def narrow(
        self, generation: int, keep: Collection[str], *, facets: FacetSelection, numeric: NumericFilters
    ) -> None:
        """Show only ``keep`` under the given filters, unless a newer load replaced the rows."""
        if self._view.generation != generation:
            return
        current = self._view
        if self._pre_narrow_view is None:
            self._pre_narrow_view = current
        updated = replace(
            current,
            narrowed_from=self._view.narrowed_from if self._view.narrowed_from is not None else len(self._view.rows),
            rows=tuple(row for row in self._view.rows if row.trace_id in keep),
            facets=facets,
            numeric=numeric,
            page=0,
        )
        self._view = updated if updated == current else replace(updated, version=next(_SEQUENCE))

    async def restore_narrowed(self) -> ExplorerView:
        """Restore the rows and user filters saved before the first AI narrowing of this generation.

        Filter edits made while the AI-narrowed results are showing are discarded with that narrowed
        state; a later run after restoring saves a fresh snapshot.
        """
        previous = self._pre_narrow_view
        if previous is None or previous.generation != self._view.generation:
            return self._view
        self._pre_narrow_view = None
        updated = replace(
            self._view,
            rows=previous.rows,
            facets=previous.facets,
            numeric=previous.numeric,
            narrowed_from=None,
            page=0,
            sort=None if self._view.sort == MATCH else self._view.sort,
            descending=True if self._view.sort == MATCH else self._view.descending,
        )
        self._view = updated if updated == self._view else replace(updated, version=next(_SEQUENCE))
        return self._view

    async def clear_ai_view_state(self) -> ExplorerView:
        """Clear table controls that only make sense while AI results are active."""
        current = self._view
        updated = replace(
            current,
            sort=None if current.sort == MATCH else current.sort,
            descending=True if current.sort == MATCH else current.descending,
            matched_only=False,
            quick_view='all' if current.quick_view == 'matches' else current.quick_view,
            page=0,
        )
        if updated != current:
            updated = replace(updated, version=next(_SEQUENCE))
        self._view = updated
        return self._view

    async def records(self, trace_ids: Sequence[str]) -> dict[str, TraceRecord | None]:
        """Hydrate requested rows; successful records and empty traces are cached until the next load."""
        async with self._records_lock:
            generation = self._view.generation
            requested = set(trace_ids)
            wanted = [row for row in self._view.rows if row.trace_id in requested and row.trace_id not in self._records]
            key: tuple[str, ...] | None = None
            if wanted:
                key = tuple(sorted(row.trace_id for row in wanted))
                task = self._hydrations.get(key)
                if task is None:

                    async def hydrate_rows() -> dict[str, TraceRecord | None]:
                        return await self._hydrate(wanted)

                    task = asyncio.create_task(hydrate_rows())
                    self._hydrations[key] = task
            else:
                task = None
        if task is not None:
            try:
                hydrated = await asyncio.shield(task)
            finally:
                async with self._records_lock:
                    if key is not None and self._hydrations.get(key) is task and task.done():
                        del self._hydrations[key]
            async with self._records_lock:
                # A load may replace rows while hydration is in progress. Its cache belongs
                # to that load only, so late results from the previous generation are discarded.
                if self._view.generation == generation:
                    for trace_id, record in hydrated.items():
                        cached = self._records.get(trace_id)
                        if trace_id not in self._records or (cached is None and record is not None):
                            self._records[trace_id] = record
                    models = {
                        trace_id: record.model
                        for trace_id in hydrated
                        if (record := self._records.get(trace_id)) is not None
                        if record.model not in {'', 'unknown'}
                    }
                    if models:
                        rows = tuple(
                            row.model_copy(
                                update={
                                    'models': tuple(
                                        model.strip() for model in models[row.trace_id].split(',') if model.strip()
                                    )
                                }
                            )
                            if not row.models and row.trace_id in models
                            else row
                            for row in self._view.rows
                        )
                        if rows != self._view.rows:
                            self._view = replace(self._view, rows=rows, version=next(_SEQUENCE))
                    return {trace_id: self._records.get(trace_id) for trace_id in trace_ids}
                return {trace_id: hydrated.get(trace_id) for trace_id in trace_ids}
        async with self._records_lock:
            return {trace_id: self._records.get(trace_id) for trace_id in trace_ids}

    def cached_records(self, trace_ids: Sequence[str]) -> dict[str, TraceRecord | None]:
        """Return only trajectory records this load has already hydrated."""
        return {trace_id: self._records[trace_id] for trace_id in trace_ids if trace_id in self._records}

    async def message_counts(self) -> ExplorerView:
        generation = self._view.generation
        records = await self.records([row.trace_id for row in self._view.rows])
        if self._view.generation == generation:
            self._view = replace(
                self._view,
                message_counts=MappingProxyType({
                    trace_id: len(record.messages) for trace_id, record in records.items() if record is not None
                }),
            )
        return self._view

    async def start_message_counting(self, *, retry: bool = False) -> ExplorerView:
        """Count loaded-row messages in the background, scoped to the current load generation."""
        if retry and self._view.message_count_error:
            self._view = replace(self._view, message_count_error=None)
        if (
            self._view.message_counts
            or self._view.message_counting
            or self._view.message_count_error
            or not self._view.rows
        ):
            return self._view
        generation = self._view.generation
        self._view = replace(self._view, message_counting=True, message_count_error=None, version=next(_SEQUENCE))

        async def count() -> None:
            try:
                counted = await self.message_counts()
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 — counting should leave the loaded table usable
                if self._view.generation == generation:
                    logger.warning('Counting conversation messages failed: {}', error)
                    self._view = replace(
                        self._view,
                        message_counting=False,
                        message_count_error=str(error),
                        version=next(_SEQUENCE),
                    )
                return
            if self._view.generation == generation and counted.generation == generation:
                self._view = replace(
                    self._view,
                    message_counting=False,
                    message_count_error=None,
                    version=next(_SEQUENCE),
                )

        self._message_count_task = asyncio.create_task(count())
        return self._view

    async def row(self, trace_id: str) -> TraceRow | None:
        return next((row for row in self._view.rows if row.trace_id == trace_id), None)

    async def has_cached_record(self, trace_id: str) -> bool:
        """Return whether this load cached a record or a successfully empty trace."""
        async with self._records_lock:
            return trace_id in self._records

    async def close(self) -> None:
        await self._cancel()
        self._records.clear()

    async def _cancel(self) -> None:
        tasks = [
            task
            for task in (self._task, self._prewarm_task, self._message_count_task)
            if task is not None and not task.done()
        ]
        self._task = None
        self._prewarm_task = None
        self._message_count_task = None
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
