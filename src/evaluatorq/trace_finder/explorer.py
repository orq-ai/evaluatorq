"""Loaded explorer rows, their UI view state and a per-trace message cache."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, Protocol

from loguru import logger

from .columns import sort_rows
from .models import FacetSelection, NumericFilters

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from datetime import datetime

    from .models import TraceClassification, TraceRecord
    from .rows import TraceRow

ViewMode = Literal['table', 'trajectories']
QuickView = Literal['all', 'errors', 'matches']
ExplorerState = Literal['idle', 'loading', 'loaded', 'failed']
PAGE_ROWS = 100


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
    state: ExplorerState = 'idle'
    rows: tuple[TraceRow, ...] = ()
    limit: int = 0
    start: datetime | None = None
    end: datetime | None = None
    facets: FacetSelection = field(default_factory=FacetSelection)
    numeric: NumericFilters = field(default_factory=NumericFilters)
    error: str | None = None
    sort: str | None = None
    descending: bool = True
    page: int = 0
    view: ViewMode = 'table'
    matched_only: bool = False
    quick_view: QuickView = 'all'

    def visible_rows(self, results: Mapping[str, TraceClassification] | None = None) -> tuple[TraceRow, ...]:
        rows = self.rows
        if self.quick_view == 'errors':
            rows = tuple(row for row in rows if row.is_error)
        elif self.quick_view == 'matches':
            rows = tuple(
                row
                for row in rows
                if results is not None and (result := results.get(row.trace_id)) is not None and result.matched
            )
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


class ExplorerStore:
    """One replaceable load; a newer load's generation makes an older one's callbacks no-ops."""

    def __init__(self, *, search: RowSearch, hydrate: RowHydrator) -> None:
        self._search = search
        self._hydrate = hydrate
        self._view = ExplorerView()
        self._task: asyncio.Task[None] | None = None
        self._records: dict[str, TraceRecord | None] = {}
        self._records_lock = asyncio.Lock()

    async def load(
        self,
        start: datetime,
        end: datetime,
        limit: int,
        *,
        facets: FacetSelection,
        numeric: NumericFilters,
        wait: bool = False,
    ) -> ExplorerView:
        await self._cancel()
        self._records.clear()
        generation = self._view.generation + 1
        self._view = ExplorerView(
            generation=generation,
            state='loading',
            limit=limit,
            start=start,
            end=end,
            facets=facets,
            numeric=numeric,
            view=self._view.view,
        )

        def on_page(rows: tuple[TraceRow, ...]) -> None:
            if self._view.generation == generation:
                self._view = replace(self._view, rows=rows)

        async def run() -> None:
            try:
                rows = await self._search(start, end, limit, facets=facets, numeric=numeric, on_page=on_page)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 — keep rows loaded so far and name the failure
                if self._view.generation == generation:
                    logger.warning('Explorer load failed after {} row(s): {}', len(self._view.rows), error)
                    self._view = replace(self._view, state='failed', error=str(error))
                return
            if self._view.generation == generation:
                self._view = replace(self._view, state='loaded', rows=rows)

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
        matched_only: bool | None = None,
        quick_view: QuickView | None = None,
    ) -> ExplorerView:
        """Change display state; ``page`` is clamped when rendered because matches affect row count."""
        current = self._view
        self._view = replace(
            current,
            sort=sort if sort is not None else current.sort,
            descending=descending if descending is not None else current.descending,
            page=max(0, page if page is not None else current.page),
            view=view if view is not None else current.view,
            matched_only=matched_only if matched_only is not None else current.matched_only,
            quick_view=quick_view if quick_view is not None else current.quick_view,
        )
        return self._view

    async def records(self, trace_ids: Sequence[str]) -> dict[str, TraceRecord | None]:
        """Hydrate requested rows; successes and failures are cached until the next load."""
        async with self._records_lock:
            generation = self._view.generation
            requested = set(trace_ids)
            wanted = [row for row in self._view.rows if row.trace_id in requested and row.trace_id not in self._records]
            if wanted:
                hydrated = await self._hydrate(wanted)
                # A load may replace rows while hydration is in progress. Its cache belongs
                # to that load only, so late results from the previous generation are discarded.
                if self._view.generation == generation:
                    self._records.update(hydrated)
                    return {trace_id: self._records.get(trace_id) for trace_id in trace_ids}
                return {trace_id: hydrated.get(trace_id) for trace_id in trace_ids}
            return {trace_id: self._records.get(trace_id) for trace_id in trace_ids}

    async def row(self, trace_id: str) -> TraceRow | None:
        return next((row for row in self._view.rows if row.trace_id == trace_id), None)

    async def close(self) -> None:
        await self._cancel()
        self._records.clear()

    async def _cancel(self) -> None:
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
