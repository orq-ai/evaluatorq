from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import pytest

from evaluatorq.trace_finder import FacetSelection, NumericFilters
from evaluatorq.trace_finder.explorer import ExplorerStore
from evaluatorq.trace_finder.models import TraceClassification, TraceRecord
from evaluatorq.trace_finder.rows import TraceRow

START = datetime(2026, 9, 27, tzinfo=timezone.utc)
END = datetime(2026, 9, 27, 1, tzinfo=timezone.utc)


def record(trace_id: str) -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id=trace_id,
        span_id='s',
        timestamp=START,
        messages=({'role': 'user', 'content': 'hi'},),
        project='project',
        model='model',
        provider='provider',
        status='ok',
        product='product',
        trace_type='span.chat_completion',
    )


class FakeSource:
    def __init__(self, rows: tuple[TraceRow, ...], *, fail_after: int | None = None) -> None:
        self.rows = rows
        self.fail_after = fail_after
        self.hydrate_calls: list[tuple[str, ...]] = []
        self.missing: set[str] = set()

    async def search(
        self,
        start: datetime,
        end: datetime,
        limit: int,
        *,
        facets: Any,
        numeric: Any,
        on_page: Any = None,
    ) -> tuple[TraceRow, ...]:
        if self.fail_after is not None:
            on_page(self.rows[: self.fail_after])
            raise RuntimeError('page 2 timed out')
        on_page(self.rows)
        return self.rows

    async def hydrate_rows(self, rows: Any) -> dict[str, TraceRecord | None]:
        self.hydrate_calls.append(tuple(row.trace_id for row in rows))
        return {row.trace_id: None if row.trace_id in self.missing else record(row.trace_id) for row in rows}


def store_for(source: FakeSource) -> ExplorerStore:
    return ExplorerStore(search=source.search, hydrate=source.hydrate_rows)


@pytest.mark.asyncio
async def test_load_then_page_and_sort() -> None:
    rows = tuple(TraceRow(trace_id=f't{i}', tokens_in=i) for i in range(250))
    store = store_for(FakeSource(rows))

    view = await store.load(START, END, 250, facets=FacetSelection(), numeric=NumericFilters(), wait=True)
    assert view.state == 'loaded'
    assert view.page_count() == 3
    assert len(view.page_rows()) == 100

    view = await store.set_view(sort='tokens_in', descending=True, page=0)
    assert view.page_rows()[0].trace_id == 't249'


@pytest.mark.asyncio
async def test_matched_only_filters_with_results_and_is_ignored_without() -> None:
    rows = tuple(TraceRow(trace_id=f't{i}') for i in range(3))
    store = store_for(FakeSource(rows))
    await store.load(START, END, 3, facets=FacetSelection(), numeric=NumericFilters(), wait=True)
    view = await store.set_view(matched_only=True)
    results = {
        't1': TraceClassification(trace_id='t1', span_id='s', matched=True, raw_result={}),
        't2': TraceClassification(trace_id='t2', span_id='s', matched=False, raw_result={}),
    }
    assert [row.trace_id for row in view.visible_rows(results)] == ['t1']
    assert len(view.visible_rows(None)) == 3


@pytest.mark.asyncio
async def test_failed_load_keeps_partial_rows_and_error() -> None:
    rows = tuple(TraceRow(trace_id=f't{i}') for i in range(5))
    store = store_for(FakeSource(rows, fail_after=2))

    view = await store.load(START, END, 5, facets=FacetSelection(), numeric=NumericFilters(), wait=True)

    assert view.state == 'failed'
    assert [row.trace_id for row in view.rows] == ['t0', 't1']
    assert view.error is not None and 'page 2 timed out' in view.error


@pytest.mark.asyncio
async def test_records_caches_successes() -> None:
    source = FakeSource((TraceRow(trace_id='a'), TraceRow(trace_id='b')))
    store = store_for(source)
    await store.load(START, END, 2, facets=FacetSelection(), numeric=NumericFilters(), wait=True)

    await store.records(['a', 'b'])
    await store.records(['a', 'b'])

    assert source.hydrate_calls == [('a', 'b')]


@pytest.mark.asyncio
async def test_records_caches_failures_until_next_load() -> None:
    source = FakeSource((TraceRow(trace_id='a'),))
    source.missing = {'a'}
    store = store_for(source)
    await store.load(START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), wait=True)

    assert (await store.records(['a'])) == {'a': None}
    await store.records(['a'])

    assert source.hydrate_calls == [('a',)]


@pytest.mark.asyncio
async def test_stale_hydration_does_not_fill_the_next_load_cache() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    old_calls = 0

    async def slow_hydrate(rows: Any) -> dict[str, TraceRecord | None]:
        nonlocal old_calls
        old_calls += 1
        entered.set()
        await release.wait()
        return {row.trace_id: record(row.trace_id) for row in rows}

    new_source = FakeSource((TraceRow(trace_id='same'),))
    store = ExplorerStore(search=FakeSource((TraceRow(trace_id='same'),)).search, hydrate=slow_hydrate)
    await store.load(START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), wait=True)
    stale_request = asyncio.create_task(store.records(['same']))
    await entered.wait()

    store._search = new_source.search  # pyright: ignore[reportPrivateUsage]
    store._hydrate = new_source.hydrate_rows  # pyright: ignore[reportPrivateUsage]
    await store.load(START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), wait=True)
    release.set()
    await stale_request

    await store.records(['same'])
    assert old_calls == 1
    assert new_source.hydrate_calls == [('same',)]


@pytest.mark.asyncio
async def test_concurrent_records_requests_share_hydration() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    source = FakeSource((TraceRow(trace_id='same'),))

    async def slow_hydrate(rows: Any) -> dict[str, TraceRecord | None]:
        source.hydrate_calls.append(tuple(row.trace_id for row in rows))
        entered.set()
        await release.wait()
        return {row.trace_id: record(row.trace_id) for row in rows}

    store = ExplorerStore(search=source.search, hydrate=slow_hydrate)
    await store.load(START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), wait=True)

    first = asyncio.create_task(store.records(['same']))
    await entered.wait()
    second = asyncio.create_task(store.records(['same']))
    await asyncio.sleep(0)
    release.set()
    await asyncio.gather(first, second)

    assert source.hydrate_calls == [('same',)]


@pytest.mark.asyncio
async def test_new_load_supersedes_the_old_one() -> None:
    gate = asyncio.Event()

    class Slow(FakeSource):
        async def search(self, *args: Any, **kwargs: Any) -> tuple[TraceRow, ...]:
            await gate.wait()
            return await super().search(*args, **kwargs)

    store = ExplorerStore(search=Slow((TraceRow(trace_id='old'),)).search, hydrate=FakeSource(()).hydrate_rows)
    await store.load(START, END, 1, facets=FacetSelection(), numeric=NumericFilters())
    store._search = FakeSource((TraceRow(trace_id='new'),)).search  # pyright: ignore[reportPrivateUsage]
    view = await store.load(START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), wait=True)
    gate.set()
    await asyncio.sleep(0)

    assert [row.trace_id for row in (await store.view()).rows] == ['new']
    assert view.state == 'loaded'
