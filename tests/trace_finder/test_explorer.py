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
async def test_failed_hydration_is_retried_and_can_succeed() -> None:
    rows = (TraceRow(trace_id='retry-me'),)
    calls = 0

    async def hydrate(requested: Any) -> dict[str, TraceRecord | None]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {}
        return {row.trace_id: record(row.trace_id) for row in requested}

    store = ExplorerStore(search=FakeSource(rows).search, hydrate=hydrate)
    await store.load(START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), wait=True, warm_trajectories=False)

    assert await store.records(['retry-me']) == {'retry-me': None}
    assert not await store.has_cached_record('retry-me')
    assert await store.records(['retry-me']) == {'retry-me': record('retry-me')}
    assert calls == 2


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
async def test_summary_load_finishes_before_first_page_trajectory_warmup() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    rows = tuple(TraceRow(trace_id=f't{i}') for i in range(100))

    async def hydrate(rows: Any) -> dict[str, TraceRecord | None]:
        entered.set()
        await release.wait()
        return {row.trace_id: record(row.trace_id) for row in rows}

    store = ExplorerStore(search=FakeSource(rows).search, hydrate=hydrate)
    view = await store.load(START, END, 100, facets=FacetSelection(), numeric=NumericFilters(), wait=True)
    await entered.wait()

    assert view.state == 'loaded'
    assert view.trajectory_warming
    assert store.cached_records(['t0']) == {}

    release.set()
    prewarm_task = store._prewarm_task  # pyright: ignore[reportPrivateUsage]
    assert prewarm_task is not None
    await prewarm_task

    view = await store.view()
    assert view.state == 'loaded'
    assert not view.trajectory_warming
    assert store.cached_records(['t0']) == {'t0': record('t0')}


@pytest.mark.asyncio
async def test_failed_trajectory_warmup_clears_flag_and_can_retry() -> None:
    rows = tuple(TraceRow(trace_id=f't{i}') for i in range(100))
    calls = 0

    async def flaky_hydrate(rows: Any) -> dict[str, TraceRecord | None]:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(0)
            raise RuntimeError('trajectory store unavailable')
        return {row.trace_id: record(row.trace_id) for row in rows}

    store = ExplorerStore(search=FakeSource(rows).search, hydrate=flaky_hydrate)
    view = await store.load(START, END, 100, facets=FacetSelection(), numeric=NumericFilters(), wait=True)
    prewarm_task = store._prewarm_task  # pyright: ignore[reportPrivateUsage]
    assert prewarm_task is not None
    await prewarm_task

    view = await store.view()
    assert view.state == 'loaded'
    assert not view.trajectory_warming
    assert calls == 1
    assert await store.records(['t0']) == {'t0': record('t0')}
    assert calls == 2


@pytest.mark.asyncio
async def test_switching_quick_view_clears_sort() -> None:
    store = store_for(FakeSource((TraceRow(trace_id='a'),)))
    await store.load(START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), wait=True)
    await store.set_view(sort='tokens_in')
    view = await store.set_view(quick_view='slow')
    assert view.sort is None


@pytest.mark.asyncio
async def test_narrow_updates_rows_filters_and_page_only_for_current_generation() -> None:
    rows = tuple(TraceRow(trace_id=f't{i}') for i in range(3))
    store = store_for(FakeSource(rows))
    view = await store.load(START, END, 3, facets=FacetSelection(), numeric=NumericFilters(), wait=True)
    await store.set_view(page=2)

    facets = FacetSelection(model=frozenset({'gpt-5'}))
    numeric = NumericFilters(tokens_min=100)
    await store.narrow(view.generation, {'t1'}, facets=facets, numeric=numeric)

    narrowed = await store.view()
    assert [row.trace_id for row in narrowed.rows] == ['t1']
    assert narrowed.facets == facets
    assert narrowed.numeric == numeric
    assert narrowed.page == 0
    assert narrowed.narrowed_from == 3

    await store.narrow(view.generation, {'t1'}, facets=facets, numeric=numeric)
    assert (await store.view()).narrowed_from == 3

    await store.narrow(view.generation - 1, {'t2'}, facets=FacetSelection(), numeric=NumericFilters())
    assert await store.view() == narrowed


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
    await store.load(START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), wait=True, warm_trajectories=False)
    stale_request = asyncio.create_task(store.records(['same']))
    await entered.wait()

    store._search = new_source.search  # pyright: ignore[reportPrivateUsage]
    store._hydrate = new_source.hydrate_rows  # pyright: ignore[reportPrivateUsage]
    await store.load(START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), wait=True, warm_trajectories=False)
    release.set()
    await stale_request

    await store.records(['same'])
    assert old_calls == 1
    assert new_source.hydrate_calls == [('same',)]


@pytest.mark.asyncio
async def test_message_counts_ignores_stale_generation() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_hydrate(rows: Any) -> dict[str, TraceRecord | None]:
        entered.set()
        await release.wait()
        return {row.trace_id: record(row.trace_id) for row in rows}

    store = ExplorerStore(search=FakeSource((TraceRow(trace_id='old'),)).search, hydrate=slow_hydrate)
    await store.load(START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), wait=True, warm_trajectories=False)
    counts = asyncio.create_task(store.message_counts())
    await entered.wait()
    store._search = FakeSource((TraceRow(trace_id='new'),)).search  # pyright: ignore[reportPrivateUsage]
    await store.load(START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), wait=True, warm_trajectories=False)
    release.set()
    await counts

    assert (await store.view()).message_counts == {}


@pytest.mark.asyncio
async def test_message_counts_fill_from_hydrated_records() -> None:
    rows = tuple(TraceRow(trace_id=trace_id) for trace_id in ('a', 'b', 'c'))

    async def hydrate(rows: Any) -> dict[str, TraceRecord | None]:
        sizes = {'a': 3, 'b': 7}
        return {
            row.trace_id: record(row.trace_id).model_copy(update={'messages': ({'role': 'user'},) * sizes[row.trace_id]})
            if row.trace_id in sizes
            else None
            for row in rows
        }

    store = ExplorerStore(search=FakeSource(rows).search, hydrate=hydrate)
    await store.load(START, END, 3, facets=FacetSelection(), numeric=NumericFilters(), wait=True)
    view = await store.message_counts()
    assert view.message_counts == {'a': 3, 'b': 7}


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
    await store.load(START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), wait=True, warm_trajectories=False)

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


def test_top_share_views_keep_heaviest_tenth_and_skip_rows_without_the_metric() -> None:
    from evaluatorq.trace_finder.explorer import ExplorerView
    from evaluatorq.trace_finder.rows import TraceRow

    rows = tuple(
        TraceRow(trace_id=f't{i}', duration_ms=i * 10, cost_total=None if i == 19 else i / 100, tokens_in=i, tokens_out=None)
        for i in range(20)
    )
    slow = ExplorerView(state='loaded', rows=rows, quick_view='slow').visible_rows()
    assert [row.trace_id for row in slow] == ['t19', 't18']
    costly = ExplorerView(state='loaded', rows=rows, quick_view='costly').visible_rows()
    assert [row.trace_id for row in costly] == ['t18', 't17']
    assert ExplorerView(state='loaded', rows=(TraceRow(trace_id='x'),), quick_view='tokens').visible_rows() == ()


def test_largest_context_view_keeps_largest_tenth_and_skips_missing_context() -> None:
    from evaluatorq.trace_finder.explorer import ExplorerView

    rows = tuple(TraceRow(trace_id=f't{i}', context_tokens=None if i == 19 else i) for i in range(20))

    visible = ExplorerView(state='loaded', rows=rows, quick_view='context').visible_rows()

    assert [row.trace_id for row in visible] == ['t18', 't17']


def test_top_conversations_groups_falls_back_excludes_unvalued_and_orders_rows() -> None:
    from evaluatorq.trace_finder.explorer import top_conversations

    rows = [
        TraceRow(
            trace_id='thread-late',
            thread_id='thread',
            started_at=START.replace(hour=2),
            cost_total=1,
        ),
        TraceRow(
            trace_id='thread-early',
            thread_id='thread',
            started_at=START.replace(hour=1),
            cost_total=4,
        ),
        TraceRow(trace_id='session-top', session_id='session', cost_total=3),
        TraceRow(trace_id='unidentified', cost_total=100),
        TraceRow(trace_id='unvalued', thread_id='unvalued'),
    ]
    rows.extend(TraceRow(trace_id=f'low-{i}', session_id=f'low-{i}', cost_total=0.1) for i in range(9))

    selected = top_conversations(rows, lambda group, _counts: sum(row.cost_total or 0 for row in group))

    assert [row.trace_id for row in selected] == ['thread-early', 'thread-late', 'session-top']

    no_metric = top_conversations(
        [TraceRow(trace_id='missing', thread_id='missing')],
        lambda group, _counts: None,
    )
    assert no_metric == ()


def test_longest_conversations_use_maximum_message_count_and_skip_missing_records() -> None:
    from evaluatorq.trace_finder.explorer import ExplorerView

    rows = tuple(
        TraceRow(trace_id=trace_id, thread_id=conversation)
        for trace_id, conversation in (
            ('a-early', 'a'), ('a-late', 'a'), ('b', 'b'), ('c', 'c'), ('missing', 'missing'),
        )
    )
    view = ExplorerView(
        state='loaded',
        rows=rows,
        quick_view='conv_longest',
        message_counts={'a-early': 3, 'a-late': 7, 'b': 6, 'c': 1},
    )

    assert [row.trace_id for row in view.visible_rows()] == ['a-early', 'a-late']


@pytest.mark.asyncio
async def test_span_loader_is_lazy_and_uses_injected_source() -> None:
    calls: list[str] = []

    async def load_spans(trace_id: str) -> list[Any]:
        calls.append(trace_id)
        return [{'span_id': 's1'}]

    source = FakeSource(())
    store = ExplorerStore(search=source.search, hydrate=source.hydrate_rows, load_spans=load_spans)
    assert calls == []
    assert await store.spans('trace-1') == [{'span_id': 's1'}]
    assert calls == ['trace-1']


@pytest.mark.asyncio
async def test_view_version_grows_only_on_visible_changes_and_generations_are_unique_per_load() -> None:
    rows = tuple(TraceRow(trace_id=f't{i}') for i in range(3))
    first = store_for(FakeSource(rows))
    second = store_for(FakeSource(rows))
    loaded = await first.load(START, END, 3, facets=FacetSelection(), numeric=NumericFilters(), wait=True)
    other = await second.load(START, END, 3, facets=FacetSelection(), numeric=NumericFilters(), wait=True)
    # Two sessions' stores never share a generation, so per-load caches cannot collide.
    assert loaded.generation != other.generation
    assert loaded.version == loaded.generation

    # A poll calls set_view with nothing to change; that must not look like a newer view.
    assert (await first.set_view()).version == loaded.version
    sorted_view = await first.set_view(sort='tokens_in')
    assert sorted_view.version > loaded.version
    assert (await first.set_view(sort='tokens_in')).version == sorted_view.version
    assert (await first.set_view(quick_view='errors')).version > sorted_view.version
