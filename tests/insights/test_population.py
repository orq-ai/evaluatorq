"""Unit tests for `evaluatorq.insights.population` — resolving a population via the trace finder."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from evaluatorq.insights import population as population_module
from evaluatorq.insights.models import InsightsPopulation
from evaluatorq.insights.population import PopulationError, resolve_population
from evaluatorq.trace_finder.compiler import CompiledPlan
from evaluatorq.trace_finder.export import (
    ExportCounts,
    ExportFilters,
    ExportNumericFilters,
    ExportTask,
    ExportTimes,
    ExportTrace,
    ExportValuesSelection,
    RunExport,
)
from evaluatorq.trace_finder.filter_selector import FilterSelectionResult
from evaluatorq.trace_finder.models import (
    CompiledQuery,
    FacetCatalogue,
    FacetSelection,
    NumericFilters,
    Snapshot,
    TraceRecord,
    ValueSelection,
)


def make_trace(trace_id: str) -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id=trace_id,
        span_id=f'span-{trace_id}',
        timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
        messages=({'role': 'user', 'content': 'hello'},),
        project='default',
        model='gpt-5',
        provider='openai',
        status='completed',
        product='chat',
        trace_type='conversation',
    )


def _client() -> Any:
    return object()


def _orq() -> Any:
    return object()


class FakeSource:
    """Fake `OrqTraceSource` recording every `load_async` call for merge/window assertions."""

    calls: list[dict[str, Any]] = []
    snapshot: Snapshot = Snapshot(traces=())
    closed = False

    def __init__(self, orq: Any) -> None:
        del orq

    async def load_async(
        self, start: Any, end: Any, limit: Any, *, facets: FacetSelection, numeric: NumericFilters
    ) -> Snapshot:
        FakeSource.calls.append({'start': start, 'end': end, 'limit': limit, 'facets': facets, 'numeric': numeric})
        return FakeSource.snapshot

    def close(self) -> None:
        FakeSource.closed = True


@pytest.fixture(autouse=True)
def _reset_fake_source() -> None:
    FakeSource.calls = []
    FakeSource.snapshot = Snapshot(traces=())
    FakeSource.closed = False


@pytest.mark.asyncio
async def test_filter_only_path_loads_directly_with_no_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(population_module, 'OrqTraceSource', FakeSource)
    traces = (make_trace('t1'), make_trace('t2'))
    FakeSource.snapshot = Snapshot(traces=traces)

    pop = InsightsPopulation(facets=FacetSelection(agent_name=frozenset({'support-bot'})), limit=50)
    resolved = await resolve_population(
        pop, orq=_orq(), client=_client(), compiler_model='compiler', classifier_model='classifier'
    )

    assert resolved.compiled is None
    assert [t.trace_id for t in resolved.traces] == ['t1', 't2']
    assert resolved.n_scanned == 2
    assert resolved.echo['mode'] == 'filter'
    assert FakeSource.calls[0]['facets'].agent_name == frozenset({'support-bot'})
    assert FakeSource.closed is True


@pytest.mark.asyncio
async def test_query_path_merges_explicit_facets_and_numeric_over_generated(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.common.judge import ClassifyQuestion

    compiled_query = CompiledQuery(
        task=ClassifyQuestion(
            kind='choice', instructions='classify', criteria={'billing': 'billing help', 'technical': 'tech help'}, state={}
        ),
        selection=ValueSelection(kind='values', values=('billing',)),
    )

    async def fake_compile_query(client: Any, model: str, query: str, *, cfg: Any = None) -> CompiledPlan:
        return CompiledPlan(compiled=compiled_query, numeric=NumericFilters(tokens_min=5, tokens_max=100))

    async def fake_load_facet_catalogue(orq: Any, *, start: Any, end: Any, limit: int) -> FacetCatalogue:
        return FacetCatalogue()

    async def fake_select_filters_with_response(
        client: Any, model: str, catalogue: FacetCatalogue, query: str, *, cfg: Any = None
    ) -> FilterSelectionResult:
        return FilterSelectionResult(FacetSelection(agent_name=frozenset({'generated-bot'}), model=frozenset({'gpt-5'})))

    monkeypatch.setattr(population_module, 'compile_query', fake_compile_query)
    monkeypatch.setattr(population_module, 'load_facet_catalogue', fake_load_facet_catalogue)
    monkeypatch.setattr(population_module, 'select_filters_with_response', fake_select_filters_with_response)
    monkeypatch.setattr(population_module, 'OrqTraceSource', FakeSource)
    FakeSource.snapshot = Snapshot(traces=(make_trace('t1'),))

    # Explicit agent_name wins over generated; explicit tokens_min is unset so the generated bound wins.
    pop = InsightsPopulation(
        query='refund requests', facets=FacetSelection(agent_name=frozenset({'explicit-bot'})), limit=50
    )
    resolved = await resolve_population(
        pop, orq=_orq(), client=_client(), compiler_model='compiler', classifier_model='classifier'
    )

    assert resolved.compiled == compiled_query
    call = FakeSource.calls[0]
    assert call['facets'].agent_name == frozenset({'explicit-bot'})
    assert call['facets'].model == frozenset({'gpt-5'})
    assert call['numeric'].tokens_min == 5
    assert call['numeric'].tokens_max == 100
    assert resolved.echo['mode'] == 'query'
    assert resolved.echo['query'] == 'refund requests'


@pytest.mark.asyncio
async def test_query_path_degrades_when_filter_selection_fails(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from evaluatorq.common.judge import ClassifyQuestion

    compiled_query = CompiledQuery(
        task=ClassifyQuestion(
            kind='choice', instructions='classify', criteria={'billing': 'billing help', 'technical': 'tech help'}, state={}
        ),
        selection=ValueSelection(kind='values', values=('billing',)),
    )

    async def fake_compile_query(client: Any, model: str, query: str, *, cfg: Any = None) -> CompiledPlan:
        return CompiledPlan(compiled=compiled_query, numeric=NumericFilters())

    async def unavailable_catalogue(orq: Any, *, start: Any, end: Any, limit: int) -> FacetCatalogue:
        raise RuntimeError('facet service unavailable')

    monkeypatch.setattr(population_module, 'compile_query', fake_compile_query)
    monkeypatch.setattr(population_module, 'load_facet_catalogue', unavailable_catalogue)
    monkeypatch.setattr(population_module, 'OrqTraceSource', FakeSource)
    FakeSource.snapshot = Snapshot(traces=())

    pop = InsightsPopulation(query='refund requests', limit=50)

    with caplog.at_level('WARNING'):
        resolved = await resolve_population(
            pop, orq=_orq(), client=_client(), compiler_model='compiler', classifier_model='classifier'
        )

    assert resolved.compiled == compiled_query
    assert resolved.echo['filter_selection_error'] == 'facet service unavailable'
    assert 'facet service unavailable' in caplog.text


@pytest.mark.asyncio
async def test_query_path_compile_failure_raises_population_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failing_compile_query(client: Any, model: str, query: str, *, cfg: Any = None) -> CompiledPlan:
        raise ValueError('the compiler produced no structured document')

    monkeypatch.setattr(population_module, 'compile_query', failing_compile_query)
    pop = InsightsPopulation(query='refund requests', limit=50)

    with pytest.raises(PopulationError, match='compiling the population query failed'):
        await resolve_population(pop, orq=_orq(), client=_client(), compiler_model='compiler', classifier_model='classifier')


@pytest.mark.asyncio
async def test_loader_failure_raises_population_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingSource:
        def __init__(self, orq: Any) -> None:
            del orq

        async def load_async(self, *args: Any, **kwargs: Any) -> Snapshot:
            raise RuntimeError('live Orq trace loading failed')

        def close(self) -> None:
            pass

    monkeypatch.setattr(population_module, 'OrqTraceSource', FailingSource)
    pop = InsightsPopulation(limit=50)

    with pytest.raises(PopulationError, match='loading the trace population failed'):
        await resolve_population(pop, orq=_orq(), client=_client(), compiler_model='compiler', classifier_model='classifier')


def _run_export(matched_trace_ids: list[str], *, filters: ExportFilters | None = None) -> RunExport:
    return RunExport(
        query='refund requests',
        task=ExportTask(kind='choice', instructions='classify', criteria={'billing': 'billing help'}, state={}, noul_threshold=0.5),
        selection=ExportValuesSelection(kind='values', values=('billing',)),
        generated_filters=ExportFilters(),
        filters=filters if filters is not None else ExportFilters(agent_name=('support-bot',)),
        generated_numeric=ExportNumericFilters(),
        numeric=ExportNumericFilters(tokens_min=5),
        start=datetime(2026, 8, 25, tzinfo=timezone.utc),
        end=datetime(2026, 9, 1, tzinfo=timezone.utc),
        limit=500,
        parallelism=100,
        times=ExportTimes(elapsed=1.0, rate=1.0),
        counts=ExportCounts(total=3, completed=3, failed=0, matched=len(matched_trace_ids), active=0, queued=0, percent=100.0),
        traces=tuple(
            ExportTrace(trace_id=trace_id, span_id=f'span-{trace_id}', timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc), matched=True)
            for trace_id in matched_trace_ids
        ),
        matched_trace_ids=matched_trace_ids,
    )


@pytest.mark.asyncio
async def test_export_path_keeps_matched_only_and_warns_on_missing_ids(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    export = _run_export(['t1', 't2', 't3'])
    export_path = tmp_path / 'export.json'
    export_path.write_text(export.model_dump_json())

    monkeypatch.setattr(population_module, 'OrqTraceSource', FakeSource)
    # t3 is not found in the reloaded window/filters.
    FakeSource.snapshot = Snapshot(traces=(make_trace('t1'), make_trace('t2')))

    pop = InsightsPopulation.from_finder_export(export_path)

    with caplog.at_level('WARNING'):
        resolved = await resolve_population(
            pop, orq=_orq(), client=_client(), compiler_model='compiler', classifier_model='classifier'
        )

    assert resolved.compiled is None
    assert [t.trace_id for t in resolved.traces] == ['t1', 't2']
    assert resolved.echo['mode'] == 'export'
    assert resolved.echo['n_missing_export_ids'] == 1
    assert 'could not find 1' in caplog.text
    call = FakeSource.calls[0]
    assert call['facets'].agent_name == frozenset({'support-bot'})
    assert call['numeric'].tokens_min == 5


@pytest.mark.asyncio
async def test_export_path_bad_json_raises_population_error(tmp_path: Any) -> None:
    export_path = tmp_path / 'export.json'
    export_path.write_text('not json')

    pop = InsightsPopulation.from_finder_export(export_path)

    with pytest.raises(PopulationError, match='loading finder export'):
        await resolve_population(pop, orq=_orq(), client=_client(), compiler_model='compiler', classifier_model='classifier')
