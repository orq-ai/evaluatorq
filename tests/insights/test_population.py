"""Unit tests for `evaluatorq.insights.population` — resolving a population via the trace finder."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from evaluatorq.common.trace_document import TraceDocument, ensure_trace_document
from evaluatorq.insights import population as population_module
from evaluatorq.insights.models import InsightsPopulation
from evaluatorq.insights.population import PopulationError, resolve_population
from evaluatorq.trace_finder.compiler import CompiledPlan
from evaluatorq.trace_finder.export import (
    ExportCounts,
    ExportDimension,
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


def make_document(trace_id: str) -> TraceDocument:
    """A session-style ATIF document that is not a `TraceRecord`."""
    return ensure_trace_document(make_trace(trace_id))


@pytest.mark.asyncio
async def test_local_snapshot_resolves_records_and_session_documents(tmp_path: Path) -> None:
    path = tmp_path / 'traces.json'
    path.write_text(
        Snapshot(traces=(make_trace('record-1'),), documents=(make_document('session-1'),)).model_dump_json(),
        encoding='utf-8',
    )
    resolved = await resolve_population(
        InsightsPopulation.from_snapshot(path),
        orq=_orq(),
        client=_client(),
        compiler_model='compiler',
        classifier_model='classifier',
    )
    assert [trace.metadata.trace_id for trace in resolved.traces] == ['record-1', 'session-1']
    assert resolved.echo['mode'] == 'snapshot'
    assert resolved.echo['snapshot_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert resolved.n_scanned == 2


def test_preview_snapshot_counts_documents(tmp_path: Path) -> None:
    path = tmp_path / 'traces.json'
    path.write_text(Snapshot(traces=(), documents=(make_document('session-1'),)).model_dump_json(), encoding='utf-8')
    assert population_module.preview_snapshot(str(path))['n_traces'] == 1


def test_describe_local_send_names_count_size_file_and_models() -> None:
    text = population_module.describe_local_send(
        n_traces=3,
        n_bytes=1536,
        file_name='s.json',
        models={'summary': 'a/s', 'classifier': 'a/c', 'embedding': 'a/e'},
    )
    assert text == 'Sending 3 traces (1.5 KB) from s.json to models: summary a/s, classifier a/c, embedding a/e.'
    big = population_module.describe_local_send(
        n_traces=1, n_bytes=3 * 1024 * 1024, file_name='s.json', models={'summary': 'a', 'classifier': 'b', 'embedding': 'c'}
    )
    assert '(3.0 MB)' in big



def _orq() -> Any:
    return object()


class FakeSource:
    """Fake `OrqTraceSource` recording every `load_async` call for merge/window assertions."""

    calls: list[dict[str, Any]] = []
    snapshot: Snapshot = Snapshot(traces=())
    closed = False
    enrichment_calls: list[tuple[TraceRecord, ...]] = []

    def __init__(self, orq: Any) -> None:
        del orq

    async def load_async(
        self,
        start: Any,
        end: Any,
        limit: Any,
        *,
        facets: FacetSelection,
        numeric: NumericFilters,
        target_trace_ids: set[str] | frozenset[str] | None = None,
    ) -> Snapshot:
        FakeSource.calls.append({
            'start': start,
            'end': end,
            'limit': limit,
            'facets': facets,
            'numeric': numeric,
            'target_trace_ids': target_trace_ids,
        })
        return FakeSource.snapshot

    async def enrich_selected_signal_spans(self, records: tuple[TraceRecord, ...]) -> tuple[TraceRecord, ...]:
        FakeSource.enrichment_calls.append(records)
        return records

    def close(self) -> None:
        FakeSource.closed = True


@pytest.fixture(autouse=True)
def _reset_fake_source() -> None:
    FakeSource.calls = []
    FakeSource.snapshot = Snapshot(traces=())
    FakeSource.closed = False
    FakeSource.enrichment_calls = []


@pytest.mark.asyncio
async def test_targeted_reload_rejects_incomplete_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(population_module, 'OrqTraceSource', FakeSource)
    FakeSource.snapshot = Snapshot(
        traces=(make_trace('t1'),), capture_metadata={'incomplete_reason': 'target_deadline'}
    )

    with pytest.raises(PopulationError, match='reloading Finder export traces was incomplete: target_deadline'):
        await population_module._load_traces(
            _orq(),
            start=None,
            end=None,
            limit=2,
            facets=FacetSelection(),
            numeric=NumericFilters(),
            target_trace_ids={'t1', 't2'},
        )
    assert FakeSource.closed


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
    assert all(isinstance(trace, TraceDocument) for trace in resolved.traces)
    assert FakeSource.enrichment_calls == [traces]
    assert resolved.echo['mode'] == 'filter'
    assert FakeSource.calls[0]['facets'].agent_name == frozenset({'support-bot'})
    assert FakeSource.closed is True


@pytest.mark.asyncio
async def test_local_snapshot_uses_messages_without_orq(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / 'traces.json'
    path.write_text(Snapshot(traces=(make_trace('local-1'),)).model_dump_json(), encoding='utf-8')

    def unexpected_orq_source(_orq: Any) -> None:
        raise AssertionError('local snapshots must not fetch Orq traces')

    monkeypatch.setattr(population_module, 'OrqTraceSource', unexpected_orq_source)
    resolved = await resolve_population(
        InsightsPopulation.from_snapshot(path),
        orq=_orq(),
        client=_client(),
        compiler_model='compiler',
        classifier_model='classifier',
    )

    assert [trace.trace_id for trace in resolved.traces] == ['local-1']
    assert isinstance(resolved.traces[0], TraceDocument)
    assert resolved.traces[0].messages[0]['content'] == 'hello'
    assert resolved.echo['mode'] == 'snapshot'
    assert resolved.echo['snapshot_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert resolved.echo['n_projection_truncated'] == 0
    assert resolved.echo['n_source_messages'] == 1
    assert resolved.echo['n_omitted_messages'] == 0


@pytest.mark.asyncio
async def test_local_snapshot_reports_truncated_projection(tmp_path: Path) -> None:
    path = tmp_path / 'traces.json'
    long_trace = make_trace('long').model_copy(
        update={'messages': ({'role': 'user', 'content': 'a' * 600_000}, {'role': 'assistant', 'content': 'done'})}
    )
    path.write_text(Snapshot(traces=(long_trace,)).model_dump_json(), encoding='utf-8')

    resolved = await resolve_population(
        InsightsPopulation.from_snapshot(path),
        orq=_orq(),
        client=_client(),
        compiler_model='compiler',
        classifier_model='classifier',
    )

    assert resolved.echo['n_projection_truncated'] == 1
    assert resolved.echo['n_source_messages'] == 2
    assert resolved.echo['n_omitted_messages'] == 1
    assert resolved.echo['source_bytes'] > resolved.echo['projected_bytes']


@pytest.mark.asyncio
async def test_live_population_reports_the_same_projection_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(population_module, 'OrqTraceSource', FakeSource)
    FakeSource.snapshot = Snapshot(traces=(
        make_trace('long').model_copy(
            update={'messages': ({'role': 'user', 'content': 'a' * 600_000}, {'role': 'assistant', 'content': 'done'})}
        ),
    ))

    resolved = await resolve_population(
        InsightsPopulation(), orq=_orq(), client=_client(), compiler_model='compiler', classifier_model='classifier'
    )

    assert resolved.echo['mode'] == 'filter'
    assert resolved.echo['n_projection_truncated'] == 1
    assert resolved.echo['n_omitted_messages'] == 1


@pytest.mark.asyncio
async def test_live_population_rejects_duplicate_trace_ids_before_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(population_module, 'OrqTraceSource', FakeSource)
    FakeSource.snapshot = Snapshot(
        traces=(make_trace('duplicate'), make_trace('duplicate').model_copy(update={'span_id': 'another-span'}))
    )

    with pytest.raises(PopulationError, match='select one span per trace'):
        await resolve_population(
            InsightsPopulation(), orq=_orq(), client=_client(), compiler_model='compiler', classifier_model='classifier'
        )

    assert FakeSource.closed


@pytest.mark.asyncio
async def test_invalid_local_snapshot_reports_population_failure(tmp_path: Path) -> None:
    path = tmp_path / 'traces.json'
    path.write_text('{"traces": [{"trace_id": "missing-fields"}]}', encoding='utf-8')
    with pytest.raises(PopulationError, match='loading local trace snapshot'):
        await resolve_population(
            InsightsPopulation.from_snapshot(path),
            orq=_orq(),
            client=_client(),
            compiler_model='compiler',
            classifier_model='classifier',
        )


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
        return CompiledPlan(dimensions=(compiled_query,), numeric=NumericFilters(tokens_min=5, tokens_max=100))

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

    assert resolved.compiled == (compiled_query,)
    assert resolved.echo['compiled_dimensions'][0]['name'] == 'AI match'
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
        return CompiledPlan(dimensions=(compiled_query,), numeric=NumericFilters())

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

    assert resolved.compiled == (compiled_query,)
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


@pytest.mark.asyncio
async def test_cleanup_failure_keeps_the_loaded_traces_and_the_original_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BadCloseSource(FakeSource):
        def close(self) -> None:
            raise RuntimeError('hook unregistration failed')

    monkeypatch.setattr(population_module, 'OrqTraceSource', BadCloseSource)
    FakeSource.snapshot = Snapshot(traces=(make_trace('t1'),))
    kwargs: dict[str, Any] = {
        'start': None,
        'end': None,
        'limit': 5,
        'facets': FacetSelection(),
        'numeric': NumericFilters(),
    }

    traces = await population_module._load_traces(_orq(), **kwargs)
    assert [t.trace_id for t in traces] == ['t1']

    class FailingBadCloseSource(BadCloseSource):
        async def load_async(self, *args: Any, **kwargs: Any) -> Snapshot:
            raise RuntimeError('live load failed')

    monkeypatch.setattr(population_module, 'OrqTraceSource', FailingBadCloseSource)
    with pytest.raises(PopulationError, match='live load failed'):
        await population_module._load_traces(_orq(), **kwargs)


def _run_export(matched_trace_ids: list[str], *, filters: ExportFilters | None = None) -> RunExport:
    return RunExport(
        query='refund requests',
        dimensions=(ExportDimension(name='intent', task=ExportTask(kind='choice', instructions='classify', criteria={'billing': 'billing help'}, state={}, noul_threshold=0.5), selection=ExportValuesSelection(kind='values', values=('billing',))),),
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
async def test_export_path_echo_records_an_absolute_path(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'export.json').write_text(_run_export(['t1']).model_dump_json())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(population_module, 'OrqTraceSource', FakeSource)
    FakeSource.snapshot = Snapshot(traces=(make_trace('t1'),))

    resolved = await resolve_population(
        InsightsPopulation.from_finder_export(Path('export.json')),
        orq=_orq(),
        client=_client(),
        compiler_model='compiler',
        classifier_model='classifier',
    )

    recorded = Path(resolved.echo['finder_export'])
    assert recorded.is_absolute()
    assert recorded == (tmp_path / 'export.json').resolve()


@pytest.mark.asyncio
async def test_export_path_reloads_merged_filters_and_pinned_time_range(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = _run_export(['t1'], filters=ExportFilters(agent_name=('explicit-bot',)))
    export = export.model_copy(
        update={
            'generated_filters': ExportFilters(agent_name=('generated-bot',), model=('gpt-5',)),
            'generated_numeric': ExportNumericFilters(tokens_min=20, tokens_max=300),
            'start': None,
            'end': None,
        }
    )
    export_path = tmp_path / 'export.json'
    export_path.write_text(export.model_dump_json())
    monkeypatch.setattr(population_module, 'OrqTraceSource', FakeSource)
    FakeSource.snapshot = Snapshot(traces=(make_trace('t1'),))

    resolved = await resolve_population(
        InsightsPopulation.from_finder_export(export_path),
        orq=_orq(),
        client=_client(),
        compiler_model='compiler',
        classifier_model='classifier',
    )

    call = FakeSource.calls[0]
    assert call['facets'].agent_name == frozenset({'explicit-bot'})
    assert call['facets'].model == frozenset({'gpt-5'})
    assert call['numeric'].tokens_min == 5
    assert call['numeric'].tokens_max == 300
    assert call['start'] < export.traces[0].timestamp < call['end']
    assert call['limit'] == export.limit
    assert call['target_trace_ids'] == {'t1'}
    assert [trace.trace_id for trace in resolved.traces] == ['t1']


@pytest.mark.asyncio
async def test_export_path_with_no_matched_ids_skips_live_reload(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    export_path = tmp_path / 'export.json'
    export_path.write_text(_run_export([]).model_dump_json())
    monkeypatch.setattr(population_module, 'OrqTraceSource', FakeSource)

    resolved = await resolve_population(
        InsightsPopulation.from_finder_export(export_path),
        orq=_orq(),
        client=_client(),
        compiler_model='compiler',
        classifier_model='classifier',
    )

    assert resolved.traces == []
    assert resolved.n_scanned == 0
    assert FakeSource.calls == []


@pytest.mark.asyncio
async def test_export_path_uses_validated_snapshot_if_file_changes(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    export = _run_export(['t1'])
    export_path = tmp_path / 'export.json'
    export_path.write_text(export.model_dump_json())
    population = InsightsPopulation.from_finder_export(export_path, export=export)
    export_path.write_text('not json')
    monkeypatch.setattr(population_module, 'OrqTraceSource', FakeSource)
    FakeSource.snapshot = Snapshot(traces=(make_trace('t1'),))

    resolved = await resolve_population(
        population, orq=_orq(), client=_client(), compiler_model='compiler', classifier_model='classifier'
    )

    assert [trace.trace_id for trace in resolved.traces] == ['t1']


@pytest.mark.asyncio
async def test_export_path_bad_json_raises_population_error(tmp_path: Any) -> None:
    export_path = tmp_path / 'export.json'
    export_path.write_text('not json')

    pop = InsightsPopulation.from_finder_export(export_path)

    with pytest.raises(PopulationError, match='loading finder export'):
        await resolve_population(pop, orq=_orq(), client=_client(), compiler_model='compiler', classifier_model='classifier')
