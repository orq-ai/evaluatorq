"""Unit tests for `evaluatorq.insights.population` — resolving a population via the trace finder."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import pytest

from evaluatorq.common.trace_document import TraceDocument, ensure_trace_document
from evaluatorq.insights import population as population_module
from evaluatorq.insights.models import InsightsPopulation
from evaluatorq.insights.population import PopulationError, resolve_population
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
from evaluatorq.trace_finder.models import (
    CompiledQuery,
    FacetSelection,
    NumericFilters,
    PopulationRequest,
    Snapshot,
    TraceRecord,
    ValueSelection,
)
from evaluatorq.trace_finder.planner import PlanningResult


def make_trace(trace_id: str, *, span_id: str | None = None, content: str = 'hello') -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id=trace_id,
        span_id=span_id or f'span-{trace_id}',
        timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
        messages=({'role': 'user', 'content': content},),
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


@pytest.mark.parametrize('classifier_model', ['openai/gpt-6-luna', 'typesafe/jev-latest'])
def test_preview_snapshot_counts_documents_and_uses_selected_renderers(
    tmp_path: Path, classifier_model: str
) -> None:
    path = tmp_path / 'traces.json'
    document = ensure_trace_document(
        make_trace('session-1').model_copy(
            update={'messages': ({'role': 'user', 'content': 'session request ' * 10_000},)}
        )
    )
    snapshot = Snapshot(traces=(make_trace('record-1'),), documents=(document,))
    path.write_text(snapshot.model_dump_json(), encoding='utf-8')
    questions = {'label': {'kind': 'noul', 'instructions': 'q' * 3_000}}
    options: dict[str, Any] = {
        'trace_input_chars': 10_000,
        'classifier_model': classifier_model,
        'classifier_questions': questions,
    }

    coverage = population_module.preview_snapshot(str(path), **options)

    assert coverage == population_module.projection_coverage(
        population_module.snapshot_documents(snapshot), **options
    )
    assert coverage['n_traces'] == 2
    assert coverage['classifier_question_reserve'] == 'applied'
    assert coverage['n_classifier_truncated'] == 1
    assert coverage['classifier_request_chars'] <= 2 * options['trace_input_chars']


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
    long_body = 'long trace body with stable context. ' * 10_000
    long_trace = make_trace('long').model_copy(
        update={'messages': ({'role': 'user', 'content': long_body}, {'role': 'assistant', 'content': long_body}, {'role': 'user', 'content': long_body})}
    )
    path.write_text(Snapshot(traces=(long_trace,)).model_dump_json(), encoding='utf-8')

    resolved = await resolve_population(
        InsightsPopulation.from_snapshot(path),
        orq=_orq(),
        client=_client(),
        compiler_model='compiler',
        classifier_model='classifier',
        label_classifier_model='typesafe/jev-latest',
    )

    assert resolved.echo['n_projection_truncated'] == 1
    assert resolved.echo['n_source_messages'] == 3
    assert resolved.echo['n_omitted_messages'] == 1
    assert resolved.echo['source_chars'] > resolved.echo['classifier_state_chars']


@pytest.mark.asyncio
async def test_live_population_reports_the_same_projection_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(population_module, 'OrqTraceSource', FakeSource)
    long_body = 'long trace body with stable context. ' * 10_000
    FakeSource.snapshot = Snapshot(traces=(
        make_trace('long').model_copy(
            update={'messages': ({'role': 'user', 'content': long_body}, {'role': 'assistant', 'content': long_body}, {'role': 'user', 'content': long_body})}
        ),
    ))

    resolved = await resolve_population(
        InsightsPopulation(),
        orq=_orq(),
        client=_client(),
        compiler_model='compiler',
        classifier_model='classifier',
        label_classifier_model='typesafe/jev-latest',
    )

    assert resolved.echo['mode'] == 'filter'
    assert resolved.echo['n_projection_truncated'] == 1
    assert resolved.echo['n_omitted_messages'] == 1




def test_projection_coverage_uses_the_actual_non_jev_insights_classifier_state() -> None:
    from evaluatorq.insights.labeling import prepare_classifier_state

    trace = make_trace('input').model_copy(
        update={
            'messages': (
                {'role': 'user', 'content': 'REQUEST-START'},
                {'role': 'assistant', 'tool_calls': [{'id': 'call-1', 'function': {'name': 'lookup', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': 'call-1', 'content': 'FULL-TOOL-RESULT ' + 'r' * 8_000},
            )
        }
    )
    cap = 20_000

    coverage = population_module.projection_coverage(
        [trace], trace_input_chars=cap, classifier_model='openai/gpt-6-luna'
    )
    state = prepare_classifier_state(trace, model='openai/gpt-6-luna', char_cap=cap)

    assert isinstance(state, str)
    assert 'REQUEST-START' in state
    assert 'FULL-TOOL-RESULT' in state  # each result keeps its start and end
    assert '[... 7717 chars left out ...]' in state
    assert coverage['n_classifier_truncated'] == 1
    assert coverage['classifier_state_chars'] == len(state)
    assert coverage['classifier_state_format'] == 'text'
    assert coverage['classifier_question_reserve'] == 'not_available'


@pytest.mark.parametrize('classifier_model', ['openai/gpt-6-luna', 'typesafe/jev-latest'])
def test_projection_coverage_does_not_mistake_literal_markers_for_truncation(classifier_model: str) -> None:
    from evaluatorq.common.model_input import serialized_chars, serialized_question_chars

    trace = make_trace('literal-marker').model_copy(
        update={'messages': ({'role': 'user', 'content': 'Literal [... 123 chars left out ...] text.'},)}
    )
    cap = 20_000
    questions = {
        'marker_check': {
            'kind': 'noul',
            'instructions': 'Classify the request.',
            'criteria': None,
            'state': {'must_not_count': True},
            'noul_threshold': 0.5,
        }
    }

    coverage = population_module.projection_coverage(
        [trace],
        trace_input_chars=cap,
        classifier_model=classifier_model,
        classifier_questions=questions,
    )

    assert coverage['n_summary_truncated'] == 0
    assert coverage['n_classifier_truncated'] == 0
    assert coverage['n_projection_truncated'] == 0
    assert coverage['classifier_request_chars'] == coverage['classifier_state_chars'] + serialized_question_chars(
        questions
    )
    assert coverage['classifier_request_chars'] <= cap


def test_projection_coverage_applies_available_jev_question_reserve() -> None:
    from evaluatorq.common.model_input import (
        JEV_STATE_ALL_QUESTIONS_CHARS,
        JEV_STATE_CHARS,
        JEV_STATE_QUESTION_CHARS,
        fit_classifier_input,
        serialized_chars,
        serialized_question_chars,
    )
    from evaluatorq.insights.labeling import prepare_classifier_state

    trace = make_trace('question-reserve').model_copy(
        update={'messages': ({'role': 'user', 'content': 'request ' + 'x' * 100_000},)}
    )
    questions = {'label': {'kind': 'noul', 'instructions': 'q' * 20_000}}

    coverage = population_module.projection_coverage(
        [trace], trace_input_chars=500_000, classifier_model='typesafe/jev-latest', classifier_questions=questions
    )
    state = fit_classifier_input(
        prepare_classifier_state(trace, model='typesafe/jev-latest', char_cap=500_000),
        model='typesafe/jev-latest',
        questions=questions,
        global_char_cap=500_000,
    )

    state_chars = serialized_chars(state)
    question_chars = serialized_question_chars(questions)
    longest_question = max(serialized_chars(question) for question in questions.values())
    assert coverage['classifier_question_reserve'] == 'applied'
    assert coverage['classifier_state_chars'] == state_chars
    assert state_chars <= JEV_STATE_CHARS
    assert state_chars + longest_question <= JEV_STATE_QUESTION_CHARS
    assert state_chars + question_chars <= JEV_STATE_ALL_QUESTIONS_CHARS
    assert state_chars + question_chars <= 500_000
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
@pytest.mark.parametrize('mode', ['llm', 'llm+jev'])
async def test_query_path_uses_shared_planner_and_generated_default_time(
    mode: Literal['llm', 'llm+jev'], monkeypatch: pytest.MonkeyPatch
) -> None:
    from evaluatorq.common.judge import ClassifyQuestion

    compiled_query = CompiledQuery(
        task=ClassifyQuestion(
            kind='choice', instructions='classify', criteria={'billing': 'billing help', 'technical': 'tech help'}, state={}
        ),
        selection=ValueSelection(kind='values', values=('billing',)),
    )
    generated = (
        datetime(2026, 9, 2, tzinfo=timezone.utc),
        datetime(2026, 9, 8, tzinfo=timezone.utc),
    )
    generated_facets = FacetSelection(agent_name=frozenset({'generated-bot'}), model=frozenset({'gpt-5'}))

    async def fake_plan(
        query: str, population: PopulationRequest, **kwargs: Any
    ) -> PlanningResult:
        assert query == 'refund requests'
        assert kwargs['filter_selection'] == mode
        assert population.time_origin == 'default'
        effective = population.model_copy(update={
            'start': generated[0],
            'end': generated[1],
            'time_origin': 'explicit',
            'facets': FacetSelection(agent_name=frozenset({'explicit-bot'}), model=frozenset({'gpt-5'})),
            'numeric': NumericFilters(tokens_min=5, tokens_max=100),
        })
        return PlanningResult(
            dimensions=(compiled_query,),
            generated_facets=generated_facets,
            generated_numeric=NumericFilters(tokens_min=5, tokens_max=100),
            generated_time=generated,
            effective_population=effective,
            selector_response=None,
            selector_error=None,
            provenance={'filter_mode': mode, 'time': 'query'},
            usage={},
        )

    monkeypatch.setattr(population_module, 'plan_query', fake_plan)
    monkeypatch.setattr(population_module, 'OrqTraceSource', FakeSource)
    FakeSource.snapshot = Snapshot(traces=(make_trace('t1'),))
    pop = InsightsPopulation(
        query='refund requests',
        facets=FacetSelection(agent_name=frozenset({'explicit-bot'})),
        filter_selection=mode, limit=50
    )
    resolved = await resolve_population(
        pop, orq=_orq(), client=_client(), compiler_model='compiler', classifier_model='classifier'
    )

    assert resolved.compiled == (compiled_query,)
    assert resolved.echo['filter_selection'] == mode
    assert resolved.echo['filter_selection_error'] is None
    assert resolved.echo['filter_model_role'] == (
        'fast/finder.filters' if mode == 'llm' else 'classifier/finder.classifier'
    )
    assert resolved.echo['generated_time'] == [bound.isoformat() for bound in generated]
    assert resolved.echo['time_provenance'] == 'query'
    assert resolved.echo['time_origin'] == 'explicit'
    call = FakeSource.calls[0]
    assert (call['start'], call['end']) == generated
    assert call['facets'].agent_name == frozenset({'explicit-bot'})
    assert call['facets'].model == frozenset({'gpt-5'})
    assert call['numeric'] == NumericFilters(tokens_min=5, tokens_max=100)


@pytest.mark.asyncio
async def test_query_path_preserves_explicit_window_and_selector_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit = (
        datetime(2026, 9, 3, tzinfo=timezone.utc),
        datetime(2026, 9, 7, tzinfo=timezone.utc),
    )
    generated = (datetime(2026, 9, 1, tzinfo=timezone.utc), datetime(2026, 9, 9, tzinfo=timezone.utc))

    async def fake_plan(query: str, population: PopulationRequest, **kwargs: Any) -> PlanningResult:
        assert kwargs['filter_selection'] == 'llm'
        assert population.time_origin == 'explicit'
        return PlanningResult(
            dimensions=(),
            generated_facets=FacetSelection(),
            generated_numeric=NumericFilters(),
            generated_time=generated,
            effective_population=population,
            selector_response=None,
            selector_error='selector unavailable',
            provenance={'time': 'explicit', 'filter_mode': 'llm'},
            usage={},
        )

    monkeypatch.setattr(population_module, 'plan_query', fake_plan)
    monkeypatch.setattr(population_module, 'OrqTraceSource', FakeSource)
    FakeSource.snapshot = Snapshot(traces=())
    pop = InsightsPopulation(
        query='refund requests', start=explicit[0], end=explicit[1], time_origin='explicit', filter_selection='llm'
    )
    resolved = await resolve_population(
        pop, orq=_orq(), client=_client(), compiler_model='compiler', classifier_model='classifier'
    )

    assert (FakeSource.calls[0]['start'], FakeSource.calls[0]['end']) == explicit
    assert resolved.echo['generated_time'] == [bound.isoformat() for bound in generated]
    assert resolved.echo['time_provenance'] == 'explicit'
    assert resolved.echo['filter_selection_error'] == 'selector unavailable'


@pytest.mark.asyncio
async def test_query_path_planner_failure_raises_population_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failing_plan(*args: Any, **kwargs: Any) -> PlanningResult:
        raise ValueError('the compiler produced no structured document')

    monkeypatch.setattr(population_module, 'plan_query', failing_plan)
    pop = InsightsPopulation(query='refund requests', limit=50)

    with pytest.raises(PopulationError, match='planning the population query failed'):
        await resolve_population(pop, orq=_orq(), client=_client(), compiler_model='compiler', classifier_model='classifier')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('origin', 'expected_origin'),
    [('default', 'default'), ('explicit', 'explicit'), (None, 'explicit')],
)
async def test_query_path_respects_declared_window_origin(
    monkeypatch: pytest.MonkeyPatch,
    origin: Literal['default', 'explicit'] | None,
    expected_origin: Literal['default', 'explicit'],
) -> None:
    seen: dict[str, Any] = {}

    async def fake_plan(query: str, population: PopulationRequest, **kwargs: Any) -> PlanningResult:
        seen.update(population=population, kwargs=kwargs)
        return PlanningResult(
            dimensions=(),
            generated_facets=FacetSelection(),
            generated_numeric=NumericFilters(),
            generated_time=None,
            effective_population=population,
            selector_response=None,
            selector_error=None,
            provenance={'time': expected_origin},
            usage={},
        )

    monkeypatch.setattr(population_module, 'plan_query', fake_plan)
    monkeypatch.setattr(population_module, 'OrqTraceSource', FakeSource)
    FakeSource.snapshot = Snapshot(traces=())
    population = InsightsPopulation(query='refund requests', window_days=14)
    if origin is not None:
        population = InsightsPopulation(query='refund requests', window_days=14, time_origin=origin)
    await resolve_population(
        population,
        orq=_orq(),
        client=_client(),
        compiler_model='compiler',
        classifier_model='classifier',
    )

    population = seen['population']
    assert population.time_origin == expected_origin
    assert population.end - population.start == timedelta(days=14)

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
async def test_export_path_reloads_merged_filters_and_original_time_range(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = _run_export(['t1'], filters=ExportFilters(agent_name=('explicit-bot',)))
    export = export.model_copy(
        update={
            'generated_filters': ExportFilters(agent_name=('generated-bot',), model=('gpt-5',)),
            'generated_numeric': ExportNumericFilters(tokens_min=20, tokens_max=300),
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
    assert call['start'] == export.start
    assert call['end'] == export.end
    assert call['limit'] == export.limit
    assert call['target_trace_ids'] == {'t1'}
    assert [trace.trace_id for trace in resolved.traces] == ['t1']


@pytest.mark.asyncio
async def test_export_without_original_window_fails_instead_of_guessing_a_recent_range(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    export = _run_export(['t1']).model_copy(update={'start': None, 'end': None})
    export_path = tmp_path / 'legacy-export.json'
    export_path.write_text(export.model_dump_json())
    monkeypatch.setattr(population_module, 'OrqTraceSource', FakeSource)

    with pytest.raises(PopulationError, match='does not record its original time window'):
        await resolve_population(
            InsightsPopulation.from_finder_export(export_path),
            orq=_orq(),
            client=_client(),
            compiler_model='compiler',
            classifier_model='classifier',
        )

    assert FakeSource.calls == []


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


@pytest.mark.asyncio
@pytest.mark.parametrize('selection_error', ['catalogue unavailable', None])
async def test_empty_query_plan_does_not_load_a_broader_population(monkeypatch, selection_error):
    from evaluatorq.common.judge import ClassifyQuestion

    dimension = CompiledQuery(
        task=ClassifyQuestion(kind='noul', instructions='The user requests a refund.', state={}),
        selection=ValueSelection(kind='values', values=(True,)),
    )

    async def empty_plan(query, population, **kwargs):
        return PlanningResult(
            dimensions=(dimension,), generated_facets=FacetSelection(),
            generated_numeric=NumericFilters(), generated_time=None,
            effective_population=population, selector_response=None,
            selector_error=selection_error, provenance={'time': 'default'}, usage={},
            empty_match=True,
        )

    async def forbidden_load(*args, **kwargs):
        pytest.fail('An empty plan must not load a broader population.')

    monkeypatch.setattr(population_module, 'plan_query', empty_plan)
    monkeypatch.setattr(population_module, '_load_traces', forbidden_load)
    resolved = await resolve_population(
        InsightsPopulation(query='refund requests'), orq=_orq(), client=_client(),
        compiler_model='compiler', classifier_model='classifier',
    )
    assert resolved.traces == []
    assert resolved.n_scanned == 0
    assert resolved.compiled == (dimension,)
    assert resolved.echo['filter_selection_error'] == selection_error


@pytest.mark.asyncio
@pytest.mark.parametrize(('end_exclusive', 'expected'), [(True, ['inside']), (False, ['inside', 'at-end'])])
async def test_query_population_honours_the_effective_end_boundary(monkeypatch, end_exclusive, expected):
    start = datetime(2026, 9, 1, tzinfo=timezone.utc)
    end = datetime(2026, 9, 2, tzinfo=timezone.utc)

    async def plan(query, population, **kwargs):
        effective = PopulationRequest(start=start, end=end, time_origin='explicit', end_exclusive=end_exclusive)
        return PlanningResult(
            dimensions=(), generated_facets=FacetSelection(), generated_numeric=NumericFilters(),
            generated_time=(start, end), effective_population=effective, selector_response=None,
            selector_error=None, provenance={'time': 'query'}, usage={},
        )

    monkeypatch.setattr(population_module, 'plan_query', plan)
    monkeypatch.setattr(population_module, 'OrqTraceSource', FakeSource)
    FakeSource.snapshot = Snapshot(traces=(
        make_trace('inside').model_copy(update={'timestamp': end - timedelta(seconds=1)}),
        make_trace('at-end').model_copy(update={'timestamp': end}),
    ))
    resolved = await resolve_population(
        InsightsPopulation(query='traces from 1 September'), orq=_orq(), client=_client(),
        compiler_model='compiler', classifier_model='classifier',
    )
    assert [trace.metadata.trace_id for trace in resolved.traces] == expected
    assert resolved.n_scanned == len(expected)
