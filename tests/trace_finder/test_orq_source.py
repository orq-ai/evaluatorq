"""Behavioral contracts for the live Orq OQL trace source."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace, TracebackType
from typing import Any, cast

import pytest

from evaluatorq.trace_finder import FacetSelection, NumericFilters
from evaluatorq.trace_finder.models import Snapshot
from evaluatorq.trace_finder.orq_source import (
    MAX_SPAN_PAGES,
    OrqSourceError,
    OrqTraceSource,
    _RawResponseCapture,
    _conversation_messages,
    build_oql,
)
from evaluatorq.trace_finder.rows import TraceRow

UTC = timezone.utc
START = datetime(2026, 9, 20, tzinfo=UTC)
END = datetime(2026, 9, 21, tzinfo=UTC)


class FakeTraces:
    def __init__(
        self,
        pages: dict[str | None, tuple[list[Any], bool, str | None]],
        *,
        spans: dict[str, list[Any]] | None = None,
        details: dict[tuple[str, str], Any] | None = None,
        raw_details: dict[tuple[str, str], dict[str, Any]] | None = None,
    ) -> None:
        self.pages = pages
        self.spans = spans or {}
        self.details = details or {}
        self.raw_details = raw_details or {}
        self.capture_hook: Any | None = None
        self.query_calls: list[dict[str, Any]] = []
        self.list_span_calls: list[dict[str, Any]] = []
        self.get_span_calls: list[dict[str, Any]] = []

    async def query_async(self, **kwargs: Any) -> Any:
        self.query_calls.append(kwargs)
        data, has_more, token = self.pages[kwargs.get('page_token')]
        return namespace(search=namespace(data=data, has_more=has_more, next_page_token=token))

    async def list_spans_async(self, *, trace_id: str, **kwargs: Any) -> Any:
        self.list_span_calls.append({'trace_id': trace_id, **kwargs})
        return namespace(data=self.spans.get(trace_id, []), has_more=False, next_page_token=None)

    async def get_span_async(self, *, trace_id: str, span_id: str, **kwargs: Any) -> Any:
        self.get_span_calls.append({'trace_id': trace_id, 'span_id': span_id, **kwargs})
        raw = self.raw_details.get((trace_id, span_id))
        if raw is not None and self.capture_hook is not None:
            response = namespace(
                request=namespace(url=namespace(path=f'/v2/traces/{trace_id}/spans/{span_id}')),
                json=lambda: raw,
            )
            self.capture_hook.after_success(namespace(operation_id='TracesGetSpan'), response)
        return namespace(span=self.details[trace_id, span_id])


class YieldingRawQueryTraces(FakeTraces):
    def __init__(self) -> None:
        super().__init__({})
        self.query_hook: Any | None = None
        self.query_number = 0

    async def query_async(self, **kwargs: Any) -> Any:
        self.query_calls.append(kwargs)
        self.query_number += 1
        query_number = self.query_number
        trace_id = f'trace-{query_number}'
        raw_marker = f'raw-{query_number}'
        raw_response = namespace(
            request=namespace(url=namespace(path='/v2/traces/query')),
            json=lambda: {'search': {'data': [{'trace_id': trace_id, 'messages': user_messages(raw_marker)}]}},
        )
        assert self.query_hook is not None
        self.query_hook.after_success(namespace(operation_id='TracesQueryOql'), raw_response)
        await asyncio.sleep(0)
        if query_number == 1:
            await asyncio.sleep(0)
        return namespace(search=namespace(data=[summary(trace_id, messages=[])], has_more=False, next_page_token=None))


class FakeProjects:
    def __init__(self, projects: list[Any] | None = None) -> None:
        self.projects = projects or []
        self.calls: list[dict[str, Any]] = []

    async def list_async(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return namespace(data=self.projects, has_more=False)


class FakeOrq:
    def __init__(self, traces: FakeTraces, projects: FakeProjects | None = None) -> None:
        self.traces = traces
        self.projects = projects or FakeProjects()
        self.enter_calls = 0
        self.exit_calls: list[tuple[Any, Any, Any]] = []
        self.sdk_configuration: Any = None

    async def __aenter__(self) -> 'FakeOrq':
        self.enter_calls += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.exit_calls.append((exc_type, exc_value, traceback))


class FakeHooks:
    def __init__(self) -> None:
        self.after_success_hooks: list[Any] = []

    def register_after_success_hook(self, hook: Any) -> None:
        self.after_success_hooks.append(hook)


def with_hooks(client: FakeOrq) -> FakeOrq:
    client.sdk_configuration = namespace(_hooks=FakeHooks())
    return client


def test_build_oql_includes_categorical_and_numeric_filters() -> None:
    facets = FacetSelection(
        project=frozenset({'Research'}),
        model=frozenset({'gpt-5'}),
        provider=frozenset({'openai'}),
        status=frozenset({'error'}),
        product=frozenset({'ai-gateway'}),
        trace_type=frozenset({'span.responses'}),
        agent_name=frozenset({'support-agent'}),
        tool_name=frozenset({'search'}),
    )
    numeric = NumericFilters(tokens_min=500, tokens_max=5000, duration_ms_min=10, duration_ms_max=1000)

    assert build_oql(facets, numeric, {'project-123': 'Research'}) == (
        'fetch traces | filter project_id in ("project-123") '
        '| filter model in ("gpt-5") '
        '| filter provider in ("openai") '
        '| filter status in ("error") '
        '| filter product in ("ai-gateway") '
        '| filter attributes.orq.leading_span.span_type in ("span.responses") '
        '| filter agent_name in ("support-agent") '
        '| filter tool_name in ("search") '
        '| filter total_tokens >= 500 '
        '| filter total_tokens <= 5000 '
        '| filter duration_ms >= 10 '
        '| filter duration_ms <= 1000 '
        '| sort end_time desc'
    )


def test_build_oql_keeps_the_base_filter_unless_a_model_or_provider_is_picked() -> None:
    base = 'operation not_in ("generate_content")'

    assert base in build_oql(FacetSelection(status=frozenset({'error'})), NumericFilters(), {})
    # jev and other bare model calls are generate_content traces; the base filter would hide all of them.
    assert base not in build_oql(FacetSelection(model=frozenset({'jev-latest'})), NumericFilters(), {})
    assert base not in build_oql(FacetSelection(provider=frozenset({'typesafe'})), NumericFilters(), {})


def test_build_oql_rejects_a_project_name_that_cannot_be_resolved() -> None:
    with pytest.raises(OrqSourceError, match='cannot resolve selected project names'):
        build_oql(FacetSelection(project=frozenset({'Missing'})), NumericFilters(), {})


def test_saved_project_id_does_not_expand_a_duplicate_project_name() -> None:
    facets = FacetSelection(project_id='project-b', project=frozenset({'Same'}))

    query = build_oql(facets, NumericFilters(), {'project-a': 'Same', 'project-b': 'Same'})

    assert 'project_id in ("project-b")' in query
    assert 'project-a' not in query


def test_project_facet_with_duplicate_name_selects_only_its_id() -> None:
    facets = FacetSelection(project=frozenset({'Same (project-b)'}))

    query = build_oql(facets, NumericFilters(), {'project-a': 'Same', 'project-b': 'Same'})

    assert 'project_id in ("project-b")' in query
    assert 'project-a' not in query


@pytest.mark.asyncio
async def test_selected_project_discards_cross_project_query_results_before_hydration() -> None:
    traces = FakeTraces({
        None: ([summary('wrong', project_id='project-a')], True, 'page-2'),
        'page-2': ([summary('right', project_id='project-b')], False, None),
    })
    projects = FakeProjects([
        namespace(project_id='project-a', name='Other'),
        namespace(project_id='project-b', name='Selected'),
    ])

    snapshot = await make_source(FakeOrq(traces, projects)).load_async(
        START, END, 1, facets=FacetSelection(project_id='project-b'), numeric=NumericFilters()
    )

    assert [trace.trace_id for trace in snapshot.traces] == ['right']
    assert traces.list_span_calls == []
    assert all('project_id in ("project-b")' in call['oql'] for call in traces.query_calls)


@pytest.mark.asyncio
async def test_selected_project_reports_when_query_returns_only_other_projects() -> None:
    traces = FakeTraces({None: ([summary('wrong', project_id='project-a')], False, None)})
    projects = FakeProjects([namespace(project_id='project-b', name='Selected')])

    with pytest.raises(OrqSourceError, match='only traces outside the selected project'):
        await make_source(FakeOrq(traces, projects)).load_async(
            START, END, 1, facets=FacetSelection(project_id='project-b'), numeric=NumericFilters()
        )

    assert traces.list_span_calls == []


@pytest.mark.asyncio
async def test_loads_pages_newest_first_and_uses_bounded_requests() -> None:
    traces = FakeTraces({
        None: ([summary('older', minute=1), summary('newer', minute=3)], True, 'page-2'),
        'page-2': ([summary('middle', minute=2)], False, None),
    })

    snapshot = await make_source(FakeOrq(traces)).load_async(
        START,
        END,
        3,
        facets=FacetSelection(),
        numeric=NumericFilters(),
    )

    assert isinstance(snapshot, Snapshot)
    assert [trace.trace_id for trace in snapshot.traces] == ['newer', 'middle', 'older']
    assert [call['page_token'] for call in traces.query_calls] == [None, 'page-2']
    assert [call['limit'] for call in traces.query_calls] == [3, 1]
    assert all(call['from_'] == START and call['to'] == END for call in traces.query_calls)
    assert all(call['timeout_ms'] == 30_000 for call in traces.query_calls)


@pytest.mark.asyncio
async def test_targeted_load_scans_past_unrelated_summaries_without_hydrating_them() -> None:
    traces = FakeTraces({
        None: ([summary('newer-a', messages=[]), summary('newer-b', messages=[])], True, 'page-2'),
        'page-2': ([summary('pinned', minute=1)], False, None),
    })

    snapshot = await make_source(FakeOrq(traces)).load_async(
        START,
        END,
        1,
        facets=FacetSelection(),
        numeric=NumericFilters(),
        target_trace_ids={'pinned'},
    )

    assert [trace.trace_id for trace in snapshot.traces] == ['pinned']
    assert [call['page_token'] for call in traces.query_calls] == [None, 'page-2']
    assert all(call['limit'] == 200 for call in traces.query_calls)
    assert not traces.list_span_calls
    assert not traces.get_span_calls


@pytest.mark.asyncio
async def test_targeted_load_stops_starting_pages_after_budget_when_id_is_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = FakeTraces({None: ([summary('unrelated')], True, 'page-2')})
    ticks = iter((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 31.0))
    monkeypatch.setattr('evaluatorq.trace_finder.orq_source.time', SimpleNamespace(monotonic=lambda: next(ticks)))

    snapshot = await make_source(FakeOrq(traces)).load_async(
        START,
        END,
        1,
        facets=FacetSelection(),
        numeric=NumericFilters(),
        target_trace_ids={'stale'},
    )

    assert snapshot.traces == ()
    assert len(traces.query_calls) == 1


@pytest.mark.asyncio
async def test_targeted_page_budget_includes_project_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    traces = FakeTraces({})
    ticks = iter((0.0, 31.0))
    monkeypatch.setattr('evaluatorq.trace_finder.orq_source.time', SimpleNamespace(monotonic=lambda: next(ticks)))

    snapshot = await make_source(FakeOrq(traces)).load_async(
        START,
        END,
        1,
        facets=FacetSelection(),
        numeric=NumericFilters(),
        target_trace_ids={'stale'},
    )

    assert snapshot.traces == ()
    assert not traces.query_calls


@pytest.mark.asyncio
async def test_targeted_deadline_cancels_slow_project_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.trace_finder.orq_source.TARGET_RELOAD_PAGE_BUDGET_SECONDS', 0.01)
    traces = FakeTraces({})
    client = FakeOrq(traces)
    project_started = asyncio.Event()

    async def slow_projects(**kwargs: Any) -> Any:
        project_started.set()
        await asyncio.sleep(60)

    cast('Any', client.projects).list_async = slow_projects
    snapshot = await make_source(client).load_async(
        START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), target_trace_ids={'stale'}
    )

    assert project_started.is_set()
    assert snapshot.traces == ()
    assert snapshot.capture_metadata['incomplete_reason'] == 'target_deadline'
    assert not traces.query_calls


@pytest.mark.asyncio
async def test_targeted_project_provider_timeout_is_not_treated_as_deadline() -> None:
    traces = FakeTraces({})
    client = FakeOrq(traces)

    async def provider_timeout(**_kwargs: Any) -> Any:
        raise asyncio.TimeoutError('provider project lookup timed out')

    cast('Any', client.projects).list_async = provider_timeout

    with pytest.raises(OrqSourceError, match='provider project lookup timed out') as error:
        await make_source(client).load_async(
            START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), target_trace_ids={'stale'}
        )
    assert isinstance(error.value.__cause__, asyncio.TimeoutError)


@pytest.mark.asyncio
async def test_targeted_deadline_cancels_slow_query_and_passes_remaining_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr('evaluatorq.trace_finder.orq_source.TARGET_RELOAD_PAGE_BUDGET_SECONDS', 1.0)
    # Advance only the source's clock; asyncio still enforces real cancellation.
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr('evaluatorq.trace_finder.orq_source.time', SimpleNamespace(monotonic=lambda: clock.now))
    traces = FakeTraces({})
    client = FakeOrq(traces)
    query_started = asyncio.Event()
    query_cancelled = asyncio.Event()
    query_timeouts: list[int] = []

    async def projects_with_elapsed_budget(**kwargs: Any) -> Any:
        result = await FakeProjects().list_async(**kwargs)
        clock.now = 0.75
        return result

    async def slow_query(**kwargs: Any) -> Any:
        query_started.set()
        query_timeouts.append(kwargs['timeout_ms'])
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            query_cancelled.set()
            raise

    cast('Any', client.projects).list_async = projects_with_elapsed_budget
    cast('Any', traces).query_async = slow_query
    snapshot = await make_source(client).load_async(
        START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), target_trace_ids={'stale'}
    )

    assert query_started.is_set()
    assert query_cancelled.is_set()
    assert snapshot.traces == ()
    assert snapshot.capture_metadata['incomplete_reason'] == 'target_deadline'
    assert query_timeouts == [250]


@pytest.mark.asyncio
async def test_targeted_query_provider_timeout_is_not_treated_as_deadline() -> None:
    traces = FakeTraces({})

    async def provider_timeout(**_kwargs: Any) -> Any:
        raise asyncio.TimeoutError('provider query timed out')

    cast('Any', traces).query_async = provider_timeout

    with pytest.raises(OrqSourceError, match='provider query timed out') as error:
        await make_source(FakeOrq(traces)).load_async(
            START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), target_trace_ids={'stale'}
        )
    assert isinstance(error.value.__cause__, asyncio.TimeoutError)


@pytest.mark.asyncio
async def test_targeted_deadline_cancels_slow_span_hydration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.trace_finder.orq_source.TARGET_RELOAD_PAGE_BUDGET_SECONDS', 1.0)
    ticks = iter((0.0, 0.0, 0.0, 0.0, 0.0, 0.5))

    def monotonic() -> float:
        return next(ticks, 0.5)

    monkeypatch.setattr('evaluatorq.trace_finder.orq_source.time', SimpleNamespace(monotonic=monotonic))
    traces = FakeTraces({None: ([summary('target', messages=[])], False, None)})
    hydration_started = asyncio.Event()
    hydration_timeouts: list[int] = []

    async def slow_span_list(*, trace_id: str, **kwargs: Any) -> Any:
        hydration_started.set()
        hydration_timeouts.append(kwargs['timeout_ms'])
        await asyncio.sleep(60)

    cast('Any', traces).list_spans_async = slow_span_list
    snapshot = await make_source(FakeOrq(traces)).load_async(
        START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), target_trace_ids={'target'}
    )

    assert hydration_started.is_set()
    assert snapshot.traces == ()
    assert snapshot.capture_metadata['incomplete_reason'] == 'target_deadline'
    assert len(hydration_timeouts) == 1
    assert 1 <= hydration_timeouts[0] <= 500


@pytest.mark.asyncio
async def test_targeted_hydration_provider_timeout_is_not_treated_as_deadline() -> None:
    traces = FakeTraces({None: ([summary('target', messages=[])], False, None)})

    async def provider_timeout(*, trace_id: str, **_kwargs: Any) -> Any:
        raise asyncio.TimeoutError(f'provider span timed out for {trace_id}')

    cast('Any', traces).list_spans_async = provider_timeout

    with pytest.raises(OrqSourceError, match='provider span timed out') as error:
        await make_source(FakeOrq(traces)).load_async(
            START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), target_trace_ids={'target'}
        )
    assert isinstance(error.value.__cause__, asyncio.TimeoutError)


@pytest.mark.asyncio
async def test_targeted_hydration_deadline_keeps_completed_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.trace_finder.orq_source.TARGET_RELOAD_PAGE_BUDGET_SECONDS', 0.05)
    traces = FakeTraces({None: ([summary('fast', messages=[]), summary('slow', messages=[])], False, None)})
    slow_started = asyncio.Event()
    slow_cancelled = asyncio.Event()
    warnings: list[tuple[Any, ...]] = []
    monkeypatch.setattr('evaluatorq.trace_finder.orq_source.logger.warning', lambda *args: warnings.append(args))

    async def list_spans(*, trace_id: str, **kwargs: Any) -> Any:
        traces.list_span_calls.append({'trace_id': trace_id, **kwargs})
        if trace_id == 'slow':
            slow_started.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                slow_cancelled.set()
                raise
        return namespace(data=[span(f'{trace_id}-span', minute=1)], has_more=False, next_page_token=None)

    cast('Any', traces).list_spans_async = list_spans
    traces.details['fast', 'fast-span'] = detail('fast-span', 'fast conversation', minute=1)
    snapshot = await make_source(FakeOrq(traces)).load_async(
        START,
        END,
        2,
        facets=FacetSelection(),
        numeric=NumericFilters(),
        target_trace_ids={'fast', 'slow'},
    )

    assert slow_started.is_set()
    assert slow_cancelled.is_set()
    assert [record.trace_id for record in snapshot.traces] == ['fast']
    assert snapshot.capture_metadata['incomplete_reason'] == 'target_deadline'
    assert any('during trace hydration' in str(args[0]) for args in warnings)


@pytest.mark.asyncio
async def test_targeted_scan_limit_marks_snapshot_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('evaluatorq.trace_finder.orq_source.MAX_LIVE_TRACES', 1)
    traces = FakeTraces({None: ([summary('other', messages=user_messages('other'))], True, 'next')})

    snapshot = await make_source(FakeOrq(traces)).load_async(
        START, END, 1, facets=FacetSelection(), numeric=NumericFilters(), target_trace_ids={'target'}
    )

    assert snapshot.traces == ()
    assert snapshot.capture_metadata['incomplete_reason'] == 'scan_limit'
    assert len(traces.query_calls) == 1


@pytest.mark.asyncio
async def test_populates_summary_metadata_without_listing_spans() -> None:
    trace = summary('complete', messages=user_messages('from summary'))
    trace.agent_name = 'support-agent'
    trace.tool_name = 'search'
    trace.total_tokens = 321
    trace.duration_ms = 42
    traces = FakeTraces({None: ([trace], False, None)})

    snapshot = await make_source(FakeOrq(traces)).load_async(
        START,
        END,
        1,
        facets=FacetSelection(),
        numeric=NumericFilters(),
    )

    record = snapshot.traces[0]
    assert record.messages == ({'role': 'user', 'content': 'from summary'},)
    assert record.agent_name == 'support-agent'
    assert record.tool_names == ('search',)
    assert record.total_tokens == 321
    assert record.duration_ms == 42
    assert traces.list_span_calls == []
    assert traces.get_span_calls == []


@pytest.mark.asyncio
async def test_captures_tool_definition_size_from_selected_span() -> None:
    trace = summary('with-tools', messages=[])
    hydrated = detail('completion', 'find the order', minute=1)
    raw = {
        'span': {
            'attributes': {
                'gen_ai': {
                    'input': {'messages': user_messages('find the order')},
                    'tool': {'definitions': [
                        {'type': 'function', 'function': {'name': 'lookup_order', 'parameters': {'type': 'object'}}},
                        {'type': 'function', 'function': {'name': 'refund_order', 'parameters': {'type': 'object'}}},
                    ]},
                },
            },
        },
    }
    traces = FakeTraces(
        {None: ([trace], False, None)},
        spans={'with-tools': [span('completion', minute=1)]},
        details={('with-tools', 'completion'): hydrated},
        raw_details={('with-tools', 'completion'): raw},
    )
    source = make_source(with_hooks(FakeOrq(traces)))
    traces.capture_hook = source._capture

    rows = await source.search(START, END, 1, facets=FacetSelection(), numeric=NumericFilters())
    records = await source.hydrate_rows(rows)
    record = records['with-tools']

    assert record is not None
    assert record.tool_definition_count == 2
    assert record.tool_definition_tokens > 0


@pytest.mark.asyncio
async def test_uses_raw_query_payload_for_fields_dropped_by_sdk_models() -> None:
    trace = summary('raw-fields', messages=[])
    traces = FakeTraces({None: ([trace], False, None)})
    source = make_source(FakeOrq(traces))
    raw_summary = {
        'trace_id': 'raw-fields',
        'agent_name': 'raw-agent',
        'tool_name': ['lookup', 'search'],
        'total_tokens': 654,
        'duration_ms': 77,
        'attributes': {'gen_ai': {'input': {'messages': user_messages('raw payload')}}},
    }
    source._capture.after_success(
        namespace(operation_id='TracesQueryOql'),
        namespace(
            request=namespace(url=namespace(path='/v2/traces/query')),
            json=lambda: {'search': {'data': [raw_summary]}},
        ),
    )

    snapshot = await source.load_async(
        START,
        END,
        1,
        facets=FacetSelection(),
        numeric=NumericFilters(),
    )

    record = snapshot.traces[0]
    assert record.messages == ({'role': 'user', 'content': 'raw payload'},)
    assert record.agent_name == 'raw-agent'
    assert record.tool_names == ('lookup', 'search')
    assert record.total_tokens == 654
    assert record.duration_ms == 77


@pytest.mark.asyncio
async def test_prepends_system_prompt_from_span_detail_when_summary_lacks_it() -> None:
    # Live Orq shape: the trace summary's gen_ai.input is only the last user turn (a JSON string),
    # while get_span serves gen_ai.input.messages with the system message first, as role + parts.
    trace = summary('trace', messages=[])
    trace.span_id = 'leading'
    trace.attributes = {'gen_ai': {'input': '"which traces had tool errors?"'}}
    detail_payload = namespace(
        attributes={
            'gen_ai': {
                'input': {
                    'messages': [
                        {'role': 'system', 'parts': [{'type': 'text', 'content': 'You compile queries.'}]},
                        {'role': 'user', 'parts': [{'type': 'text', 'content': 'which traces had tool errors?'}]},
                    ]
                }
            }
        }
    )
    traces = FakeTraces({None: ([trace], False, None)}, details={('trace', 'leading'): detail_payload})

    snapshot = await make_source(FakeOrq(traces)).load_async(
        START, END, 1, facets=FacetSelection(), numeric=NumericFilters()
    )

    messages = snapshot.traces[0].messages
    assert [m['role'] for m in messages] == ['system', 'user']
    assert messages[0]['parts'][0]['content'] == 'You compile queries.'
    assert messages[1]['role'] == 'user'
    assert [call['span_id'] for call in traces.get_span_calls] == ['leading']


@pytest.mark.asyncio
async def test_keeps_summary_messages_when_system_prompt_lookup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = summary('trace', messages=user_messages('hello'))
    trace.span_id = 'leading'
    traces = FakeTraces({None: ([trace], False, None)})
    warnings: list[str] = []
    monkeypatch.setattr(
        'evaluatorq.trace_finder.orq_source.logger.warning',
        lambda message, *args: warnings.append(message.format(*args)),
    )

    snapshot = await make_source(FakeOrq(traces)).load_async(
        START, END, 1, facets=FacetSelection(), numeric=NumericFilters()
    )

    assert snapshot.traces[0].messages == ({'role': 'user', 'content': 'hello'},)
    lookups = [w for w in warnings if 'system prompt lookup failed for 1 of 1 trace(s); first error:' in w]
    assert len(lookups) == 1


@pytest.mark.asyncio
async def test_hydrates_latest_eligible_span_and_maps_project_metadata() -> None:
    traces = FakeTraces(
        {None: ([summary('trace', messages=[], project_id='project-1')], False, None)},
        spans={
            'trace': [
                span('conversation', minute=2),
                span('evaluator', minute=4, span_type='span.evaluator'),
                span('eligible', minute=5, span_type='span.responses'),
                span('evaluator-child', minute=6, parent_span_id='evaluator'),
            ]
        },
        details={
            ('trace', 'eligible'): detail(
                'eligible',
                'eligible conversation',
                minute=5,
                provider='span-provider',
                model='span-model',
                agent_name='span-agent',
                tool_names=['lookup', 'search'],
                total_tokens=999,
                duration_ms=88,
            )
        },
    )
    projects = FakeProjects([namespace(project_id='project-1', name='Customer Support')])

    snapshot = await make_source(FakeOrq(traces, projects)).load_async(
        START,
        END,
        1,
        facets=FacetSelection(),
        numeric=NumericFilters(),
    )

    record = snapshot.traces[0]
    assert record.span_id == 'eligible'
    assert record.messages == ({'role': 'user', 'content': 'eligible conversation'},)
    assert record.project == 'Customer Support'
    assert record.model == 'span-model'
    assert record.provider == 'span-provider'
    assert record.agent_name == 'span-agent'
    assert record.tool_names == ('lookup', 'search')
    assert record.total_tokens == 999
    assert record.duration_ms == 88
    assert [call['span_id'] for call in traces.get_span_calls] == ['eligible']


@pytest.mark.asyncio
async def test_warns_when_raw_capture_is_unavailable_and_sdk_fallback_is_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = FakeTraces({None: ([summary('fallback')], False, None)})
    warnings: list[str] = []
    monkeypatch.setattr(
        'evaluatorq.trace_finder.orq_source.logger.warning',
        lambda message, *args: warnings.append(message.format(*args)),
    )

    snapshot = await make_source(FakeOrq(traces)).load_async(
        START,
        END,
        1,
        facets=FacetSelection(),
        numeric=NumericFilters(),
    )

    assert snapshot.traces[0].messages == ({'role': 'user', 'content': 'fallback'},)
    assert len(warnings) == 1
    assert 'raw-response capture was unavailable' in warnings[0]
    assert 'generated SDK models may omit conversation messages, agent_name, tool_name, total_tokens, and duration_ms' in warnings[0]


@pytest.mark.asyncio
async def test_two_sources_on_one_client_register_one_shared_capture() -> None:
    client = with_hooks(FakeOrq(FakeTraces({None: ([summary('trace')], False, None)})))

    first = make_source(client)
    second = make_source(client)

    hooks = client.sdk_configuration._hooks
    assert len(hooks.after_success_hooks) == 1
    assert first._capture is second._capture

    first.close()
    await second.aclose()


@pytest.mark.asyncio
async def test_concurrent_sources_pop_their_own_shared_raw_query_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = YieldingRawQueryTraces()
    client = with_hooks(FakeOrq(traces))
    first = make_source(client)
    second = make_source(client)
    traces.query_hook = client.sdk_configuration._hooks.after_success_hooks[0]
    warnings: list[str] = []
    monkeypatch.setattr(
        'evaluatorq.trace_finder.orq_source.logger.warning',
        lambda message, *args: warnings.append(message.format(*args)),
    )

    first_snapshot, second_snapshot = await asyncio.gather(
        first.load_async(START, END, 1, facets=FacetSelection(), numeric=NumericFilters()),
        second.load_async(START, END, 1, facets=FacetSelection(), numeric=NumericFilters()),
    )

    assert first_snapshot.traces[0].messages == ({'role': 'user', 'content': 'raw-1'},)
    assert second_snapshot.traces[0].messages == ({'role': 'user', 'content': 'raw-2'},)
    assert warnings == []
    first.close()
    second.close()


@pytest.mark.asyncio
async def test_missing_timestamp_drops_only_affected_trace() -> None:
    missing_time = summary('missing-time')
    missing_time.started_at = None
    missing_time.ended_at = None
    traces = FakeTraces({None: ([missing_time, summary('valid')], False, None)})

    snapshot = await make_source(FakeOrq(traces)).load_async(
        START, END, 2, facets=FacetSelection(), numeric=NumericFilters()
    )

    assert [record.trace_id for record in snapshot.traces] == ['valid']


@pytest.mark.asyncio
async def test_logs_one_warning_for_all_traces_dropped_without_usable_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    traces = FakeTraces({None: ([summary('empty-1', messages=[]), summary('empty-2', messages=[])], False, None)})
    warnings: list[str] = []
    monkeypatch.setattr(
        'evaluatorq.trace_finder.orq_source.logger.warning',
        lambda message, *args: warnings.append(message.format(*args)),
    )

    snapshot = await make_source(FakeOrq(traces)).load_async(
        START,
        END,
        2,
        facets=FacetSelection(),
        numeric=NumericFilters(),
    )

    assert snapshot.traces == ()
    dropped_warnings = [warning for warning in warnings if 'dropped' in warning]
    assert len(dropped_warnings) == 1
    assert 'dropped 2 trace(s)' in dropped_warnings[0]


@pytest.mark.asyncio
async def test_owned_client_is_closed_but_caller_owned_client_is_not() -> None:
    owned = FakeOrq(FakeTraces({None: ([summary('owned')], False, None)}))
    await make_source(owned, owns_client=True).load_async(
        START,
        END,
        1,
        facets=FacetSelection(),
        numeric=NumericFilters(),
    )
    assert owned.enter_calls == 1
    assert len(owned.exit_calls) == 1

    caller_owned = FakeOrq(FakeTraces({None: ([summary('caller')], False, None)}))
    await make_source(caller_owned).load_async(
        START,
        END,
        1,
        facets=FacetSelection(),
        numeric=NumericFilters(),
    )
    assert caller_owned.enter_calls == 0
    assert caller_owned.exit_calls == []


@pytest.mark.asyncio
async def test_stops_paging_after_bounded_empty_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    pages: dict[str | None, tuple[list[Any], bool, str | None]] = {None: ([], True, 'page-1')}
    pages.update({f'page-{index}': ([], True, f'page-{index + 1}') for index in range(1, 21)})
    traces = FakeTraces(pages)
    warnings: list[str] = []
    monkeypatch.setattr(
        'evaluatorq.trace_finder.orq_source.logger.warning',
        lambda message, *args: warnings.append(message.format(*args)),
    )

    snapshot = await make_source(FakeOrq(traces)).load_async(
        START, END, 1, facets=FacetSelection(), numeric=NumericFilters()
    )

    assert snapshot.traces == ()
    assert len(traces.query_calls) == 20
    assert any('stopped trace search' in warning for warning in warnings)


@pytest.mark.asyncio
async def test_rejects_repeated_page_token() -> None:
    traces = FakeTraces({
        None: ([summary('first')], True, 'stuck'),
        'stuck': ([], True, 'stuck'),
    })

    with pytest.raises(OrqSourceError, match='repeated.*page token'):
        await make_source(FakeOrq(traces)).load_async(
            START,
            END,
            2,
            facets=FacetSelection(),
            numeric=NumericFilters(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('targeted', [False, True])
async def test_hydration_failure_cancels_other_span_requests(targeted: bool) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class FailingSpans(FakeTraces):
        async def list_spans_async(self, *, trace_id: str, **kwargs: Any) -> Any:
            self.list_span_calls.append({'trace_id': trace_id, **kwargs})
            if len(self.list_span_calls) == 2:
                started.set()
            await started.wait()
            if trace_id == 'first':
                raise RuntimeError('span service unavailable')
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

    traces = FailingSpans({None: ([summary('first', messages=[]), summary('second', messages=[])], False, None)})

    with pytest.raises(OrqSourceError, match='span service unavailable'):
        await make_source(FakeOrq(traces)).load_async(
            START,
            END,
            2,
            facets=FacetSelection(),
            numeric=NumericFilters(),
            target_trace_ids={'first', 'second'} if targeted else None,
        )

    assert len(traces.list_span_calls) == 2
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_span_pagination_stops_at_page_limit() -> None:
    class EndlessSpans(FakeTraces):
        async def list_spans_async(self, *, trace_id: str, **kwargs: Any) -> Any:
            self.list_span_calls.append({'trace_id': trace_id, **kwargs})
            return namespace(data=[], has_more=True, next_page_token=f'page-{len(self.list_span_calls)}')

    traces = EndlessSpans({})
    source = make_source(FakeOrq(traces))

    with pytest.raises(OrqSourceError, match=f'exceeded {MAX_SPAN_PAGES} pages'):
        await source._list_spans('trace', asyncio.Semaphore(1))

    assert len(traces.list_span_calls) == MAX_SPAN_PAGES


@pytest.mark.asyncio
async def test_span_pagination_rejects_more_than_span_limit() -> None:
    class OversizeSpans(FakeTraces):
        async def list_spans_async(self, *, trace_id: str, **kwargs: Any) -> Any:
            self.list_span_calls.append({'trace_id': trace_id, **kwargs})
            return namespace(data=[namespace(span_id=f'span-{index}') for index in range(2001)], has_more=False)

    traces = OversizeSpans({})
    source = make_source(FakeOrq(traces))

    with pytest.raises(OrqSourceError, match='exceeded 2000 spans'):
        await source._list_spans('trace', asyncio.Semaphore(1))


def test_raw_response_capture_retains_only_supported_operations() -> None:
    capture = _RawResponseCapture()
    response = namespace(
        request=namespace(url=namespace(path='/v2/traces/query')),
        json=lambda: {'search': {'data': [{'trace_id': 'trace'}]}},
    )

    capture.after_success(namespace(operation_id='TracesQueryOql'), response)
    capture.after_success(namespace(operation_id='TracesSearch'), response)

    assert capture.pop('/traces/query') == {'search': {'data': [{'trace_id': 'trace'}]}}
    assert capture.pop('/traces/query') is None


def test_raw_response_capture_ignores_the_api_version_segment() -> None:
    capture = _RawResponseCapture()
    response = namespace(
        request=namespace(url=namespace(path='/v3/traces/query')),
        json=lambda: {'search': {'data': [{'trace_id': 'trace'}]}},
    )

    capture.after_success(namespace(operation_id='TracesQueryOql'), response)

    assert capture.pop('/traces/query') == {'search': {'data': [{'trace_id': 'trace'}]}}


@pytest.mark.asyncio
async def test_raw_response_capture_correlates_interleaved_requests() -> None:
    from evaluatorq.trace_finder.orq_source import _CAPTURE_REQUEST

    capture = _RawResponseCapture()
    ready = asyncio.Event()

    async def capture_one(trace_id: str, wait: bool) -> dict[str, Any] | None:
        marker = _CAPTURE_REQUEST.set(object())
        try:
            response = namespace(
                request=namespace(url=namespace(path='/v3/traces/query')),
                json=lambda: {'search': {'data': [{'trace_id': trace_id}]}},
            )
            capture.after_success(namespace(operation_id='TracesQueryOql'), response)
            if wait:
                ready.set()
                await asyncio.sleep(0)
            else:
                await ready.wait()
            return capture.pop('/traces/query')
        finally:
            _CAPTURE_REQUEST.reset(marker)

    first, second = await asyncio.gather(capture_one('first', True), capture_one('second', False))

    assert first == {'search': {'data': [{'trace_id': 'first'}]}}
    assert second == {'search': {'data': [{'trace_id': 'second'}]}}


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ('Unset', 'ok'))
async def test_first_error_message_uses_raw_sdk_status_shape_without_logging_payload(status: str) -> None:
    raw = {
        'span': {
            'span_id': 'failed-span',
            'attributes': {
                'otlp': {'status': {'message': '<observed error text>'}},
                'otel': {'status_description': 'fallback text'},
            },
        }
    }
    traces = FakeTraces({}, details={('trace', 'failed-span'): {}}, raw_details={('trace', 'failed-span'): raw})
    client = with_hooks(FakeOrq(traces))
    source = make_source(client)
    traces.capture_hook = client.sdk_configuration._hooks.after_success_hooks[0]

    message = await source.first_error_message(
        'trace', [
            {'span_id': 'failed-span', 'status': status, 'status_code': 'ERROR'},
            {'span_id': 'later', 'status': 'error'},
        ]
    )

    assert message == '<observed error text>'
    assert [call['span_id'] for call in traces.get_span_calls] == ['failed-span']
    assert traces.get_span_calls[0]['timeout_ms'] > 0


def test_conversation_messages_accepts_direct_messages() -> None:
    assert _conversation_messages({'messages': [{'role': 'user', 'content': 'direct'}]}) == [
        {'role': 'user', 'content': 'direct'}
    ]


def test_conversation_messages_keeps_output_with_direct_input_messages() -> None:
    assert _conversation_messages({
        'messages': [{'role': 'user', 'content': 'question'}],
        'attributes': {'gen_ai': {'output': {'messages': [{'role': 'assistant', 'content': 'answer'}]}}},
    }) == [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}]


def test_conversation_messages_omits_empty_reasoning_part_before_visible_answer() -> None:
    assert _conversation_messages({
        'attributes': {'gen_ai': {
            'input': {'messages': [{'role': 'system', 'content': 'instructions'}]},
            'output': {'messages': [
                {'role': 'assistant', 'parts': [{'type': 'reasoning', 'content': '[encrypted]'}]},
                {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'answer'}]},
            ]},
        }},
    }) == [
        {'role': 'system', 'content': 'instructions'},
        {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'answer'}]},
    ]


def test_conversation_messages_uses_input_when_direct_messages_are_blank() -> None:
    assert _conversation_messages({
        'messages': [{'role': 'user', 'content': ''}],
        'input': {'messages': [{'role': 'user', 'content': 'usable'}]},
    }) == [{'role': 'user', 'content': 'usable'}]


def test_conversation_messages_decodes_json_conversation_strings() -> None:
    assert _conversation_messages({'input': '{"messages": [{"role": "user", "content": "json"}]}'}) == [
        {'role': 'user', 'content': 'json'}
    ]


def test_conversation_messages_extracts_prompt_and_output() -> None:
    assert _conversation_messages({'input': {'prompt': 'prompt'}, 'output': 'output'}) == [
        {'role': 'user', 'content': 'prompt'},
        {'role': 'assistant', 'content': 'output'},
    ]


def test_conversation_messages_extracts_completion_choices() -> None:
    assert _conversation_messages({
        'output': {
            'choices': [
                {'message': {'role': 'assistant', 'content': 'choice message'}},
                {'text': 'choice text'},
            ]
        }
    }) == [
        {'role': 'assistant', 'content': 'choice message'},
        {'content': 'choice text', 'role': 'assistant'},
    ]


def test_conversation_messages_preserves_tool_calls() -> None:
    tool_calls = [{'id': 'call-1', 'type': 'function', 'function': {'name': 'search'}}]
    assert _conversation_messages({'messages': [{'role': 'assistant', 'tool_calls': tool_calls}]}) == [
        {'role': 'assistant', 'tool_calls': tool_calls}
    ]


def test_conversation_messages_preserves_content_parts() -> None:
    parts = [{'type': 'text', 'text': 'part'}]
    assert _conversation_messages({'messages': [{'role': 'user', 'parts': parts}]}) == [
        {'role': 'user', 'parts': parts, 'content': 'part'}
    ]


LONG = 'Why was my invoice 4411 charged twice? ' * 60


def _otel_payload() -> dict[str, Any]:
    return {
        'attributes': {
            'gen_ai': {
                'input': [{'role': 'user', 'parts': [{'type': 'text', 'content': LONG}]}],
                'output': [
                    {
                        'role': 'assistant',
                        'parts': [
                            {
                                'type': 'tool_call',
                                'id': 'c1',
                                'name': 'lookup_invoice',
                                'arguments': {'id': 4411},
                            }
                        ],
                    },
                    {
                        'role': 'tool',
                        'parts': [
                            {
                                'type': 'tool_call_response',
                                'id': 'c1',
                                'response': 'Error: invoice service unavailable',
                            }
                        ],
                    },
                    {'role': 'assistant', 'parts': [{'type': 'text', 'content': 'Your invoice was paid.'}]},
                ],
            }
        }
    }


def test_conversation_messages_normalises_otel_parts() -> None:
    messages = _conversation_messages(_otel_payload())
    assert messages[0] == {'role': 'user', 'content': LONG}
    assert messages[1]['tool_calls'][0]['function']['name'] == 'lookup_invoice'
    assert messages[2]['role'] == 'tool' and messages[2]['tool_call_id'] == 'c1'
    assert 'invoice service unavailable' in messages[2]['content']
    assert messages[3] == {'role': 'assistant', 'content': 'Your invoice was paid.'}


def test_conversation_messages_normalises_otel_parts_given_as_json_string() -> None:
    payload = _otel_payload()
    payload['attributes']['gen_ai'] = {
        key: json.dumps(value) for key, value in payload['attributes']['gen_ai'].items()
    }
    assert _conversation_messages(payload) == _conversation_messages(_otel_payload())


def test_conversation_messages_normalises_responses_items() -> None:
    payload = {
        'input': [
            {'role': 'user', 'content': 'refund order 9'},
            {'type': 'function_call', 'call_id': 'c9', 'name': 'refund', 'arguments': '{"order": 9}'},
            {'type': 'function_call_output', 'call_id': 'c9', 'output': 'Error: order not found'},
        ]
    }
    messages = _conversation_messages(payload)
    assert any(m.get('tool_calls') and m['tool_calls'][0]['function']['name'] == 'refund' for m in messages)
    assert any(m['role'] == 'tool' and 'order not found' in m['content'] for m in messages)


def test_conversation_messages_keeps_responses_calls_beside_chat_parts() -> None:
    payload = {
        'input': [
            {'role': 'user', 'parts': [{'type': 'text', 'text': 'find order 9'}]},
            {'type': 'function_call', 'call_id': 'c9', 'name': 'lookup_order', 'arguments': '{"id": 9}'},
            {'type': 'function_call_output', 'call_id': 'c9', 'output': 'order found'},
        ]
    }

    messages = _conversation_messages(payload)

    assert messages[0] == {'role': 'user', 'content': 'find order 9'}
    assert messages[1]['role'] == 'assistant'
    assert messages[1]['tool_calls'][0]['function']['name'] == 'lookup_order'
    assert messages[2] == {'role': 'tool', 'tool_call_id': 'c9', 'content': 'order found'}


def test_conversation_messages_keeps_mixed_responses_envelope_sides() -> None:
    payload = {
        'input': {
            'input': [
                {'role': 'user', 'parts': [{'type': 'text', 'text': 'find order 9'}]},
                {'type': 'function_call_output', 'call_id': 'c9', 'output': 'order found'},
            ]
        },
        'output': {
            'output': [
                {'role': 'assistant', 'parts': [{'type': 'text', 'text': 'found it'}]},
                {'type': 'function_call', 'call_id': 'c10', 'name': 'send_update', 'arguments': '{}'},
            ]
        },
    }

    messages = _conversation_messages(payload)

    assert messages[0] == {'role': 'user', 'content': 'find order 9'}
    assert messages[1] == {'role': 'tool', 'tool_call_id': 'c9', 'content': 'order found'}
    assert messages[2] == {'role': 'assistant', 'content': 'found it'}
    assert messages[3]['tool_calls'][0]['function']['name'] == 'send_update'


def namespace(**values: Any) -> SimpleNamespace:
    return SimpleNamespace(**values)


def make_source(client: FakeOrq, **kwargs: Any) -> OrqTraceSource:
    return OrqTraceSource(cast(Any, client), **kwargs)


def summary(
    trace_id: str,
    *,
    minute: int = 0,
    messages: list[dict[str, Any]] | None = None,
    project_id: str = 'project-unknown',
) -> Any:
    timestamp = START + timedelta(minutes=minute)
    return namespace(
        trace_id=trace_id,
        root_span_id=f'root-{trace_id}',
        leading_span_id=None,
        started_at=timestamp,
        ended_at=timestamp,
        project_id=project_id,
        status='trace-status',
        product='trace-product',
        providers=['trace-provider'],
        models=['trace-model'],
        type='trace',
        attributes=conversation_attributes(messages if messages is not None else user_messages(trace_id)),
    )


def span(
    span_id: str,
    *,
    minute: int,
    span_type: str = 'span.chat_completion',
    parent_span_id: str | None = None,
) -> Any:
    timestamp = START + timedelta(minutes=minute)
    return namespace(
        span_id=span_id,
        parent_span_id=parent_span_id,
        type=span_type,
        operation=span_type,
        started_at=timestamp,
        ended_at=timestamp,
        provider='provider',
        model='model',
        status='completed',
        has_detail=True,
    )


def detail(
    span_id: str,
    content: str,
    *,
    minute: int,
    provider: str = 'provider',
    model: str = 'model',
    agent_name: str = '',
    tool_names: list[str] | None = None,
    total_tokens: int | None = None,
    duration_ms: int | None = None,
) -> Any:
    summary_value = span(span_id, minute=minute)
    summary_value.provider = provider
    summary_value.model = model
    summary_value.agent_name = agent_name
    summary_value.tool_name = tool_names or []
    summary_value.total_tokens = total_tokens
    summary_value.duration_ms = duration_ms
    return namespace(summary=summary_value, attributes=conversation_attributes(user_messages(content)))


def user_messages(content: str) -> list[dict[str, Any]]:
    return [{'role': 'user', 'content': content}]


def conversation_attributes(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {'gen_ai': {'input': {'messages': messages}}}


@pytest.mark.asyncio
@pytest.mark.parametrize('summary_messages', [
    user_messages('short question'),
    [
        {'role': 'user', 'content': 'earlier question'},
        {'role': 'assistant', 'content': 'earlier answer'},
        {'role': 'user', 'content': 'short question'},
    ],
])
async def test_hydrate_rows_fetches_full_span_when_summary_omits_reported_reply(
    summary_messages: list[dict[str, Any]],
) -> None:
    trace = summary('partial', messages=summary_messages)
    trace.usage = namespace(prompt_tokens=1339, completion_tokens=262, prompt_cached_tokens=1336)
    hydrated = detail('reply', '', minute=1)
    hydrated.attributes = {
        'gen_ai': {
            'input': {'messages': [
                {'role': 'system', 'parts': [{'type': 'text', 'content': 'full instructions'}]},
                {'role': 'user', 'parts': [{'type': 'text', 'content': 'short question'}]},
            ]},
            'output': {'messages': [{'role': 'assistant', 'parts': [{'type': 'text', 'content': 'full answer'}]}]},
        }
    }
    traces = FakeTraces(
        {None: ([trace], False, None)},
        spans={'partial': [span('reply', minute=1, span_type='span.responses')]},
        details={('partial', 'reply'): hydrated},
    )
    source = make_source(FakeOrq(traces))

    rows = await source.search(START, END, 1, facets=FacetSelection(), numeric=NumericFilters())
    records = await source.hydrate_rows(rows)

    record = records['partial']
    assert record is not None
    assert record.span_id == 'reply'
    assert [message['role'] for message in record.messages] == ['system', 'user', 'assistant']
    assert record.messages[-1]['parts'][0]['content'] == 'full answer'
    assert [call['span_id'] for call in traces.get_span_calls] == ['reply']


@pytest.mark.asyncio
async def test_hydrate_rows_warns_and_keeps_summary_if_reported_reply_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = summary('partial', messages=user_messages('short question'))
    trace.usage = namespace(prompt_tokens=10, completion_tokens=5)
    warnings: list[str] = []
    monkeypatch.setattr(
        'evaluatorq.trace_finder.orq_source.logger.warning',
        lambda message, *args: warnings.append(message.format(*args)),
    )
    source = make_source(FakeOrq(FakeTraces({None: ([trace], False, None)})))

    rows = await source.search(START, END, 1, facets=FacetSelection(), numeric=NumericFilters())
    records = await source.hydrate_rows(rows)

    assert records['partial'] is not None
    assert records['partial'].messages == ({'role': 'user', 'content': 'short question'},)
    assert any('report output tokens but their hydrated spans contain no visible reply' in warning for warning in warnings)


@pytest.mark.asyncio
async def test_hydrate_rows_uses_typed_output_tokens_when_raw_usage_is_missing() -> None:
    traces = FakeTraces({})
    source = make_source(FakeOrq(traces))
    row = TraceRow(
        trace_id='typed-fallback',
        started_at=START,
        ended_at=START,
        models=('openai/gpt-5.6-luna',),
        tokens_out=5,
        raw={'trace_id': 'typed-fallback', 'messages': user_messages('question'), 'models': ['openai/gpt-5.6-luna']},
    )

    records = await source.hydrate_rows((row,))

    assert records['typed-fallback'] is not None
    assert len(traces.list_span_calls) == 1


@pytest.mark.asyncio
async def test_search_pages_without_hydration_and_keeps_server_order() -> None:
    first = summary('newer', minute=3)
    first.usage = namespace(prompt_tokens=100, completion_tokens=5, prompt_cached_tokens=50)
    traces = FakeTraces({
        None: ([first, summary('older', minute=1)], True, 'page-2'),
        'page-2': ([summary('oldest', minute=0)], False, None),
    })
    pages: list[int] = []

    rows = await make_source(FakeOrq(traces)).search(
        START,
        END,
        3,
        facets=FacetSelection(),
        numeric=NumericFilters(),
        on_page=lambda so_far: pages.append(len(so_far)),
    )

    assert [row.trace_id for row in rows] == ['newer', 'older', 'oldest']
    assert rows[0].tokens_in == 100
    assert pages == [2, 3]
    assert traces.list_span_calls == []
    assert traces.get_span_calls == []
    assert [call['limit'] for call in traces.query_calls] == [3, 1]


@pytest.mark.asyncio
async def test_search_stops_at_limit() -> None:
    traces = FakeTraces({None: ([summary('a'), summary('b')], True, 'page-2')})

    rows = await make_source(FakeOrq(traces)).search(
        START, END, 2, facets=FacetSelection(), numeric=NumericFilters()
    )

    assert len(rows) == 2
    assert len(traces.query_calls) == 1


@pytest.mark.asyncio
async def test_search_rejects_repeated_page_token() -> None:
    traces = FakeTraces({None: ([summary('a')], True, 'same'), 'same': ([summary('b')], True, 'same')})

    with pytest.raises(OrqSourceError, match='repeated OQL page token'):
        await make_source(FakeOrq(traces)).search(
            START, END, 10, facets=FacetSelection(), numeric=NumericFilters()
        )


@pytest.mark.asyncio
async def test_search_keeps_the_project_guard() -> None:
    traces = FakeTraces({None: ([summary('wrong', project_id='project-a')], False, None)})
    projects = FakeProjects([namespace(project_id='project-b', name='Selected')])

    with pytest.raises(OrqSourceError, match='only traces outside the selected project'):
        await make_source(FakeOrq(traces, projects)).search(
            START, END, 5, facets=FacetSelection(project_id='project-b'), numeric=NumericFilters()
        )


@pytest.mark.asyncio
async def test_search_warns_once_for_exclusive_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = summary('claude-code')
    trace.usage = namespace(prompt_tokens=1150, completion_tokens=10, prompt_cached_tokens=65_600_000)
    warnings: list[str] = []
    monkeypatch.setattr(
        'evaluatorq.trace_finder.orq_source.logger.warning',
        lambda msg, *a: warnings.append(msg.format(*a)),
    )

    rows = await make_source(FakeOrq(FakeTraces({None: ([trace], False, None)}))).search(
        START, END, 5, facets=FacetSelection(), numeric=NumericFilters()
    )

    assert rows[0].tokens_in == 1150 + 65_600_000
    assert sum('outside prompt_tokens' in warning for warning in warnings) == 1


@pytest.mark.asyncio
async def test_search_warns_when_raw_summary_capture_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        'evaluatorq.trace_finder.orq_source.logger.warning',
        lambda message, *args: warnings.append(message.format(*args)),
    )
    source = make_source(FakeOrq(FakeTraces({None: ([summary('fallback')], False, None)})))

    rows = await source.search(START, END, 1, facets=FacetSelection(), numeric=NumericFilters())

    assert rows[0].trace_id == 'fallback'
    assert any('SDK-model fallback' in warning and 'optional explorer fields' in warning for warning in warnings)


class SpanLookupFailingTraces(FakeTraces):
    def __init__(self, *args: Any, failing: set[str], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.failing = failing

    async def list_spans_async(self, *, trace_id: str, **kwargs: Any) -> Any:
        if trace_id in self.failing:
            raise RuntimeError('span API down')
        return await super().list_spans_async(trace_id=trace_id, **kwargs)


class SpanDetailFailingTraces(FakeTraces):
    async def get_span_async(self, *, trace_id: str, span_id: str, **kwargs: Any) -> Any:
        raise RuntimeError('span detail API down')


def with_output_tokens(trace: Any) -> Any:
    trace.usage = namespace(prompt_tokens=10, completion_tokens=5)
    return trace


@pytest.mark.asyncio
async def test_hydrate_rows_keeps_summary_messages_when_the_span_lookup_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        'evaluatorq.trace_finder.orq_source.logger.warning',
        lambda message, *args: warnings.append(message.format(*args)),
    )
    trace = with_output_tokens(summary('partial', messages=user_messages('short question')))
    source = make_source(FakeOrq(SpanLookupFailingTraces({None: ([trace], False, None)}, failing={'partial'})))

    rows = await source.search(START, END, 1, facets=FacetSelection(), numeric=NumericFilters())
    records = await source.hydrate_rows(rows)

    assert records['partial'] is not None
    assert records['partial'].messages == ({'role': 'user', 'content': 'short question'},)
    assert sum('span lookup failed' in warning for warning in warnings) == 1


@pytest.mark.asyncio
async def test_hydrate_rows_keeps_summary_messages_when_span_detail_lookup_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        'evaluatorq.trace_finder.orq_source.logger.warning',
        lambda message, *args: warnings.append(message.format(*args)),
    )
    trace = with_output_tokens(summary('detail-failure', messages=user_messages('short question')))
    spans = {'detail-failure': [span('reply', minute=0, span_type='span.responses')]}
    source = make_source(FakeOrq(SpanDetailFailingTraces({None: ([trace], False, None)}, spans=spans)))

    rows = await source.search(START, END, 1, facets=FacetSelection(), numeric=NumericFilters())
    records = await source.hydrate_rows(rows)

    record = records['detail-failure']
    assert record is not None
    assert record.messages == ({'role': 'user', 'content': 'short question'},)
    assert sum('span lookup failed' in warning for warning in warnings) == 1


@pytest.mark.asyncio
async def test_hydrate_rows_omits_failed_trace_when_summary_is_unusable() -> None:
    trace = with_output_tokens(summary('empty', messages=[]))
    source = make_source(FakeOrq(SpanLookupFailingTraces({None: ([trace], False, None)}, failing={'empty'})))

    rows = await source.search(START, END, 1, facets=FacetSelection(), numeric=NumericFilters())
    records = await source.hydrate_rows(rows)

    assert records == {}


@pytest.mark.asyncio
async def test_one_failing_span_lookup_does_not_fail_the_rest_of_the_batch() -> None:
    bad = with_output_tokens(summary('bad', minute=1, messages=user_messages('bad question')))
    good = summary('good', minute=0, messages=user_messages('good question'))
    traces = SpanLookupFailingTraces({None: ([bad, good], False, None)}, failing={'bad'})
    source = make_source(FakeOrq(traces))

    rows = await source.search(START, END, 5, facets=FacetSelection(), numeric=NumericFilters())
    records = await source.hydrate_rows(rows)

    assert records['good'] is not None
    assert records['bad'] is not None
    assert records['bad'].messages == ({'role': 'user', 'content': 'bad question'},)


@pytest.mark.asyncio
async def test_hydrate_rows_returns_none_for_a_failed_trace() -> None:
    ok = summary('ok', messages=user_messages('hello'))
    traces = FakeTraces({None: ([ok, summary('empty', messages=[])], False, None)})
    source = make_source(FakeOrq(traces))
    rows = await source.search(START, END, 5, facets=FacetSelection(), numeric=NumericFilters())

    records = await source.hydrate_rows(rows)

    assert records['ok'] is not None
    assert records['ok'].messages[0]['content'] == 'hello'
    assert records['empty'] is None


@pytest.mark.asyncio
async def test_hydrate_rows_reports_progress_to_the_run() -> None:
    from evaluatorq.trace_finder.progress import set_load_reporter

    traces = FakeTraces({None: ([summary('a'), summary('b')], False, None)})
    source = make_source(FakeOrq(traces))
    rows = await source.search(START, END, 5, facets=FacetSelection(), numeric=NumericFilters())
    seen: list[tuple[int, int]] = []

    async def hydrate() -> None:
        set_load_reporter(lambda done, total: seen.append((done, total)))
        await source.hydrate_rows(rows)

    await asyncio.create_task(hydrate())

    assert sorted(seen) == [(1, 2), (2, 2)]


@pytest.mark.asyncio
async def test_search_rejects_limit_below_one() -> None:
    source = make_source(FakeOrq(FakeTraces({})))

    with pytest.raises(OrqSourceError, match='limit must be at least 1'):
        await source.search(START, END, 0, facets=FacetSelection(), numeric=NumericFilters())


@pytest.mark.asyncio
async def test_search_rejects_start_after_end() -> None:
    source = make_source(FakeOrq(FakeTraces({})))

    with pytest.raises(OrqSourceError, match='start must not be after end'):
        await source.search(END, START, 1, facets=FacetSelection(), numeric=NumericFilters())


@pytest.mark.asyncio
async def test_search_rejects_naive_datetime() -> None:
    source = make_source(FakeOrq(FakeTraces({})))

    with pytest.raises(OrqSourceError, match='start must include a timezone offset'):
        await source.search(datetime(2026, 9, 20), END, 1, facets=FacetSelection(), numeric=NumericFilters())
