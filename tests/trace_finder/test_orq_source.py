"""Behavioral contracts for the live Orq OQL trace source."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace, TracebackType
from typing import Any, cast

import pytest

from evaluatorq.trace_finder import FacetSelection, NumericFilters
from evaluatorq.trace_finder.models import Snapshot
from evaluatorq.trace_finder.orq_source import (
    OrqSourceError,
    OrqTraceSource,
    _RawResponseCapture,
    _conversation_messages,
    build_oql,
)

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
    ) -> None:
        self.pages = pages
        self.spans = spans or {}
        self.details = details or {}
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
        'fetch traces | filter operation not_in ("generate_content") '
        'and project_id in ("project-123") '
        'and model in ("gpt-5") '
        'and provider in ("openai") '
        'and status in ("error") '
        'and product in ("ai-gateway") '
        'and attributes.orq.leading_span.span_type in ("span.responses") '
        'and agent_name in ("support-agent") '
        'and tool_name in ("search") '
        '| filter total_tokens >= 500 '
        '| filter total_tokens <= 5000 '
        '| filter duration_ms >= 10 '
        '| filter duration_ms <= 1000 '
        '| sort end_time desc'
    )


def test_build_oql_rejects_a_project_name_that_cannot_be_resolved() -> None:
    with pytest.raises(OrqSourceError, match='cannot resolve selected project names'):
        build_oql(FacetSelection(project=frozenset({'Missing'})), NumericFilters(), {})


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


def test_conversation_messages_accepts_direct_messages() -> None:
    assert _conversation_messages({'messages': [{'role': 'user', 'content': 'direct'}]}) == [
        {'role': 'user', 'content': 'direct'}
    ]


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
        {'role': 'user', 'parts': parts}
    ]


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
