"""Integration test verifying OTel span output for the simulation module.

Uses an in-memory span exporter to capture real spans and validate names,
attributes, and hierarchy of the simulation tracing helpers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


class _CollectingExporter(SpanExporter):
    """Minimal in-memory exporter that collects finished spans."""

    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


@pytest.fixture
def span_collector():
    """Set up an in-memory OTel TracerProvider; patch the simulation tracer."""
    exporter = _CollectingExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('evaluatorq-simulation-test')

    with patch('evaluatorq.simulation.tracing.get_tracer', return_value=tracer):
        yield exporter

    provider.shutdown()


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _find(exporter: _CollectingExporter, name: str) -> ReadableSpan:
    for s in exporter.spans:
        if s.name == name:
            return s
    raise AssertionError(f'span {name!r} not found; got {[s.name for s in exporter.spans]}')


# ---------------------------------------------------------------------------
# with_simulation_span
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simulation_span_creates_span_with_attrs(
    span_collector: _CollectingExporter,
):
    from evaluatorq.simulation.tracing import with_simulation_span

    async with with_simulation_span(
        'orq.simulation.run',
        {'orq.simulation.persona': 'alice', 'orq.simulation.max_turns': 5},
    ) as span:
        assert span is not None

    s = _find(span_collector, 'orq.simulation.run')
    a = _attrs(s)
    assert a['orq.simulation.persona'] == 'alice'
    assert a['orq.simulation.max_turns'] == 5


@pytest.mark.asyncio
async def test_simulation_span_skips_none_attrs(
    span_collector: _CollectingExporter,
):
    from evaluatorq.simulation.tracing import with_simulation_span

    async with with_simulation_span('orq.simulation.run', {'orq.simulation.persona': None, 'kept': 'yes'}):
        pass

    s = _find(span_collector, 'orq.simulation.run')
    a = _attrs(s)
    assert 'orq.simulation.persona' not in a
    assert a['kept'] == 'yes'


@pytest.mark.asyncio
async def test_simulation_span_records_exception(
    span_collector: _CollectingExporter,
):
    from evaluatorq.simulation.tracing import with_simulation_span

    with pytest.raises(RuntimeError):
        async with with_simulation_span('orq.simulation.run', None):
            raise RuntimeError('boom')

    s = _find(span_collector, 'orq.simulation.run')
    a = _attrs(s)
    assert a['error.type'] == 'RuntimeError'
    # span ended with ERROR status (status set by helper)
    assert s.status.status_code.name == 'ERROR'


# ---------------------------------------------------------------------------
# with_llm_span
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_span_name_and_attrs(span_collector: _CollectingExporter):
    from evaluatorq.simulation.tracing import with_llm_span

    async with with_llm_span(
        model='azure/gpt-4o-mini',
        operation='chat',
        temperature=0.7,
        max_tokens=512,
        purpose='judge',
    ) as span:
        assert span is not None

    s = _find(span_collector, 'chat azure/gpt-4o-mini')
    a = _attrs(s)
    assert a['gen_ai.operation.name'] == 'chat'
    assert a['gen_ai.system'] == 'azure.ai.openai'
    assert a['gen_ai.provider.name'] == 'azure.ai.openai'
    assert a['gen_ai.request.model'] == 'azure/gpt-4o-mini'
    assert a['gen_ai.request.temperature'] == 0.7
    assert a['gen_ai.request.max_tokens'] == 512
    assert a['orq.simulation.llm_purpose'] == 'judge'
    # Domain-neutral key is dual-emitted (parity with openresponses.with_llm_span).
    assert a['orq.llm.purpose'] == 'judge'


@pytest.mark.asyncio
async def test_llm_span_responses_operation(span_collector: _CollectingExporter):
    from evaluatorq.simulation.tracing import with_llm_span

    async with with_llm_span(model='openai/gpt-4o', operation='responses'):
        pass

    s = _find(span_collector, 'responses openai/gpt-4o')
    assert _attrs(s)['gen_ai.operation.name'] == 'responses'


# ---------------------------------------------------------------------------
# Recording helpers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_token_usage_sets_genai_attrs(
    span_collector: _CollectingExporter,
):
    from evaluatorq.common.tracing import record_token_usage
    from evaluatorq.simulation.tracing import with_llm_span

    async with with_llm_span(model='openai/gpt-4o') as span:
        record_token_usage(span, prompt_tokens=10, completion_tokens=20, total_tokens=30)

    a = _attrs(_find(span_collector, 'chat openai/gpt-4o'))
    assert a['gen_ai.usage.input_tokens'] == 10
    assert a['gen_ai.usage.output_tokens'] == 20
    assert a['gen_ai.usage.total_tokens'] == 30


@pytest.mark.asyncio
async def test_record_llm_input_serializes_messages(
    span_collector: _CollectingExporter,
):
    from evaluatorq.common.tracing import record_llm_input
    from evaluatorq.simulation.tracing import with_llm_span

    async with with_llm_span(model='openai/gpt-4o') as span:
        record_llm_input(span, [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'yo'}])

    a = _attrs(_find(span_collector, 'chat openai/gpt-4o'))
    assert 'gen_ai.input.messages' in a
    parsed = json.loads(a['gen_ai.input.messages'])
    assert parsed == [
        {'role': 'user', 'content': 'hi'},
        {'role': 'assistant', 'content': 'yo'},
    ]
    # Platform fallback alias
    assert a['input'] == a['gen_ai.input.messages']


@pytest.mark.asyncio
async def test_record_llm_input_truncates_long_content(
    span_collector: _CollectingExporter,
    monkeypatch: pytest.MonkeyPatch,
):
    from evaluatorq.common.tracing import (
        _RECOMMENDED_SPAN_MAX_TEXT_CHARS,
        _default_span_max_text_chars,
        record_llm_input,
    )
    from evaluatorq.simulation.tracing import with_llm_span

    # Truncation is off by default (capture all); opt in via the env var.
    monkeypatch.setenv('EVALUATORQ_SPAN_MAX_TEXT_CHARS', str(_RECOMMENDED_SPAN_MAX_TEXT_CHARS))
    _default_span_max_text_chars.cache_clear()
    try:
        long_content = 'x' * 10000
        async with with_llm_span(model='openai/gpt-4o') as span:
            record_llm_input(span, [{'role': 'user', 'content': long_content}])

        a = _attrs(_find(span_collector, 'chat openai/gpt-4o'))
        serialized = a['gen_ai.input.messages']
        parsed = json.loads(serialized)
        # Content gets truncated to the opted-in cap (8192).
        assert len(parsed[0]['content']) == _RECOMMENDED_SPAN_MAX_TEXT_CHARS
        assert parsed[0]['content'].endswith('... [truncated]')
    finally:
        _default_span_max_text_chars.cache_clear()


@pytest.mark.asyncio
async def test_record_token_usage_preserves_zero_prompt_tokens(
    span_collector: _CollectingExporter,
):
    """Zero input_tokens (e.g. fully-cached request) must not fall back to the
    truthy `prompt_tokens` alias. Regression guard for the falsy-or chain bug:
    `_usage_first_int` is presence-based (`('input_tokens', 'prompt_tokens', ...)`,
    first *present* numeric wins), so it must select the present `0` over the
    conflicting nonzero alias rather than treating `0` as "absent" and falling
    through to `prompt_tokens=99`."""
    from evaluatorq.common.tracing import record_llm_response
    from evaluatorq.simulation.tracing import with_llm_span

    class _Usage:
        input_tokens = 0  # fully-cached request: zero input tokens is a real value
        prompt_tokens = 99  # would be selected if `input_tokens or prompt_tokens`
        output_tokens = 5
        total_tokens = 5
        prompt_tokens_details = None

    class _Resp:
        id = 'r'
        model = 'm'
        usage = _Usage()
        choices = []

    async with with_llm_span(model='openai/gpt-4o') as span:
        record_llm_response(span, _Resp())

    a = _attrs(_find(span_collector, 'chat openai/gpt-4o'))
    assert a['gen_ai.usage.input_tokens'] == 0
    assert a['gen_ai.usage.output_tokens'] == 5


@pytest.mark.asyncio
async def test_record_llm_response_responses_api_shape(
    span_collector: _CollectingExporter,
):
    """recordLLMResponse with a Responses API-shaped object: output items
    have content[].text, not flat .text on the item."""
    from evaluatorq.common.tracing import record_llm_response
    from evaluatorq.simulation.tracing import with_llm_span

    class _ContentPart:
        text = 'hello world'

    class _OutputItem:
        # No .text on the item itself; SDK shape uses .content[*].text
        content = [_ContentPart()]

    class _Usage:
        input_tokens = 4
        output_tokens = 2
        total_tokens = 6
        prompt_tokens_details = None

    class _Resp:
        id = 'resp_1'
        model = 'openai/gpt-4o'
        usage = _Usage()
        # No .choices; this is the Responses API shape
        output = [_OutputItem()]

    async with with_llm_span(model='openai/gpt-4o', operation='responses') as span:
        record_llm_response(span, _Resp())

    a = _attrs(_find(span_collector, 'responses openai/gpt-4o'))
    assert 'gen_ai.output.messages' in a
    parsed = json.loads(a['gen_ai.output.messages'])
    assert parsed == [{'role': 'assistant', 'content': 'hello world'}]
    # Falls back from .input_tokens since prompt_tokens is absent
    assert a['gen_ai.usage.input_tokens'] == 4
    assert a['gen_ai.usage.output_tokens'] == 2


@pytest.mark.asyncio
async def test_record_llm_response_dict_responses_api_shape(
    span_collector: _CollectingExporter,
):
    from evaluatorq.common.tracing import record_llm_response
    from evaluatorq.simulation.tracing import with_llm_span

    response = {
        'id': 'resp_dict',
        'model': 'openai/gpt-4o',
        'status': 'completed',
        'usage': {'input_tokens': 4, 'output_tokens': 2, 'total_tokens': 6},
        'output': [
            {
                'type': 'message',
                'content': [{'type': 'output_text', 'text': 'hello dict'}],
            }
        ],
    }

    async with with_llm_span(model='openai/gpt-4o', operation='responses') as span:
        record_llm_response(span, response)

    a = _attrs(_find(span_collector, 'responses openai/gpt-4o'))
    assert a['gen_ai.response.id'] == 'resp_dict'
    assert a['gen_ai.response.model'] == 'openai/gpt-4o'
    assert a['gen_ai.usage.input_tokens'] == 4
    assert a['gen_ai.usage.output_tokens'] == 2
    assert a['gen_ai.response.finish_reasons'] == ('completed',)
    parsed = json.loads(a['gen_ai.output.messages'])
    assert parsed == [{'role': 'assistant', 'content': 'hello dict'}]


@pytest.mark.asyncio
async def test_record_openresponses_request_sets_max_tokens(
    span_collector: _CollectingExporter,
):
    from evaluatorq.openresponses.tracing import record_openresponses_request
    from evaluatorq.simulation.tracing import with_llm_span

    async with with_llm_span(model='openai/gpt-4o', operation='responses') as span:
        record_openresponses_request(
            span,
            {
                'model': 'openai/gpt-4o',
                'input': [{'role': 'user', 'content': 'hi'}],
                'max_output_tokens': 123,
            },
        )

    a = _attrs(_find(span_collector, 'responses openai/gpt-4o'))
    assert a['gen_ai.request.max_tokens'] == 123
    assert 'orq.openresponses.request' in a


@pytest.mark.asyncio
async def test_record_llm_response_tool_call_only_chat_output(
    span_collector: _CollectingExporter,
):
    from evaluatorq.common.tracing import record_llm_response
    from evaluatorq.simulation.tracing import with_llm_span

    class _Function:
        name = 'finish_conversation'
        arguments = '{"reason":"done"}'

    class _ToolCall:
        function = _Function()

    class _Message:
        role = 'assistant'
        content = None
        tool_calls = [_ToolCall()]

    class _Choice:
        finish_reason = 'tool_calls'
        message = _Message()

    class _Resp:
        id = 'resp_tool'
        model = 'openai/gpt-4o'
        usage = None
        choices = [_Choice()]

    async with with_llm_span(model='openai/gpt-4o') as span:
        record_llm_response(span, _Resp())

    a = _attrs(_find(span_collector, 'chat openai/gpt-4o'))
    parsed = json.loads(a['gen_ai.output.messages'])
    assert parsed[0]['role'] == 'assistant'
    payload = json.loads(parsed[0]['content'])
    assert payload['tool_calls'] == [{'name': 'finish_conversation', 'arguments': '{"reason":"done"}'}]


@pytest.mark.asyncio
async def test_simulation_span_records_asyncio_cancellation(
    span_collector: _CollectingExporter,
):
    """asyncio.CancelledError (a BaseException, not Exception) must be
    recorded on the span — otherwise timed-out simulations end with UNSET
    status and look like normal completions in the dashboard."""
    import asyncio

    from evaluatorq.simulation.tracing import with_simulation_span

    with pytest.raises(asyncio.CancelledError):
        async with with_simulation_span('orq.simulation.run', None):
            raise asyncio.CancelledError()

    s = _find(span_collector, 'orq.simulation.run')
    assert s.status.status_code.name == 'ERROR'
    assert _attrs(s)['error.type'] == 'CancelledError'


@pytest.mark.asyncio
async def test_llm_span_records_asyncio_cancellation(
    span_collector: _CollectingExporter,
):
    """Mirror of the with_simulation_span cancellation test for the LLM
    span path (timed-out LLM calls must record as ERROR on the span)."""
    import asyncio

    from evaluatorq.simulation.tracing import with_llm_span

    with pytest.raises(asyncio.CancelledError):
        async with with_llm_span(model='openai/gpt-4o'):
            raise asyncio.CancelledError()

    s = _find(span_collector, 'chat openai/gpt-4o')
    assert s.status.status_code.name == 'ERROR'
    assert _attrs(s)['error.type'] == 'CancelledError'


@pytest.mark.asyncio
async def test_record_llm_response_responses_api_finish_reason(
    span_collector: _CollectingExporter,
):
    """gen_ai.response.finish_reasons must be set from response.status when
    the response has no .choices (Responses API shape)."""
    from evaluatorq.common.tracing import record_llm_response
    from evaluatorq.simulation.tracing import with_llm_span

    class _Resp:
        id = 'resp_x'
        model = 'openai/gpt-4o'
        usage = None
        status = 'completed'
        # No .choices

    async with with_llm_span(model='openai/gpt-4o', operation='responses') as span:
        record_llm_response(span, _Resp())

    a = _attrs(_find(span_collector, 'responses openai/gpt-4o'))
    assert a['gen_ai.response.finish_reasons'] == ('completed',)


@pytest.mark.asyncio
async def test_llm_span_records_exception(span_collector: _CollectingExporter):
    """Exceptions inside with_llm_span are recorded with ERROR status."""
    from evaluatorq.simulation.tracing import with_llm_span

    with pytest.raises(ValueError, match='boom'):
        async with with_llm_span(model='openai/gpt-4o'):
            raise ValueError('boom')

    s = _find(span_collector, 'chat openai/gpt-4o')
    assert _attrs(s)['error.type'] == 'ValueError'
    assert s.status.status_code.name == 'ERROR'


@pytest.mark.asyncio
async def test_record_llm_response_chat_completions(
    span_collector: _CollectingExporter,
):
    """recordLLMResponse with a Chat Completions-shaped object."""
    from evaluatorq.common.tracing import record_llm_response
    from evaluatorq.simulation.tracing import with_llm_span

    class _Usage:
        prompt_tokens = 7
        completion_tokens = 11
        total_tokens = 18
        prompt_tokens_details = None

    class _Msg:
        role = 'assistant'
        content = 'hello'

    class _Choice:
        finish_reason = 'stop'
        message = _Msg()

    class _Resp:
        id = 'resp-123'
        model = 'azure/gpt-4o-mini'
        usage = _Usage()
        choices = [_Choice()]

    async with with_llm_span(model='azure/gpt-4o-mini') as span:
        record_llm_response(span, _Resp())

    a = _attrs(_find(span_collector, 'chat azure/gpt-4o-mini'))
    assert a['gen_ai.response.id'] == 'resp-123'
    assert a['gen_ai.response.model'] == 'azure/gpt-4o-mini'
    assert a['gen_ai.usage.input_tokens'] == 7
    assert a['gen_ai.usage.output_tokens'] == 11
    assert a['gen_ai.response.finish_reasons'] == ('stop',)
    parsed = json.loads(a['gen_ai.output.messages'])
    assert parsed == [{'role': 'assistant', 'content': 'hello'}]


# ---------------------------------------------------------------------------
# Hierarchy: nested simulation spans share the same trace
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nested_spans_share_trace(span_collector: _CollectingExporter):
    from evaluatorq.simulation.tracing import with_llm_span, with_simulation_span

    async with with_simulation_span('Evaluatorq - Agent Simulation', None):
        async with with_simulation_span('orq.simulation.run', None):
            async with with_simulation_span('orq.simulation.turn', None):
                async with with_llm_span(model='openai/gpt-4o', purpose='target'):
                    pass

    pipeline = _find(span_collector, 'Evaluatorq - Agent Simulation')
    run = _find(span_collector, 'orq.simulation.run')
    turn = _find(span_collector, 'orq.simulation.turn')
    llm = _find(span_collector, 'chat openai/gpt-4o')

    # All four spans share the same trace_id
    trace_ids = {pipeline.context.trace_id, run.context.trace_id, turn.context.trace_id, llm.context.trace_id}  # pyright: ignore[reportOptionalMemberAccess]
    assert len(trace_ids) == 1

    # Parent chain: llm → turn → run → pipeline → root
    assert llm.parent.span_id == turn.context.span_id  # pyright: ignore[reportOptionalMemberAccess]
    assert turn.parent.span_id == run.context.span_id  # pyright: ignore[reportOptionalMemberAccess]
    assert run.parent.span_id == pipeline.context.span_id  # pyright: ignore[reportOptionalMemberAccess]
    assert pipeline.parent is None


# ---------------------------------------------------------------------------
# get_tracer is None (tracing disabled): helpers degrade to no-ops
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_simulation_produces_full_span_tree(
    span_collector: _CollectingExporter,
    monkeypatch: pytest.MonkeyPatch,
):
    """End-to-end smoke test: run a 2-turn simulation with mocks and verify
    the full span hierarchy is produced (pipeline → run → turn → leaf calls).

    Prints the captured span tree to stdout for visual inspection (run with
    ``-s`` to see).
    """
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    from evaluatorq.contracts import TokenUsage
    from evaluatorq.simulation.runner.simulation import SimulationRunner
    from evaluatorq.simulation.tracing import with_simulation_span
    from evaluatorq.simulation.types import (
        CommunicationStyle,
        Message,
        Persona,
        Scenario,
        SimulationDatapoint,
    )

    # Build mock judge: should_terminate after the 2nd turn
    def _judgment(*, terminate: bool) -> MagicMock:
        j = MagicMock()
        j.should_terminate = terminate
        j.goal_achieved = terminate
        j.goal_completion_score = 1.0 if terminate else 0.5
        j.rules_broken = []
        j.reason = 'done' if terminate else 'keep going'
        j.response_quality = 0.9
        j.hallucination_risk = 0.1
        j.tone_appropriateness = 0.9
        j.factual_accuracy = 0.9
        return j

    judge = MagicMock()
    judge.evaluate = AsyncMock(side_effect=[_judgment(terminate=False), _judgment(terminate=True)])
    judge.get_usage = MagicMock(return_value=TokenUsage())

    user_sim = MagicMock()
    user_sim.generate_first_message = AsyncMock(return_value='Hi, I need help.')
    user_sim.respond_async = AsyncMock(return_value='ok thanks')
    user_sim.get_usage = MagicMock(return_value=TokenUsage())

    target_calls: list[int] = []

    async def target_cb(messages: list[Message]) -> str:
        target_calls.append(len(messages))
        return f'agent reply #{len(target_calls)}'

    runner = SimulationRunner(
        target=target_cb,
        model='azure/gpt-4o-mini',
        max_turns=3,
        user_simulator=user_sim,
        judge=judge,
    )

    persona = Persona(
        name='Tester',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='A test user',
    )
    scenario = Scenario(name='Smoke', goal='Get help')
    dp = SimulationDatapoint(
        id='dp-smoke',
        persona=persona,
        scenario=scenario,
        user_system_prompt='sys',
        first_message='Hi, I need help.',
    )

    # Wrap the run() in an outer pipeline span so the smoke output mirrors
    # what simulate() would produce.
    async with with_simulation_span(
        'Evaluatorq - Agent Simulation',
        {'orq.simulation.evaluation_name': 'smoke', 'orq.simulation.max_turns': 3},
    ):
        result = await runner.run(datapoint=dp)

    assert result.terminated_by.value == 'judge'
    assert result.turn_count == 2
    assert target_calls == [1, 3]

    names = [s.name for s in span_collector.spans]
    # Required spans
    assert 'Evaluatorq - Agent Simulation' in names
    assert 'orq.simulation.run' in names
    assert names.count('orq.simulation.turn') == 2
    assert names.count('orq.simulation.target_call') == 2
    assert names.count('orq.simulation.judge_evaluation') == 2
    # User simulator only fires when not the last turn AND not terminated:
    # turn 1 (not terminated, not last) -> 1 call; turn 2 (terminated) -> 0
    assert names.count('orq.simulation.user_simulator_call') == 1

    # Hierarchy: each turn span is a child of run, which is a child of pipeline
    pipeline = _find(span_collector, 'Evaluatorq - Agent Simulation')
    run = _find(span_collector, 'orq.simulation.run')
    assert run.parent.span_id == pipeline.context.span_id  # pyright: ignore[reportOptionalMemberAccess]

    turn_spans = [s for s in span_collector.spans if s.name == 'orq.simulation.turn']
    for t in turn_spans:
        assert t.parent.span_id == run.context.span_id  # pyright: ignore[reportOptionalMemberAccess]

    # target_call spans are children of their turn
    for tc in [s for s in span_collector.spans if s.name == 'orq.simulation.target_call']:
        assert any(tc.parent.span_id == t.context.span_id for t in turn_spans)  # pyright: ignore[reportOptionalMemberAccess]

    # Run span carries termination attrs but NO token usage: usage lives on the
    # per-call LLM spans, and the sink aggregates. Emitting a rolled-up total here
    # too would double-count it against the children it was summed from.
    run_attrs = _attrs(run)
    assert run_attrs['orq.simulation.terminated_by'] == 'judge'
    assert run_attrs['orq.simulation.turn_count'] == 2
    assert not [k for k in run_attrs if k.startswith('gen_ai.usage.')]

    # Pretty-print the span tree for visual inspection
    print('\n=== Captured span tree (smoke test) ===')
    by_id = {s.context.span_id: s for s in span_collector.spans}  # pyright: ignore[reportOptionalMemberAccess]
    children: dict[int | None, list[ReadableSpan]] = {}
    for s in span_collector.spans:
        parent_id = s.parent.span_id if s.parent else None
        children.setdefault(parent_id, []).append(s)

    def _print(node: ReadableSpan, depth: int) -> None:
        attrs = _attrs(node)
        # Show a couple of high-signal attrs inline
        hint_keys = (
            'orq.simulation.turn',
            'orq.simulation.terminated_by',
            'gen_ai.operation.name',
            'orq.simulation.llm_purpose',
            'gen_ai.usage.total_tokens',
        )
        hints = ' '.join(f'{k.split(".")[-1]}={attrs[k]}' for k in hint_keys if k in attrs)
        print(f'{"  " * depth}- {node.name}' + (f'  [{hints}]' if hints else ''))
        for child in children.get(node.context.span_id, []):  # pyright: ignore[reportOptionalMemberAccess]
            _print(child, depth + 1)

    for root in children.get(None, []):
        _print(root, 0)
    print(f'=== {len(span_collector.spans)} spans total ===\n')


@pytest.mark.asyncio
async def test_run_span_carries_no_usage_and_result_carries_the_aggregate(
    span_collector: _CollectingExporter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Usage belongs on the per-call LLM spans; the sink aggregates from those.

    The run span deliberately carries none, so a rolled-up total is never
    double-counted against the children it was summed from. The aggregate is
    still computed — it lives on the SimulationResult, which is what the
    reports read. Pins both halves: absent on the span, exact on the result."""
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    from evaluatorq.contracts import TokenUsage
    from evaluatorq.simulation.runner.simulation import SimulationRunner
    from evaluatorq.simulation.types import CommunicationStyle, Message, Persona, Scenario, SimulationDatapoint

    def _judgment(*, terminate: bool) -> MagicMock:
        j = MagicMock()
        j.should_terminate = terminate
        j.goal_achieved = terminate
        j.goal_completion_score = 1.0 if terminate else 0.5
        j.rules_broken = []
        j.reason = 'done' if terminate else 'keep going'
        j.response_quality = 0.9
        j.hallucination_risk = 0.1
        j.tone_appropriateness = 0.9
        j.factual_accuracy = 0.9
        return j

    # judge + user simulator each report 1 call per turn (2 turns -> 2 calls each).
    judge = MagicMock()
    judge.evaluate = AsyncMock(side_effect=[_judgment(terminate=False), _judgment(terminate=True)])
    judge.get_usage = MagicMock(return_value=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, calls=2))

    user_sim = MagicMock()
    user_sim.generate_first_message = AsyncMock(return_value='Hi, I need help.')
    user_sim.respond_async = AsyncMock(return_value='ok thanks')
    user_sim.get_usage = MagicMock(return_value=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2, calls=1))

    async def target_cb(messages: list[Message]) -> str:
        return 'agent reply'

    runner = SimulationRunner(target=target_cb, model='azure/gpt-4o-mini', max_turns=3, user_simulator=user_sim, judge=judge)

    persona = Persona(
        name='Tester',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='A test user',
    )
    scenario = Scenario(name='Smoke', goal='Get help')
    dp = SimulationDatapoint(
        id='dp-smoke',
        persona=persona,
        scenario=scenario,
        user_system_prompt='sys',
        first_message='Hi, I need help.',
    )

    result = await runner.run(datapoint=dp)
    assert result.terminated_by.value == 'judge'

    run_attrs = _attrs(_find(span_collector, 'orq.simulation.run'))
    assert not [k for k in run_attrs if k.startswith('gen_ai.usage.')]
    # target_cb has no usage_fn, so it contributes no usage; the run's total
    # calls is judge.get_usage().calls + user_sim.get_usage().calls == 2 + 1 == 3.
    assert result.token_usage.calls == 3


@pytest.mark.asyncio
async def test_pipeline_aggregate_span_carries_summed_call_count(
    span_collector: _CollectingExporter,
) -> None:
    """The pipeline span built by `_simulate_via_evaluatorq` in
    simulation/api.py aggregates every result's `token_usage` (including
    `.calls`) via `sum((r.token_usage for r in results), Usage())` and
    records it with `record_token_usage(pipeline_span, usage=...)`. This
    exercises that exact aggregation + recording against real
    `SimulationResult`/`Usage` objects and a real span, pinning
    `gen_ai.usage.calls` to the true sum rather than an accidental leftover
    from the `usage=` migration."""
    from evaluatorq.common.tracing import record_token_usage
    from evaluatorq.contracts import Usage
    from evaluatorq.simulation.tracing import with_simulation_span
    from evaluatorq.simulation.types import SimulationResult, TerminatedBy

    def _result(calls: int) -> SimulationResult:
        return SimulationResult(
            messages=[],
            terminated_by=TerminatedBy.judge,
            reason='done',
            goal_achieved=True,
            goal_completion_score=1.0,
            rules_broken=[],
            turn_count=2,
            token_usage=Usage(input_tokens=10, output_tokens=10, total_tokens=20, calls=calls),
            turn_metrics=[],
            metadata={},
        )

    results = [_result(1), _result(2), _result(3)]

    async with with_simulation_span('orq.simulation.pipeline_test', None) as pipeline_span:
        # Same expression as simulation/api.py's _simulate_via_evaluatorq.
        record_token_usage(pipeline_span, usage=sum((r.token_usage for r in results), Usage()))

    pipeline = _find(span_collector, 'orq.simulation.pipeline_test')
    assert _attrs(pipeline)['gen_ai.usage.calls'] == 6


@pytest.mark.asyncio
async def test_traceparent_injected_into_chat_completions_call(
    span_collector: _CollectingExporter,
    monkeypatch: pytest.MonkeyPatch,
):
    """When tracing is enabled, _call_chat_completions should inject
    traceparent into the OpenAI client request via extra_headers."""
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    captured: dict[str, object] = {}

    class _Choice:
        finish_reason = 'stop'

        class message:  # noqa: N801
            content = 'ok'
            tool_calls = None
            role = 'assistant'

    class _Usage:
        prompt_tokens = 1
        completion_tokens = 1
        total_tokens = 2
        prompt_tokens_details = None

    class _Resp:
        id = 'r'
        model = 'test'
        usage = _Usage()
        choices = [_Choice()]

    async def fake_create(**kwargs: object) -> _Resp:
        captured.update(kwargs)
        return _Resp()

    fake_client = MagicMock()
    fake_client.chat = MagicMock()
    fake_client.chat.completions = MagicMock()
    fake_client.chat.completions.create = AsyncMock(side_effect=fake_create)

    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.simulation.agents.base import BaseAgent
    from evaluatorq.simulation.tracing import with_simulation_span
    from evaluatorq.simulation.types import Message

    class _A(BaseAgent):
        @property
        def name(self) -> str:
            return 'T'

        @property
        def system_prompt(self) -> str:
            return 'sys'

    a = _A(LLMCallConfig(model='test', client=fake_client))

    # Wrap in an outer span so traceparent is meaningful to inject.
    async with with_simulation_span('orq.simulation.run', None):
        await a._call_chat_completions([Message(role='user', content='hi')])

    headers = captured.get('extra_headers')
    assert isinstance(headers, dict)
    assert 'traceparent' in headers, f'expected traceparent in {headers}'


@pytest.mark.asyncio
async def test_traceparent_injected_into_first_message_generation_call(
    span_collector: _CollectingExporter,
    monkeypatch: pytest.MonkeyPatch,
):
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    captured: dict[str, object] = {}

    class _Usage:
        input_tokens = 2
        output_tokens = 3
        total_tokens = 5
        input_tokens_details = None

    class _Resp:
        id = 'fm_1'
        model = 'test'
        usage = _Usage()
        output_text = 'Need help with my order'

    async def fake_create(**kwargs: object) -> _Resp:
        captured.update(kwargs)
        return _Resp()

    fake_client = MagicMock()
    fake_client.responses = MagicMock()
    fake_client.responses.create = AsyncMock(side_effect=fake_create)

    from evaluatorq.simulation.generators.first_message_generator import (
        FirstMessageGenerator,
    )
    from evaluatorq.simulation.tracing import with_simulation_span
    from evaluatorq.simulation.types import (
        CommunicationStyle,
        Persona,
        Scenario,
    )

    persona = Persona(
        name='Tester',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='A test user',
    )
    scenario = Scenario(name='Order', goal='Get order help')
    gen = FirstMessageGenerator(model='test', client=fake_client)

    async with with_simulation_span('orq.simulation.first_message_generation', None):
        message = await gen.generate(persona, scenario)

    assert message == 'Need help with my order'
    headers = captured.get('extra_headers')
    assert isinstance(headers, dict)
    assert 'traceparent' in headers, f'expected traceparent in {headers}'
    llm = _find(span_collector, 'responses test')
    parent = _find(span_collector, 'orq.simulation.first_message_generation')
    assert llm.parent.span_id == parent.context.span_id  # pyright: ignore[reportOptionalMemberAccess]


@pytest.mark.asyncio
async def test_generated_datapoint_first_message_has_simulation_span(
    span_collector: _CollectingExporter,
    monkeypatch: pytest.MonkeyPatch,
):
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    class _Usage:
        input_tokens = 1
        output_tokens = 1
        total_tokens = 2
        input_tokens_details = None

    class _Resp:
        id = 'dp_1'
        model = 'test'
        usage = _Usage()
        output_text = 'Hello there'

    fake_client = MagicMock()
    fake_client.responses = MagicMock()
    fake_client.responses.create = AsyncMock(return_value=_Resp())

    from evaluatorq.simulation.api import _generate_single_datapoint
    from evaluatorq.simulation.generators.first_message_generator import (
        FirstMessageGenerator,
    )
    from evaluatorq.simulation.tracing import with_simulation_span
    from evaluatorq.simulation.types import (
        CommunicationStyle,
        Persona,
        Scenario,
    )

    persona = Persona(
        name='Tester',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='A test user',
    )
    scenario = Scenario(name='Billing', goal='Fix a billing issue')
    gen = FirstMessageGenerator(model='test', client=fake_client)

    # The generation phase gets ONE span for all persona x scenario pairs; the
    # per-pair helper adds no span of its own.
    async with with_simulation_span('Evaluatorq - Agent Simulation', None):
        async with with_simulation_span('orq.simulation.first_message_generation', None):
            datapoint = await _generate_single_datapoint(gen, persona, scenario)

    assert datapoint.first_message == 'Hello there'
    pipeline = _find(span_collector, 'Evaluatorq - Agent Simulation')
    first_msg = _find(span_collector, 'orq.simulation.first_message_generation')
    llm = _find(span_collector, 'responses test')
    assert first_msg.parent.span_id == pipeline.context.span_id  # pyright: ignore[reportOptionalMemberAccess]
    assert llm.parent.span_id == first_msg.context.span_id  # pyright: ignore[reportOptionalMemberAccess]


@pytest.mark.asyncio
async def test_run_span_records_error_metadata_and_usage(
    span_collector: _CollectingExporter,
    monkeypatch: pytest.MonkeyPatch,
):
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    from evaluatorq.contracts import TokenUsage
    from evaluatorq.simulation.runner.simulation import SimulationRunner
    from evaluatorq.simulation.tracing import with_simulation_span
    from evaluatorq.simulation.types import (
        CommunicationStyle,
        Message,
        Persona,
        Scenario,
        SimulationDatapoint,
        TerminatedBy,
    )

    judge = MagicMock()
    judge.evaluate = AsyncMock(side_effect=RuntimeError('judge failed'))
    judge.get_usage = MagicMock(return_value=TokenUsage(prompt_tokens=5, completion_tokens=7, total_tokens=12))

    user_sim = MagicMock()
    user_sim.get_usage = MagicMock(return_value=TokenUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5))
    user_sim.generate_first_message = AsyncMock(return_value='hello')

    async def target_cb(messages: list[Message]) -> str:
        return 'agent reply'

    runner = SimulationRunner(
        target=target_cb,
        model='test',
        max_turns=1,
        user_simulator=user_sim,
        judge=judge,
    )

    persona = Persona(
        name='Tester',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='A test user',
    )
    scenario = Scenario(name='Failure', goal='Get help')
    dp = SimulationDatapoint(
        id='dp-error',
        persona=persona,
        scenario=scenario,
        user_system_prompt='sys',
        first_message='Hello',
    )

    async with with_simulation_span('Evaluatorq - Agent Simulation', None):
        result = await runner.run(datapoint=dp)

    assert result.terminated_by == TerminatedBy.error
    assert result.turn_count == 1
    assert result.token_usage.total_tokens == 17

    run = _find(span_collector, 'orq.simulation.run')
    attrs = _attrs(run)
    assert attrs['orq.simulation.terminated_by'] == 'error'
    assert attrs['orq.simulation.turn_count'] == 1
    assert not [k for k in attrs if k.startswith('gen_ai.usage.')]


@pytest.mark.asyncio
async def test_target_agent_usage_aggregated_into_run_result(
    span_collector: _CollectingExporter,
    monkeypatch: pytest.MonkeyPatch,
):
    """`AgentTarget.respond().usage` must flow into `SimulationResult.token_usage`.

    Regression guard for the aggregation added in `bf840ed` — without this
    test, a future refactor can silently drop target token spend from cost
    reporting again (the bug `arianpasquali` flagged).
    """
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    from evaluatorq.contracts import AgentResponse, AgentTarget, Message, TokenUsage
    from evaluatorq.simulation.runner.simulation import SimulationRunner
    from evaluatorq.simulation.tracing import with_simulation_span
    from evaluatorq.simulation.types import (
        CommunicationStyle,
        Persona,
        Scenario,
        SimulationDatapoint,
    )

    class _FakeAgentTarget(AgentTarget):
        def __init__(self, usage: TokenUsage) -> None:
            super().__init__()
            self._usage = usage
            self.call_count = 0

        async def respond(self, messages: list[Message]) -> AgentResponse:
            self.call_count += 1
            return AgentResponse(text=f'reply #{self.call_count}', usage=self._usage)

        def new(self) -> '_FakeAgentTarget':
            # Shared on purpose: this test asserts usage aggregation on one
            # instance, not clone isolation (covered in test_target_isolation).
            return self

    target_usage = TokenUsage(
        input_tokens=9,
        output_tokens=11,
        total_tokens=20,
        cached_tokens=2,
        reasoning_tokens=3,
        calls=1,
    )
    user_sim_usage = TokenUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5, calls=1)
    judge_usage = TokenUsage(prompt_tokens=5, completion_tokens=7, total_tokens=12, calls=1)

    def _judgment(*, terminate: bool) -> MagicMock:
        j = MagicMock()
        j.should_terminate = terminate
        j.goal_achieved = terminate
        j.goal_completion_score = 1.0 if terminate else 0.5
        j.rules_broken = []
        j.reason = 'done' if terminate else 'keep going'
        j.response_quality = 0.9
        j.hallucination_risk = 0.1
        j.tone_appropriateness = 0.9
        j.factual_accuracy = 0.9
        return j

    judge = MagicMock()
    judge.evaluate = AsyncMock(return_value=_judgment(terminate=True))
    judge.get_usage = MagicMock(return_value=judge_usage)

    user_sim = MagicMock()
    user_sim.generate_first_message = AsyncMock(return_value='Hi')
    user_sim.respond_async = AsyncMock(return_value='ok')
    user_sim.get_usage = MagicMock(return_value=user_sim_usage)

    target = _FakeAgentTarget(target_usage)

    runner = SimulationRunner(
        target_agent=target,
        model='test',
        max_turns=2,
        user_simulator=user_sim,
        judge=judge,
    )

    persona = Persona(
        name='Tester',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='A test user',
    )
    scenario = Scenario(name='UsageRollup', goal='Get help')
    dp = SimulationDatapoint(
        id='dp-usage',
        persona=persona,
        scenario=scenario,
        user_system_prompt='sys',
        first_message='Hi',
    )

    async with with_simulation_span('Evaluatorq - Agent Simulation', None):
        result = await runner.run(datapoint=dp)

    assert target.call_count >= 1
    expected_prompt = (
        user_sim_usage.prompt_tokens + judge_usage.prompt_tokens + target_usage.prompt_tokens * target.call_count
    )
    expected_completion = (
        user_sim_usage.completion_tokens
        + judge_usage.completion_tokens
        + target_usage.completion_tokens * target.call_count
    )
    expected_total = (
        user_sim_usage.total_tokens + judge_usage.total_tokens + target_usage.total_tokens * target.call_count
    )
    expected_calls = user_sim_usage.calls + judge_usage.calls + target_usage.calls * target.call_count

    assert result.token_usage.prompt_tokens == expected_prompt
    assert result.token_usage.completion_tokens == expected_completion
    assert result.token_usage.total_tokens == expected_total
    assert result.token_usage.calls == expected_calls

    # Neither the run span nor the target_call wrapper carries usage — both are
    # aggregates over the per-call LLM spans beneath them.
    for name in ('orq.simulation.run', 'orq.simulation.target_call'):
        attrs = _attrs(_find(span_collector, name))
        assert not [k for k in attrs if k.startswith('gen_ai.usage.')]


@pytest.mark.asyncio
async def test_run_span_records_cancellation_metadata_and_usage(
    span_collector: _CollectingExporter,
    monkeypatch: pytest.MonkeyPatch,
):
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    from evaluatorq.contracts import TokenUsage
    from evaluatorq.simulation.runner.simulation import SimulationRunner
    from evaluatorq.simulation.tracing import with_simulation_span
    from evaluatorq.simulation.types import (
        CommunicationStyle,
        Message,
        Persona,
        Scenario,
        SimulationDatapoint,
    )

    judge = MagicMock()
    judge.evaluate = AsyncMock(side_effect=asyncio.CancelledError())
    judge.get_usage = MagicMock(return_value=TokenUsage(prompt_tokens=5, completion_tokens=7, total_tokens=12))

    user_sim = MagicMock()
    user_sim.get_usage = MagicMock(return_value=TokenUsage(prompt_tokens=2, completion_tokens=3, total_tokens=5))
    user_sim.generate_first_message = AsyncMock(return_value='hello')

    async def target_cb(messages: list[Message]) -> str:
        return 'agent reply'

    runner = SimulationRunner(
        target=target_cb,
        model='test',
        max_turns=1,
        user_simulator=user_sim,
        judge=judge,
    )

    persona = Persona(
        name='Tester',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='A test user',
    )
    scenario = Scenario(name='Cancelled', goal='Get help')
    dp = SimulationDatapoint(
        id='dp-cancelled',
        persona=persona,
        scenario=scenario,
        user_system_prompt='sys',
        first_message='Hello',
    )

    with pytest.raises(asyncio.CancelledError):
        async with with_simulation_span('Evaluatorq - Agent Simulation', None):
            await runner.run(datapoint=dp)

    run = _find(span_collector, 'orq.simulation.run')
    attrs = _attrs(run)
    assert attrs['orq.simulation.terminated_by'] == 'error'
    assert attrs['orq.simulation.turn_count'] == 1
    assert not [k for k in attrs if k.startswith('gen_ai.usage.')]
    assert attrs['error.type'] == 'CancelledError'


@pytest.mark.asyncio
async def test_concurrent_runs_share_pipeline_parent(
    span_collector: _CollectingExporter,
):
    """Two concurrent simulation spans started under one pipeline span both
    have the pipeline as their parent — confirms asyncio.gather doesn't leak
    OTel context across tasks."""
    import asyncio

    from evaluatorq.simulation.tracing import with_simulation_span

    async def _sub(name: str) -> None:
        async with with_simulation_span(name, None):
            await asyncio.sleep(0)

    async with with_simulation_span('Evaluatorq - Agent Simulation', None):
        await asyncio.gather(_sub('orq.simulation.run'), _sub('orq.simulation.run'))

    pipeline = _find(span_collector, 'Evaluatorq - Agent Simulation')
    runs = [s for s in span_collector.spans if s.name == 'orq.simulation.run']
    assert len(runs) == 2
    for r in runs:
        assert r.parent.span_id == pipeline.context.span_id  # pyright: ignore[reportOptionalMemberAccess]


@pytest.mark.asyncio
async def test_helpers_noop_when_tracing_disabled():
    from evaluatorq.common.tracing import (
        get_trace_context_headers,
        record_llm_input,
        record_llm_response,
        record_token_usage,
    )
    from evaluatorq.simulation.tracing import (
        with_llm_span,
        with_simulation_span,
    )

    with patch('evaluatorq.simulation.tracing.get_tracer', return_value=None):
        async with with_simulation_span('orq.simulation.run', {'x': 'y'}) as span:
            assert span is None

        async with with_llm_span(model='openai/gpt-4o') as span:
            assert span is None

        # Recorder helpers must accept None without raising
        record_token_usage(None, prompt_tokens=1, completion_tokens=2, total_tokens=3)
        record_llm_input(None, [{'role': 'user', 'content': 'x'}])
        record_llm_response(None, object())

        headers = await get_trace_context_headers()
        # propagator may still inject something even without a tracer; just
        # assert it returns a dict
        assert isinstance(headers, dict)


# ---------------------------------------------------------------------------
# executive-summary LLM span
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_executive_summary_emits_llm_span(span_collector: _CollectingExporter):
    """populate_run_executive_summary wraps its LLM call in a GenAI span."""
    from types import SimpleNamespace

    from evaluatorq.simulation.reports.executive_summary import populate_run_executive_summary

    async def _create(**_kwargs: Any) -> Any:
        msg = SimpleNamespace(content='One paragraph summary.')
        usage = SimpleNamespace(prompt_tokens=12, completion_tokens=7, total_tokens=19)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))
    run = SimpleNamespace(
        executive_summary=None,
        results=[SimpleNamespace(goal_achieved=True, rules_broken=[], reason='')],
    )

    await populate_run_executive_summary(
        run,
        enabled=True,
        model='openai/gpt-4o',
        resolve_client=lambda: SimpleNamespace(client=client),
    )

    assert run.executive_summary == 'One paragraph summary.'
    span = _find(span_collector, 'chat openai/gpt-4o')
    a = _attrs(span)
    assert a['orq.llm.purpose'] == 'executive_summary'
    assert a['orq.simulation.llm_purpose'] == 'executive_summary'
    assert a['gen_ai.request.model'] == 'openai/gpt-4o'
    # Span reports the exact request params generate_executive_summary sends.
    from evaluatorq.common.reports.executive_summary import (
        EXECUTIVE_SUMMARY_MAX_TOKENS,
        EXECUTIVE_SUMMARY_TEMPERATURE,
    )

    assert a['gen_ai.request.temperature'] == EXECUTIVE_SUMMARY_TEMPERATURE
    assert a['gen_ai.request.max_tokens'] == EXECUTIVE_SUMMARY_MAX_TOKENS
    # Token usage from the response lands on the span.
    assert a['gen_ai.usage.input_tokens'] == 12
    assert a['gen_ai.usage.output_tokens'] == 7
    from opentelemetry.trace import StatusCode

    assert span.status.status_code == StatusCode.OK


@pytest.mark.asyncio
async def test_executive_summary_no_span_when_disabled(span_collector: _CollectingExporter):
    """enabled=False short-circuits before opening any span."""
    from types import SimpleNamespace

    from evaluatorq.simulation.reports.executive_summary import populate_run_executive_summary

    run = SimpleNamespace(executive_summary=None, results=[])
    await populate_run_executive_summary(run, enabled=False, model='openai/gpt-4o')

    assert run.executive_summary is None
    assert not [s for s in span_collector.spans if s.name == 'chat openai/gpt-4o']


@pytest.mark.asyncio
async def test_batched_first_message_generation_uses_one_span(
    span_collector: _CollectingExporter,
    monkeypatch: pytest.MonkeyPatch,
):
    """The persona x scenario sweep opens ONE generation span; every pair's LLM
    span nests under it, and the counts land on that span.
    """
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    class _Usage:
        input_tokens = 1
        output_tokens = 1
        total_tokens = 2
        input_tokens_details = None

    class _Resp:
        id = 'dp_1'
        model = 'test'
        usage = _Usage()
        output_text = 'Hello there'

    fake_client = MagicMock()
    fake_client.responses = MagicMock()
    fake_client.responses.create = AsyncMock(return_value=_Resp())

    from evaluatorq.simulation.api import _resolve_or_generate_datapoints
    from evaluatorq.simulation.types import CommunicationStyle, Persona, Scenario

    personas = [
        Persona(
            name=f'P{i}',
            patience=0.5,
            assertiveness=0.5,
            politeness=0.5,
            technical_level=0.5,
            communication_style=CommunicationStyle.casual,
            background='A test user',
        )
        for i in range(2)
    ]
    scenarios = [Scenario(name=f'S{i}', goal='Fix it') for i in range(3)]

    datapoints = await _resolve_or_generate_datapoints(
        caller='simulate',
        datapoints=None,
        personas=personas,
        scenarios=scenarios,
        dataset_id=None,
        model='test',
        generation_client=fake_client,
    )

    assert len(datapoints) == 6
    gen_spans = [s for s in span_collector.spans if s.name == 'orq.simulation.first_message_generation']
    assert len(gen_spans) == 1
    a = _attrs(gen_spans[0])
    assert a['orq.simulation.pair_count'] == 6
    assert a['orq.simulation.persona_count'] == 2
    assert a['orq.simulation.scenario_count'] == 3
    assert a['orq.simulation.generated_count'] == 6
    assert a['orq.simulation.failed_count'] == 0

    llm_spans = [s for s in span_collector.spans if s.name == 'responses test']
    assert len(llm_spans) == 6
    assert all(s.parent.span_id == gen_spans[0].context.span_id for s in llm_spans)  # pyright: ignore[reportOptionalMemberAccess]


@pytest.mark.asyncio
async def test_first_message_generation_span_records_failures(
    span_collector: _CollectingExporter,
    monkeypatch: pytest.MonkeyPatch,
):
    """Every pair failing marks the phase span ERROR and records one event per
    failed pair (the per-pair spans that used to carry this are gone).
    """
    from unittest.mock import AsyncMock, MagicMock

    monkeypatch.setenv('ORQ_API_KEY', 'test-key')

    fake_client = MagicMock()
    fake_client.responses = MagicMock()
    fake_client.responses.create = AsyncMock(side_effect=RuntimeError('boom'))

    from evaluatorq.simulation.api import _resolve_or_generate_datapoints
    from evaluatorq.simulation.types import CommunicationStyle, Persona, Scenario

    persona = Persona(
        name='P0',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='A test user',
    )
    scenario = Scenario(name='S0', goal='Fix it')

    with pytest.raises(RuntimeError, match='produced no datapoints'):
        await _resolve_or_generate_datapoints(
            caller='simulate',
            datapoints=None,
            personas=[persona],
            scenarios=[scenario],
            dataset_id=None,
            model='test',
            generation_client=fake_client,
        )

    gen = _find(span_collector, 'orq.simulation.first_message_generation')
    assert gen.status.status_code.name == 'ERROR'
    assert _attrs(gen)['orq.simulation.failed_count'] == 1
    events = [e for e in gen.events if e.name == 'orq.simulation.first_message_generation_failed']
    assert len(events) == 1
    assert dict(events[0].attributes or {})['orq.simulation.persona'] == 'P0'
