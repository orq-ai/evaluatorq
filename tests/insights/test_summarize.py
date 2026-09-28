"""Unit tests for `evaluatorq.insights.summarize` — the per-trace summary pass."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any, cast

import pytest

from evaluatorq.common.structured_output import StructuredResult
from evaluatorq.contracts import Usage
from evaluatorq.insights import summarize as summarize_module
from evaluatorq.insights.cache import InsightsCache, prompt_hash
from evaluatorq.insights.models import TraceSummary
from evaluatorq.insights.summarize import SUMMARY_PROMPT, summarize_traces
from evaluatorq.insights.usage import UsageLedger
from evaluatorq.trace_finder.models import TraceRecord

if TYPE_CHECKING:
    from openai import AsyncOpenAI


def make_trace(trace_id: str, content: str = 'hello') -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id=trace_id,
        span_id=f'span-{trace_id}',
        timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
        messages=({'role': 'user', 'content': content},),
        project='default',
        model='gpt-5',
        provider='openai',
        status='completed',
        product='chat',
        trace_type='conversation',
    )


def make_summary(summary_text: str = 'A user asked a question.') -> TraceSummary:
    return TraceSummary(
        summary=summary_text,
        request='What is the refund policy?',
        task='answer question',
        topic='refunds',
        sentiment_explanation='The user is calm.',
    )


class _BilledError(RuntimeError):
    def __init__(self, message: str, *, usage: Usage) -> None:
        super().__init__(message)
        self.usage = usage


@pytest.fixture
def cache(tmp_path: Path) -> Iterator[InsightsCache]:
    instance = InsightsCache(tmp_path / 'insights.sqlite')
    try:
        yield instance
    finally:
        instance.close()


class _FakeClient:
    """Placeholder object — `generate_structured` is monkeypatched, so this is never called."""


def fake_client() -> AsyncOpenAI:
    """Typed as `AsyncOpenAI` for basedpyright; `generate_structured` is always monkeypatched in these tests."""
    return cast('AsyncOpenAI', cast(object, _FakeClient()))


@pytest.mark.asyncio
async def test_summarize_traces_calls_generate_structured_and_caches(
    monkeypatch: pytest.MonkeyPatch, cache: InsightsCache
) -> None:
    trace = make_trace('trace-1')
    summary = make_summary()
    captured_messages: list[list[dict[str, Any]]] = []

    async def fake_generate_structured(client: Any, **kwargs: Any) -> StructuredResult[TraceSummary]:
        captured_messages.append(kwargs['messages'])
        assert kwargs['response_format'] is TraceSummary
        assert kwargs['max_tokens'] == 4096
        assert kwargs['label'] == 'insights.summary'
        return StructuredResult(parsed=summary, raw='')

    monkeypatch.setattr(summarize_module, 'generate_structured', fake_generate_structured)

    result = await summarize_traces([trace], client=fake_client(), model='openai/gpt-6-luna', cache=cache)

    assert result == {'trace-1': summary}
    assert len(captured_messages) == 1
    content = captured_messages[0][0]['content']
    assert '<conversation>' in content
    assert '</conversation>' in content

    cached = cache.get_summary(
        'trace-1', 'span-trace-1', 'openai/gpt-6-luna', prompt_hash(captured_messages[0][0]['content'])
    )
    assert cached == summary


@pytest.mark.asyncio
async def test_cache_reads_and_writes_leave_event_loop_responsive(
    monkeypatch: pytest.MonkeyPatch, cache: InsightsCache
) -> None:
    read_released = Event()
    write_released = Event()
    original_get = cache.get_summary
    original_put = cache.put_summary

    def slow_get(*args: Any) -> TraceSummary | None:
        assert read_released.wait(1), 'cache read blocked the event loop'
        return original_get(*args)

    def slow_put(*args: Any) -> None:
        assert write_released.wait(1), 'cache write blocked the event loop'
        original_put(*args)

    async def fake_generate_structured(client: Any, **kwargs: Any) -> StructuredResult[TraceSummary]:
        return StructuredResult(parsed=make_summary(), raw='')

    async def release_io() -> None:
        await asyncio.sleep(0.01)
        read_released.set()
        await asyncio.sleep(0.01)
        write_released.set()

    monkeypatch.setattr(cache, 'get_summary', slow_get)
    monkeypatch.setattr(cache, 'put_summary', slow_put)
    monkeypatch.setattr(summarize_module, 'generate_structured', fake_generate_structured)

    release = asyncio.create_task(release_io())
    result = await summarize_traces([make_trace('trace-1')], client=fake_client(), model='m', cache=cache)
    await release

    assert result == {'trace-1': make_summary()}


@pytest.mark.asyncio
async def test_summarize_traces_reports_each_completed_trace(
    monkeypatch: pytest.MonkeyPatch, cache: InsightsCache
) -> None:
    traces = [make_trace(f'trace-{i}') for i in range(3)]

    async def fake_generate_structured(client: Any, **kwargs: Any) -> StructuredResult[TraceSummary]:
        return StructuredResult(parsed=make_summary(), raw='')

    monkeypatch.setattr(summarize_module, 'generate_structured', fake_generate_structured)
    updates: list[tuple[int, int]] = []

    await summarize_traces(
        traces,
        client=fake_client(),
        model='openai/gpt-6-luna',
        cache=cache,
        on_progress=lambda completed, total: updates.append((completed, total)),
    )

    assert sorted(updates) == [(1, 3), (2, 3), (3, 3)]


@pytest.mark.asyncio
async def test_summarize_progress_callback_failure_does_not_fail_run(
    monkeypatch: pytest.MonkeyPatch, cache: InsightsCache
) -> None:
    trace = make_trace('trace-1')

    async def fake_generate_structured(client: Any, **kwargs: Any) -> StructuredResult[TraceSummary]:
        return StructuredResult(parsed=make_summary(), raw='')

    monkeypatch.setattr(summarize_module, 'generate_structured', fake_generate_structured)

    def fail_progress(completed: int, total: int) -> None:
        raise RuntimeError('progress unavailable')

    result = await summarize_traces(
        [trace], client=fake_client(), model='openai/gpt-6-luna', cache=cache, on_progress=fail_progress
    )

    assert result == {'trace-1': make_summary()}


@pytest.mark.asyncio
async def test_summarize_traces_cache_hit_skips_the_call(monkeypatch: pytest.MonkeyPatch, cache: InsightsCache) -> None:
    trace = make_trace('trace-1')
    summary = make_summary('Cached summary.')
    cache.put_summary(
        'trace-1',
        'span-trace-1',
        'openai/gpt-6-luna',
        prompt_hash(summarize_module._build_prompt(trace)),
        summary,
    )

    calls = 0

    async def fake_generate_structured(client: Any, **kwargs: Any) -> StructuredResult[TraceSummary]:
        nonlocal calls
        calls += 1
        raise AssertionError('generate_structured must not be called on a cache hit')

    monkeypatch.setattr(summarize_module, 'generate_structured', fake_generate_structured)
    ledger = UsageLedger()

    result = await summarize_traces([trace], client=fake_client(), model='openai/gpt-6-luna', cache=cache, usage=ledger)

    assert result == {'trace-1': summary}
    assert calls == 0
    assert ledger.totals() == {}


@pytest.mark.asyncio
async def test_summarize_traces_unparseable_output_yields_error_string(
    monkeypatch: pytest.MonkeyPatch, cache: InsightsCache
) -> None:
    trace = make_trace('trace-1')

    async def fake_generate_structured(client: Any, **kwargs: Any) -> StructuredResult[TraceSummary]:
        return StructuredResult(parsed=None, raw='not valid json')

    monkeypatch.setattr(summarize_module, 'generate_structured', fake_generate_structured)

    result = await summarize_traces([trace], client=fake_client(), model='openai/gpt-6-luna', cache=cache)

    assert result == {'trace-1': 'summary: unparseable model output'}
    # An unparseable reply must not be cached.
    assert (
        cache.get_summary(
            'trace-1', 'span-trace-1', 'openai/gpt-6-luna', prompt_hash(summarize_module._build_prompt(trace))
        )
        is None
    )


@pytest.mark.asyncio
async def test_summarize_traces_exception_yields_error_and_other_traces_still_succeed(
    monkeypatch: pytest.MonkeyPatch, cache: InsightsCache
) -> None:
    summary = make_summary('It went fine.')
    failing_trace = make_trace('trace-fail', content='fail-marker')
    ok_trace = make_trace('trace-ok', content='ok-marker')

    async def fake_dispatch(client: Any, **kwargs: Any) -> StructuredResult[TraceSummary]:
        content = kwargs['messages'][0]['content']
        await asyncio.sleep(0)
        if 'fail-marker' in content:
            raise RuntimeError('boom')
        return StructuredResult(parsed=summary, raw='')

    monkeypatch.setattr(summarize_module, 'generate_structured', fake_dispatch)

    result = await summarize_traces(
        [failing_trace, ok_trace], client=fake_client(), model='openai/gpt-6-luna', cache=cache
    )

    assert result['trace-fail'] == 'summary: boom'
    assert result['trace-ok'] == summary


@pytest.mark.asyncio
async def test_summarize_records_usage_from_billed_exception(monkeypatch: pytest.MonkeyPatch, cache: InsightsCache) -> None:
    trace = make_trace('trace-fail')
    billed = Usage(input_tokens=10, output_tokens=2, total_tokens=12, total_cost=0.01, calls=1, priced_calls=1)

    async def fail_after_billing(client: Any, **kwargs: Any) -> StructuredResult[TraceSummary]:
        raise _BilledError('provider failed after billing', usage=billed)

    monkeypatch.setattr(summarize_module, 'generate_structured', fail_after_billing)
    ledger = UsageLedger()

    result = await summarize_traces(
        [trace], client=fake_client(), model='openai/gpt-6-luna', cache=cache, usage=ledger
    )

    assert isinstance(result['trace-fail'], str)
    assert ledger.totals()['summary'] == billed


def test_summary_prompt_has_no_scalar_fields_from_upstream() -> None:
    # The old taxonomy/scalar fields are now classifier labels, not part of the
    # summary schema or its prompt.
    for dropped in ('user_frustration', 'customer_satisfaction', 'made_errors', 'concerning_score'):
        assert dropped not in SUMMARY_PROMPT


@pytest.mark.asyncio
async def test_changed_conversation_misses_the_cache(monkeypatch: pytest.MonkeyPatch, cache: InsightsCache) -> None:
    calls: list[str] = []

    async def fake_generate_structured(client: Any, **kwargs: Any) -> StructuredResult[TraceSummary]:
        calls.append(kwargs['messages'][0]['content'])
        return StructuredResult(parsed=make_summary(), raw='')

    monkeypatch.setattr(summarize_module, 'generate_structured', fake_generate_structured)
    first = make_trace('trace-cache', content='hi')
    second = first.model_copy(update={'messages': (*first.messages, {'role': 'assistant', 'content': 'done'})})

    await summarize_traces([first], client=fake_client(), model='m', cache=cache)
    await summarize_traces([second], client=fake_client(), model='m', cache=cache)

    assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('parallelism', [0, -1])
async def test_summarize_parallelism_must_be_positive(parallelism: int, cache: InsightsCache) -> None:
    with pytest.raises(ValueError, match='parallelism must be greater than zero'):
        await summarize_traces([], client=fake_client(), model='openai/gpt-6-luna', cache=cache, parallelism=parallelism)
