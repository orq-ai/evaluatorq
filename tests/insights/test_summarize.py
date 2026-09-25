"""Unit tests for `evaluatorq.insights.summarize` — the per-trace summary pass."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from evaluatorq.common.structured_output import StructuredResult
from evaluatorq.insights import summarize as summarize_module
from evaluatorq.insights.cache import InsightsCache
from evaluatorq.insights.models import TraceSummary
from evaluatorq.insights.summarize import SUMMARY_HASH, SUMMARY_PROMPT, summarize_traces
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


@pytest.fixture
def cache(tmp_path: Path) -> InsightsCache:
    return InsightsCache(tmp_path / 'insights.sqlite')


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
        assert kwargs['max_tokens'] == 1200
        assert kwargs['label'] == 'insights.summary'
        return StructuredResult(parsed=summary, raw='')

    monkeypatch.setattr(summarize_module, 'generate_structured', fake_generate_structured)

    result = await summarize_traces([trace], client=fake_client(), model='openai/gpt-6-luna', cache=cache)

    assert result == {'trace-1': summary}
    assert len(captured_messages) == 1
    content = captured_messages[0][0]['content']
    assert '<conversation>' in content
    assert '</conversation>' in content

    # Cached under (trace_id, span_id, model, SUMMARY_HASH).
    cached = cache.get_summary('trace-1', 'span-trace-1', 'openai/gpt-6-luna', SUMMARY_HASH)
    assert cached == summary


@pytest.mark.asyncio
async def test_summarize_traces_cache_hit_skips_the_call(monkeypatch: pytest.MonkeyPatch, cache: InsightsCache) -> None:
    trace = make_trace('trace-1')
    summary = make_summary('Cached summary.')
    cache.put_summary('trace-1', 'span-trace-1', 'openai/gpt-6-luna', SUMMARY_HASH, summary)

    calls = 0

    async def fake_generate_structured(client: Any, **kwargs: Any) -> StructuredResult[TraceSummary]:
        nonlocal calls
        calls += 1
        raise AssertionError('generate_structured must not be called on a cache hit')

    monkeypatch.setattr(summarize_module, 'generate_structured', fake_generate_structured)

    result = await summarize_traces([trace], client=fake_client(), model='openai/gpt-6-luna', cache=cache)

    assert result == {'trace-1': summary}
    assert calls == 0


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
    assert cache.get_summary('trace-1', 'span-trace-1', 'openai/gpt-6-luna', SUMMARY_HASH) is None


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


def test_summary_prompt_has_no_scalar_fields_from_upstream() -> None:
    # The old taxonomy/scalar fields are now classifier labels, not part of the
    # summary schema or its prompt.
    for dropped in ('user_frustration', 'customer_satisfaction', 'made_errors', 'concerning_score'):
        assert dropped not in SUMMARY_PROMPT
