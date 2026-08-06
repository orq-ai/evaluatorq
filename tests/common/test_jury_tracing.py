"""RES-985: per-judge OTEL spans + jury aggregate attributes for run_jury().

Asserts the span *shape* (one ``llm.judge`` child per judge, all parented to a
single ``llm.jury`` span) and the attribute keys/values the ticket enumerates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from evaluatorq.common.jury import Prediction, VerdictKind, run_jury
from evaluatorq.contracts import TokenUsage


class _Exporter(SpanExporter):
    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


@pytest.fixture
def span_collector():
    exporter = _Exporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('test')
    # run_jury opens spans via evaluatorq.common.tracing.get_tracer.
    with patch('evaluatorq.common.tracing.get_tracer', return_value=tracer):
        yield exporter
    provider.shutdown()


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _span_id(span: ReadableSpan) -> int:
    assert span.context is not None
    return span.context.span_id


def _by_name(exporter: _Exporter, name: str) -> list[ReadableSpan]:
    return [s for s in exporter.spans if s.name == name]


def _make_judge_fn(verdicts: dict[str, str | None], *, fail: set[str] | None = None):
    """Judge that returns a fixed verdict per model, abstains on None, or fails."""
    fail = fail or set()

    async def judge_fn(model: str) -> Prediction:
        if model in fail:
            raise RuntimeError(f'{model} is down')
        v = verdicts.get(model)
        if v is None:
            return Prediction(abstained=True, explanation='no opinion', token_usage=TokenUsage(input_tokens=1))
        return Prediction(
            value=v, explanation=f'{model} says {v}', token_usage=TokenUsage(input_tokens=3, output_tokens=2)
        )

    return judge_fn


@pytest.mark.asyncio
async def test_one_judge_span_per_judge_parented_to_jury(span_collector) -> None:
    exporter = span_collector
    judge_fn = _make_judge_fn({'gpt-a': 'yes', 'gpt-b': 'yes', 'gpt-c': None})

    await run_jury(judge_fn=judge_fn, panel=['gpt-a', 'gpt-b', 'gpt-c'])

    jury_spans = _by_name(exporter, 'llm.jury')
    judge_spans = _by_name(exporter, 'llm.judge')
    assert len(jury_spans) == 1
    assert len(judge_spans) == 3

    # Every judge span is a child of the one jury span.
    jury_span_id = _span_id(jury_spans[0])
    for js in judge_spans:
        assert js.parent is not None
        assert js.parent.span_id == jury_span_id


@pytest.mark.asyncio
async def test_judge_span_attributes(span_collector) -> None:
    exporter = span_collector
    judge_fn = _make_judge_fn({'gpt-a': 'yes', 'gpt-b': None})

    await run_jury(judge_fn=judge_fn, panel=['gpt-a', 'gpt-b'])

    by_model = {_attrs(s)['judge.model']: _attrs(s) for s in _by_name(exporter, 'llm.judge')}

    decisive = by_model['gpt-a']
    assert decisive['judge.name'] == 'gpt-a'
    assert decisive['judge.verdict'] == 'yes'
    assert decisive['judge.success'] is True
    assert decisive['judge.abstained'] is False
    assert decisive['judge.replacement'] is False
    assert 'judge.latency_ms' in decisive
    assert decisive['total_tokens'] == 5  # token usage emitted (3 in + 2 out)

    abstained = by_model['gpt-b']
    assert abstained['judge.success'] is True
    assert abstained['judge.abstained'] is True
    assert 'judge.verdict' not in abstained  # None value is not stamped


@pytest.mark.asyncio
async def test_replacement_judge_span_is_flagged(span_collector) -> None:
    exporter = span_collector
    # gpt-a fails mechanically; stand-in is promoted and casts a real verdict.
    judge_fn = _make_judge_fn({'gpt-b': 'yes', 'stand-in': 'yes'}, fail={'gpt-a'})

    await run_jury(
        judge_fn=judge_fn,
        panel=['gpt-a', 'gpt-b'],
        replacement_judges=['stand-in'],
    )

    by_model = {_attrs(s)['judge.model']: _attrs(s) for s in _by_name(exporter, 'llm.judge')}
    assert by_model['gpt-a']['judge.success'] is False
    assert by_model['gpt-a']['judge.replacement'] is False
    assert by_model['stand-in']['judge.replacement'] is True
    assert by_model['stand-in']['judge.verdict'] == 'yes'
    # The replacement span still parents to the jury span.
    jury_span_id = _span_id(_by_name(exporter, 'llm.jury')[0])
    stand_in_span = next(s for s in _by_name(exporter, 'llm.judge') if _attrs(s)['judge.model'] == 'stand-in')
    assert stand_in_span.parent is not None
    assert stand_in_span.parent.span_id == jury_span_id


@pytest.mark.asyncio
async def test_jury_aggregate_attributes(span_collector) -> None:
    exporter = span_collector
    judge_fn = _make_judge_fn({'gpt-a': 'yes', 'gpt-b': 'yes', 'gpt-c': 'no'})

    await run_jury(judge_fn=judge_fn, panel=['gpt-a', 'gpt-b', 'gpt-c'])

    jury = _attrs(_by_name(exporter, 'llm.jury')[0])
    assert jury['jury.judges_configured'] == 3
    assert jury['jury.judges_succeeded'] == 3
    assert jury['jury.tie'] is False
    assert jury['jury.inconclusive'] is False
    # 2 of 3 decisive votes in the largest bloc.
    assert jury['jury.raw_agreement'] == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_inconclusive_jury_marks_span(span_collector) -> None:
    exporter = span_collector
    # Every judge abstains -> inconclusive; raw_agreement omitted (None).
    judge_fn = _make_judge_fn({'gpt-a': None, 'gpt-b': None})

    await run_jury(judge_fn=judge_fn, panel=['gpt-a', 'gpt-b'])

    jury = _attrs(_by_name(exporter, 'llm.jury')[0])
    assert jury['jury.inconclusive'] is True
    assert jury['jury.judges_succeeded'] == 0
    assert 'jury.raw_agreement' not in jury  # None is not stamped


@pytest.mark.asyncio
async def test_no_spans_when_tracing_disabled() -> None:
    """With get_tracer() returning None the spans are a no-op; the verdict is
    still produced (spans never change behavior)."""
    judge_fn = _make_judge_fn({'gpt-a': 'yes'})
    with patch('evaluatorq.common.tracing.get_tracer', return_value=None):
        deliberation = await run_jury(judge_fn=judge_fn, panel=['gpt-a'], verdict_kind=VerdictKind.CATEGORICAL)
    assert deliberation.verdict == 'yes'
