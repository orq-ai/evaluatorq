"""RES-985: per-judge OTEL spans + jury aggregate attributes for run_jury().

Asserts the span *shape* (one ``orq.judge`` child per judge, all parented to a
single ``orq.jury`` span) and the attribute keys/values the ticket enumerates.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import StatusCode

from evaluatorq.common.jury import Prediction, VerdictKind, run_jury
from evaluatorq.common.tracing import with_span
from evaluatorq.contracts import TokenUsage
from evaluatorq.pairwise import run_pairwise
from evaluatorq.processings import process_evaluator
from evaluatorq.types import DataPoint, EvaluationResult


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
    # run_jury opens spans via evaluatorq.common.tracing.get_tracer; the core
    # runner's evaluation span comes from evaluatorq.tracing.spans.get_tracer.
    with (
        patch('evaluatorq.common.tracing.get_tracer', return_value=tracer),
        patch('evaluatorq.tracing.spans.get_tracer', return_value=tracer),
    ):
        yield exporter
    provider.shutdown()


def _attrs(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _span_id(span: ReadableSpan) -> int:
    assert span.context is not None
    return span.context.span_id


def _parent_id(span: ReadableSpan) -> int | None:
    return span.parent.span_id if span.parent else None


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

    jury_spans = _by_name(exporter, 'orq.jury')
    judge_spans = _by_name(exporter, 'orq.judge')
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

    by_model = {_attrs(s)['judge.model']: _attrs(s) for s in _by_name(exporter, 'orq.judge')}

    decisive = by_model['gpt-a']
    assert decisive['judge.name'] == 'gpt-a'
    assert decisive['judge.verdict'] == 'yes'
    assert decisive['judge.success'] is True
    assert decisive['judge.abstained'] is False
    assert decisive['judge.replacement'] is False
    assert 'judge.latency_ms' in decisive

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

    by_model = {_attrs(s)['judge.model']: _attrs(s) for s in _by_name(exporter, 'orq.judge')}
    assert by_model['gpt-a']['judge.success'] is False
    assert by_model['gpt-a']['judge.replacement'] is False
    assert by_model['stand-in']['judge.replacement'] is True
    assert by_model['stand-in']['judge.verdict'] == 'yes'
    # The replacement span still parents to the jury span.
    jury_span_id = _span_id(_by_name(exporter, 'orq.jury')[0])
    stand_in_span = next(s for s in _by_name(exporter, 'orq.judge') if _attrs(s)['judge.model'] == 'stand-in')
    assert stand_in_span.parent is not None
    assert stand_in_span.parent.span_id == jury_span_id


@pytest.mark.asyncio
async def test_jury_aggregate_attributes(span_collector) -> None:
    exporter = span_collector
    judge_fn = _make_judge_fn({'gpt-a': 'yes', 'gpt-b': 'yes', 'gpt-c': 'no'})

    await run_jury(judge_fn=judge_fn, panel=['gpt-a', 'gpt-b', 'gpt-c'])

    jury = _attrs(_by_name(exporter, 'orq.jury')[0])
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

    jury = _attrs(_by_name(exporter, 'orq.jury')[0])
    assert jury['jury.inconclusive'] is True
    assert jury['jury.judges_succeeded'] == 0
    assert 'jury.raw_agreement' not in jury  # None is not stamped


@pytest.mark.asyncio
async def test_failed_judge_span_is_error_status(span_collector) -> None:
    exporter = span_collector
    judge_fn = _make_judge_fn({'gpt-b': 'yes'}, fail={'gpt-a'})

    await run_jury(judge_fn=judge_fn, panel=['gpt-a', 'gpt-b'])

    spans = {_attrs(s)['judge.model']: s for s in _by_name(exporter, 'orq.judge')}
    failed = spans['gpt-a']
    assert failed.status.status_code is StatusCode.ERROR
    assert 'gpt-a is down' in _attrs(failed)['judge.error']
    assert _attrs(failed)['judge.repetitions_failed'] == 1
    assert spans['gpt-b'].status.status_code is StatusCode.OK


@pytest.mark.asyncio
async def test_propagated_error_span_keeps_judge_identity(span_collector) -> None:
    exporter = span_collector
    judge_fn = _make_judge_fn({}, fail={'gpt-a'})

    with pytest.raises(RuntimeError):
        await run_jury(judge_fn=judge_fn, panel=['gpt-a'], propagate_errors=True)

    judge = _by_name(exporter, 'orq.judge')[0]
    assert _attrs(judge)['judge.model'] == 'gpt-a'
    assert _attrs(judge)['judge.replacement'] is False
    assert judge.status.status_code is StatusCode.ERROR


@pytest.mark.asyncio
async def test_usage_and_cost_stay_off_judge_and_jury_spans(span_collector) -> None:
    """Usage/cost belong on the ``chat`` spans that incurred them.

    Stamping them on the judge and jury spans too made the same tokens appear
    at three levels of one trace, so anything summing across a trace
    triple-counted. The consumer rolls up from the leaves instead.
    """
    exporter = span_collector

    async def judge_fn(model: str) -> Prediction:
        return Prediction(
            value='yes',
            token_usage=TokenUsage(
                input_tokens=10, output_tokens=4, cached_tokens=6, reasoning_tokens=3, calls=1, cost_usd=0.5
            ),
        )

    await run_jury(judge_fn=judge_fn, panel=['gpt-a'], repetitions=2)

    for name in ('orq.judge', 'orq.jury'):
        attrs = _attrs(_by_name(exporter, name)[0])
        assert not [k for k in attrs if k.startswith('gen_ai.usage.')], name
        assert 'judge.cost' not in attrs
        assert 'jury.cost' not in attrs


@pytest.mark.asyncio
async def test_jury_span_records_failure_aggregates(span_collector) -> None:
    exporter = span_collector
    judge_fn = _make_judge_fn({'gpt-b': 'yes', 'stand-in': 'yes'}, fail={'gpt-a'})

    await run_jury(judge_fn=judge_fn, panel=['gpt-a', 'gpt-b'], replacement_judges=['stand-in'])

    jury = _attrs(_by_name(exporter, 'orq.jury')[0])
    assert jury['jury.judges_failed'] == 1
    assert jury['jury.replacements_used'] == 1


@pytest.mark.asyncio
async def test_jury_span_records_outcome(span_collector) -> None:
    exporter = span_collector
    judge_fn = _make_judge_fn({'gpt-a': 'yes', 'gpt-b': 'yes'})

    await run_jury(judge_fn=judge_fn, panel=['gpt-a', 'gpt-b'], aggregator='majority', min_successful_judges=2)

    jury = _attrs(_by_name(exporter, 'orq.jury')[0])
    assert jury['jury.verdict'] == 'yes'
    assert jury['jury.aggregator'] == 'majority'
    assert jury['jury.min_successful_judges'] == 2


@pytest.mark.asyncio
async def test_pairwise_emits_one_jury_span_for_both_orderings(span_collector) -> None:
    """A comparison runs each judge twice (A/B then B/A). Those orderings must
    NOT each open their own orq.jury span — one span per comparison, with the
    ordering recorded on the judge spans instead."""
    exporter = span_collector

    async def judge_fn(first: str, second: str, model: str) -> Prediction:
        # Consistent preference for the response that reads 'better'.
        winner = 'A' if first == 'better' else 'B'
        return Prediction(value=winner, token_usage=TokenUsage(input_tokens=1, output_tokens=1))

    comparison = await run_pairwise(
        judge_fn=judge_fn, panel=['gpt-a', 'gpt-b'], response_a='better', response_b='worse'
    )
    assert comparison.winner == 'A'

    assert len(_by_name(exporter, 'orq.jury')) == 1
    judge_spans = _by_name(exporter, 'orq.judge')
    assert len(judge_spans) == 4  # 2 judges x 2 orderings

    jury_span_id = _span_id(_by_name(exporter, 'orq.jury')[0])
    for js in judge_spans:
        assert js.parent is not None
        assert js.parent.span_id == jury_span_id

    swapped = [_attrs(s)['judge.label_swapped'] for s in judge_spans]
    assert sorted(swapped) == [False, False, True, True]

    jury = _attrs(_by_name(exporter, 'orq.jury')[0])
    assert jury['jury.verdict'] == 'A'
    assert jury['jury.judges_configured'] == 2
    assert jury['jury.judges_succeeded'] == 2
    assert jury['jury.flipped'] == 0
    assert jury['jury.swap'] is True


@pytest.mark.asyncio
async def test_pairwise_span_records_position_bias(span_collector) -> None:
    exporter = span_collector

    async def judge_fn(first: str, second: str, model: str) -> Prediction:
        # Always picks the first slot — pure position bias, so it self-contradicts.
        return Prediction(value='A')

    comparison = await run_pairwise(judge_fn=judge_fn, panel=['gpt-a'], response_a='x', response_b='y')
    assert comparison.winner == 'inconclusive'

    jury = _attrs(_by_name(exporter, 'orq.jury')[0])
    assert jury['jury.flipped'] == 1
    assert jury['jury.flipped_judges'] == 'gpt-a'
    assert jury['jury.inconclusive'] is True
    # A flip is position bias, NOT a failure — the judge answered both times.
    assert jury['jury.judges_failed'] == 0


@pytest.mark.asyncio
async def test_pairwise_span_separates_failure_from_flip(span_collector) -> None:
    """One judge dies mechanically, another flips, a stand-in covers. The two
    outcomes must not collapse into the same counter."""
    exporter = span_collector

    async def judge_fn(first: str, second: str, model: str) -> Prediction:
        if model == 'dead':
            raise RuntimeError('dead is down')
        if model == 'flipper':
            return Prediction(value='A')  # always the first slot -> self-contradicts
        return Prediction(value='A' if first == 'better' else 'B')

    comparison = await run_pairwise(
        judge_fn=judge_fn,
        panel=['dead', 'flipper', 'steady'],
        replacement_judges=['stand-in'],
        response_a='better',
        response_b='worse',
    )
    assert comparison.winner == 'A'

    assert len(_by_name(exporter, 'orq.jury')) == 1
    jury = _attrs(_by_name(exporter, 'orq.jury')[0])
    assert jury['jury.judges_failed'] == 1  # 'dead' only
    assert jury['jury.flipped'] == 1
    assert jury['jury.flipped_judges'] == 'flipper'
    assert jury['jury.replacements_used'] == 1
    assert jury['jury.judges_configured'] == 3

    # The stand-in ran in both orderings and parents to the same jury span.
    stand_in_spans = [s for s in _by_name(exporter, 'orq.judge') if _attrs(s)['judge.model'] == 'stand-in']
    assert len(stand_in_spans) == 2
    jury_span_id = _span_id(_by_name(exporter, 'orq.jury')[0])
    for s in stand_in_spans:
        assert _parent_id(s) == jury_span_id
        assert _attrs(s)['judge.replacement'] is True


@pytest.mark.asyncio
async def test_pairwise_propagates_errors_and_marks_the_jury_span(span_collector) -> None:
    exporter = span_collector

    async def judge_fn(first: str, second: str, model: str) -> Prediction:
        raise RuntimeError('judge is down')

    with pytest.raises(RuntimeError):
        await run_pairwise(judge_fn=judge_fn, panel=['gpt-a'], response_a='x', response_b='y', propagate_errors=True)

    jury_spans = _by_name(exporter, 'orq.jury')
    assert len(jury_spans) == 1
    assert jury_spans[0].status.status_code is StatusCode.ERROR
    # The judge span still names who died.
    judge = _by_name(exporter, 'orq.judge')[0]
    assert _attrs(judge)['judge.model'] == 'gpt-a'
    assert judge.status.status_code is StatusCode.ERROR


@pytest.mark.asyncio
async def test_judge_error_is_truncated(span_collector) -> None:
    exporter = span_collector

    async def judge_fn(model: str) -> Prediction:
        raise RuntimeError('x' * 5000)

    with patch('evaluatorq.common.tracing._default_span_max_text_chars', return_value=100):
        await run_jury(judge_fn=judge_fn, panel=['gpt-a'])

    error = _attrs(_by_name(exporter, 'orq.judge')[0])['judge.error']
    assert len(error) == 100
    assert error.endswith('... [truncated]')


@pytest.mark.asyncio
async def test_explicit_root_parent_context_is_not_swapped_for_ambient(span_collector) -> None:
    """An empty Context is falsy (it subclasses dict). Passing one explicitly
    must still root the span, not silently inherit whatever is ambient."""
    exporter = span_collector
    from opentelemetry.context import Context

    async with with_span('outer'):
        async with with_span('inner', parent_context=Context()):
            pass

    inner = _by_name(exporter, 'inner')[0]
    assert _parent_id(inner) is None


@pytest.mark.asyncio
async def test_full_hierarchy_evaluation_jury_judge_llm(span_collector) -> None:
    """The shipped tree is orq.evaluation -> orq.jury -> orq.judge -> chat {model}.

    The evaluation span comes from process_evaluator, so this pins the seam
    between the core runner's span and the jury's — nothing threads a parent
    context across it, it rides the ambient OTel context.
    """
    exporter = span_collector

    async def judge_fn(model: str) -> Prediction:
        # Stands in for the real judge's with_llm_span(...) call.
        async with with_span(f'chat {model}'):
            return Prediction(value='yes', token_usage=TokenUsage(input_tokens=1, output_tokens=1))

    async def scorer(_params) -> EvaluationResult:
        deliberation = await run_jury(judge_fn=judge_fn, panel=['gpt-a'])
        return EvaluationResult(value=str(deliberation.verdict))

    score = await process_evaluator(
        {'name': 'jury-eval', 'scorer': scorer},
        DataPoint(inputs={}, expected_output=None),
        'some output',
    )
    assert score.error is None

    evaluation = next(s for s in exporter.spans if s.name.startswith('orq.evaluation'))
    jury = _by_name(exporter, 'orq.jury')[0]
    judge = _by_name(exporter, 'orq.judge')[0]
    chat = _by_name(exporter, 'chat gpt-a')[0]

    assert _parent_id(jury) == _span_id(evaluation)
    assert _parent_id(judge) == _span_id(jury)
    assert _parent_id(chat) == _span_id(judge)


@pytest.mark.asyncio
async def test_no_spans_when_tracing_disabled() -> None:
    """With get_tracer() returning None the spans are a no-op; the verdict is
    still produced (spans never change behavior)."""
    judge_fn = _make_judge_fn({'gpt-a': 'yes'})
    with patch('evaluatorq.common.tracing.get_tracer', return_value=None):
        deliberation = await run_jury(judge_fn=judge_fn, panel=['gpt-a'], verdict_kind=VerdictKind.CATEGORICAL)
    assert deliberation.verdict == 'yes'
