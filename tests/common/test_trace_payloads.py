"""Job and evaluator payloads survive real OTel span recording."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from evaluatorq.common.tracing import _default_span_max_text_chars
from evaluatorq.processings import process_evaluator, process_job
from evaluatorq.types import DataPoint, EvaluationResult, ScorerParameter


@pytest.fixture
def spans(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer('payload-tests')
    monkeypatch.setattr('evaluatorq.common.tracing.get_tracer', lambda: tracer)
    monkeypatch.setenv('EVALUATORQ_CAPTURE_MESSAGE_CONTENT', 'true')
    monkeypatch.delenv('EVALUATORQ_SPAN_MAX_TEXT_CHARS', raising=False)
    _default_span_max_text_chars.cache_clear()
    yield exporter
    provider.shutdown()
    _default_span_max_text_chars.cache_clear()


def _span(exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    return next(s for s in exporter.get_finished_spans() if s.name == name)


def _payload(span: ReadableSpan, key: str) -> Any:
    return json.loads(str((span.attributes or {})[key]))


@pytest.mark.asyncio
@pytest.mark.parametrize(('output', 'captured'), [
    ('hello', {'value': 'hello'}), ('', {'value': ''}), (False, {'value': False}),
    (0, {'value': 0}), (None, {'value': None}), ({}, {'value': {}}), ([], {'value': []}),
    ({'nested': [False, 0, '']}, {'nested': [False, 0, '']}),
])
async def test_job_records_datapoint_and_falsy_output(spans: InMemorySpanExporter, output: Any, captured: Any) -> None:
    data = DataPoint(inputs={'question': 'hello'}, expected_output='reference')

    async def job(_data: DataPoint, _row: int) -> dict[str, Any]:
        return {'name': 'echo', 'output': output}

    result = await process_job(job, data, row_index=4)

    span = _span(spans, 'orq.job')
    assert result.error is None
    assert result.output == output
    assert _payload(span, 'gen_ai.input') == data.model_dump()
    assert _payload(span, 'gen_ai.output') == captured


@pytest.mark.asyncio
async def test_job_records_pydantic_message_fields(spans: InMemorySpanExporter) -> None:
    from evaluatorq.contracts import Message

    messages = [Message(role='assistant', content='A structured reply')]

    async def job(_data: DataPoint, _row: int) -> dict[str, Any]:
        return {'name': 'messages', 'output': messages}

    result = await process_job(job, DataPoint(inputs={}), row_index=0)

    assert result.error is None
    output = _payload(_span(spans, 'orq.job'), 'gen_ai.output')
    assert output[0]['role'] == 'assistant'
    assert output[0]['content'] == 'A structured reply'


@pytest.mark.asyncio
async def test_job_exception_keeps_input_and_error(spans: InMemorySpanExporter) -> None:
    async def job(_data: DataPoint, _row: int) -> dict[str, Any]:
        raise RuntimeError('target unavailable')

    result = await process_job(job, DataPoint(inputs={'request': 'test'}), row_index=0)

    span = _span(spans, 'orq.job')
    assert result.error == 'target unavailable'
    assert span.status.status_code == StatusCode.ERROR
    assert _payload(span, 'gen_ai.input')['inputs'] == {'request': 'test'}
    assert 'gen_ai.output' not in (span.attributes or {})


@pytest.mark.asyncio
async def test_returned_job_error_keeps_output_and_skips_evaluator(spans: InMemorySpanExporter) -> None:
    async def job(_data: DataPoint, _row: int) -> dict[str, Any]:
        return {'name': 'failed', 'output': {'partial': 'reply'}, 'error': 'target failed'}

    async def scorer(_params: ScorerParameter) -> EvaluationResult:
        pytest.fail('A failed job must not be scored')

    result = await process_job(job, DataPoint(inputs={}), row_index=0, evaluators=[{'name': 'skip', 'scorer': scorer}])

    span = _span(spans, 'orq.job')
    assert result.error == 'target failed'
    assert result.evaluator_scores == []
    assert span.status.status_code == StatusCode.ERROR
    assert _payload(span, 'gen_ai.output') == {'partial': 'reply'}


@pytest.mark.asyncio
async def test_custom_evaluator_records_input_and_classifies_result(spans: InMemorySpanExporter) -> None:
    data = DataPoint(inputs={'question': 'test'}, expected_output='answer')

    async def scorer(params: ScorerParameter) -> EvaluationResult:
        assert params['row'] == 2
        return EvaluationResult(value=False, explanation='The answer differs', pass_=False)

    result = await process_evaluator({'name': 'match', 'scorer': scorer}, data, 'actual', row_index=2)

    span = _span(spans, 'orq.evaluation match')
    assert result.error is None
    assert _payload(span, 'orq.evaluation.input') == {'data': data.model_dump(), 'output': 'actual', 'row': 2}
    assert _payload(span, 'orq.evaluation.output') == {
        'passed': False, 'value': False, 'type': 'python_eval', 'reason': 'The answer differs',
    }
    assert (span.attributes or {})['orq.span_type'] == 'span.evaluator'


@pytest.mark.asyncio
async def test_simulation_evaluator_keeps_structured_criteria(spans: InMemorySpanExporter) -> None:
    from evaluatorq.contracts import TokenUsage
    from evaluatorq.simulation.api import _adapt_simulation_scorer
    from evaluatorq.simulation.contracts import SimulationResult, TerminatedBy
    from evaluatorq.simulation.evaluators import get_evaluator

    data = DataPoint(inputs={'persona': 'synthetic customer'})
    simulation = SimulationResult(
        messages=[], terminated_by=TerminatedBy.judge, reason='Finished', goal_achieved=True,
        goal_completion_score=1.0, rules_broken=[], turn_count=1, token_usage=TokenUsage(),
        turn_metrics=[], criteria_verified=True, criteria_results={'greeted': True},
        metadata={'criteria_meta': [{
            'id': 'greeting', 'description': 'greeted', 'type': 'must_happen',
            'passed': True, 'audited': True, 'evidence': 'Hello',
        }]},
    )
    evaluator = _adapt_simulation_scorer('criteria_met', get_evaluator('criteria_met'), {id(data): simulation})
    response = {'object': 'response', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Hello'}]}]}

    result = await process_evaluator(evaluator, data, response)

    span = _span(spans, 'orq.evaluation criteria_met')
    assert result.error is None
    assert result.score.value == 1.0
    assert _payload(span, 'orq.evaluation.input')['output'] == response
    output = _payload(span, 'orq.evaluation.output')
    assert output['passed'] is True
    assert output['value'] == 1.0
    assert output['raw_output']['criteria'][0]['evidence'] == 'Hello'
    assert output['raw_output']['criteria_verified'] is True


@pytest.mark.asyncio
async def test_capture_gate_suppresses_job_and_evaluator_payloads(spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EVALUATORQ_CAPTURE_MESSAGE_CONTENT', 'false')

    async def job(_data: DataPoint, _row: int) -> dict[str, Any]:
        return {'name': 'private', 'output': {'secret': 'reply'}}

    async def scorer(_params: ScorerParameter) -> EvaluationResult:
        return EvaluationResult(value=1, explanation='private reason', raw_output={'secret': 'details'})

    result = await process_job(job, DataPoint(inputs={'secret': 'request'}), row_index=0, evaluators=[{'name': 'private', 'scorer': scorer}])

    assert result.error is None
    assert result.evaluator_scores is not None
    assert result.evaluator_scores[0].error is None
    for span in spans.get_finished_spans():
        attrs = span.attributes or {}
        for key in ('gen_ai.input', 'gen_ai.output', 'orq.evaluation.input', 'orq.evaluation.output'):
            assert key not in attrs
    assert (_span(spans, 'orq.evaluation private').attributes or {})['gen_ai.evaluation.score.value'] == 1.0


@pytest.mark.asyncio
async def test_scorer_failure_retains_input(spans: InMemorySpanExporter) -> None:
    async def scorer(_params: ScorerParameter) -> EvaluationResult:
        raise RuntimeError('judge unavailable')

    result = await process_evaluator({'name': 'broken', 'scorer': scorer}, DataPoint(inputs={}), 'reply')

    span = _span(spans, 'orq.evaluation broken')
    assert result.error == 'judge unavailable'
    assert _payload(span, 'orq.evaluation.input')['output'] == 'reply'
    assert _payload(span, 'orq.evaluation.output') == {'error': 'judge unavailable'}
    assert span.status.status_code == StatusCode.ERROR
    assert (span.attributes or {})['orq.span_type'] == 'span.evaluator'


@pytest.mark.asyncio
async def test_text_truncation_preserves_json_structure(spans: InMemorySpanExporter, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EVALUATORQ_SPAN_MAX_TEXT_CHARS', '24')
    _default_span_max_text_chars.cache_clear()

    async def job(_data: DataPoint, _row: int) -> dict[str, Any]:
        return {'name': 'long', 'output': {'nested': ['hello "quoted" \\ world ' * 20, False]}}

    await process_job(job, DataPoint(inputs={'text': 'x' * 100}), row_index=0)

    span = _span(spans, 'orq.job')
    output = _payload(span, 'gen_ai.output')
    assert len(output['nested'][0]) <= 24
    assert output['nested'][0].endswith('... [truncated]')
    assert output['nested'][1] is False
    assert len(_payload(span, 'gen_ai.input')['inputs']['text']) <= 24


@pytest.mark.asyncio
async def test_circular_output_does_not_fail_job(spans: InMemorySpanExporter) -> None:
    circular: dict[str, Any] = {}
    circular['self'] = circular

    async def job(_data: DataPoint, _row: int) -> dict[str, Any]:
        return {'name': 'circular', 'output': circular}

    result = await process_job(job, DataPoint(inputs={}), row_index=0)

    span = _span(spans, 'orq.job')
    assert result.error is None
    assert span.status.status_code == StatusCode.OK
    assert 'gen_ai.input' in (span.attributes or {})
    assert 'gen_ai.output' not in (span.attributes or {})


@pytest.mark.asyncio
@pytest.mark.parametrize('nonfinite', [float('nan'), float('inf'), float('-inf')], ids=['nan', 'inf', 'negative-inf'])
async def test_nonfinite_payload_is_skipped_without_failing_job(spans: InMemorySpanExporter, nonfinite: float) -> None:
    async def job(_data: DataPoint, _row: int) -> dict[str, Any]:
        return {'name': 'nonfinite', 'output': {'measurement': nonfinite}}

    result = await process_job(job, DataPoint(inputs={}), row_index=0)

    span = _span(spans, 'orq.job')
    assert result.error is None
    assert span.status.status_code == StatusCode.OK
    assert 'gen_ai.input' in (span.attributes or {})
    assert 'gen_ai.output' not in (span.attributes or {})


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_detail', ['nan', 'inf', 'negative-inf', 'circular'])
async def test_invalid_optional_diagnostics_preserve_verdict(
    spans: InMemorySpanExporter, caplog: pytest.LogCaptureFixture, bad_detail: str,
) -> None:
    diagnostics: dict[str, Any] = {}
    if bad_detail == 'circular':
        diagnostics['self'] = diagnostics
    else:
        diagnostics['measurement'] = float({'nan': 'nan', 'inf': 'inf', 'negative-inf': '-inf'}[bad_detail])

    async def scorer(_params: ScorerParameter) -> EvaluationResult:
        return EvaluationResult(value=0.75, pass_=True, explanation='valid verdict', raw_output=diagnostics)

    result = await process_evaluator({'name': 'diagnostics', 'scorer': scorer}, DataPoint(inputs={}), 'reply')

    span = _span(spans, 'orq.evaluation diagnostics')
    assert result.error is None
    assert result.score.value == 0.75
    assert _payload(span, 'orq.evaluation.output') == {
        'passed': True, 'value': 0.75, 'type': 'python_eval', 'reason': 'valid verdict',
    }
    assert 'raw_output' in caplog.text
    assert 'optional' in caplog.text
    assert span.status.status_code == StatusCode.OK
