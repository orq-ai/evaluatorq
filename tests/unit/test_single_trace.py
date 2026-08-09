"""``evaluatorq(single_trace=...)`` — one trace per run vs one trace per row.

Without the flag every row's ``orq.job`` is its own root, so an N-row run lands
as N separate traces. With it, one ``evaluatorq.run`` span brackets the run and
all rows share a trace.
"""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from evaluatorq import DataPoint, evaluatorq, job


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
    with patch('evaluatorq.tracing.spans.get_tracer', return_value=tracer):
        yield exporter
    provider.shutdown()


@job('echo')
async def echo(data: DataPoint, _row: int) -> str:
    return str(data.inputs.get('text', ''))


DATA = [DataPoint(inputs={'text': 'a'}), DataPoint(inputs={'text': 'b'})]


def _by_name(exporter: _Exporter, name: str) -> list[ReadableSpan]:
    return [s for s in exporter.spans if s.name == name]


def _trace_ids(spans: Sequence[ReadableSpan]) -> set[int]:
    return {s.context.trace_id for s in spans if s.context is not None}


async def _run(*, single_trace: bool):
    return await evaluatorq(
        'single-trace-test',
        data=DATA,
        jobs=[echo],
        single_trace=single_trace,
        print_results=False,
        _send_results=False,
        _exit_on_failure=False,
    )


@pytest.mark.asyncio
async def test_rows_are_separate_traces_by_default(span_collector) -> None:
    exporter = span_collector

    await _run(single_trace=False)

    assert _by_name(exporter, 'evaluatorq.run') == []
    jobs = _by_name(exporter, 'orq.job')
    assert len(jobs) == 2
    # Each job is its own root, and therefore its own trace.
    assert all(j.parent is None for j in jobs)
    assert len(_trace_ids(jobs)) == 2


@pytest.mark.asyncio
async def test_single_trace_groups_every_row_under_one_run_span(span_collector) -> None:
    exporter = span_collector

    await _run(single_trace=True)

    run_spans = _by_name(exporter, 'evaluatorq.run')
    assert len(run_spans) == 1
    run = run_spans[0]
    assert run.parent is None
    assert run.context is not None

    jobs = _by_name(exporter, 'orq.job')
    assert len(jobs) == 2
    for j in jobs:
        assert j.parent is not None
        assert j.parent.span_id == run.context.span_id
    # One trace for the whole run, rows included.
    assert _trace_ids([run, *jobs]) == {run.context.trace_id}


@pytest.mark.asyncio
async def test_run_span_carries_run_identity(span_collector) -> None:
    exporter = span_collector

    await _run(single_trace=True)

    attrs = dict(_by_name(exporter, 'evaluatorq.run')[0].attributes or {})
    assert attrs['orq.run_name'] == 'single-trace-test'
    assert attrs['orq.trace_type'] == 'evaluatorq'
    # Same run id under both keys: orq.run_id matches the job spans, and
    # orq.evaluatorq_run_id is the cross-surface root key.
    assert attrs['orq.run_id'] == attrs['orq.evaluatorq_run_id']
    job_run_ids = {dict(j.attributes or {})['orq.run_id'] for j in _by_name(exporter, 'orq.job')}
    assert job_run_ids == {attrs['orq.run_id']}
