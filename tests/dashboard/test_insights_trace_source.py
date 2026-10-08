"""Re-reading one Insights trace's conversation from its snapshot file or Orq."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest
from loguru import logger

from evaluatorq.dashboard.insights_trace_source import load_trace_record
from evaluatorq.insights.models import InsightsRun, reads_snapshot
from evaluatorq.insights.population import _load_traces
from evaluatorq.trace_finder import FacetSelection, NumericFilters
from evaluatorq.trace_finder.models import Snapshot, TraceRecord
from evaluatorq.trace_finder.orq_source import OrqTraceSource
from tests.insights.conftest import minimal_run  # noqa: F401 — fixture reused from the insights suite
from tests.trace_finder.test_orq_source import FakeOrq, FakeTraces, detail, span, summary


def _record(trace_id: str, span_id: str, content: str) -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id=trace_id,
        span_id=span_id,
        timestamp=datetime(2026, 9, 1, tzinfo=timezone.utc),
        messages=({'role': 'user', 'content': content},),
        project='default',
        model='m',
        provider='p',
        status='ok',
        product='deployments',
        trace_type='trace',
    )


def _write_snapshot(path: Path, records: list[TraceRecord]) -> str:
    path.write_bytes(Snapshot(traces=tuple(records)).model_dump_json().encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_run(run: InsightsRun, path: Path, digest: str | None) -> InsightsRun:
    population: dict[str, Any] = {'mode': 'snapshot', 'snapshot_path': str(path)}
    if digest is not None:
        population['snapshot_sha256'] = digest
    return run.model_copy(update={'population': population})


@pytest.fixture
def logged_warnings() -> Any:
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level='WARNING', format='{message}')
    yield messages
    logger.remove(sink)


@pytest.mark.asyncio
async def test_snapshot_run_returns_the_record_with_the_analysed_span(minimal_run: InsightsRun, tmp_path: Path) -> None:
    path = tmp_path / 'snap.json'
    digest = _write_snapshot(
        path, [_record('trace-1', 'span-0', 'other span'), _record('trace-1', 'span-1', 'analysed span')]
    )
    run = _snapshot_run(minimal_run, path, digest)

    result = await load_trace_record(run, run.traces[0], source=None)

    assert isinstance(result, TraceRecord)
    assert result.span_id == 'span-1'
    assert result.messages[0]['content'] == 'analysed span'


@pytest.mark.asyncio
async def test_snapshot_file_deleted_gives_a_reason_without_the_path(
    minimal_run: InsightsRun, tmp_path: Path, logged_warnings: list[str]
) -> None:
    path = tmp_path / 'gone.json'
    run = _snapshot_run(minimal_run, path, 'abc')

    result = await load_trace_record(run, run.traces[0], source=None)

    assert result == 'The snapshot file this run read is no longer available or could not be read.'
    assert str(tmp_path) not in result
    assert len(logged_warnings) == 1
    assert str(tmp_path) in logged_warnings[0]


@pytest.mark.asyncio
async def test_snapshot_rewritten_after_the_run_is_reported(minimal_run: InsightsRun, tmp_path: Path) -> None:
    path = tmp_path / 'snap.json'
    digest = _write_snapshot(path, [_record('trace-1', 'span-1', 'before')])
    _write_snapshot(path, [_record('trace-1', 'span-1', 'after')])
    run = _snapshot_run(minimal_run, path, digest)

    result = await load_trace_record(run, run.traces[0], source=None)

    assert result == 'The snapshot file changed after this run, so it may not match the analysis.'


@pytest.mark.asyncio
async def test_trace_absent_from_the_snapshot_is_reported(minimal_run: InsightsRun, tmp_path: Path) -> None:
    path = tmp_path / 'snap.json'
    digest = _write_snapshot(path, [_record('trace-2', 'span-2', 'only other')])
    run = _snapshot_run(minimal_run, path, digest)

    result = await load_trace_record(run, run.traces[0], source=None)

    assert result == 'This trace is not in the snapshot file.'


@pytest.mark.asyncio
async def test_snapshot_run_without_a_recorded_path_is_reported(minimal_run: InsightsRun) -> None:
    run = minimal_run.model_copy(update={'population': {'mode': 'snapshot'}})

    result = await load_trace_record(run, run.traces[0], source=None)

    assert result == 'This run does not record where its snapshot was read from.'


@pytest.mark.asyncio
async def test_orq_run_without_a_source_asks_for_an_account_and_leaves_logging_to_the_caller(
    minimal_run: InsightsRun, logged_warnings: list[str]
) -> None:
    result = await load_trace_record(minimal_run, minimal_run.traces[0], source=None)

    assert result == 'Connect an Orq account in Settings to load this conversation.'
    assert logged_warnings == []


@pytest.mark.asyncio
async def test_orq_run_hydrates_the_conversation(minimal_run: InsightsRun) -> None:
    traces = FakeTraces(
        pages={None: ([summary('trace-1', messages=[])], False, None)},
        spans={'trace-1': [span('span-9', minute=1)]},
        details={('trace-1', 'span-9'): detail('span-9', 'hello from orq', minute=1)},
    )
    source = OrqTraceSource(cast(Any, FakeOrq(traces)))

    result = await load_trace_record(minimal_run, minimal_run.traces[0], source=source)

    assert isinstance(result, TraceRecord)
    assert result.span_id == 'span-9'
    assert result.messages == ({'role': 'user', 'content': 'hello from orq'},)


@pytest.mark.asyncio
async def test_orq_run_queries_with_the_run_filters_around_the_analysed_time(minimal_run: InsightsRun) -> None:
    run = minimal_run.model_copy(update={'population': {'mode': 'filter', 'facets': {'status': ['error']}}})
    traces = FakeTraces(pages={None: ([summary('trace-1')], False, None)})
    analysed_at = run.traces[0].timestamp
    assert analysed_at is not None

    await load_trace_record(run, run.traces[0], source=OrqTraceSource(cast(Any, FakeOrq(traces))))

    [query] = traces.query_calls
    assert query['from_'] == analysed_at - timedelta(seconds=1)
    assert query['to'] == analysed_at + timedelta(seconds=1)
    assert 'filter status in ("error")' in query['oql']


@pytest.mark.asyncio
async def test_orq_failure_gives_the_could_not_load_reason(minimal_run: InsightsRun, logged_warnings: list[str]) -> None:
    class FailingTraces(FakeTraces):
        async def query_async(self, **kwargs: Any) -> Any:
            raise RuntimeError('orq is down')

    source = OrqTraceSource(cast(Any, FakeOrq(FailingTraces(pages={}))))

    result = await load_trace_record(minimal_run, minimal_run.traces[0], source=source)

    assert result == 'Could not load the conversation from Orq. Open it in Orq instead.'
    assert any('conversation unavailable' in message and 'orq is down' in message for message in logged_warnings)


@pytest.mark.asyncio
@pytest.mark.parametrize('page', [[], [summary('trace-1', messages=[])]], ids=['absent', 'no-messages'])
async def test_orq_trace_without_a_conversation_is_reported(minimal_run: InsightsRun, page: list[Any]) -> None:
    traces = FakeTraces(pages={None: (page, False, None)}, spans={'trace-1': []})
    source = OrqTraceSource(cast(Any, FakeOrq(traces)))

    result = await load_trace_record(minimal_run, minimal_run.traces[0], source=source)

    assert result == 'Orq no longer returns a conversation for this trace.'


def _summary_with_transcript_and_child_llm_span() -> FakeTraces:
    transcript = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]
    return FakeTraces(
        pages={None: ([summary('trace-1', messages=transcript)], False, None)},
        spans={'trace-1': [span('child-llm', minute=1, parent_span_id='root-trace-1')]},
        details={('trace-1', 'child-llm'): detail('child-llm', 'child only', minute=1)},
    )


@pytest.mark.asyncio
async def test_orq_run_reads_the_same_span_the_pipeline_analysed(minimal_run: InsightsRun) -> None:
    [analysed] = await _load_traces(
        cast(Any, FakeOrq(_summary_with_transcript_and_child_llm_span())),
        start=None,
        end=None,
        limit=10,
        facets=FacetSelection(),
        numeric=NumericFilters(),
    )
    trace = minimal_run.traces[0].model_copy(update={'span_id': analysed.span_id})
    source = OrqTraceSource(cast(Any, FakeOrq(_summary_with_transcript_and_child_llm_span())))

    result = await load_trace_record(minimal_run, trace, source=source)

    assert isinstance(result, TraceRecord)
    assert result.span_id == analysed.span_id
    assert [message['content'] for message in result.messages] == ['q', 'a']


@pytest.mark.parametrize(
    ('population', 'expected'),
    [
        ({'mode': 'snapshot'}, True),
        ({'mode': 'export', 'snapshot_path': '/x'}, True),
        ({'mode': 'query'}, False),
    ],
)
def test_reads_snapshot(population: dict[str, object], expected: bool) -> None:
    assert reads_snapshot(population) is expected
