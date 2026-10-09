"""Re-reading one Insights trace's conversation from its snapshot file or Orq."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from loguru import logger

from evaluatorq.dashboard.insights_trace_source import (
    ORQ_SPAN_CHANGED,
    conversation_rereadable,
    conversation_unavailable_reason,
    load_orq_record,
    load_snapshot_record,
    source_not_rereadable,
)
from evaluatorq.dashboard.insights_uploads import uploads_dir
from evaluatorq.insights.models import InsightsRun, population_source, reads_orq, reads_snapshot
from evaluatorq.insights.population import _load_traces
from evaluatorq.insights.store import get_insights_runs_dir
from evaluatorq.trace_finder import FacetSelection, NumericFilters
from evaluatorq.trace_finder.models import TraceRecord
from evaluatorq.trace_finder.orq_source import OrqTraceSource
from tests.dashboard.insights_trace_helpers import snapshot_run, write_snapshot
from tests.dashboard.test_insights_page import minimal_run  # noqa: F401 — fixture reused
from tests.insights.test_population import make_trace
from tests.trace_finder.test_orq_source import FakeOrq, FakeTraces, detail, span, summary


@pytest.fixture
def orq_run(minimal_run: InsightsRun) -> InsightsRun:
    """An Orq query run without project facets, which the fake client cannot resolve to project ids."""
    return minimal_run.model_copy(update={'population': {'mode': 'query'}})


@pytest.fixture
def logged_warnings() -> Any:
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level='WARNING', format='{message}')
    yield messages
    logger.remove(sink)


def _opened(source: OrqTraceSource | str) -> Any:
    @asynccontextmanager
    async def open_source() -> Any:
        yield source

    return open_source


@pytest.mark.asyncio
async def test_snapshot_run_returns_the_record_with_the_analysed_span(minimal_run: InsightsRun, tmp_path: Path) -> None:
    path = tmp_path / 'snap.json'
    digest = write_snapshot(
        path,
        [
            make_trace('trace-1', span_id='span-0', content='other span'),
            make_trace('trace-1', span_id='span-1', content='analysed span'),
        ],
    )
    run = snapshot_run(minimal_run, path, digest)

    result = await load_snapshot_record(run, run.traces[0])

    assert isinstance(result, TraceRecord)
    assert result.span_id == 'span-1'
    assert result.messages[0]['content'] == 'analysed span'


@pytest.mark.asyncio
async def test_snapshot_file_deleted_gives_a_reason_without_the_path(
    minimal_run: InsightsRun, tmp_path: Path, logged_warnings: list[str]
) -> None:
    path = tmp_path / 'gone.json'
    run = snapshot_run(minimal_run, path, 'abc')

    result = await load_snapshot_record(run, run.traces[0])

    assert result == 'The snapshot file this run read is no longer available or could not be read.'
    assert str(tmp_path) not in result
    assert len(logged_warnings) == 1
    assert str(tmp_path) in logged_warnings[0]


@pytest.mark.asyncio
async def test_snapshot_rewritten_after_the_run_is_reported(minimal_run: InsightsRun, tmp_path: Path) -> None:
    path = tmp_path / 'snap.json'
    digest = write_snapshot(path, [make_trace('trace-1', span_id='span-1', content='before')])
    write_snapshot(path, [make_trace('trace-1', span_id='span-1', content='after')])
    run = snapshot_run(minimal_run, path, digest)

    result = await load_snapshot_record(run, run.traces[0])

    assert result == 'The snapshot file changed after this run, so it may not match the analysis.'


@pytest.mark.asyncio
async def test_trace_absent_from_the_snapshot_is_reported(minimal_run: InsightsRun, tmp_path: Path) -> None:
    path = tmp_path / 'snap.json'
    digest = write_snapshot(path, [make_trace('trace-2', span_id='span-2', content='only other')])
    run = snapshot_run(minimal_run, path, digest)

    result = await load_snapshot_record(run, run.traces[0])

    assert result == 'This trace is not in the snapshot file.'


@pytest.mark.asyncio
async def test_snapshot_run_without_a_recorded_path_is_reported(minimal_run: InsightsRun) -> None:
    run = minimal_run.model_copy(update={'population': {'mode': 'snapshot'}})

    result = await load_snapshot_record(run, run.traces[0])

    assert result == 'This run does not record where its snapshot was read from.'


@pytest.mark.asyncio
async def test_orq_run_returns_the_reason_its_source_could_not_open_without_logging_again(
    orq_run: InsightsRun, logged_warnings: list[str]
) -> None:
    result = await load_orq_record(
        orq_run, orq_run.traces[0], open_source=_opened('Connect an Orq account in Settings to load this conversation.')
    )

    assert result == 'Connect an Orq account in Settings to load this conversation.'
    assert logged_warnings == []


@pytest.mark.asyncio
async def test_orq_run_hydrates_the_conversation(orq_run: InsightsRun) -> None:
    traces = FakeTraces(
        pages={None: ([summary('trace-1', messages=[])], False, None)},
        spans={'trace-1': [span('span-9', minute=1)]},
        details={('trace-1', 'span-9'): detail('span-9', 'hello from orq', minute=1)},
    )
    source = OrqTraceSource(cast(Any, FakeOrq(traces)))
    trace = orq_run.traces[0].model_copy(update={'span_id': 'span-9'})
    result = await load_orq_record(orq_run, trace, open_source=_opened(source))

    assert isinstance(result, TraceRecord)
    assert result.span_id == 'span-9'
    assert result.messages == ({'role': 'user', 'content': 'hello from orq'},)


@pytest.mark.asyncio
@pytest.mark.parametrize('saved_span_id', ['span-1', None])
async def test_orq_run_reports_a_changed_span_without_returning_its_conversation(
    orq_run: InsightsRun, saved_span_id: str | None
) -> None:
    traces = FakeTraces(
        pages={None: ([summary('trace-1', messages=[])], False, None)},
        spans={'trace-1': [span('span-current', minute=1)]},
        details={('trace-1', 'span-current'): detail('span-current', 'new conversation', minute=1)},
    )
    trace = orq_run.traces[0].model_copy(update={'span_id': saved_span_id})
    source = OrqTraceSource(cast(Any, FakeOrq(traces)))

    result = await load_orq_record(orq_run, trace, open_source=_opened(source))

    assert result == ORQ_SPAN_CHANGED
    assert 'new conversation' not in str(result)


@pytest.mark.asyncio
async def test_orq_run_queries_with_the_run_filters_around_the_analysed_time(minimal_run: InsightsRun) -> None:
    run = minimal_run.model_copy(update={'population': {'mode': 'filter', 'facets': {'status': ['error']}}})
    traces = FakeTraces(pages={None: ([summary('trace-1')], False, None)})
    analysed_at = run.traces[0].timestamp
    assert analysed_at is not None

    await load_orq_record(run, run.traces[0], open_source=_opened(OrqTraceSource(cast(Any, FakeOrq(traces)))))

    [query] = traces.query_calls
    assert query['from_'] == analysed_at - timedelta(seconds=1)
    assert query['to'] == analysed_at + timedelta(seconds=1)
    assert 'filter status in ("error")' in query['oql']


@pytest.mark.asyncio
async def test_orq_failure_gives_the_could_not_load_reason(orq_run: InsightsRun, logged_warnings: list[str]) -> None:
    class FailingTraces(FakeTraces):
        async def query_async(self, **kwargs: Any) -> Any:
            raise RuntimeError('orq is down')

    source = OrqTraceSource(cast(Any, FakeOrq(FailingTraces(pages={}))))

    result = await load_orq_record(orq_run, orq_run.traces[0], open_source=_opened(source))

    assert result == 'Could not load the conversation from Orq. Open it in Orq instead.'
    assert any('conversation unavailable' in message and 'orq is down' in message for message in logged_warnings)


@pytest.mark.asyncio
@pytest.mark.parametrize('page', [[], [summary('trace-1', messages=[])]], ids=['absent', 'no-messages'])
async def test_orq_trace_without_a_conversation_is_reported(orq_run: InsightsRun, page: list[Any]) -> None:
    traces = FakeTraces(pages={None: (page, False, None)}, spans={'trace-1': []})
    source = OrqTraceSource(cast(Any, FakeOrq(traces)))

    result = await load_orq_record(orq_run, orq_run.traces[0], open_source=_opened(source))

    assert result == (
        'Orq did not return this trace in the workspace used for this request. '
        'Check that the account has access to that workspace and that the trace still exists.'
    )


def _summary_with_transcript_and_child_llm_span() -> FakeTraces:
    transcript = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a'}]
    return FakeTraces(
        pages={None: ([summary('trace-1', messages=transcript)], False, None)},
        spans={'trace-1': [span('child-llm', minute=1, parent_span_id='root-trace-1')]},
        details={('trace-1', 'child-llm'): detail('child-llm', 'child only', minute=1)},
    )


@pytest.mark.asyncio
async def test_orq_run_reads_the_same_span_the_pipeline_analysed(orq_run: InsightsRun) -> None:
    [analysed] = await _load_traces(
        cast(Any, FakeOrq(_summary_with_transcript_and_child_llm_span())),
        start=None,
        end=None,
        limit=10,
        facets=FacetSelection(),
        numeric=NumericFilters(),
    )
    trace = orq_run.traces[0].model_copy(update={'span_id': analysed.span_id})
    source = OrqTraceSource(cast(Any, FakeOrq(_summary_with_transcript_and_child_llm_span())))

    result = await load_orq_record(orq_run, trace, open_source=_opened(source))

    assert isinstance(result, TraceRecord)
    assert result.span_id == analysed.span_id
    assert [message['content'] for message in result.messages] == ['q', 'a']


@pytest.mark.asyncio
async def test_orq_run_does_not_return_a_different_analyzed_span(
    orq_run: InsightsRun, logged_warnings: list[str]
) -> None:
    trace = orq_run.traces[0].model_copy(update={'span_id': 'analysed-span'})
    source = OrqTraceSource(cast(Any, FakeOrq(_summary_with_transcript_and_child_llm_span())))

    result = await load_orq_record(orq_run, trace, open_source=_opened(source))

    assert result == 'The conversation span analyzed by this run is no longer the span returned by Orq.'
    assert len([message for message in logged_warnings if 'conversation unavailable' in message]) == 1


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


@pytest.mark.parametrize(
    ('population', 'expected'),
    [
        ({'mode': 'query'}, True),
        ({'mode': 'export'}, True),
        ({'mode': 'filter'}, True),
        ({'mode': 'export', 'snapshot_path': '/x'}, False),
        ({'mode': 'snapshot'}, False),
        ({'mode': 'dataset'}, False),
        ({'request': None, 'query': None}, False),
    ],
)
def test_reads_orq(population: dict[str, object], expected: bool) -> None:
    assert reads_orq(population) is expected


def _uploaded_snapshot_path() -> Path:
    directory = uploads_dir(get_insights_runs_dir())
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f'snapshot-{"0" * 32}.json'


@pytest.mark.asyncio
async def test_orq_run_without_an_analysed_time_is_reported_before_opening_a_source(
    orq_run: InsightsRun, logged_warnings: list[str]
) -> None:
    trace = orq_run.traces[0].model_copy(update={'timestamp': None})

    def open_source() -> Any:
        raise AssertionError('a trace without a timestamp must not look up credentials')

    result = await load_orq_record(orq_run, trace, open_source=open_source)

    assert result == 'This run did not record when the trace ran, so its conversation cannot be re-read from Orq.'
    assert len(logged_warnings) == 1


def test_run_from_an_unknown_source_is_not_re_read(minimal_run: InsightsRun, logged_warnings: list[str]) -> None:
    run = minimal_run.model_copy(update={'population': {'mode': 'dataset'}})

    assert source_not_rereadable(run, run.traces[0]) == "This run's trace source cannot be re-read."
    assert len(logged_warnings) == 1
    assert not conversation_rereadable(run)
    assert conversation_unavailable_reason(run) == "This run's trace source cannot be re-read."


@pytest.mark.asyncio
async def test_uploaded_snapshot_still_on_disk_is_read(minimal_run: InsightsRun) -> None:
    path = _uploaded_snapshot_path()
    digest = write_snapshot(path, [make_trace('trace-1', span_id='span-1', content='still here')])
    run = snapshot_run(minimal_run, path, digest)

    result = await load_snapshot_record(run, run.traces[0])

    assert isinstance(result, TraceRecord)
    assert result.messages[0]['content'] == 'still here'
    assert conversation_rereadable(run)


@pytest.mark.asyncio
async def test_deleted_uploaded_snapshot_says_the_dashboard_deleted_the_file(
    minimal_run: InsightsRun, logged_warnings: list[str]
) -> None:
    run = snapshot_run(minimal_run, _uploaded_snapshot_path(), 'abc')
    deleted = (
        'The dashboard deletes uploaded snapshot files after the run, so this conversation is no longer available.'
    )

    result = await load_snapshot_record(run, run.traces[0])

    assert result == deleted
    assert len(logged_warnings) == 1
    assert not conversation_rereadable(run)
    assert conversation_unavailable_reason(run) == deleted


def test_conversation_rereadable_for_orq_runs_and_snapshot_files_that_exist(
    minimal_run: InsightsRun, tmp_path: Path
) -> None:
    kept = tmp_path / 'snap.json'
    write_snapshot(kept, [])

    assert conversation_rereadable(minimal_run)
    assert conversation_rereadable(snapshot_run(minimal_run, kept, None))
    assert not conversation_rereadable(snapshot_run(minimal_run, tmp_path / 'gone.json', None))
    assert conversation_unavailable_reason(snapshot_run(minimal_run, tmp_path / 'gone.json', None)) == (
        'The snapshot file this run read is no longer available or could not be read.'
    )
    assert not conversation_rereadable(minimal_run.model_copy(update={'population': {'mode': 'snapshot'}}))


@pytest.mark.parametrize(
    ('population', 'expected'),
    [
        ({'mode': 'snapshot'}, 'snapshot'),
        ({'mode': 'export', 'snapshot_path': '/x'}, 'snapshot'),
        ({'mode': 'query'}, 'query'),
        ({'mode': 'export', 'finder_export': '/x'}, 'export'),
        ({'mode': 'filter'}, 'filter'),
        ({'mode': 'dataset'}, 'unknown'),
        ({'mode': 'finder', 'finder_export': '/x'}, 'unknown'),
        ({}, 'unknown'),
    ],
)
def test_population_source(population: dict[str, object], expected: str) -> None:
    assert population_source(population) == expected
