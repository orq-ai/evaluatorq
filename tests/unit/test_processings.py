from __future__ import annotations

from typing import Any, cast
import pytest

from evaluatorq.processings import process_data_point
from evaluatorq.progress import Phase, ProgressService
from evaluatorq.types import DataPoint


class _FailingAfterResolveProgressService(ProgressService):
    """Raises when the progress display is updated after resolution."""

    async def update_progress(
        self,
        total_data_points: int | None = None,
        current_data_point: int | None = None,
        current_job: str | None = None,
        current_evaluator: str | None = None,
        phase: Phase | None = None,
    ) -> None:
        raise RuntimeError('progress update blew up')


@pytest.mark.asyncio
async def test_process_data_point_ignores_progress_display_failure(caplog) -> None:
    """A progress display failure must not prevent the resolved data point from running."""

    resolved = DataPoint(inputs={'text': 'hello'})

    async def resolving_promise() -> DataPoint:
        return resolved

    def job(_data: DataPoint, _row: int):
        return {'name': 'job', 'output': 'output'}

    with caplog.at_level('WARNING'):
        results = await process_data_point(
            data_promise=resolving_promise(),
            row_index=3,
            jobs=cast('Any', [job]),
            evaluators=None,
            datapoint_parallelism=1,
            progress_service=_FailingAfterResolveProgressService(),
        )

    assert len(results) == 1
    result = results[0]
    assert result.error is None
    assert result.data_point is resolved
    assert result.data_point.inputs == {'text': 'hello'}
    assert 'Progress display datapoint update failed; continuing' in caplog.text
    assert 'RuntimeError' in caplog.text


@pytest.mark.asyncio
async def test_process_data_point_fallback_carries_row_index_when_unresolvable() -> None:
    """When the data point itself fails to resolve, the fallback DataPoint must
    carry the row_index in its inputs so multiple failing rows don't collide
    (Task 7's converter guard relies on this to build a unique attack.id)."""

    async def failing_promise() -> DataPoint:
        raise ValueError('cannot resolve data point')

    results = await process_data_point(
        data_promise=failing_promise(),
        row_index=7,
        jobs=[],
        evaluators=None,
        datapoint_parallelism=1,
    )

    assert len(results) == 1
    result = results[0]
    assert result.error is not None
    assert result.data_point.inputs == {'row_index': 7}


@pytest.mark.asyncio
async def test_process_job_honours_a_job_reported_error() -> None:
    """A job that reports a failure it handled must not count as a successful row.

    The failure is returned, not raised, so nothing else in the run can see it: this
    is the branch that keeps a dead target from reporting a 100% pass rate.
    """
    from evaluatorq.evaluatorq import check_pass_failures
    from evaluatorq.processings import process_job
    from evaluatorq.types import DataPointResult

    async def reporting_job(_data: DataPoint, _row: int) -> dict[str, Any]:
        return {'name': 'sim', 'output': {'status': 'failed'}, 'error': '401 unauthorized'}

    result = await process_job(reporting_job, DataPoint(inputs={'text': 'hi'}), row_index=0)

    assert result.error == '401 unauthorized'
    # The output survives the error — it is the transcript you diagnose from.
    assert result.output == {'status': 'failed'}
    assert check_pass_failures(
        [DataPointResult(data_point=DataPoint(inputs={'text': 'hi'}), job_results=[result])],
        treat_errors_as_failure=True,
    )


@pytest.mark.asyncio
async def test_process_job_leaves_error_unset_on_success() -> None:
    """A job that omits the key, or sets it to None, still reports a clean row."""
    from evaluatorq.processings import process_job

    async def clean_job(_data: DataPoint, _row: int) -> dict[str, Any]:
        return {'name': 'sim', 'output': 'fine', 'error': None}

    async def silent_job(_data: DataPoint, _row: int) -> dict[str, Any]:
        return {'name': 'sim', 'output': 'fine'}

    for job in (clean_job, silent_job):
        result = await process_job(job, DataPoint(inputs={'text': 'hi'}), row_index=0)
        assert result.error is None


def test_job_reported_error_never_loses_a_reported_failure() -> None:
    """Presence is the failure signal — a blank payload must not flatten to a clean row.

    An empty message is exactly the case that put this PR here: nothing raised, so a
    row whose ``error`` flattened to ``None`` counted as a success over a dead target.
    Flattening delegates to ``output_error_text`` so a job-level and an output-level
    error payload cannot disagree about what the same dict means.
    """
    from evaluatorq.processings import _UNREADABLE_JOB_ERROR, _job_reported_error

    assert _job_reported_error('boom') == 'boom'
    assert _job_reported_error({'message': 'boom', 'code': 401}) == 'boom'
    # A present-but-falsy message is the payload's answer, and it still failed.
    assert _job_reported_error({'message': 0}) == '0'
    assert _job_reported_error({'message': ''}) == _UNREADABLE_JOB_ERROR
    assert _job_reported_error('') == _UNREADABLE_JOB_ERROR
    # Absent only when the key itself says so.
    assert _job_reported_error(None) is None
    # A dict with no readable message keeps its content rather than being dropped.
    unreadable = _job_reported_error({'weird': 'shape'})
    assert unreadable is not None and 'shape' in unreadable
    assert _job_reported_error(False) == 'False'


def test_summary_table_counts_a_job_reported_error() -> None:
    """The user-visible claim: `Failed Jobs` and `Success Rate` move off 0 and 100%."""
    from evaluatorq.table_display import create_summary_display
    from evaluatorq.types import DataPointResult, JobResult

    dead = DataPointResult(
        data_point=DataPoint(inputs={'text': 'hi'}),
        job_results=[JobResult(job_name='sim', output='dead', error='401 unauthorized', evaluator_scores=[])],
    )
    rows = {cells[0]: cells[1].plain for cells in zip(*(col._cells for col in create_summary_display([dead]).columns))}

    assert rows['Failed Jobs'] == '1'
    assert rows['Success Rate'] == '0%'


@pytest.mark.asyncio
async def test_process_job_skips_evaluators_on_a_reported_error_row() -> None:
    """A reported-error row keeps its output but is not scored.

    Scoring a transcript already known to be dead buys nothing and costs an LLM judge
    call per row — the raise path has never scored one either.
    """
    from evaluatorq.processings import process_job
    from evaluatorq.types import EvaluationResult, Evaluator, ScorerParameter

    async def reporting_job(_data: DataPoint, _row: int) -> dict[str, Any]:
        return {'name': 'sim', 'output': 'dead', 'error': 'boom'}

    calls: list[str] = []

    async def scorer(_params: ScorerParameter) -> EvaluationResult:  # noqa: RUF029
        calls.append('scored')
        return EvaluationResult(value=1)

    evaluator: Evaluator = {'name': 'always', 'scorer': scorer}
    result = await process_job(reporting_job, DataPoint(inputs={'text': 'hi'}), row_index=0, evaluators=[evaluator])

    assert result.error == 'boom'
    # The output survives for diagnosis even though nothing scored it.
    assert result.output == 'dead'
    assert result.evaluator_scores == []
    assert calls == []
