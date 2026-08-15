from __future__ import annotations

import pytest

from evaluatorq.processings import process_data_point
from evaluatorq.progress import Phase, ProgressService
from evaluatorq.types import DataPoint


class _FailingAfterResolveProgressService(ProgressService):
    """Raises once the data point has already resolved, simulating a failure
    that happens after resolution but before job execution."""

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
async def test_process_data_point_logs_and_preserves_resolved_datapoint(caplog) -> None:
    """When a data point resolves but something before job execution fails, the
    resolved DataPoint identity must be preserved on the error result, and a
    warning naming the row index must be logged."""

    resolved = DataPoint(inputs={'text': 'hello'})

    async def resolving_promise() -> DataPoint:
        return resolved

    def job(_data: DataPoint, _row: int):
        raise AssertionError('job should never run')

    with caplog.at_level('WARNING'):
        results = await process_data_point(
            data_promise=resolving_promise(),
            row_index=3,
            jobs=[job],
            evaluators=None,
            parallelism=1,
            progress_service=_FailingAfterResolveProgressService(),
        )

    assert len(results) == 1
    result = results[0]
    assert result.error is not None
    # The resolved DataPoint identity must be preserved, not collapsed to an
    # empty placeholder.
    assert result.data_point is resolved
    assert result.data_point.inputs == {'text': 'hello'}

    assert any('3' in record.getMessage() for record in caplog.records)


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
        parallelism=1,
    )

    assert len(results) == 1
    result = results[0]
    assert result.error is not None
    assert result.data_point.inputs == {'row_index': 7}
