from __future__ import annotations

import pytest

from evaluatorq.progress import Phase, with_progress


@pytest.mark.asyncio
async def test_with_progress_final_update_failure_returns_result(caplog: pytest.LogCaptureFixture) -> None:
    """A completion display fault must not hide a successful coroutine result."""

    class FinalUpdateFailureService:
        async def start_spinner(self) -> None:
            return None

        async def update_progress(self, **kwargs: object) -> None:
            if kwargs.get('phase') == Phase.COMPLETED:
                raise BrokenPipeError('completion progress pipe closed')

        async def stop_spinner(self) -> None:
            return None

    async def completed_work() -> list[str]:
        return ['completed']

    with caplog.at_level('WARNING'):
        result = await with_progress(completed_work(), FinalUpdateFailureService())

    assert result == ['completed']
    assert 'completion progress pipe closed' in caplog.text
    assert 'BrokenPipeError' in caplog.text
