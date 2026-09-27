from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from evaluatorq.contracts import ManifestStatus, ManifestSurface, RunManifest, StageRecord
from evaluatorq.dashboard.insights_views import progress


def _manifest(status: ManifestStatus, current: ManifestStatus) -> RunManifest:
    now = datetime.now(timezone.utc)
    stages = [
        StageRecord(name='fetch', status=ManifestStatus.COMPLETED, started_at=now - timedelta(minutes=3)),
        StageRecord(name='cluster', status=current, started_at=now - timedelta(seconds=75)),
    ]
    return RunManifest(
        run_id='r', surface=ManifestSurface.INSIGHTS, run_name='r', status=status, stage='cluster',
        stages=stages, planned_stages=['fetch', 'cluster', 'report'], started_at=now, updated_at=now,
    )


def test_running_stage_gets_flag_with_elapsed_time_and_count(monkeypatch: pytest.MonkeyPatch) -> None:
    fixed_now = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return fixed_now

    monkeypatch.setattr('evaluatorq.dashboard.insights_views.datetime', FrozenDateTime)
    manifest = _manifest(ManifestStatus.RUNNING, ManifestStatus.RUNNING)
    manifest.stages[1].started_at = fixed_now - timedelta(seconds=75)
    html = progress(manifest)
    assert 'Running · 1m 15s' in html
    assert '1 of 3' in html
    assert '<li class="insights-stage pending"' in html


@pytest.mark.parametrize(('field', 'value'), [('completed', True), ('total', False)])
def test_stage_progress_rejects_boolean_counts(field: str, value: bool) -> None:
    now = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        StageRecord(name='cluster', started_at=now, **{field: value})


def test_failed_stage_gets_failed_flag_and_later_stages_skip() -> None:
    html = progress(_manifest(ManifestStatus.ERROR, ManifestStatus.ERROR))
    assert '<span class="insights-stage-flag">Failed</span>' in html
    assert '<li class="insights-stage skipped"' in html
