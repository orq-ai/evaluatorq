from __future__ import annotations

from datetime import datetime, timedelta, timezone

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


def test_running_stage_gets_flag_with_elapsed_time_and_count() -> None:
    html = progress(_manifest(ManifestStatus.RUNNING, ManifestStatus.RUNNING))
    assert 'Running · 1m 15s' in html
    assert '1 of 3' in html
    assert '<li class="insights-stage pending"' in html


def test_failed_stage_gets_failed_flag_and_later_stages_skip() -> None:
    html = progress(_manifest(ManifestStatus.ERROR, ManifestStatus.ERROR))
    assert '<span class="insights-stage-flag">Failed</span>' in html
    assert '<li class="insights-stage skipped"' in html
