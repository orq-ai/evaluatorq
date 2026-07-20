"""Red-team runner ↔ lifecycle-manifest integration: a run must never stay
'running' once it has finished, whatever fails and wherever."""

from __future__ import annotations

import warnings
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from evaluatorq.common.run_manifest import list_manifests
from evaluatorq.redteam.exceptions import CancelledError
from evaluatorq.redteam.hooks import DefaultHooks
from evaluatorq.redteam.runner import get_runs_dir, red_team


def _fake_report() -> MagicMock:
    # A stand-in report: only the attributes red_team touches post-pipeline.
    report = MagicMock()
    report.pipeline_warnings = []
    return report


@pytest.mark.asyncio
async def test_validation_error_leaves_no_manifest() -> None:
    """A config error before the run starts creates no manifest at all — there
    is nothing to be stuck 'running'."""
    with pytest.raises(ValueError, match='at least one target'):
        await red_team([], mode='dynamic')
    assert list_manifests(get_runs_dir()) == []


@pytest.mark.asyncio
async def test_pipeline_failure_marks_manifest_errored() -> None:
    with patch('evaluatorq.redteam.runner._run_dynamic_or_hybrid') as mock_dynamic:

        async def _boom(**_kwargs: Any) -> MagicMock:
            raise RuntimeError('pipeline boom')

        mock_dynamic.side_effect = _boom
        with pytest.raises(RuntimeError, match='pipeline boom'):
            await red_team('agent:test', mode='dynamic', generate_executive_summary=False)

    [m] = list_manifests(get_runs_dir())
    assert m.status == 'error'
    assert m.error == 'pipeline boom'
    assert m.ended_at is not None


@pytest.mark.asyncio
async def test_on_complete_raising_does_not_report_completed() -> None:
    """complete() runs only after on_complete succeeds — a raising on_complete
    leaves the manifest 'error', matching the error the caller receives."""

    class BadComplete(DefaultHooks):
        async def on_complete(self, report: Any, **_kw: Any) -> None:
            raise RuntimeError('complete boom')

    with patch('evaluatorq.redteam.runner._run_dynamic_or_hybrid') as mock_dynamic:

        async def _fake(**_kwargs: Any) -> MagicMock:
            return _fake_report()

        mock_dynamic.side_effect = _fake
        with pytest.raises(RuntimeError, match='complete boom'):
            await red_team('agent:test', mode='dynamic', hooks=BadComplete(), generate_executive_summary=False)

    [m] = list_manifests(get_runs_dir())
    assert m.status == 'error'
    assert m.error == 'complete boom'


@pytest.mark.asyncio
async def test_declined_run_marks_manifest_cancelled() -> None:
    """A run declined at the confirm gate (surfaced as CancelledError) is a
    distinct terminal status — 'cancelled', not 'error'."""
    with patch('evaluatorq.redteam.runner._run_dynamic_or_hybrid') as mock_dynamic:

        async def _declined(**_kwargs: Any) -> MagicMock:
            raise CancelledError('Execution cancelled by confirmation callback')

        mock_dynamic.side_effect = _declined
        with pytest.raises(CancelledError):
            await red_team('agent:test', mode='dynamic', generate_executive_summary=False)

    [m] = list_manifests(get_runs_dir())
    assert m.status == 'cancelled'
    assert m.error is None
    assert m.ended_at is not None


@pytest.mark.asyncio
async def test_sync_user_hook_warns_but_composite_does_not_misfire() -> None:
    """warn_if_sync_hooks runs per user child BEFORE composing, so a sync child
    still triggers the deprecation warning — and the async composite itself
    never trips it."""

    class SyncHook(DefaultHooks):
        # Sync override of the now-async on_confirm — intentional (this is what
        # trips warn_if_sync_hooks).
        def on_confirm(self, payload: Any) -> bool:  # pyright: ignore[reportIncompatibleMethodOverride]
            return True

        async def on_complete(self, report: Any, **_kw: Any) -> None:
            # No-op: the fake report is a MagicMock, so skip DefaultHooks'
            # summary formatting.
            return None

    with patch('evaluatorq.redteam.runner._run_dynamic_or_hybrid') as mock_dynamic:

        async def _fake(**_kwargs: Any) -> MagicMock:
            return _fake_report()

        mock_dynamic.side_effect = _fake

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            await red_team('agent:test', mode='dynamic', hooks=SyncHook(), generate_executive_summary=False)

    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    sync_warnings = [m for m in messages if 'sync hook' in m]
    assert any('SyncHook' in m for m in sync_warnings), sync_warnings
    # The async composite must never be reported as a sync-hook offender.
    assert not any('CompositePipelineHooks' in m for m in sync_warnings), sync_warnings
