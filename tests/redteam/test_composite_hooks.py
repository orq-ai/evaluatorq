"""Unit tests for the redteam multi-hooks composite + manifest stage hook.

Covers CompositePipelineHooks fan-out semantics (void methods + on_confirm
combine), the uniform run-all-then-reraise exception policy, and the
ManifestStageHooks that records stage transitions into a ManifestWriter
(manifest-first ordering + Dec2 per-target keying)."""

from __future__ import annotations

from typing import Any

import pytest

from evaluatorq.common.run_manifest import ManifestStatus, start_manifest
from evaluatorq.redteam.hooks import CompositePipelineHooks, ManifestStageHooks
from evaluatorq.redteam.runner import get_runs_dir


class _RecordingHook:
    """Async hook that records every call; on_confirm returns a fixed verdict."""

    def __init__(self, name: str, calls: list[Any], *, confirm: bool = True) -> None:
        self._name = name
        self._calls = calls
        self._confirm = confirm

    async def on_stage_start(self, stage: Any, meta: dict[str, Any]) -> None:
        self._calls.append((self._name, 'on_stage_start', stage))

    async def on_stage_end(self, stage: Any, meta: dict[str, Any]) -> None:
        self._calls.append((self._name, 'on_stage_end', stage))

    async def on_confirm(self, payload: Any) -> bool:
        self._calls.append((self._name, 'on_confirm'))
        return self._confirm

    async def on_complete(self, report: Any, **_kw: Any) -> None:
        self._calls.append((self._name, 'on_complete'))


class _RaisingHook:
    """Async hook whose every method raises after recording that it ran."""

    def __init__(self, name: str, calls: list[Any], exc: BaseException) -> None:
        self._name = name
        self._calls = calls
        self._exc = exc

    async def on_stage_start(self, stage: Any, meta: dict[str, Any]) -> None:
        self._calls.append((self._name, 'on_stage_start'))
        raise self._exc

    async def on_stage_end(self, stage: Any, meta: dict[str, Any]) -> None:
        self._calls.append((self._name, 'on_stage_end'))
        raise self._exc

    async def on_confirm(self, payload: Any) -> bool:
        self._calls.append((self._name, 'on_confirm'))
        raise self._exc

    async def on_complete(self, report: Any, **_kw: Any) -> None:
        self._calls.append((self._name, 'on_complete'))
        raise self._exc


# ---------------------------------------------------------------------------
# CompositePipelineHooks — void fan-out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_void_fan_out_calls_every_child() -> None:
    calls: list[Any] = []
    composite = CompositePipelineHooks([_RecordingHook('a', calls), _RecordingHook('b', calls)])
    await composite.on_stage_start('context_retrieval', {})
    assert calls == [('a', 'on_stage_start', 'context_retrieval'), ('b', 'on_stage_start', 'context_retrieval')]


@pytest.mark.asyncio
async def test_void_fan_out_raising_child_still_runs_later_child() -> None:
    """A child that raises must not prevent a later child from running; the first
    exception is re-raised only after the whole loop has run."""
    calls: list[Any] = []
    boom = RuntimeError('boom')
    composite = CompositePipelineHooks([
        _RaisingHook('a', calls, boom),
        _RecordingHook('b', calls),
    ])
    with pytest.raises(RuntimeError, match='boom') as excinfo:
        await composite.on_stage_start('context_retrieval', {})
    assert excinfo.value is boom
    # 'b' ran even though 'a' raised first.
    assert ('a', 'on_stage_start') in calls
    assert ('b', 'on_stage_start', 'context_retrieval') in calls


@pytest.mark.asyncio
async def test_void_fan_out_reraises_first_exception() -> None:
    calls: list[Any] = []
    first = RuntimeError('first')
    second = RuntimeError('second')
    composite = CompositePipelineHooks([
        _RaisingHook('a', calls, first),
        _RaisingHook('b', calls, second),
    ])
    with pytest.raises(RuntimeError) as excinfo:
        await composite.on_stage_end('report_generation', {})
    assert excinfo.value is first
    assert ('a', 'on_stage_end') in calls
    assert ('b', 'on_stage_end') in calls


# ---------------------------------------------------------------------------
# CompositePipelineHooks — on_confirm combine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_confirm_all_true_proceeds() -> None:
    calls: list[Any] = []
    composite = CompositePipelineHooks([
        _RecordingHook('a', calls, confirm=True),
        _RecordingHook('b', calls, confirm=True),
    ])
    assert await composite.on_confirm({}) is True
    assert ('a', 'on_confirm') in calls
    assert ('b', 'on_confirm') in calls


@pytest.mark.asyncio
async def test_on_confirm_any_false_vetoes() -> None:
    calls: list[Any] = []
    composite = CompositePipelineHooks([
        _RecordingHook('a', calls, confirm=True),
        _RecordingHook('b', calls, confirm=False),
    ])
    assert await composite.on_confirm({}) is False
    # Every child was consulted (run-all, not short-circuit).
    assert ('a', 'on_confirm') in calls
    assert ('b', 'on_confirm') in calls


@pytest.mark.asyncio
async def test_on_confirm_single_child_identity() -> None:
    """Backward compat: one hook behaves identically (all([x]) == bool(x))."""
    calls: list[Any] = []
    assert await CompositePipelineHooks([_RecordingHook('a', calls, confirm=False)]).on_confirm({}) is False
    assert await CompositePipelineHooks([_RecordingHook('a', calls, confirm=True)]).on_confirm({}) is True


@pytest.mark.asyncio
async def test_on_confirm_raising_child_still_runs_later_child() -> None:
    calls: list[Any] = []
    boom = RuntimeError('confirm boom')
    composite = CompositePipelineHooks([
        _RaisingHook('a', calls, boom),
        _RecordingHook('b', calls, confirm=True),
    ])
    with pytest.raises(RuntimeError, match='confirm boom') as excinfo:
        await composite.on_confirm({})
    assert excinfo.value is boom
    # Later child ran despite the earlier raise (same policy as void methods).
    assert ('b', 'on_confirm') in calls


# ---------------------------------------------------------------------------
# ManifestStageHooks — manifest-first ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_manifest_first_records_stage_before_user_hook_raises() -> None:
    """With the manifest hook registered first, a user hook raising in
    on_stage_start still leaves the stage durably recorded."""
    writer = start_manifest(run_id='r1', surface='redteam', run_name='t', runs_dir=get_runs_dir())
    calls: list[Any] = []
    composite = CompositePipelineHooks([
        ManifestStageHooks(writer),
        _RaisingHook('user', calls, RuntimeError('user boom')),
    ])
    with pytest.raises(RuntimeError, match='user boom'):
        await composite.on_stage_start('attack_execution', {})
    # Manifest ran first, so the stage is recorded even though the user hook blew up.
    assert [s.name for s in writer.manifest.stages] == ['attack_execution']
    assert writer.manifest.stages[0].status == ManifestStatus.RUNNING


@pytest.mark.asyncio
async def test_manifest_stage_hooks_on_confirm_never_vetoes() -> None:
    writer = start_manifest(run_id='r2', surface='redteam', run_name='t', runs_dir=get_runs_dir())
    assert await ManifestStageHooks(writer).on_confirm({}) is True


@pytest.mark.asyncio
async def test_manifest_stage_hooks_records_error_from_meta() -> None:
    writer = start_manifest(run_id='r3', surface='redteam', run_name='t', runs_dir=get_runs_dir())
    hook = ManifestStageHooks(writer)
    await hook.on_stage_start('attack_execution', {})
    await hook.on_stage_end('attack_execution', {'error': RuntimeError('stage failed')})
    [rec] = writer.manifest.stages
    assert rec.status == ManifestStatus.ERROR


# ---------------------------------------------------------------------------
# Dec2 — per-target stage keying (no cross-target corruption)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_target_stages_do_not_corrupt_each_other() -> None:
    """Two targets' stages, opened before either closes, must be tracked
    independently and closed against the right record."""
    writer = start_manifest(run_id='r4', surface='redteam', run_name='t', runs_dir=get_runs_dir())
    hook = ManifestStageHooks(writer)

    # Interleave two targets: open A, open B, close A, close B.
    await hook.on_stage_start('datapoint_generation', {'target': 'agent:a'})
    await hook.on_stage_start('datapoint_generation', {'target': 'agent:b'})
    await hook.on_stage_end('datapoint_generation', {'target': 'agent:a'})
    await hook.on_stage_end('datapoint_generation', {'target': 'agent:b', 'error': RuntimeError('b failed')})

    by_target = {s.target: s for s in writer.manifest.stages}
    assert set(by_target) == {'agent:a', 'agent:b'}
    # Each target's outcome is recorded against its own record — no cross-close.
    assert by_target['agent:a'].status == ManifestStatus.COMPLETED
    assert by_target['agent:b'].status == ManifestStatus.ERROR
    assert by_target['agent:a'].ended_at is not None
    assert by_target['agent:b'].ended_at is not None
