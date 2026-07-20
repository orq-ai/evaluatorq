"""Run lifecycle manifests: track a run's stage + status while it executes.

Reports only land on disk when a run *completes* — a run that is still running
or that crashed leaves no artifact. The manifest fills that gap: a tiny record
written when a run starts (``status='running'``), patched as stages advance, and
finalised to ``'completed'`` or ``'error'``. It lives in a ``.manifests/``
sidecar inside the same runs dir the report is saved to, so it never pollutes
the existing ``*.json`` report globs (which are non-recursive).

Everything here is best-effort: a manifest write must never raise into — or slow
— the actual run. All disk ops swallow-and-log. A run that is hard-killed
(SIGKILL, power loss) leaves a stale ``running`` manifest behind.
# ponytail: no heartbeat/pid liveness check — a hard-killed run reads as
# "running" forever. Add a pid + mtime staleness check to the reader if that
# becomes a real problem.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path

Surface = Literal['sim', 'redteam']
Status = Literal['running', 'completed', 'error']

MANIFESTS_DIR_NAME = '.manifests'


class StageRecord(BaseModel):
    """One pipeline stage's own status + timing within a run."""

    name: str
    status: Status = 'running'
    started_at: datetime
    ended_at: datetime | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()


class RunManifest(BaseModel):
    """Lifecycle record for a single run. One file per run, keyed by run_id."""

    run_id: str
    surface: Surface
    run_name: str
    status: Status = 'running'
    stage: str | None = None  # name of the current / most-recent stage
    stages: list[StageRecord] = Field(default_factory=list)
    started_at: datetime
    updated_at: datetime
    ended_at: datetime | None = None  # set when the run reaches a terminal status
    error: str | None = None
    report_path: str | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()


def _manifests_dir(runs_dir: Path) -> Path:
    return runs_dir / MANIFESTS_DIR_NAME


class ManifestWriter:
    """Holds one manifest and flushes it to disk on each transition.

    Construct via :func:`start_manifest`. Every method is best-effort: a disk
    failure is logged, never raised, so manifest bookkeeping can't break a run.
    """

    def __init__(self, manifest: RunManifest, path: Path) -> None:
        self.manifest = manifest
        self.path = path

    def flush(self) -> None:
        self.manifest.updated_at = datetime.now(tz=timezone.utc)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(self.manifest.model_dump_json(indent=2), encoding='utf-8')
        except OSError as exc:
            logger.debug(f'Failed to write run manifest {self.path}: {exc}')

    def _open_stage(self, name: str | None = None) -> StageRecord | None:
        """Most-recent still-open stage record (optionally matching *name*)."""
        for rec in reversed(self.manifest.stages):
            if rec.ended_at is None and (name is None or rec.name == name):
                return rec
        return None

    def start_stage(self, name: str | None) -> None:
        """Open a new stage record. Leaves the run's overall status unchanged."""
        if name is None or self.manifest.status != 'running':
            return
        now = datetime.now(tz=timezone.utc)
        # Defensive: close any dangling open stage (a missing on_stage_end)
        # rather than leaving two stages 'running' at once.
        prev = self._open_stage()
        if prev is not None:
            prev.status = 'completed'
            prev.ended_at = now
        self.manifest.stages.append(StageRecord(name=str(name), status='running', started_at=now))
        self.manifest.stage = str(name)
        self.flush()

    def end_stage(self, name: str | None, *, status: Status = 'completed') -> None:
        """Close the matching open stage record with *status* + an end time."""
        if self.manifest.status != 'running':
            return
        rec = self._open_stage(str(name) if name is not None else None)
        if rec is None:
            return
        rec.status = status
        rec.ended_at = datetime.now(tz=timezone.utc)
        self.flush()

    def complete(self, report_path: str | Path | None = None) -> None:
        now = datetime.now(tz=timezone.utc)
        # Close any stage left open (e.g. a final stage with no on_stage_end).
        for rec in self.manifest.stages:
            if rec.ended_at is None:
                rec.status = 'completed'
                rec.ended_at = now
        self.manifest.status = 'completed'
        self.manifest.ended_at = now
        if report_path is not None:
            self.manifest.report_path = str(report_path)
        self.flush()

    def fail(self, error: str, stage: Any = None) -> None:
        now = datetime.now(tz=timezone.utc)
        # Normalize an enum stage to its value so it matches the record keys
        # opened by start_stage (which uses ``getattr(stage, 'value', stage)``).
        stage = getattr(stage, 'value', stage)
        # Mark the failing stage errored (prefer a named match, else the open
        # one); the run failed here, so this is the meaningful stage to flag.
        rec = self._open_stage(str(stage) if stage is not None else None) or self._open_stage()
        if rec is not None:
            rec.status = 'error'
            rec.ended_at = now
        elif stage is not None:
            self.manifest.stage = str(stage)
        self.manifest.status = 'error'
        self.manifest.error = error
        self.manifest.ended_at = now
        self.flush()


def start_manifest(*, run_id: str, surface: Surface, run_name: str, runs_dir: Path) -> ManifestWriter:
    """Create + persist a ``running`` manifest, returning its writer."""
    now = datetime.now(tz=timezone.utc)
    manifest = RunManifest(
        run_id=run_id,
        surface=surface,
        run_name=run_name,
        status='running',
        started_at=now,
        updated_at=now,
    )
    writer = ManifestWriter(manifest, _manifests_dir(runs_dir) / f'{run_id}.json')
    writer.flush()
    return writer


def list_manifests(runs_dir: Path) -> list[RunManifest]:
    """Read all manifests in *runs_dir*, newest first. Bad files are skipped."""
    mdir = _manifests_dir(runs_dir)
    if not mdir.is_dir():
        return []
    out: list[RunManifest] = []
    for p in mdir.glob('*.json'):
        try:
            out.append(RunManifest.model_validate_json(p.read_text(encoding='utf-8')))
        except (OSError, ValueError) as exc:  # noqa: PERF203 — tiny loop, best-effort read
            logger.debug(f'Skipping unreadable manifest {p}: {exc}')
    return sorted(out, key=lambda m: m.started_at, reverse=True)


def active_manifests(runs_dir: Path) -> list[RunManifest]:
    """Manifests for runs that are not completed (i.e. running or errored)."""
    return [m for m in list_manifests(runs_dir) if m.status != 'completed']


def format_active_lines(runs_dir: Path) -> list[str]:
    """Human-readable one-liners for running/errored runs, newest first.

    Empty when nothing is active. Completed runs are omitted — their report file
    already shows in the runs listing.
    """
    now = datetime.now(tz=timezone.utc)
    lines: list[str] = []
    for m in active_manifests(runs_dir):
        done = sum(1 for s in m.stages if s.status == 'completed')
        if m.status == 'running':
            elapsed = (now - m.started_at).total_seconds()
            stage = m.stage or 'starting'
            detail = f'stage {done + 1}: {stage} — {elapsed:.0f}s elapsed'
        else:  # error
            where = f' at {m.stage}' if m.stage else ''
            detail = f'error{where}: {m.error}' if m.error else f'error{where}'
        lines.append(f'  • {m.run_name} [{m.status}] — {detail}')
    return lines


class _ManifestHooks:
    """Wrap a hooks object, recording per-stage status + timing on the manifest.

    Delegates every other attribute to the wrapped hooks unchanged, so it drops
    in for any sim/redteam hooks implementation. This is the hook-integration
    seam: the ``on_stage_start``/``on_stage_end`` transitions the pipeline
    already emits open and close the manifest's per-stage records, without the
    runner having to know each stage.

    Both overrides return the wrapped call's result verbatim — a coroutine for
    async hooks, None for sync ones — preserving the wrapped hook's sync/async
    nature (callers use ``await_maybe``), so a sync hook that raises still
    raises synchronously.
    """

    def __init__(self, wrapped: Any, writer: ManifestWriter) -> None:
        self._wrapped = wrapped
        self._writer = writer

    def on_stage_start(self, stage: Any, meta: dict[str, Any]) -> Any:
        self._writer.start_stage(getattr(stage, 'value', stage))
        return self._wrapped.on_stage_start(stage, meta)

    def on_stage_end(self, stage: Any, meta: dict[str, Any]) -> Any:
        self._writer.end_stage(getattr(stage, 'value', stage))
        return self._wrapped.on_stage_end(stage, meta)

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes not defined above — forward verbatim.
        return getattr(self._wrapped, name)


def wrap_hooks(hooks: Any, writer: ManifestWriter | None) -> Any:
    """Return *hooks* wrapped for manifest stage updates, or unchanged if no writer."""
    if writer is None:
        return hooks
    return _ManifestHooks(hooks, writer)
