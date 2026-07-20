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
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.contracts import (
    ManifestStatus,
    RunManifest,
    StageRecord,
)
from evaluatorq.contracts import (
    ManifestSurface as Surface,
)

if TYPE_CHECKING:
    from pathlib import Path

MANIFESTS_DIR_NAME = '.manifests'


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

    def _open_stage(self, name: str | None = None, target: str | None = None) -> StageRecord | None:
        """Most-recent still-open stage record matching *name* and *target*.

        A ``None`` filter matches any value, so per-target concurrent stages
        (Dec2) never close each other's records.
        """
        for rec in reversed(self.manifest.stages):
            if (
                rec.ended_at is None
                and (name is None or rec.name == name)
                and (target is None or rec.target == target)
            ):
                return rec
        return None

    def _close(self, rec: StageRecord, status: ManifestStatus) -> None:
        """Close a stage record: set its terminal status + end time."""
        rec.status = status
        rec.ended_at = datetime.now(tz=timezone.utc)

    def start_stage(self, stage: Any, *, target: str | None = None) -> None:
        """Open a new stage record. Leaves the run's overall status unchanged."""
        name = getattr(stage, 'value', stage)  # normalize enum → value once (R3)
        if name is None or self.manifest.status != ManifestStatus.RUNNING:
            return
        now = datetime.now(tz=timezone.utc)
        # Defensive: close any dangling open stage for this target (a missing
        # on_stage_end) rather than leaving two stages 'running' at once.
        prev = self._open_stage(target=target)
        if prev is not None:
            self._close(prev, ManifestStatus.COMPLETED)
        self.manifest.stages.append(
            StageRecord(name=str(name), target=target, status=ManifestStatus.RUNNING, started_at=now)
        )
        self.manifest.stage = str(name)
        self.flush()

    def end_stage(self, stage: Any, *, target: str | None = None, error: Any = None) -> None:
        """Close the matching open stage record. ``error`` → status ``error``."""
        name = getattr(stage, 'value', stage)  # normalize enum → value once (R3)
        if self.manifest.status != ManifestStatus.RUNNING:
            return
        rec = self._open_stage(str(name) if name is not None else None, target)
        if rec is None:
            return
        self._close(rec, ManifestStatus.ERROR if error else ManifestStatus.COMPLETED)
        self.flush()

    def complete(self, report_path: str | Path | None = None) -> None:
        if self.manifest.status != ManifestStatus.RUNNING:
            return  # terminal transitions are idempotent — first one wins
        now = datetime.now(tz=timezone.utc)
        # Close any stage left open (e.g. a final stage with no on_stage_end).
        for rec in self.manifest.stages:
            if rec.ended_at is None:
                rec.status = ManifestStatus.COMPLETED
                rec.ended_at = now
        self.manifest.status = ManifestStatus.COMPLETED
        self.manifest.ended_at = now
        if report_path is not None:
            self.manifest.report_path = str(report_path)
        self.flush()

    def cancel(self) -> None:
        """Terminate as ``cancelled`` (declined run). Stages are left untouched.

        A finished stage stays truthful — cancellation happens outside a stage,
        so nothing is relabeled (Dec1).
        """
        if self.manifest.status != ManifestStatus.RUNNING:
            return  # terminal transitions are idempotent — first one wins
        self.manifest.status = ManifestStatus.CANCELLED
        self.manifest.ended_at = datetime.now(tz=timezone.utc)
        self.flush()

    def fail(self, error: str, stage: Any = None) -> None:
        if self.manifest.status != ManifestStatus.RUNNING:
            return  # terminal transitions are idempotent — first one wins
        now = datetime.now(tz=timezone.utc)
        # Close every still-open stage as errored (R2). A stage that already
        # finished stays 'completed' — never relabel a succeeded stage (R1).
        any_open = False
        for rec in self.manifest.stages:
            if rec.ended_at is None:
                rec.status = ManifestStatus.ERROR
                rec.ended_at = now
                any_open = True
        # If nothing was open, only record where we failed — do NOT flip a
        # closed stage's status (R1 revised).
        if not any_open and stage is not None:
            self.manifest.stage = getattr(stage, 'value', stage)
        self.manifest.status = ManifestStatus.ERROR
        self.manifest.error = error
        self.manifest.ended_at = now
        self.flush()


def start_manifest(*, run_id: str, surface: Surface | str, run_name: str, runs_dir: Path) -> ManifestWriter:
    """Create + persist a ``running`` manifest, returning its writer."""
    now = datetime.now(tz=timezone.utc)
    manifest = RunManifest(
        run_id=run_id,
        surface=Surface(surface),
        run_name=run_name,
        status=ManifestStatus.RUNNING,
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
    """Manifests for runs that are not completed (running, errored, cancelled)."""
    return [m for m in list_manifests(runs_dir) if m.status != ManifestStatus.COMPLETED]


def format_active_lines(runs_dir: Path) -> list[str]:
    """Human-readable one-liners for non-completed runs, newest first.

    Empty when nothing is active. Completed runs are omitted — their report file
    already shows in the runs listing. Running, errored, and cancelled runs are
    all surfaced.
    """
    now = datetime.now(tz=timezone.utc)
    lines: list[str] = []
    for m in active_manifests(runs_dir):
        done = sum(1 for s in m.stages if s.status == ManifestStatus.COMPLETED)
        if m.status == ManifestStatus.RUNNING:
            elapsed = (now - m.started_at).total_seconds()
            stage = m.stage or 'starting'
            detail = f'stage {done + 1}: {stage} — {elapsed:.0f}s elapsed'
        elif m.status == ManifestStatus.CANCELLED:
            where = f' at {m.stage}' if m.stage else ''
            detail = f'cancelled{where}'
        else:  # error
            where = f' at {m.stage}' if m.stage else ''
            detail = f'error{where}: {m.error}' if m.error else f'error{where}'
        lines.append(f'  • {m.run_name} [{m.status}] — {detail}')
    return lines


def echo_active_runs(runs_dir: Path) -> None:
    """Print the 'Active runs' block (running/errored) for *runs_dir*, if any.

    Shared by the ``redteam`` / ``sim`` ``runs`` CLI commands so the two never
    drift. ``typer`` is imported lazily to keep this module — which the runners
    import at runtime — free of the CLI-only dependency.
    """
    import typer

    lines = format_active_lines(runs_dir)
    if not lines:
        return
    typer.echo('Active runs:')
    for line in lines:
        typer.echo(line)
    typer.echo('')
