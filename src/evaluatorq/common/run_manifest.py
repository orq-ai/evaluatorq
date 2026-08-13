"""Run lifecycle manifests: track a run's stage + status while it executes.

Canonical run-state record for both surfaces — no sidecar status dict of your own.

Reports only land on disk when a run *completes* — a run that is still running
or that crashed leaves no artifact. The manifest fills that gap: a tiny record
written when a run starts (``status='running'``), patched as stages advance, and
finalised to a terminal ``'completed'``, ``'error'``, or ``'cancelled'``. It
lives in a ``.manifests/``
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

import contextlib
import operator
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.contracts import (
    RUN_SUMMARY_VERSION,
    ManifestStatus,
    RunManifest,
    RunSummary,
    StageRecord,
)
from evaluatorq.contracts import (
    ManifestSurface as Surface,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

MANIFESTS_DIR_NAME = '.manifests'
# Stage artifacts (save='detail') sit beside reports but are not runs.
_ARTIFACT_PREFIXES = ('01_', '02_', '03_')


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
        # Write to a temp file in the same dir then atomically rename over the
        # target, so a SIGKILL mid-write can never leave truncated JSON that the
        # reader would silently drop (the whole point of the manifest is to
        # surface crashed runs). Best-effort: on failure clean up the temp file.
        tmp = self.path.with_name(f'{self.path.name}.{os.getpid()}.tmp')
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(self.manifest.model_dump_json(indent=2), encoding='utf-8')
            os.replace(tmp, self.path)  # noqa: PTH105 — atomic rename is the whole point
        except OSError as exc:
            logger.debug(f'Failed to write run manifest {self.path}: {exc}')
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)

    def _open_stage(self, name: str | None = None, target: str | None = None) -> StageRecord | None:
        """Most-recent still-open stage record matching *name* and *target*.

        ``target`` is an exact part of the key: ``None`` matches only aggregate
        stages whose target is also ``None``. This prevents targetless events
        from closing one of several concurrent per-target stages.
        """
        for rec in reversed(self.manifest.stages):
            if rec.ended_at is None and (name is None or rec.name == name) and rec.target == target:
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

    def complete(self, report_path: str | Path | None = None, summary: RunSummary | None = None) -> None:
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
        # Compact headline stats so a run-list row can be built from the manifest
        # alone — no full-report read needed for completed runs. Stamped with the
        # shape's version here, the one place a summary is stored, so no writer
        # can persist an unversioned (or stale-shaped) one.
        if summary is not None:
            self.manifest.summary = summary
            self.manifest.summary_version = RUN_SUMMARY_VERSION
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
            # A genuinely unreadable/corrupt manifest is a run we can no longer
            # surface — make it visible (warning), not silent (debug).
            logger.warning(f'Skipping unreadable manifest {p}: {exc}')
    return sorted(out, key=lambda m: m.started_at, reverse=True)


def summary_is_current(manifest: RunManifest) -> bool:
    """Whether *manifest* carries a summary of the current shape.

    A stamp comparison, not a shape guess: the summary is usable only when it was
    written by a ``manifest_summary()`` of today's ``RUN_SUMMARY_VERSION``.
    Anything older — an unstamped sidecar from before versioning, or the thin
    ``{'total_results': N}`` an early dashboard backfill wrote — is not usable,
    so callers fall back to the full report (and the dashboard rewrites the
    sidecar on its next scan). Bumping ``RUN_SUMMARY_VERSION`` when a surface's
    summary shape changes re-reads every stored run exactly once.
    """
    return bool(manifest.summary) and manifest.summary_version == RUN_SUMMARY_VERSION


def iter_report_files(runs_dir: Path) -> Iterator[Path]:
    """Report files in *runs_dir*, sorted: non-recursive ``*.json`` minus stage artifacts.

    The one definition of "is a report" shared by every reader — the run listings
    here, the dashboard's scan and its store fingerprint. ``save='detail'`` writes
    ``01_``/``02_``/``03_`` stage artifacts next to reports; they are not runs, and
    a reader that globs them lists phantom rows (and invalidates caches) for files
    nothing renders.

    Yields:
        Each report file in *runs_dir*, in sorted path order. Nothing when the
        directory does not exist.
    """
    if not runs_dir.is_dir():
        return
    for p in sorted(runs_dir.glob('*.json')):
        if not p.name.startswith(_ARTIFACT_PREFIXES):
            yield p


def list_run_records(runs_dir: Path) -> list[tuple[RunManifest | None, Path | None]]:
    """Unified, manifest-first run listing for a runs dir, newest first.

    Returns ``(manifest, report_path)`` pairs:

    * A manifest-backed run yields ``(manifest, report_path_or_None)``. Completed
      runs carry a ``report_path``; in-flight (running/error/cancelled) runs have
      ``None`` — they render from the manifest's status/stage/summary alone.
    * A LEGACY report with no manifest yields ``(None, report_path)`` — the
      backwards-compatible path (read the full report for its stats). A completed
      manifest whose ``summary`` is too thin to build a row (see
      :func:`summary_is_current`) is demoted to this path rather than listed
      with blank columns.

    Reports already covered by a manifest's ``report_path`` are de-duplicated out
    of the legacy set, so a run is never listed twice. Sorted newest-first by
    manifest ``started_at`` (manifest rows) or file mtime (legacy rows).
    """
    records: list[tuple[RunManifest | None, Path | None, float]] = []
    covered: set[Path] = set()
    for m in list_manifests(runs_dir):
        report_path = Path(m.report_path) if m.report_path else None
        if m.status == ManifestStatus.COMPLETED and report_path is not None and not summary_is_current(m):
            # Thin sidecar: leave the report uncovered so the legacy full-report
            # row below carries the run instead of a half-blank manifest row.
            continue
        if report_path is not None:
            covered.add(report_path.resolve())
        records.append((m, report_path, m.started_at.timestamp()))

    if runs_dir.is_dir():
        for p in iter_report_files(runs_dir):
            try:
                if p.resolve() in covered:
                    continue
                mtime = p.stat().st_mtime
            except OSError as exc:
                # A legacy report we can't stat/resolve is a run we're dropping —
                # log it (warning) instead of swallowing it silently.
                logger.warning(f'Skipping unreadable legacy report {p}: {exc}')
                continue
            records.append((None, p, mtime))

    records.sort(key=operator.itemgetter(2), reverse=True)
    return [(m, p) for m, p, _ in records]
