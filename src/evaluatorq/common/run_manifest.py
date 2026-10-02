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
import sys
import threading
import time
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
    from collections.abc import Callable, Iterator

MANIFESTS_DIR_NAME = '.manifests'
# Stage artifacts (save='detail') sit beside reports but are not runs.
_ARTIFACT_PREFIXES = ('01_', '02_', '03_')


def _manifests_dir(runs_dir: Path) -> Path:
    return runs_dir / MANIFESTS_DIR_NAME


@contextlib.contextmanager
def _manifest_lock(path: Path) -> Iterator[None]:
    """Hold a cross-process advisory lock on ``<path>.lock`` for one read-modify-write.

    Several processes can write one run's manifest (the run itself, the dashboard launcher,
    the insights worker's failure path), and an unguarded read-modify-write loses whichever
    update lands first. The lock lives in a sibling file because the manifest is replaced
    atomically, which would orphan a lock held on the manifest's own inode. The lock file is
    never deleted, for the same reason. A lock that cannot be taken is logged and skipped:
    manifest bookkeeping must never break a run.
    """
    lock_path = path.with_name(f'{path.name}.lock')
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open('a+b')
    except OSError as exc:
        logger.warning('Could not open manifest lock {}: {}; writing without it', lock_path, exc)
        yield
        return
    locked = False
    try:
        try:
            if sys.platform == 'win32':
                import msvcrt

                while not locked:
                    try:
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                        locked = True
                    except OSError:  # noqa: PERF203 — LK_LOCK gives up after ~10s; keep waiting
                        continue
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                locked = True
        except OSError as exc:
            logger.warning('Could not lock manifest {}: {}; writing without it', path, exc)
        yield
    finally:
        # Closing the handle releases the lock on both platforms.
        handle.close()


def _write_atomic(manifest: RunManifest, path: Path) -> bool:
    """Write *manifest* to *path* via temp file + atomic rename; False on disk failure."""
    manifest.updated_at = datetime.now(tz=timezone.utc)
    # Write to a temp file in the same dir then atomically rename over the
    # target, so a SIGKILL mid-write can never leave truncated JSON that the
    # reader would silently drop (the whole point of the manifest is to
    # surface crashed runs). Best-effort: on failure clean up the temp file.
    tmp = path.with_name(f'{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(manifest.model_dump_json(indent=2), encoding='utf-8')
        os.replace(tmp, path)  # noqa: PTH105 — atomic rename is the whole point
        return True
    except OSError as exc:
        logger.warning('Failed to write run manifest {}: {}; later transitions will retry', path, exc)
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        return False


def _read_manifest(path: Path) -> RunManifest | None:
    try:
        return RunManifest.model_validate_json(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        logger.warning('Could not read run manifest {}: {}', path, exc)
        return None


def update_manifest(path: Path, mutate: Callable[[RunManifest], None]) -> RunManifest | None:
    """Read, mutate and atomically rewrite the manifest at *path* under the cross-process lock.

    Use this instead of read-then-`ManifestWriter` anywhere a second process may be writing the
    same run: the read happens inside the lock, so a concurrent update to another field is
    kept rather than overwritten. Returns the written manifest, or ``None`` when the file is
    missing, unreadable or the write failed (each logged).
    """
    with _manifest_lock(path):
        manifest = _read_manifest(path)
        if manifest is None:
            return None
        mutate(manifest)
        return manifest if _write_atomic(manifest, path) else None


def fail_if_running(path: Path, error: str, stage: Any = None) -> bool:
    """Mark the manifest at *path* errored unless another writer already finished it.

    The status check and the write share one lock, so a run that completes between the
    caller's own read and this write is not clobbered. Returns True when this call failed it.
    """
    failed = False

    def apply(manifest: RunManifest) -> None:
        nonlocal failed
        if manifest.status == ManifestStatus.RUNNING:
            ManifestWriter(manifest, path)._apply_fail(error, stage)  # noqa: SLF001
            failed = True

    return update_manifest(path, apply) is not None and failed


class ManifestWriter:
    """Holds one manifest and flushes it to disk on each transition.

    Construct via `start_manifest`. Every method is best-effort: a disk
    failure is logged, never raised, so manifest bookkeeping can't break a run.
    """

    def __init__(self, manifest: RunManifest, path: Path, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.manifest = manifest
        self.path = path
        self._persisted_manifest = manifest.model_copy(deep=True)
        self._clock = clock
        self._last_progress_flush: dict[int, float] = {}

    def flush(self) -> bool:
        """Persist the latest in-memory state under the manifest lock; False when the disk view lags.

        Changes made since this writer's last flush are merged onto the current disk view.
        This keeps independent updates from a long-lived writer from replacing newer fields.
        The first terminal status on disk also wins, including when this writer has already
        changed its local status before flushing.
        """
        with _manifest_lock(self.path):
            on_disk = _read_manifest(self.path)
            if on_disk is not None:
                if (
                    self._persisted_manifest.status == ManifestStatus.RUNNING
                    and on_disk.status != ManifestStatus.RUNNING
                ):
                    logger.warning(
                        'Run manifest {} was already finished as {} by another writer; keeping it',
                        self.path,
                        on_disk.status.value,
                    )
                    self.manifest = on_disk
                    self._persisted_manifest = on_disk.model_copy(deep=True)
                    return True

                base = self._persisted_manifest.model_dump(mode='python')
                local = self.manifest.model_dump(mode='python')
                merged = on_disk.model_copy(deep=True)
                for field in RunManifest.model_fields:
                    if local[field] != base[field]:
                        setattr(merged, field, getattr(self.manifest, field))
                self.manifest = merged

            if not _write_atomic(self.manifest, self.path):
                return False
            self._persisted_manifest = self.manifest.model_copy(deep=True)
            return True

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

    def stage_progress(self, stage: Any, completed: int, total: int, *, target: str | None = None) -> None:
        """Record progress with one-second write throttling.

        A failed write keeps the latest counts in memory and does not advance
        the throttle clock, so the next callback or stage transition retries
        them. The disk view can lag until that write succeeds.
        """
        if (
            isinstance(completed, bool)
            or not isinstance(completed, int)
            or isinstance(total, bool)
            or not isinstance(total, int)
            or completed < 0
            or total < 0
            or completed > total
        ):
            logger.warning(
                'Ignoring invalid progress for stage {}: completed={!r}, total={!r}', stage, completed, total
            )
            return
        name = getattr(stage, 'value', stage)
        rec = self._open_stage(str(name), target)
        if rec is None or self.manifest.status != ManifestStatus.RUNNING:
            return
        rec.completed, rec.total = completed, total
        now = self._clock()
        record_key = id(rec)
        last_flush = self._last_progress_flush.get(record_key)
        if completed >= total or last_flush is None or now - last_flush >= 1.0:
            if self.flush():
                self._last_progress_flush[record_key] = now
            else:
                # Forget the old success so the next callback retries inside the throttle window.
                self._last_progress_flush.pop(record_key, None)

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
        self._apply_fail(error, stage)
        self.flush()

    def _apply_fail(self, error: str, stage: Any = None) -> None:
        """Mutate the in-memory manifest into the errored state without touching disk."""
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


def start_manifest(
    *,
    run_id: str,
    surface: Surface | str,
    run_name: str,
    runs_dir: Path,
    planned_stages: list[str] | None = None,
    stage_labels: dict[str, str] | None = None,
    parallelism: int | None = None,
) -> ManifestWriter:
    """Create + persist a ``running`` manifest, returning its writer."""
    now = datetime.now(tz=timezone.utc)
    manifest = RunManifest(
        run_id=run_id,
        surface=Surface(surface),
        run_name=run_name,
        status=ManifestStatus.RUNNING,
        planned_stages=planned_stages or [],
        stage_labels=stage_labels or {},
        parallelism=parallelism,
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
      `summary_is_current`) is demoted to this path rather than listed
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
