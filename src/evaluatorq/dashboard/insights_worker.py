"""Detached worker for Insights runs started from the dashboard."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path

from loguru import logger

from evaluatorq.common.run_manifest import ManifestWriter
from evaluatorq.contracts import ManifestStatus, RunManifest
from evaluatorq.dashboard.insights_launch import (
    _MANIFEST_ENV,
    _SNAPSHOT_ENV,
    MAX_FINDER_EXPORT_BYTES,
    finder_export_reference_path,
    read_launch_payload,
    start_worker_heartbeat,
    validate_private_finder_reference,
    worker_state_path,
)
from evaluatorq.insights.models import InsightsPopulation
from evaluatorq.insights.pipeline import insights
from evaluatorq.trace_finder.export import RunExport


def _fail_running(path: Path, error: str) -> None:
    try:
        manifest = RunManifest.model_validate_json(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        logger.warning('Could not update Insights manifest {} after worker failure: {}', path, exc)
        return
    if manifest.status == ManifestStatus.RUNNING:
        ManifestWriter(manifest, path).fail(error, stage='setup')


def _cleanup_snapshot(path: Path) -> None:
    directory = path.parent
    try:
        info = directory.lstat()
        snapshot_info = path.lstat()
        expected_root = Path(tempfile.gettempdir()).resolve()
        if (
            path.name != 'finder-export.json'
            or directory.is_symlink()
            or not stat.S_ISREG(snapshot_info.st_mode)
            or not directory.name.startswith('evaluatorq-finder-snapshot-')
            or directory.resolve().parent != expected_root
            or (os.name != 'nt' and stat.S_IMODE(info.st_mode) != 0o700)
            or (os.name != 'nt' and stat.S_IMODE(snapshot_info.st_mode) != 0o600)
            or (hasattr(os, 'getuid') and info.st_uid != os.getuid())
            or (hasattr(os, 'getuid') and snapshot_info.st_uid != os.getuid())
            or {entry.name for entry in directory.iterdir()} != {path.name}
        ):
            logger.warning('Leaving untrusted Finder snapshot path in place: {}', path)
            return
        shutil.rmtree(directory)
    except OSError as exc:
        logger.warning('Could not remove Finder snapshot directory {}: {}', directory, exc)


def _snapshot_candidate_from_request() -> Path | None:
    """Recover a private Finder snapshot path when payload decoding fails.

    The separate environment value survives malformed request JSON. Treat both
    as untrusted: `_cleanup_snapshot` verifies the private temp directory,
    owner, permissions, and expected filename before deleting anything.
    """
    raw = os.environ.get('EVALUATORQ_INSIGHTS_LAUNCH_REQUEST')
    try:
        request = json.loads(raw) if raw is not None else None
    except (TypeError, ValueError):
        request = None
    if isinstance(request, dict):
        spec = request.get('spec')
        snapshot = request.get('finder_export_snapshot')
        if isinstance(spec, dict) and spec.get('source') == 'finder' and isinstance(snapshot, str):
            return Path(snapshot)
    snapshot = os.environ.get(_SNAPSHOT_ENV)
    return Path(snapshot) if snapshot else None


def _cleanup_finder_reference(runs_dir: Path, run_id: str) -> None:
    """Release a Finder export after its Insights worker reaches a terminal state."""
    try:
        reference = finder_export_reference_path(runs_dir, run_id)
        if reference.exists():
            validate_private_finder_reference(reference)
            reference.unlink(missing_ok=True)
    except (OSError, ValueError) as exc:
        logger.warning('Could not remove Finder export reference for run {}: {}', run_id, exc)


def _cleanup_finder_reference_for_manifest(manifest_path: Path) -> None:
    """Release a Finder lease when a launch payload could not be decoded."""
    try:
        if manifest_path.parent.name != '.manifests':
            return
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
        if manifest.surface.value != 'insights' or manifest_path.name != f'{manifest.run_id}.json':
            return
        _cleanup_finder_reference(manifest_path.parent.parent, manifest.run_id)
    except (OSError, ValueError) as exc:
        logger.warning('Could not recover Finder export reference from manifest {}: {}', manifest_path, exc)


def main() -> int:
    manifest_env = os.environ.get(_MANIFEST_ENV)
    heartbeat = start_worker_heartbeat(Path(manifest_env)) if manifest_env else None
    payload = None
    unvalidated_snapshot = _snapshot_candidate_from_request()
    try:
        payload = read_launch_payload()
        spec = payload.spec
        if spec.source == 'finder':
            snapshot = payload.finder_export_snapshot
            if snapshot is None:
                raise ValueError('Missing or oversized validated Finder export snapshot')
            with snapshot.open('rb') as snapshot_file:
                raw_snapshot = snapshot_file.read(MAX_FINDER_EXPORT_BYTES + 1)
            if len(raw_snapshot) > MAX_FINDER_EXPORT_BYTES:
                raise ValueError('Missing or oversized validated Finder export snapshot')
            validated_export = RunExport.model_validate_json(raw_snapshot)
            run = asyncio.run(
                insights(
                    InsightsPopulation.from_finder_export(snapshot, export=validated_export),
                    labels=spec.label_specs(),
                    dimensions=spec.dimension_names(),
                    parallelism=spec.parallelism,
                    run_name=payload.run_name,
                    runs_dir=payload.runs_dir,
                    _run_id=payload.run_id,
                    _finder_export_source=Path(spec.finder_export),
                    _finder_export_sha256=hashlib.sha256(raw_snapshot).hexdigest(),
                )
            )
        else:
            run = asyncio.run(
                insights(
                    spec.population(),
                    labels=spec.label_specs(),
                    dimensions=spec.dimension_names(),
                    parallelism=spec.parallelism,
                    run_name=payload.run_name,
                    runs_dir=payload.runs_dir,
                    _run_id=payload.run_id,
                )
            )
    except Exception as exc:  # noqa: BLE001 — record any failure before the pipeline owns its manifest
        logger.exception('Dashboard Insights worker failed')
        if payload is not None:
            _fail_running(payload.runs_dir / '.manifests' / f'{payload.run_id}.json', str(exc))
        else:
            manifest_path = os.environ.get(_MANIFEST_ENV)
            if manifest_path:
                _fail_running(Path(manifest_path), str(exc))
        return 1
    finally:
        if heartbeat is not None:
            heartbeat_stop, heartbeat_thread = heartbeat
            heartbeat_stop.set()
            heartbeat_thread.join()
        if payload is not None and payload.finder_export_snapshot is not None:
            _cleanup_snapshot(payload.finder_export_snapshot)
        elif unvalidated_snapshot is not None:
            _cleanup_snapshot(unvalidated_snapshot)
        if payload is not None and payload.spec.source == 'finder':
            _cleanup_finder_reference(payload.runs_dir, payload.run_id)
        elif payload is None and manifest_env:
            _cleanup_finder_reference_for_manifest(Path(manifest_env))
        state_run_id = payload.run_id if payload is not None else Path(manifest_env).stem if manifest_env else None
        if state_run_id is not None:
            try:
                state_runs_dir = payload.runs_dir if payload is not None else Path(manifest_env).parent.parent
                worker_state_path(state_runs_dir, state_run_id).unlink(missing_ok=True)
            except (OSError, ValueError) as exc:
                logger.warning('Could not remove Insights worker state for run {}: {}', state_run_id, exc)
    return 0 if run.status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
