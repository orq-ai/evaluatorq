"""Detached worker for Insights runs started from the dashboard."""

from __future__ import annotations

import asyncio
import os
import shutil
import stat
import tempfile
from pathlib import Path

from loguru import logger

from evaluatorq.common.run_manifest import ManifestWriter
from evaluatorq.contracts import ManifestStatus, RunManifest
from evaluatorq.dashboard.insights_launch import _MANIFEST_ENV, MAX_FINDER_EXPORT_BYTES, read_launch_payload
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
        info = directory.stat()
        expected_root = Path(tempfile.gettempdir()).resolve()
        if (
            path.name != 'finder-export.json'
            or directory.is_symlink()
            or not directory.name.startswith('evaluatorq-finder-snapshot-')
            or directory.resolve().parent != expected_root
            or (os.name != 'nt' and stat.S_IMODE(info.st_mode) != 0o700)
            or (hasattr(os, 'getuid') and info.st_uid != os.getuid())
        ):
            logger.warning('Leaving untrusted Finder snapshot path in place: {}', path)
            return
        shutil.rmtree(directory)
    except OSError as exc:
        logger.warning('Could not remove Finder snapshot directory {}: {}', directory, exc)


def main() -> int:
    payload = None
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
            RunExport.model_validate_json(raw_snapshot)
            run = asyncio.run(
                insights(
                    InsightsPopulation.from_finder_export(snapshot),
                    labels=spec.label_specs(),
                    dimensions=spec.dimension_names(),
                    parallelism=spec.parallelism,
                    run_name=payload.run_name,
                    runs_dir=payload.runs_dir,
                    _run_id=payload.run_id,
                    _finder_export_source=Path(spec.finder_export),
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
        if payload is not None and payload.finder_export_snapshot is not None:
            _cleanup_snapshot(payload.finder_export_snapshot)
    return 0 if run.status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
