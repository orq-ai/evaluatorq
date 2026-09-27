"""Detached worker for Insights runs started from the dashboard."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.common.run_manifest import ManifestWriter
from evaluatorq.contracts import ManifestStatus, RunManifest
from evaluatorq.dashboard.insights_launch import read_launch_payload
from evaluatorq.insights.pipeline import insights

if TYPE_CHECKING:
    from pathlib import Path


def _fail_running(path: Path, error: str) -> None:
    try:
        manifest = RunManifest.model_validate_json(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        logger.warning('Could not update Insights manifest {} after worker failure: {}', path, exc)
        return
    if manifest.status == ManifestStatus.RUNNING:
        ManifestWriter(manifest, path).fail(error, stage='setup')


def main() -> int:
    payload = read_launch_payload()
    spec = payload.spec
    try:
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
        _fail_running(payload.runs_dir / '.manifests' / f'{payload.run_id}.json', str(exc))
        return 1
    return 0 if run.status == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
