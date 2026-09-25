"""JSON run persistence for trace insights."""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
from datetime import timezone
from pathlib import Path

from loguru import logger

from evaluatorq.common.run_store_dir import get_store_dir
from evaluatorq.insights.models import InsightsRun


def get_insights_runs_dir() -> Path:
    """Return the Insights run directory, honoring ``EVALUATORQ_DIR``."""
    return get_store_dir('insights-runs')


def _slug(value: str) -> str:
    return re.sub(r'-+', '-', re.sub(r'[^a-z0-9]+', '-', value.lower())).strip('-')[:64] or 'insights'


def save_run(run: InsightsRun, runs_dir: Path | None = None) -> Path:
    """Atomically save a run JSON and return its path."""
    directory = runs_dir or get_insights_runs_dir()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = run.created_at.astimezone(timezone.utc).strftime('%Y%m%d-%H%M%S')
    base_name = f'insights_{stamp}_{_slug(run.run_name)}'
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=directory, prefix=f'.{base_name}.', suffix='.tmp', delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(run.model_dump_json(indent=2))
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        suffix = 1
        while True:
            name = base_name if suffix == 1 else f'{base_name}-{suffix}'
            path = directory / f'{name}.json'
            try:
                # A hard link creates the final name atomically only when it is
                # unused, so simultaneous same-second runs cannot replace one
                # another. The temp file is in the same directory/filesystem.
                os.link(temporary, path)
                break
            except FileExistsError:
                suffix += 1
        with contextlib.suppress(OSError):
            temporary.unlink()
    except Exception:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)
        raise
    return path


def load_run(path: Path) -> InsightsRun:
    """Read and validate an Insights run JSON."""
    return InsightsRun.model_validate_json(path.read_text(encoding='utf-8'))


def list_run_paths(runs_dir: Path | None = None) -> list[Path]:
    """Return Insights run paths newest first, without reading or validating their JSON."""
    directory = runs_dir or get_insights_runs_dir()
    try:
        paths = list(directory.glob('insights_*.json'))
    except OSError as exc:
        logger.warning('Could not list Insights run files in {}: {}', directory, exc)
        return []

    def mtime(path: Path) -> int:
        try:
            return path.stat().st_mtime_ns
        except OSError as exc:
            logger.warning('Could not stat Insights run file {}: {}', path, exc)
            return 0

    return sorted(paths, key=mtime, reverse=True)


def list_runs(runs_dir: Path | None = None) -> list[tuple[Path, InsightsRun | str]]:
    """Load run files newest first, retaining corrupt files as visible errors."""
    result: list[tuple[Path, InsightsRun | str]] = []
    for path in list_run_paths(runs_dir):
        try:
            result.append((path, load_run(path)))
        except (OSError, ValueError) as exc:  # noqa: PERF203 - preserve each corrupt file in the listing
            message = f'{type(exc).__name__}: {exc}'
            logger.warning('Unreadable Insights run {}: {}', path, message)
            result.append((path, message))
    return result
