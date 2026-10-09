"""Snapshot helpers shared by the Insights Trace-tab source and route tests."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from evaluatorq.trace_finder.models import Snapshot

if TYPE_CHECKING:
    from pathlib import Path

    from evaluatorq.insights.models import InsightsRun
    from evaluatorq.trace_finder.models import TraceRecord


def write_snapshot(path: Path, records: list[TraceRecord]) -> str:
    path.write_bytes(Snapshot(traces=tuple(records)).model_dump_json().encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_run(run: InsightsRun, path: Path, digest: str | None) -> InsightsRun:
    population: dict[str, Any] = {'mode': 'snapshot', 'snapshot_path': str(path)}
    if digest is not None:
        population['snapshot_sha256'] = digest
    return run.model_copy(update={'population': population})
