"""Re-read one Insights trace's conversation from the source its run analysed."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.dashboard.insights_uploads import is_uploaded_source
from evaluatorq.insights.models import population_source, reads_orq
from evaluatorq.insights.population import TARGETED_RELOAD_MARGIN, PopulationError, read_snapshot
from evaluatorq.insights.store import get_insights_runs_dir
from evaluatorq.trace_finder.models import FacetSelection, NumericFilters
from evaluatorq.trace_finder.orq_source import OrqSpanMismatchError

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager

    from evaluatorq.insights.models import InsightsRun, TraceInsight
    from evaluatorq.trace_finder.models import TraceRecord
    from evaluatorq.trace_finder.orq_source import OrqTraceSource

SOURCE_NOT_REREADABLE = "This run's trace source cannot be re-read."
SNAPSHOT_PATH_MISSING = 'This run does not record where its snapshot was read from.'
SNAPSHOT_UNREADABLE = 'The snapshot file this run read is no longer available or could not be read.'
UPLOAD_DELETED = (
    'The dashboard deletes uploaded snapshot files after the run, so this conversation is no longer available.'
)
ORQ_SPAN_CHANGED = 'Orq now selects a different span for this trace than the one this run analysed.'


def _gone_snapshot_reason(path: Path) -> str:
    return UPLOAD_DELETED if is_uploaded_source(get_insights_runs_dir(), path) else SNAPSHOT_UNREADABLE


def conversation_unavailable_reason(run: InsightsRun) -> str | None:
    """Say why the Trace tab cannot re-read this run's conversations, or None when it can; a snapshot costs one stat."""
    source = population_source(run.population)
    if source == 'snapshot':
        path = run.population.get('snapshot_path')
        if not path:
            return SNAPSHOT_PATH_MISSING
        return None if Path(str(path)).is_file() else _gone_snapshot_reason(Path(str(path)))
    return None if reads_orq(run.population) else SOURCE_NOT_REREADABLE


def conversation_rereadable(run: InsightsRun) -> bool:
    """Whether the Trace tab can re-read this run's conversations: from Orq, or from a snapshot file that still exists."""
    return conversation_unavailable_reason(run) is None


def _unavailable(trace: TraceInsight, reason: str, cause: object = None) -> str:
    logger.warning('Insights trace {} conversation unavailable: {}', trace.trace_id, cause or reason)
    return reason


def source_not_rereadable(run: InsightsRun, trace: TraceInsight) -> str:
    """Log and return the reason a run whose population is neither a snapshot nor an Orq read cannot be re-read."""
    return _unavailable(trace, SOURCE_NOT_REREADABLE, f'population mode {run.population.get("mode")!r}')


async def load_orq_record(
    run: InsightsRun,
    trace: TraceInsight,
    *,
    open_source: Callable[[], AbstractAsyncContextManager[OrqTraceSource | str]],
) -> TraceRecord | str:
    """Re-run the pipeline's query for this one trace, or return a user-facing reason it cannot be shown.

    The query keeps the run's filters and a window around the analysed timestamp. The result is returned only when
    Orq selects the saved span; a changed selection becomes an unavailable reason. ``open_source`` is entered only
    once the trace has a timestamp; a string it yields is a reason it has logged.
    """
    timestamp = trace.timestamp
    if timestamp is None:
        return _unavailable(
            trace, 'This run did not record when the trace ran, so its conversation cannot be re-read from Orq.'
        )
    async with open_source() as source:
        if isinstance(source, str):
            return source
        try:
            record = await source.load_trace(
                trace.trace_id,
                expected_span_id=trace.span_id,
                start=timestamp - TARGETED_RELOAD_MARGIN,
                end=timestamp + TARGETED_RELOAD_MARGIN,
                facets=FacetSelection.model_validate(run.population.get('facets') or {}),
                numeric=NumericFilters.model_validate(run.population.get('numeric') or {}),
            )
        except OrqSpanMismatchError as error:
            return _unavailable(trace, ORQ_SPAN_CHANGED, error)
        except Exception as error:  # noqa: BLE001 — Orq and SDK failures vary; the tab must degrade visibly.
            return _unavailable(
                trace,
                'Could not load the conversation from Orq. Open it in Orq instead.',
                f'{type(error).__name__}: {error}',
            )
    if record is None:
        return _unavailable(
            trace,
            'Orq did not return this trace in the workspace used for this request. '
            'Check that the account has access to that workspace and that the trace still exists.',
        )
    return record


async def load_snapshot_record(run: InsightsRun, trace: TraceInsight) -> TraceRecord | str:
    """Read the trace's record from the snapshot file the run analysed, or return a user-facing reason it cannot."""
    return await asyncio.to_thread(_from_snapshot, run, trace)


def _from_snapshot(run: InsightsRun, trace: TraceInsight) -> TraceRecord | str:
    raw_path = run.population.get('snapshot_path')
    if not raw_path:
        return _unavailable(trace, SNAPSHOT_PATH_MISSING)
    path = Path(str(raw_path))
    try:
        snapshot, digest = read_snapshot(path)
    except PopulationError as error:
        return _unavailable(trace, SNAPSHOT_UNREADABLE if path.exists() else _gone_snapshot_reason(path), error)
    saved_digest = run.population.get('snapshot_sha256')
    if saved_digest and saved_digest != digest:
        return _unavailable(
            trace,
            'The snapshot file changed after this run, so it may not match the analysis.',
            f'{path} has sha256 {digest}, the run read {saved_digest}',
        )
    for record in snapshot.traces:
        if record.trace_id == trace.trace_id and record.span_id == trace.span_id:
            return record
    return _unavailable(trace, 'This trace is not in the snapshot file.', f'not in {path}')
