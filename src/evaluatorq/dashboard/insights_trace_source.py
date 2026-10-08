"""Re-read one Insights trace's conversation from the source its run analysed."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.insights.models import reads_snapshot
from evaluatorq.insights.population import PopulationError, read_snapshot
from evaluatorq.trace_finder.models import FacetSelection, NumericFilters

if TYPE_CHECKING:
    from evaluatorq.insights.models import InsightsRun, TraceInsight
    from evaluatorq.trace_finder.models import TraceRecord
    from evaluatorq.trace_finder.orq_source import OrqTraceSource

# The pipeline's Finder-export reload queries the same margin around a trace's start timestamp.
_ANALYSED_TIME_MARGIN = timedelta(seconds=1)


def _unavailable(trace: TraceInsight, reason: str, cause: object = None) -> str:
    logger.warning('Insights trace {} conversation unavailable: {}', trace.trace_id, cause or reason)
    return reason


async def load_trace_record(
    run: InsightsRun, trace: TraceInsight, *, source: OrqTraceSource | None
) -> TraceRecord | str:
    """Return the trace's record from its source, or a user-facing reason it cannot be shown.

    An Orq run re-runs the pipeline's query for this one trace, with the run's filters and a window around the
    analysed timestamp, so the same span is selected. A snapshot run reads its saved file and ignores ``source``;
    the caller owns ``source``'s lifetime and has already logged why it is None.
    """
    if reads_snapshot(run.population):
        return await _from_snapshot(run, trace)
    if source is None:
        return 'Connect an Orq account in Settings to load this conversation.'
    failure = 'Could not load the conversation from Orq. Open it in Orq instead.'
    timestamp = trace.timestamp
    try:
        record = await source.load_trace(
            trace.trace_id,
            start=timestamp - _ANALYSED_TIME_MARGIN if timestamp else None,
            end=timestamp + _ANALYSED_TIME_MARGIN if timestamp else None,
            facets=FacetSelection.model_validate(run.population.get('facets') or {}),
            numeric=NumericFilters.model_validate(run.population.get('numeric') or {}),
        )
    except Exception as error:  # noqa: BLE001 — Orq and SDK failures vary; the tab must degrade visibly.
        return _unavailable(trace, failure, f'{type(error).__name__}: {error}')
    if record is None:
        return _unavailable(trace, 'Orq no longer returns a conversation for this trace.')
    return record


async def _from_snapshot(run: InsightsRun, trace: TraceInsight) -> TraceRecord | str:
    path = run.population.get('snapshot_path')
    if not path:
        return _unavailable(trace, 'This run does not record where its snapshot was read from.')
    try:
        snapshot, digest = await asyncio.to_thread(read_snapshot, Path(str(path)))
    except PopulationError as error:
        return _unavailable(
            trace, 'The snapshot file this run read is no longer available or could not be read.', error
        )
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
