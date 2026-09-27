"""Conversation-free JSON exports for trace-finder runs."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictStr

from .models import (
    FACET_NAMES,
    CompiledQuery,
    FacetSelection,
    NumericFilters,
    RunSnapshot,
    TraceClassification,
    TraceRecord,
    ValueSelection,
)


class ExportTask(BaseModel):
    """Allow-listed, state-free portion of a compiled classifier task."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    kind: Literal['noul', 'choice', 'score']
    instructions: str
    criteria: dict[str, str | None] | list[str] | None = None
    state: dict[str, Any]
    noul_threshold: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class ExportValuesSelection(BaseModel):
    """Allow-listed value selection from a compiled query."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    kind: Literal['values']
    values: tuple[StrictStr | StrictBool, ...] = Field(min_length=1)


class ExportThresholdSelection(BaseModel):
    """Allow-listed threshold selection from a compiled query."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    kind: Literal['threshold']
    operator: Literal['gte', 'lte']
    value: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


ExportSelection = Annotated[ExportValuesSelection | ExportThresholdSelection, Field(discriminator='kind')]


class ExportFilters(BaseModel):
    """Sorted categorical values used for the population."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    project: tuple[str, ...] = ()
    model: tuple[str, ...] = ()
    provider: tuple[str, ...] = ()
    status: tuple[str, ...] = ()
    product: tuple[str, ...] = ()
    trace_type: tuple[str, ...] = ()
    agent_name: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()


class ExportNumericFilters(BaseModel):
    """Allow-listed numeric bounds used for the population."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    tokens_min: int | None = Field(default=None, ge=0)
    tokens_max: int | None = Field(default=None, ge=0)
    duration_ms_min: int | None = Field(default=None, ge=0)
    duration_ms_max: int | None = Field(default=None, ge=0)


class ExportTimes(BaseModel):
    """Run timestamps and elapsed classification timing."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    elapsed: float
    rate: float


class ExportCounts(BaseModel):
    """Aggregate and progress counts captured by the run."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    total: int
    completed: int
    failed: int
    matched: int
    active: int
    queued: int
    percent: float


class ExportTrace(BaseModel):
    """One selected trace's non-content classification metadata."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    trace_id: str
    span_id: str
    timestamp: datetime
    agent_name: str | None = None
    tool_names: tuple[str, ...] = ()
    value: StrictBool | StrictFloat | StrictStr | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    probabilities: dict[str, float] | None = None
    matched: bool
    error: str | None = None


class RunExport(BaseModel):
    """Explicit allow-list for run provenance and classifications."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    schema_version: Literal[1] = 1
    query: str
    task: ExportTask
    selection: ExportSelection
    generated_filters: ExportFilters
    filters: ExportFilters
    generated_numeric: ExportNumericFilters
    numeric: ExportNumericFilters
    start: datetime | None = None
    end: datetime | None = None
    limit: int
    parallelism: int
    times: ExportTimes
    counts: ExportCounts
    traces: tuple[ExportTrace, ...]
    matched_trace_ids: list[str]


def build_export(run: RunSnapshot, *, matched_only: bool = False) -> RunExport:
    """Build a metadata-only export, optionally retaining only matched traces."""

    if run.request is None or run.compiled is None:
        raise ValueError('run has no request and compiled task to export')

    source_by_id = {trace.trace_id: trace for trace in run.traces}
    selected: list[TraceRecord] = []
    for trace_id in run.trace_ids:
        trace = source_by_id.get(trace_id)
        if trace is None:
            raise ValueError(f'selected trace {trace_id!r} is missing from the run snapshot')
        selected.append(trace)
    selected.sort(key=lambda trace: trace.timestamp, reverse=True)
    traces = tuple(_export_trace(trace, run.results.get(trace.trace_id)) for trace in selected)
    if matched_only:
        traces = tuple(trace for trace in traces if trace.matched and trace.error is None)

    return RunExport(
        query=run.request.query,
        task=_export_task(run.compiled),
        selection=_export_selection(run.compiled),
        generated_filters=_export_filters(run.generated_filters),
        filters=_export_filters(run.request.population.facets),
        generated_numeric=_export_numeric(run.generated_numeric),
        numeric=_export_numeric(run.request.population.numeric),
        start=run.request.population.start,
        end=run.request.population.end,
        limit=run.request.population.limit,
        parallelism=run.request.parallelism,
        times=ExportTimes(
            created_at=run.created_at,
            started_at=run.started_at,
            finished_at=run.finished_at,
            elapsed=run.elapsed,
            rate=run.rate,
        ),
        counts=ExportCounts(
            total=run.total,
            completed=run.completed,
            failed=run.failed,
            matched=run.matched,
            active=run.active,
            queued=run.queued,
            percent=run.percent,
        ),
        traces=traces,
        matched_trace_ids=[trace.trace_id for trace in traces if trace.matched and trace.error is None],
    )


def export_json(run: RunSnapshot, *, matched_only: bool = False) -> str:
    """Return one pretty-printed run, optionally with only matched trace rows."""

    return build_export(run, matched_only=matched_only).model_dump_json(indent=2)


def _export_task(compiled: CompiledQuery) -> ExportTask:
    task = compiled.task
    return ExportTask(
        kind=task.kind,
        instructions=task.instructions,
        criteria=task.criteria.copy() if isinstance(task.criteria, dict | list) else None,
        state={},
        noul_threshold=task.noul_threshold,
    )


def _export_selection(compiled: CompiledQuery) -> ExportSelection:
    selection = compiled.selection
    if isinstance(selection, ValueSelection):
        return ExportValuesSelection(kind='values', values=tuple(selection.values))
    return ExportThresholdSelection(kind='threshold', operator=selection.operator, value=selection.value)


def _export_filters(facets: FacetSelection) -> ExportFilters:
    return ExportFilters(**{
        'agent_name': tuple(sorted(facets.agent_name)),
        'tool_names': tuple(sorted(facets.tool_name)),
        **{
            name: tuple(sorted(getattr(facets, name)))
            for name in FACET_NAMES
            if name not in {'agent_name', 'tool_name'}
        },
    })


def _export_numeric(numeric: NumericFilters) -> ExportNumericFilters:
    return ExportNumericFilters(**numeric.model_dump())


def _export_trace(trace: TraceRecord, classification: TraceClassification | None) -> ExportTrace:
    if classification is None:
        return ExportTrace(
            trace_id=trace.trace_id,
            span_id=trace.span_id,
            timestamp=trace.timestamp,
            agent_name=trace.agent_name or None,
            tool_names=trace.tool_names,
            matched=False,
            error='classification not completed',
        )
    return ExportTrace(
        trace_id=trace.trace_id,
        span_id=trace.span_id,
        timestamp=trace.timestamp,
        agent_name=trace.agent_name or None,
        tool_names=trace.tool_names,
        value=classification.value,
        confidence=classification.confidence,
        probabilities=dict(classification.probabilities) if classification.probabilities is not None else None,
        matched=classification.matched,
        error=classification.error,
    )
