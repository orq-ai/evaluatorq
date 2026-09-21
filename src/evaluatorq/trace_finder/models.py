"""Immutable data contracts for trace populations and JEV classifications."""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003
from dataclasses import dataclass, field
from datetime import datetime  # noqa: TC003
from types import MappingProxyType
from typing import Annotated, Any, Literal, NoReturn

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from evaluatorq.common.judge import ClassifyQuestion  # noqa: TC001

FacetName = Literal['project', 'model', 'provider', 'status', 'product', 'trace_type', 'agent_name', 'tool_name']
FACET_NAMES: tuple[FacetName, ...] = (
    'project',
    'model',
    'provider',
    'status',
    'product',
    'trace_type',
    'agent_name',
    'tool_name',
)


class TraceRecord(BaseModel):
    """One trace captured from the Orq trace source."""

    model_config = ConfigDict(frozen=True, extra='allow')

    schema_version: Literal[1]
    trace_id: str = Field(min_length=1)
    span_id: str = Field(min_length=1)
    timestamp: datetime
    messages: tuple[dict[str, Any], ...] = Field(min_length=1)
    project: str
    model: str
    provider: str
    status: str
    product: str
    trace_type: str
    agent_name: str = ''
    tool_names: tuple[str, ...] = ()
    total_tokens: int | None = Field(default=None, ge=0)
    duration_ms: int | None = Field(default=None, ge=0)
    capture_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator('timestamp')
    @classmethod
    def require_timezone_aware_timestamp(cls, value: datetime) -> datetime:
        """Reject timestamps that cannot be compared deterministically."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('timestamp must include a timezone offset')
        return value


class JevProjection(BaseModel):
    """The bounded, serialized trace state supplied to one JEV judgment."""

    model_config = ConfigDict(frozen=True)

    payload: dict[str, Any]
    serialized: str
    estimated_tokens: int = Field(ge=0, le=25_000)
    omitted_messages: int = Field(ge=0)
    omitted_bytes: int = Field(ge=0)


class TraceClassification(BaseModel):
    """One terminal EvaluatorQ classification, including its source result."""

    model_config = ConfigDict(frozen=True)

    trace_id: str
    span_id: str
    value: bool | float | str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    probabilities: dict[str, float] | None = None
    matched: bool = False
    error: str | None = None
    summary: str | None = None
    raw_result: dict[str, Any]


class Snapshot(BaseModel):
    """A fully validated, in-memory trace snapshot."""

    model_config = ConfigDict(frozen=True)

    traces: tuple[TraceRecord, ...]
    capture_metadata: dict[str, Any] = Field(default_factory=dict)


class FacetSelection(BaseModel):
    """Selected values for independent trace facets."""

    model_config = ConfigDict(frozen=True)

    project: frozenset[str] = frozenset()
    model: frozenset[str] = frozenset()
    provider: frozenset[str] = frozenset()
    status: frozenset[str] = frozenset()
    product: frozenset[str] = frozenset()
    trace_type: frozenset[str] = frozenset()
    agent_name: frozenset[str] = frozenset()
    tool_name: frozenset[str] = frozenset()


class FacetCatalogue(BaseModel):
    """The values available for each trace metadata facet."""

    model_config = ConfigDict(frozen=True)

    project: tuple[str, ...] = ()
    status: tuple[str, ...] = ()
    product: tuple[str, ...] = ()
    trace_type: tuple[str, ...] = ()
    model: tuple[str, ...] = ()
    provider: tuple[str, ...] = ()
    agent_name: tuple[str, ...] = ()
    tool_name: tuple[str, ...] = ()


class NumericFilters(BaseModel):
    """Inclusive numeric filters for trace tokens and duration."""

    model_config = ConfigDict(frozen=True)

    tokens_min: int | None = Field(default=None, ge=0)
    tokens_max: int | None = Field(default=None, ge=0)
    duration_ms_min: int | None = Field(default=None, ge=0)
    duration_ms_max: int | None = Field(default=None, ge=0)

    @model_validator(mode='after')
    def validate_ranges(self) -> Self:
        """Reject an inclusive range whose lower bound exceeds its upper bound."""

        for minimum_name, maximum_name in (
            ('tokens_min', 'tokens_max'),
            ('duration_ms_min', 'duration_ms_max'),
        ):
            minimum = getattr(self, minimum_name)
            maximum = getattr(self, maximum_name)
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError(f'{minimum_name} must be less than or equal to {maximum_name}')
        return self


class PopulationRequest(BaseModel):
    """Filters applied before the newest matching traces are selected."""

    model_config = ConfigDict(frozen=True)

    start: datetime | None = None
    end: datetime | None = None
    facets: FacetSelection = FacetSelection()
    numeric: NumericFilters = NumericFilters()
    limit: int = Field(default=500, ge=1, le=500)

    @field_validator('start', 'end')
    @classmethod
    def require_timezone_aware_bound(cls, value: datetime | None) -> datetime | None:
        """Reject bounds that cannot be compared with captured trace timestamps."""

        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError('time bounds must include a timezone offset')
        return value


class FacetOption(BaseModel):
    """One facet value together with its compatibility count and UI state."""

    model_config = ConfigDict(frozen=True)

    value: str
    count: int = Field(ge=0)
    selected: bool
    disabled: bool


class ValueSelection(BaseModel):
    """Include classifications whose value is one of the listed values."""

    model_config = ConfigDict(frozen=True)

    kind: Literal['values']
    values: tuple[StrictStr | StrictBool, ...] = Field(min_length=1)


class ThresholdSelection(BaseModel):
    """Include normalized score classifications on one side of a threshold."""

    model_config = ConfigDict(frozen=True)

    kind: Literal['threshold']
    operator: Literal['gte', 'lte']
    value: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


SelectionRule = Annotated[ValueSelection | ThresholdSelection, Field(discriminator='kind')]


class CompiledQuery(BaseModel):
    """A semantic JEV task and its validated inclusion rule."""

    model_config = ConfigDict(frozen=True)

    task: ClassifyQuestion
    selection: SelectionRule

    @model_validator(mode='after')
    def selection_matches_task(self) -> Self:
        validate_compiled_query(self.task, self.selection)
        return self


class LegendItem(BaseModel):
    """One stable visual explanation of a compiled classification result."""

    model_config = ConfigDict(frozen=True)

    label: str
    color: str
    kind: Literal['value', 'gradient', 'threshold'] = 'value'
    threshold: float | None = None


RunState = Literal['idle', 'compiling', 'awaiting_review', 'classifying', 'completed', 'failed', 'cancelled']


class RunRequest(BaseModel):
    """The query and local population captured when a run is submitted."""

    model_config = ConfigDict(frozen=True)

    query: str
    mode: Literal['immediate', 'review']
    population: PopulationRequest
    parallelism: int = Field(default=100, ge=1, le=200)


@dataclass(frozen=True)
class RunSnapshot:
    """A read-only progress view; nested model values are detached from the owner."""

    generation: int = 0
    state: RunState = 'idle'
    request: RunRequest | None = None
    compiled: CompiledQuery | None = None
    generated_filters: FacetSelection = field(default_factory=FacetSelection)
    generated_numeric: NumericFilters = field(default_factory=NumericFilters)
    trace_ids: tuple[str, ...] = ()
    traces: tuple[TraceRecord, ...] = ()
    results: Mapping[str, TraceClassification] = field(default_factory=lambda: MappingProxyType({}))
    projections: Mapping[str, JevProjection] = field(default_factory=lambda: MappingProxyType({}))
    total: int = 0
    completed: int = 0
    failed: int = 0
    matched: int = 0
    active: int = 0
    queued: int = 0
    percent: float = 0.0
    elapsed: float = 0.0
    rate: float = 0.0
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'results', MappingProxyType(dict(self.results)))
        object.__setattr__(self, 'projections', MappingProxyType(dict(self.projections)))


@dataclass(frozen=True)
class TraceDetail:
    """Detached source trace and its inspectable projection and terminal result."""

    trace: TraceRecord
    projection: JevProjection | None
    classification: TraceClassification | None
    compiled: CompiledQuery | None = None


def validate_compiled_query(task: ClassifyQuestion, selection: SelectionRule) -> None:
    """Reject match rules that cannot be evaluated against their JEV task."""

    if task.state != {}:
        _raise_compiled_query_error(
            ('task', 'state'),
            task.state,
            'compiled JEV task state must be exactly {}',
        )

    if task.kind == 'choice':
        criteria = task.criteria
        if not isinstance(criteria, dict):
            _raise_compiled_query_error(('task', 'criteria'), criteria, 'choice criteria must be a label dictionary')
        labels = tuple(criteria)
        if not 2 <= len(labels) <= 5 or len(set(labels)) != len(labels):
            _raise_compiled_query_error(
                ('task', 'criteria'), criteria, 'choice criteria must contain two to five unique labels'
            )
        if isinstance(selection, ThresholdSelection):
            _raise_compiled_query_error(('selection',), selection, 'choice tasks require a values selection')
        invalid_values = [value for value in selection.values if type(value) is not str or value not in criteria]
        if invalid_values:
            _raise_compiled_query_error(
                ('selection', 'values'),
                selection.values,
                'choice selection values must be labels from the task criteria',
            )
        return

    if task.kind == 'noul':
        if isinstance(selection, ThresholdSelection):
            _raise_compiled_query_error(('selection',), selection, 'noul tasks require a values selection')
        if any(type(value) is not bool for value in selection.values):
            _raise_compiled_query_error(
                ('selection', 'values'), selection.values, 'noul selection values must be booleans'
            )
        return

    if isinstance(selection, ValueSelection):
        _raise_compiled_query_error(('selection',), selection, 'score tasks require a threshold selection')


def _raise_compiled_query_error(location: tuple[str, ...], input_value: object, message: str) -> NoReturn:
    """Raise a cross-field validation error at the affected public field."""

    raise ValidationError.from_exception_data(
        CompiledQuery.__name__,
        [
            {
                'type': 'value_error',
                'loc': location,
                'input': input_value,
                'ctx': {'error': ValueError(message)},
            }
        ],
    )
