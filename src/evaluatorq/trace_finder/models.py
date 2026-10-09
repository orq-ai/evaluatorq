"""Validated data contracts for trace populations and classifications."""

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

from evaluatorq.common.judge import ClassifyQuestion, ClassifyResponse  # noqa: TC001
from evaluatorq.trace_finder.settings import MAX_TRACE_INPUT_CHARS

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
NumericFacetName = Literal['tokens', 'duration_ms']
NUMERIC_FACET_NAMES: tuple[NumericFacetName, ...] = ('tokens', 'duration_ms')


# Shared limit for the persisted classifier-input cap.
MAX_INPUT_CHAR_BUDGET = MAX_TRACE_INPUT_CHARS


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
    tool_definition_count: int = Field(default=0, ge=0)
    tool_definition_tokens: int = Field(default=0, ge=0)
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


class TraceProjection(BaseModel):
    """The bounded, serialized trace state supplied to one classifier judgment."""

    model_config = ConfigDict(frozen=True)

    payload: dict[str, Any]
    serialized: str
    estimated_tokens: int = Field(ge=0, le=MAX_INPUT_CHAR_BUDGET, description='UTF-8 serialized input size estimate.')
    omitted_messages: int = Field(ge=0)
    omitted_bytes: int = Field(ge=0, description='Source bytes omitted or compressed from classifier input.')


class DimensionAnswer(BaseModel):
    """The classifier's verdict for one dimension of one trace."""

    model_config = ConfigDict(frozen=True)

    value: bool | float | str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    probabilities: dict[str, float] | None = None
    matched: bool = False
    error: str | None = None
    summary: str | None = None


class TraceClassification(BaseModel):
    """One trace's terminal result: an answer per dimension, matched only when every dimension matched.

    A run with no dimensions answers from its filters alone, so each of its traces is matched
    with no answers.
    """

    model_config = ConfigDict(frozen=True)

    trace_id: str
    span_id: str
    answers: tuple[DimensionAnswer, ...] = ()
    matched: bool = False
    error: str | None = None
    raw_result: dict[str, Any] = Field(default_factory=dict)


class Snapshot(BaseModel):
    """A fully validated, in-memory trace snapshot."""

    model_config = ConfigDict(frozen=True)

    traces: tuple[TraceRecord, ...]
    capture_metadata: dict[str, Any] = Field(default_factory=dict)


class FacetSelection(BaseModel):
    """Selected values for independent trace facets."""

    model_config = ConfigDict(frozen=True)

    project: frozenset[str] = frozenset()
    project_id: str | None = None
    model: frozenset[str] = frozenset()
    provider: frozenset[str] = frozenset()
    status: frozenset[str] = frozenset()
    product: frozenset[str] = frozenset()
    trace_type: frozenset[str] = frozenset()
    agent_name: frozenset[str] = frozenset()
    tool_name: frozenset[str] = frozenset()


class FacetCatalogue(BaseModel):
    """Available trace facet values, fields whose returned values were truncated, and per-value trace counts.

    ``value_counts`` holds Orq's trace count per value for each facet that reported one for every value it
    returned; a facet without counts is absent rather than zero.
    """

    model_config = ConfigDict(frozen=True)

    project: tuple[str, ...] = ()
    status: tuple[str, ...] = ()
    product: tuple[str, ...] = ()
    trace_type: tuple[str, ...] = ()
    model: tuple[str, ...] = ()
    provider: tuple[str, ...] = ()
    agent_name: tuple[str, ...] = ()
    tool_name: tuple[str, ...] = ()
    truncated_facets: frozenset[FacetName] = frozenset()
    value_counts: dict[str, dict[str, int]] = {}


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
    limit: int = Field(default=500, ge=1, le=5000)

    @field_validator('start', 'end')
    @classmethod
    def require_timezone_aware_bound(cls, value: datetime | None) -> datetime | None:
        """Reject bounds that cannot be compared with captured trace timestamps."""

        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError('time bounds must include a timezone offset')
        return value

    @model_validator(mode='after')
    def validate_range(self) -> Self:
        """Reject an inverted interval before planning or source access."""

        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError('start must not be after end')
        return self


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


MAX_DIMENSIONS = 3

# One rule, sent verbatim to both the filter selector and the compiler, so the two planners cannot
# disagree about whether a phrase is a metadata filter or a judgment of the conversation.
FILTER_OR_JUDGMENT_RULE = """Metadata filter or conversation judgment:
- Use a metadata filter only for an exact value the user literally names (a project, model, provider, agent, tool, product or trace type) or for an explicit trace status request ("failed traces", "status error", "errored traces").
- Anything that needs reading what happened in the conversation is a conversation judgment: a tool call failed or returned an error, the agent gave up, the user was frustrated, a hallucination, a refusal. Never also map it onto a status filter or any other metadata value.
- When a word could be either ("errors", "issues"), make it a conversation judgment and add no status filter, unless the user says status, failed trace or errored trace.
- Aggregate questions (totals, averages, counts, rankings) cannot be answered by finding traces: use no filter and no judgment.
Examples:
- "failed traces" -> status filter; no judgment.
- "tool errors" -> judgment "a tool call returned an error"; no status filter.
- "gpt-5 traces where the user is angry" -> model filter gpt-5 and judgment "the user is angry".
- "which model costs the most?" -> nothing; it is an aggregate question."""


class CompiledQuery(BaseModel):
    """One classifier dimension: a short column name, a semantic task and its validated inclusion rule."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(default='AI match', min_length=1, max_length=40)
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
RunPhase = Literal['planning', 'loading_traces', 'starting_classification']


class RunRequest(BaseModel):
    """The query and local population captured when a run is submitted."""

    model_config = ConfigDict(frozen=True)

    query: str
    mode: Literal['immediate', 'review']
    population: PopulationRequest
    parallelism: int = Field(default=100, ge=1, le=200)


@dataclass(frozen=True)
class RunSnapshot:
    """A frozen progress view; public reads detach nested mutable values from the owner."""

    generation: int = 0
    explorer_generation: int | None = None
    state: RunState = 'idle'
    phase: RunPhase | None = None
    request: RunRequest | None = None
    within_results: bool = False
    dimensions: tuple[CompiledQuery, ...] | None = None
    explicit_filters: FacetSelection = field(default_factory=FacetSelection)
    explicit_numeric: NumericFilters = field(default_factory=NumericFilters)
    generated_filters: FacetSelection = field(default_factory=FacetSelection)
    generated_numeric: NumericFilters = field(default_factory=NumericFilters)
    filter_response: ClassifyResponse | None = None
    filter_selection_error: str | None = None
    plan_warning: str | None = None
    trace_ids: tuple[str, ...] = ()
    traces: tuple[TraceRecord, ...] = ()
    results: Mapping[str, TraceClassification] = field(default_factory=lambda: MappingProxyType({}))
    projections: Mapping[str, TraceProjection] = field(default_factory=lambda: MappingProxyType({}))
    loaded: int = 0
    to_load: int = 0
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
    classifier_model: str | None = None
    """The model the store classifies with, fixed when the store was built."""

    def __post_init__(self) -> None:
        object.__setattr__(self, 'results', MappingProxyType(dict(self.results)))
        object.__setattr__(self, 'projections', MappingProxyType(dict(self.projections)))


@dataclass(frozen=True)
class TraceDetail:
    """Detached source trace and its inspectable projection and terminal result."""

    trace: TraceRecord
    projection: TraceProjection | None
    classification: TraceClassification | None
    dimensions: tuple[CompiledQuery, ...] | None = None


def selection_rule_text(compiled: CompiledQuery) -> tuple[str, str]:
    """When a dimension includes a trace, in plain words as ``(lead, value)``, e.g. ``('Answer is', 'yes')``."""
    selection = compiled.selection
    if selection.kind == 'threshold':
        return f'Score {"at least" if selection.operator == "gte" else "at most"}', f'{selection.value:g}'
    if compiled.task.kind == 'noul':
        return 'Answer is', ' or '.join('yes' if value else 'no' for value in selection.values)
    return 'Verdict', ' or '.join(str(value) for value in selection.values)


def validate_compiled_query(task: ClassifyQuestion, selection: SelectionRule) -> None:
    """Reject match rules that cannot be evaluated against their classifier task."""

    if task.state != {}:
        _raise_compiled_query_error(
            ('task', 'state'),
            task.state,
            'compiled classifier task state must be exactly {}',
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
