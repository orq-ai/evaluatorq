"""Validated data contracts for an insights run: population, labels, clusters, the run record."""

from __future__ import annotations

import re
from datetime import datetime  # noqa: TC003
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from typing_extensions import Self

from evaluatorq.common.judge import ClassifyQuestion
from evaluatorq.contracts import Usage  # noqa: TC001 — Pydantic needs the runtime model type.
from evaluatorq.insights.transcript import ToolStats  # noqa: TC001 — Pydantic needs the runtime model type.
from evaluatorq.trace_finder.models import FacetSelection, NumericFilters

if TYPE_CHECKING:
    from evaluatorq.trace_finder.export import RunExport

DimensionName = Literal['intent', 'failure', 'sentiment']
BoundedRatio = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class LabelSpec(BaseModel):
    """A fixed classifier question answered per trace: `noul` (yes/no), `choice`, or `score`.

    `name` is the key the answer is stored under on `TraceInsight.labels` and in
    `InsightsConfig.labels`; the pattern excludes `__match__`, the key `labeling.py`
    reserves for the optional population-match question sent alongside labels.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r'^[a-z][a-z0-9_]*$')
    kind: Literal['noul', 'choice', 'score']
    instructions: str
    criteria: dict[str, str | None] | list[str] | None = None
    noul_threshold: BoundedRatio = 0.5

    def to_question(self, state: dict[str, Any] | str | list[Any]) -> ClassifyQuestion:
        """Build the `/classify` question for this label against `state`, validating the criteria/kind shape."""
        return ClassifyQuestion(
            kind=self.kind,
            instructions=self.instructions,
            criteria=self.criteria,
            state=state,
            noul_threshold=self.noul_threshold,
        )


class InsightsPopulation(BaseModel):
    """The traces in an insights run, chosen by filters only — never influenced by labels.

    Either filtered live through `trace_finder` (`query`/`facets`/`numeric`/window/limit),
    hydrated from a finder export, or read from a local `Snapshot` JSON file.
    File sources fix the population and cannot be combined with live selection.
    """

    model_config = ConfigDict(frozen=True)

    query: str | None = None
    facets: FacetSelection = FacetSelection()
    numeric: NumericFilters = NumericFilters()
    start: datetime | None = None
    end: datetime | None = None
    window_days: int = Field(default=7, ge=1, le=90)
    limit: int = Field(default=500, ge=1, le=5000)
    finder_export: Path | None = None
    snapshot_path: Path | None = None
    _finder_export_snapshot: RunExport | None = PrivateAttr(default=None)

    @model_validator(mode='after')
    def _file_source_excludes_live_selection(self) -> Self:
        for name in ('start', 'end'):
            value = getattr(self, name)
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f'{name} must be timezone-aware')
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError('start must not be later than end')

        if self.finder_export is None and self.snapshot_path is None:
            return self
        if self.finder_export is not None and self.snapshot_path is not None:
            raise ValueError('finder_export and snapshot_path are mutually exclusive')
        live_fields = {
            'query': self.query is not None,
            'facets': self.facets != FacetSelection(),
            'numeric': self.numeric != NumericFilters(),
            'start': self.start is not None,
            'end': self.end is not None,
            'window_days': self.window_days != 7,
            'limit': self.limit != 500,
        }
        conflicting = [name for name, used in live_fields.items() if used]
        if conflicting:
            raise ValueError(
                'file sources cannot be combined with live population settings '
                f'({", ".join(conflicting)}): the file already fixes the trace population'
            )
        return self

    @classmethod
    def from_finder_export(cls, path: Path, *, export: RunExport | None = None) -> InsightsPopulation:
        """Build a population from a finder export; reuse a CLI-validated snapshot when provided."""
        population = cls(finder_export=path)
        population._finder_export_snapshot = export
        return population

    def finder_export_snapshot(self) -> RunExport | None:
        """Return the validated export supplied by the CLI, if any."""
        return self._finder_export_snapshot

    @classmethod
    def from_snapshot(cls, path: Path) -> InsightsPopulation:
        """Build a population from a local `Snapshot` JSON file without querying Orq."""
        return cls(snapshot_path=path)


def label_key(spec: LabelSpec, value: object) -> str:
    """The text an answer value counts, filters and displays under: `yes`/`no`, a score level's own number, or the choice.

    A score keys by the number its level text starts with ("1 calm: ...") so a 1-5
    scale does not read as 0-4; a level without a number keys by its 0-based index.
    """
    if spec.kind == 'noul':
        return 'yes' if value is True else 'no'
    if spec.kind == 'score' and isinstance(value, (int, float)) and not isinstance(value, bool):
        levels = spec.criteria if isinstance(spec.criteria, list) else []
        index = round(float(value) * max(0, len(levels) - 1))
        number = re.match(r'\d+', levels[index]) if index < len(levels) else None
        return number.group() if number else str(index)
    return str(value)


def label_order(spec: LabelSpec) -> list[str]:
    """Every value `label_key` can give for `spec`, in the spec's own order: yes before no, score levels low to high, choices as listed."""
    if spec.kind == 'noul':
        return ['yes', 'no']
    if spec.kind == 'score' and isinstance(spec.criteria, list):
        top = max(1, len(spec.criteria) - 1)
        return [label_key(spec, index / top) for index in range(len(spec.criteria))]
    return list(spec.criteria) if isinstance(spec.criteria, dict) else []


class LabelAnswer(BaseModel):
    """One label's answer for one trace. `error` set means the value could not be read — never guess a default."""

    value: bool | BoundedRatio | str | None
    confidence: BoundedRatio | None
    probabilities: dict[str, Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]] | None
    error: str | None


class TraceSummary(BaseModel):
    """The fixed per-trace text summary produced by `summarize.py`, feeding the discovered dimensions."""

    summary: str
    request: str | None
    task: str | None
    topic: str | None
    assistant_errors: list[str] = []
    sentiment_explanation: str | None
    languages: list[str] = []


_NO_ERROR_SENTINELS = frozenset({'none', 'n/a', 'no errors', 'no errors were made', 'no error'})


def real_assistant_errors(summary: TraceSummary) -> list[str]:
    """`assistant_errors` without placeholder entries a model writes for a clean trace ('None', 'No errors')."""
    return [
        error for error in summary.assistant_errors if error.strip().rstrip('.').casefold() not in _NO_ERROR_SENTINELS
    ]


class ClusterAssignment(BaseModel):
    """Which top- and base-level cluster a trace fell into for one discovered dimension. `'noise'` marks an outlier."""

    top: str
    base: str


class TraceInsight(BaseModel):
    """One trace's full result: labels, summary, cluster assignments and 3D coordinates per dimension, and any per-stage errors."""

    trace_id: str
    span_id: str
    timestamp: datetime
    agent_name: str = ''
    project: str = ''
    labels: dict[str, LabelAnswer] = {}
    summary: TraceSummary | None = None
    tool_stats: ToolStats | None = None
    assignments: dict[str, ClusterAssignment] = {}
    coords: dict[str, tuple[FiniteFloat, FiniteFloat, FiniteFloat]] = {}
    errors: dict[str, str] = {}


class Cluster(BaseModel):
    """One cluster (top or base level) within a discovered dimension."""

    id: str
    parent_id: str | None
    level: Literal['top', 'base']
    name: str
    description: str
    size: int
    trace_ids: list[str]
    example_trace_ids: list[str]
    group: str | None = None


class DimensionResult(BaseModel):
    """Clustering output for one discovered dimension."""

    name: DimensionName
    source_field: str
    clusters: list[Cluster]
    n_noise: int = 0
    n_no_signal: int = 0
    n_failed: int = 0
    warnings: list[str] = []


class LabelResult(BaseModel):
    """Aggregate stats for one label across the run."""

    spec: LabelSpec
    counts: dict[str, int]
    mean_confidence: BoundedRatio | None
    n_low_confidence: int
    n_failed: int
    n_not_asked: int = 0


class PriorityPoint(BaseModel):
    """One cluster's point in the priority matrix (volume vs. satisfaction vs. error share)."""

    cluster_id: str
    name: str
    volume: int
    mean_satisfaction: BoundedRatio
    error_share: BoundedRatio


class StageFailure(BaseModel):
    """A whole pipeline stage that failed; the run still completes with `status='error'` and partial results."""

    stage: str
    message: str
    dimension: str | None = None


class InsightsConfig(BaseModel):
    """Every knob of an insights run, defaulted per the design's Global Constraints, plus the requested labels and dimensions."""

    model_config = ConfigDict(frozen=True)

    labels: list[LabelSpec]
    dimensions: list[DimensionName]
    summary_model: str = 'openai/gpt-6-luna'
    classifier_model: str = 'typesafe/jev-latest'
    embedding_model: str = 'openai/text-embedding-3-small'
    max_clusters: int = Field(default=15, ge=1)
    max_subclusters: int = Field(default=15, ge=1)
    outlier_zscore: FiniteFloat | None = Field(default=None, ge=0)
    parallelism: int = Field(default=100, ge=1)
    priority_dimension: DimensionName = 'intent'
    cache: bool = True
    coding_analysis: bool = False
    coding_labels: list[LabelSpec] = Field(default_factory=list)

    @model_validator(mode='after')
    def _selections_are_unique(self) -> Self:
        label_names = [label.name for label in self.labels]
        if len(label_names) != len(set(label_names)):
            raise ValueError('labels must have unique names')
        coding_label_names = [label.name for label in self.coding_labels]
        if len(coding_label_names) != len(set(coding_label_names)):
            raise ValueError('coding_labels must have unique names')
        if set(label_names).intersection(coding_label_names):
            raise ValueError('labels and coding_labels must have unique names')
        if len(self.dimensions) != len(set(self.dimensions)):
            raise ValueError('dimensions must not contain duplicates')
        return self


class InsightsRun(BaseModel):
    """The full, persisted result of one insights run — written as `insights_<timestamp>_<slug>.json`."""

    schema_version: Literal[1] = 1
    # None on runs saved before the stamp existed.
    evaluatorq_version: str | None = None
    run_id: str
    run_name: str
    created_at: datetime
    status: Literal['completed', 'error']
    stage_failures: list[StageFailure]
    population: dict[str, Any]
    config: InsightsConfig
    traces: list[TraceInsight]
    dimensions: dict[str, DimensionResult]
    labels: dict[str, LabelResult]
    priority: list[PriorityPoint] | None
    priority_reason: str | None
    counts: dict[str, int]
    warnings: list[str]
    cost_by_stage: dict[str, Usage | None] = {}

    @model_validator(mode='after')
    def _label_values_match_specs(self) -> Self:
        kinds = {spec.name: spec.kind for spec in self.config.labels}
        if self.config.coding_analysis or self.config.coding_labels:
            from evaluatorq.insights.presets import CODING_LABELS

            coding_specs = (
                CODING_LABELS
                if self.config.coding_analysis
                else (
                    *CODING_LABELS[:1],
                    *self.config.coding_labels,
                )
            )
            kinds.update({spec.name: spec.kind for spec in coding_specs})
        for trace in self.traces:
            for name, answer in trace.labels.items():
                if name not in kinds:
                    raise ValueError(f'trace {trace.trace_id!r} has unconfigured label {name!r}')
                value = answer.value
                if value is None:
                    continue
                kind = kinds[name]
                valid = (
                    (kind == 'noul' and isinstance(value, bool))
                    or (kind == 'choice' and isinstance(value, str))
                    or (kind == 'score' and isinstance(value, float))
                )
                if not valid:
                    raise ValueError(f'trace {trace.trace_id!r} label {name!r} has a value incompatible with {kind!r}')
        return self

    @model_validator(mode='after')
    def _created_at_is_timezone_aware(self) -> Self:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError('created_at must be timezone-aware')
        return self
