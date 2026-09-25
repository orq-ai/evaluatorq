"""Validated data contracts for an insights run: population, labels, clusters, the run record."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003
from pathlib import Path  # noqa: TC003
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from evaluatorq.common.judge import ClassifyQuestion
from evaluatorq.trace_finder.models import FacetSelection, NumericFilters

DimensionName = Literal['intent', 'failure', 'sentiment']


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
    noul_threshold: float = 0.5

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

    Either filtered live through `trace_finder` (`query`/`facets`/`numeric`/window/limit) or
    hydrated from a previously exported finder population (`finder_export`); the two are
    mutually exclusive because a `finder_export` already pins the matched trace ids.
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

    @model_validator(mode='after')
    def _finder_export_excludes_live_selection(self) -> Self:
        if self.finder_export is None:
            return self
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
                'finder_export cannot be combined with live population settings '
                f'({", ".join(conflicting)}): a finder export already pins the matched traces'
            )
        return self

    @classmethod
    def from_finder_export(cls, path: Path) -> InsightsPopulation:
        """Build a population hydrated from a finder export JSON; no match question is asked."""
        return cls(finder_export=path)


class LabelAnswer(BaseModel):
    """One label's answer for one trace. `error` set means the value could not be read — never guess a default."""

    value: bool | float | str | None
    confidence: float | None
    probabilities: dict[str, float] | None
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
    tools_used: list[str] = []


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
    assignments: dict[str, ClusterAssignment] = {}
    coords: dict[str, tuple[float, float, float]] = {}
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
    mean_confidence: float | None
    n_low_confidence: int
    n_failed: int


class PriorityPoint(BaseModel):
    """One cluster's point in the priority matrix (volume vs. satisfaction vs. error share)."""

    cluster_id: str
    name: str
    volume: int
    mean_satisfaction: float
    error_share: float


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
    max_clusters: int = 15
    max_subclusters: int = 15
    min_cluster_size: int = 5
    outlier_zscore: float | None = None
    parallelism: int = 100
    priority_dimension: DimensionName = 'intent'
    cache: bool = True
    merge_threshold: float = 0.5
    low_confidence_threshold: float = 0.6
    umap_random_state: int = 42


class InsightsRun(BaseModel):
    """The full, persisted result of one insights run — written as `insights_<timestamp>_<slug>.json`."""

    schema_version: Literal[1] = 1
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
