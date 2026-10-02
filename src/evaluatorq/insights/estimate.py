"""Fixed-rate estimate of an Insights run's traces, cost and time, shown before the run starts.

Every figure is a ceiling built from a cap the code already enforces, so the result is "up to". A
count, price or rate that cannot be grounded stays unknown and is named in `RunEstimate.unknowns`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING, Literal

from loguru import logger

from evaluatorq.common.run_manifest import list_manifests
from evaluatorq.contracts import ManifestStatus
from evaluatorq.insights.summarize import SUMMARY_MAX_TOKENS
from evaluatorq.insights.transcript import VIEW_BUDGET

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from evaluatorq.common.model_catalogue import ModelInfo
    from evaluatorq.insights.models import DimensionName
    from evaluatorq.trace_finder.models import FacetCatalogue, FacetSelection

TraceSource = Literal['recent', 'question', 'finder', 'snapshot']

# `transcript.py` sizes VIEW_BUDGET characters at about 25k tokens, "near 3 characters a token".
CHARS_PER_TOKEN = 3
# The classifier reads the conversation view; `labeling.py` makes one `/classify` call per trace.
LABEL_INPUT_TOKENS = VIEW_BUDGET // CHARS_PER_TOKEN
# One classify answer: the recorded usage in `tests/common/test_judge_classify.py::_usage` is
# `output_tokens: 4` for a single-question reply. A trace's call carries one answer per question.
LABEL_ANSWER_TOKENS = 4
# `summarize.py::_build_prompt` sends `conversation_view(trace)` with the default VIEW_BUDGET, not the
# `projection.MAX_TOKEN_BUDGET` projection. Prompt wording around the view is not counted.
SUMMARY_INPUT_TOKENS = VIEW_BUDGET // CHARS_PER_TOKEN
# `summarize.py` passes this as `max_tokens` for the summary call.
SUMMARY_OUTPUT_TOKENS = SUMMARY_MAX_TOKENS
# Each trace's summary is embedded once per selected dimension; the summary cap bounds its length.
EMBEDDING_INPUT_TOKENS = SUMMARY_MAX_TOKENS

# stage -> (input tokens, output tokens) per trace. Label output is per answer.
TOKENS_PER_TRACE: Mapping[str, tuple[int, int]] = {
    'label': (LABEL_INPUT_TOKENS, LABEL_ANSWER_TOKENS),
    'summary': (SUMMARY_INPUT_TOKENS, SUMMARY_OUTPUT_TOKENS),
    'embedding': (EMBEDDING_INPUT_TOKENS, 0),
}

EXCLUDED = ('trace selection (question compiling and filter choice)', 'cluster naming and merging')
FIRST_RUN_NOTE = 'estimated after your first run'
PARALLELISM_NOTE = "earlier runs' parallelism is not recorded, so time is not scaled to the chosen parallelism"


@dataclass(frozen=True)
class StageModels:
    summary: str
    classifier: str
    embedding: str


@dataclass(frozen=True)
class TraceBound:
    n: int | None
    exact: bool
    basis: str


@dataclass(frozen=True)
class StageEstimate:
    name: str
    traces_low: int
    traces_high: int
    tokens_in: int
    tokens_out: int
    cost_low: float | None
    cost_high: float | None
    seconds_low: float | None
    seconds_high: float | None


@dataclass(frozen=True)
class RunEstimate:
    bound: TraceBound
    rows: tuple[StageEstimate, ...]
    cost_low: float | None
    cost_high: float | None
    cost_partial: bool
    seconds_low: float | None
    seconds_high: float | None
    time_partial: bool
    time_note: str | None
    unknowns: tuple[str, ...]
    excluded: tuple[str, ...]


def trace_bound(
    source: TraceSource,
    *,
    catalogue: FacetCatalogue,
    facets: FacetSelection,
    limit: int,
    finder_count: int | None,
    snapshot_count: int | None,
) -> TraceBound:
    """How many traces a run can load: exact for a file source, a ceiling for Recent and Question."""
    if source == 'finder':
        return _exact(finder_count, 'matched traces in the export')
    if source == 'snapshot':
        return _exact(snapshot_count, 'traces in the local file')

    selected = {
        name: values
        for name in ('project', 'model', 'provider', 'status', 'product', 'trace_type', 'agent_name', 'tool_name')
        if (values := getattr(facets, name))
    }
    capped = f'the trace limit of {limit}'
    if facets.project_id is not None and 'project' not in selected:
        return TraceBound(limit, exact=False, basis=f'{capped}; a project id filter has no count')

    needed = list(selected) or ['status']
    for name in needed:
        counts = catalogue.value_counts.get(name)
        if name in catalogue.truncated_facets or counts is None:
            return TraceBound(limit, exact=False, basis=f'{capped}; {name} counts are incomplete')

    sums = {name: sum(catalogue.value_counts[name].get(value, 0) for value in selected[name]) for name in selected}
    if not sums:
        total = sum(catalogue.value_counts['status'].values())
        return TraceBound(min(total, limit), exact=False, basis=f'status counts, up to {capped}')
    name, smallest = min(sums.items(), key=itemgetter(1))
    if len(sums) == 1:
        basis = f'selected {name} values, up to {capped}'
    else:
        basis = f'smallest of the selected facet sums ({name}), up to {capped}'
    return TraceBound(min(smallest, limit), exact=False, basis=basis)


def _exact(count: int | None, what: str) -> TraceBound:
    if count is None:
        return TraceBound(None, exact=False, basis=f'{what} unknown')
    return TraceBound(count, exact=True, basis=what)


def stage_seconds(directory: Path) -> Mapping[str, float]:
    """Median seconds per trace for each stage, from earlier Insights manifests.

    A stage counts only when it completed and recorded a positive progress ``total``; the manifest
    stores no per-trace count otherwise. Stage names are the manifest's (``label``, ``summary``,
    ``dimension:<name>``).
    """
    per_trace: dict[str, list[float]] = {}
    for manifest in list_manifests(directory):
        if manifest.surface != 'insights':
            continue
        for stage in manifest.stages:
            if stage.status != ManifestStatus.COMPLETED or stage.ended_at is None or not stage.total:
                continue
            seconds = (stage.ended_at - stage.started_at).total_seconds()
            if seconds < 0:
                logger.debug('Ignoring stage {} with negative duration in run {}', stage.name, manifest.run_id)
                continue
            per_trace.setdefault(stage.name, []).append(seconds / stage.total)
    return {name: statistics.median(values) for name, values in per_trace.items()}


def _price(info: ModelInfo | None, tokens_in: int, tokens_out: int) -> float | None:
    """USD for one trace at the model's per-1k prices; None when a needed price is missing."""
    if info is None or info.input_cost_per_1k is None or info.output_cost_per_1k is None:
        return None
    return (info.input_cost_per_1k * tokens_in + info.output_cost_per_1k * tokens_out) / 1000


def estimate_run(
    *,
    bound: TraceBound,
    dimensions: Sequence[DimensionName],
    question_count: int,
    coding: bool,
    models: StageModels,
    prices: Mapping[str, ModelInfo | None],
    seconds: Mapping[str, float],
    parallelism: int,
    query: bool,
    seconds_parallelism: int | None = None,
) -> RunEstimate:
    """Cost and time ceilings per stage.

    A Question source labels every trace in range but only matches reach later stages, so those
    stages run from zero (label only) to every trace (every trace matches).
    """
    n = bound.n
    unknowns: list[str] = []
    if n is None:
        unknowns.append(f'traces: {bound.basis}')
    high = n or 0
    later_low = 0 if query else high

    label_in, label_answer = TOKENS_PER_TRACE['label']
    summary_in, summary_out = TOKENS_PER_TRACE['summary']
    embed_in, embed_out = TOKENS_PER_TRACE['embedding']
    # (name, model, tokens in, tokens out, traces low)
    plan: list[tuple[str, str, int, int, int]] = []
    if question_count or 'sentiment' in dimensions:
        plan.append(('label', models.classifier, label_in, label_answer * max(question_count, 1), high))
    if coding:
        plan.append(('label_coding', models.classifier, label_in, label_answer, high))
    plan.append(('summary', models.summary, summary_in, summary_out, later_low))
    plan.extend((f'dimension:{name}', models.embedding, embed_in, embed_out, later_low) for name in dimensions)

    scale = 1.0
    scale_known = seconds_parallelism is not None
    if seconds_parallelism is not None:
        scale = seconds_parallelism / max(parallelism, 1)

    rows: list[StageEstimate] = []
    for name, model, tokens_in, tokens_out, low in plan:
        unit = _price(prices.get(model), tokens_in, tokens_out) if n is not None else None
        if n is not None and unit is None:
            unknowns.append(f'cost: {name} has no price for {model}')
        per_trace = seconds.get(name if name != 'label_coding' else 'label')
        if n is not None and per_trace is None and seconds:
            unknowns.append(f'time: {name} has no earlier timing')
        rows.append(
            StageEstimate(
                name=name,
                traces_low=low,
                traces_high=high,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_low=None if unit is None else unit * low,
                cost_high=None if unit is None else unit * high,
                seconds_low=None if per_trace is None or n is None else per_trace * low * scale,
                seconds_high=None if per_trace is None or n is None else per_trace * high * scale,
            )
        )

    priced = [row for row in rows if row.cost_high is not None]
    timed = [row for row in rows if row.seconds_high is not None]
    time_note = None
    if not seconds:
        time_note = FIRST_RUN_NOTE
    elif not scale_known:
        time_note = PARALLELISM_NOTE
    return RunEstimate(
        bound=bound,
        rows=tuple(rows),
        cost_low=sum(row.cost_low or 0.0 for row in priced) if priced else None,
        cost_high=sum(row.cost_high or 0.0 for row in priced) if priced else None,
        cost_partial=len(priced) != len(rows),
        seconds_low=sum(row.seconds_low or 0.0 for row in timed) if timed else None,
        seconds_high=sum(row.seconds_high or 0.0 for row in timed) if timed else None,
        time_partial=bool(timed) and len(timed) != len(rows),
        time_note=time_note,
        unknowns=tuple(unknowns),
        excluded=EXCLUDED,
    )
