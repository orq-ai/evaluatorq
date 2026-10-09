"""Fixed-rate estimate of an Insights run's traces, cost and time, shown before the run starts.

Cost estimates use configured input caps and named assumptions; some per-stage figures are
baselines rather than ceilings. A count, price or rate that cannot be grounded stays unknown and
is named in `RunEstimate.unknowns`.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from operator import itemgetter
from typing import TYPE_CHECKING, Literal

from loguru import logger

from evaluatorq.common.model_input import (
    CHARS_PER_ESTIMATED_TOKEN,
    JEV_STATE_ALL_QUESTIONS_CHARS,
    effective_trace_input_chars,
    is_jev_model,
)
from evaluatorq.common.run_manifest import list_manifests
from evaluatorq.contracts import ManifestStatus
from evaluatorq.insights.summarize import SUMMARY_MAX_TOKENS
from evaluatorq.trace_finder.settings import MAX_TRACE_INPUT_CHARS, MIN_TRACE_INPUT_CHARS

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from evaluatorq.common.model_catalogue import ModelInfo
    from evaluatorq.insights.models import DimensionName
    from evaluatorq.trace_finder.models import FacetCatalogue, FacetSelection

TraceSource = Literal['recent', 'query', 'finder', 'snapshot']

# Character caps divided by four are estimates, not measurements from a model tokenizer.
CHARS_PER_TOKEN = CHARS_PER_ESTIMATED_TOKEN
LABEL_ANSWER_TOKENS = 4
SUMMARY_OUTPUT_TOKENS = SUMMARY_MAX_TOKENS
EMBEDDING_INPUT_TOKENS = SUMMARY_MAX_TOKENS

EXCLUDED = ('trace selection (question compiling and filter choice)', 'cluster naming and merging')
FIRST_RUN_NOTE = 'estimated after your first run'
PARALLELISM_NOTE = 'an earlier run did not record its parallelism, so its time is not scaled to the chosen parallelism'


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
class StageTiming:
    """Median seconds per trace of one stage, and the parallelism that figure is for.

    ``parallelism`` is ``1`` when the figure was normalised from runs that recorded theirs (the
    time then scales as one over the chosen parallelism) and ``None`` when no earlier run did.
    """

    seconds: float
    parallelism: int | None
    runs: int


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
    basis: str


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
    time_basis: str
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


def stage_seconds(directory: Path) -> Mapping[str, StageTiming]:
    """Median seconds per trace for each stage, from earlier Insights manifests.

    A stage counts only when it completed and recorded a positive progress ``total``; the manifest
    stores no per-trace count otherwise. Stage names are the manifest's (``label``, ``summary``,
    ``dimension:<name>``). Runs that recorded their parallelism are normalised to one request at a
    time, assuming time falls in proportion to parallelism; when a stage has no such run, its
    median is returned unscaled with ``parallelism=None``.
    """
    samples: dict[str, list[tuple[float, int | None]]] = {}
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
            samples.setdefault(stage.name, []).append((seconds / stage.total, manifest.parallelism))
    timings: dict[str, StageTiming] = {}
    for name, values in samples.items():
        recorded = [seconds * parallelism for seconds, parallelism in values if parallelism]
        if recorded:
            timings[name] = StageTiming(statistics.median(recorded), 1, len(recorded))
        else:
            timings[name] = StageTiming(statistics.median(seconds for seconds, _ in values), None, len(values))
    return timings


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
    seconds: Mapping[str, StageTiming],
    parallelism: int,
    query: bool,
    sentiment_selected: bool = False,
    trace_input_chars: int | None = None,
) -> RunEstimate:
    """Cost and time ceilings per stage.

    ``question_count`` counts the chosen general and custom questions. The label call also answers
    the sentiment label the pipeline adds for the sentiment dimension (unless ``sentiment_selected``
    already has it) and, for a Question source, the population-match question.

    A Question source labels every trace in range but only matches reach later stages, so those
    stages run from zero (label only) to every trace (every trace matches).
    """
    trace_input_chars = trace_input_chars if trace_input_chars is not None else effective_trace_input_chars()
    if not MIN_TRACE_INPUT_CHARS <= trace_input_chars <= MAX_TRACE_INPUT_CHARS:
        raise ValueError(
            f'trace_input_chars must be between {MIN_TRACE_INPUT_CHARS} and {MAX_TRACE_INPUT_CHARS} characters'
        )
    n = bound.n
    unknowns: list[str] = []
    if n is None:
        unknowns.append(f'traces: {bound.basis}')
    high = n or 0
    later_low = 0 if query else high

    label_chars = (
        min(trace_input_chars, JEV_STATE_ALL_QUESTIONS_CHARS) if is_jev_model(models.classifier) else trace_input_chars
    )
    label_in = label_chars // CHARS_PER_TOKEN
    label_answer = LABEL_ANSWER_TOKENS
    summary_in = trace_input_chars // CHARS_PER_TOKEN
    summary_out = SUMMARY_OUTPUT_TOKENS
    embed_in, embed_out = EMBEDDING_INPUT_TOKENS, 0
    answers = question_count + (1 if 'sentiment' in dimensions and not sentiment_selected else 0) + (1 if query else 0)
    if coding:
        unknowns.append('cost: coding tool activity may require additional classifier chunks')
    label_basis = (
        f'one classify call per trace: up to {label_in:,} estimated input tokens at four characters per token '
        'for combined state and serialized questions, '
        f'and {answers} answer{"s" if answers != 1 else ""} of {label_answer} output tokens'
        + (', including the population-match answer' if query else '')
    )
    coding_basis = (
        f'baseline: one tool-activity classify request per coding trace ({label_in:,} input tokens, '
        f'{label_answer} output tokens); long activity can split into additional chunks'
    )
    summary_basis = (
        f'up to {summary_in:,} estimated input tokens at four characters per token, '
        f'and up to {summary_out:,} output tokens (max_tokens)'
    )
    embed_basis = f'one embedding per trace, up to {embed_in:,} tokens (summary length cap)'
    # (name, model, tokens in, tokens out, traces low, basis)
    plan: list[tuple[str, str, int, int, int, str]] = []
    if answers:
        plan.append(('label', models.classifier, label_in, label_answer * answers, high, label_basis))
    if coding:
        plan.append(('label_coding', models.classifier, label_in, label_answer, high, coding_basis))
    plan.append(('summary', models.summary, summary_in, summary_out, later_low, summary_basis))
    plan.extend(
        (f'dimension:{name}', models.embedding, embed_in, embed_out, later_low, embed_basis) for name in dimensions
    )

    rows: list[StageEstimate] = []
    unscaled = False
    for name, model, tokens_in, tokens_out, low, basis in plan:
        unit = _price(prices.get(model), tokens_in, tokens_out) if n is not None else None
        if n is not None and unit is None:
            unknowns.append(f'cost: {name} has no price for {model}')
        timing = seconds.get('label' if name == 'label_coding' else name)
        if n is not None and timing is None and seconds:
            unknowns.append(f'time: {name} has no earlier timing')
        scale = 1.0
        if timing is not None:
            if timing.parallelism is None:
                unscaled = True
            else:
                scale = timing.parallelism / max(parallelism, 1)
        unknown_time = timing is None or n is None
        rows.append(
            StageEstimate(
                name=name,
                traces_low=low,
                traces_high=high,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost_low=None if unit is None else unit * low,
                cost_high=None if unit is None else unit * high,
                seconds_low=None if timing is None or unknown_time else timing.seconds * low * scale,
                seconds_high=None if timing is None or unknown_time else timing.seconds * high * scale,
                basis=basis,
            )
        )

    priced = [row for row in rows if row.cost_high is not None]
    timed = [row for row in rows if row.seconds_high is not None]
    time_note = None
    if not seconds:
        time_note = FIRST_RUN_NOTE
    elif unscaled:
        time_note = PARALLELISM_NOTE
    earlier = max((timing.runs for timing in seconds.values()), default=0)
    runs = f'median per-trace time of up to {earlier} earlier run{"" if earlier == 1 else "s"}'
    if not seconds:
        time_basis = FIRST_RUN_NOTE
    elif unscaled:
        time_basis = f'{runs}, at the parallelism they used; rough'
    else:
        time_basis = f'{runs}, assuming time falls in proportion to the {parallelism} parallel requests; rough'
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
        time_basis=time_basis,
        unknowns=tuple(unknowns),
        excluded=EXCLUDED,
    )
