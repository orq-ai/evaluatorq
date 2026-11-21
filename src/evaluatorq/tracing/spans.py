"""
Span creation utilities for OpenTelemetry instrumentation.

Span hierarchy:
- evaluatorq.run (opt-in via ``single_trace=True``; root or child of parent context)
  └── orq.job (per job per data point — the root itself when single_trace is off)
      ├── [User's instrumented code becomes child spans]
      └── orq.evaluation (per evaluator - child of its job)
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.contracts import EVAL_ERROR_RAW_OUTPUT_KEY
from evaluatorq.types import EvaluationResultCell

from .setup import get_tracer  # noqa: F401  # backwards-compatible test/import hook

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from opentelemetry.trace import Span

# Orq's trace ingestion drops (not truncates) any dynamic string span attribute
# longer than this (genai_traces.go: dynamicAttributeMaxStringLength). Keep the
# span copy of evaluator explanations at or under it so they always render.
_SPAN_TEXT_MAX_CHARS = 512

# The GenAI conventions say gen_ai.evaluation.score.label SHOULD be low
# cardinality; a string verdict longer than this is cut to keep it a label.
_SCORE_LABEL_MAX_CHARS = 64

# Longest ``error_type`` taken verbatim from a scorer's error payload for ``error.type``.
_ERROR_TYPE_MAX_CHARS = 64
_FALLBACK_ERROR_TYPE = 'evaluation_error'

_EVALUATION_RESULT_EVENT = 'gen_ai.evaluation.result'


@dataclass
class JobSpanOptions:
    """Options for creating a job span."""

    run_id: str
    row_index: int
    job_name: str | None = None
    parent_context: Any | None = None
    trace_type: str = 'evaluatorq'


@dataclass
class RunSpanOptions:
    """Options for creating a run span."""

    run_id: str
    run_name: str
    parent_context: Any | None = None
    trace_type: str = 'evaluatorq'


@dataclass
class EvaluationSpanOptions:
    """Options for creating an evaluation span."""

    run_id: str
    evaluator_name: str
    evaluator_type: str | None = None


@asynccontextmanager
async def with_run_span(
    options: RunSpanOptions,
) -> AsyncGenerator[Span | None, None]:
    """Execute an evaluation run inside one ``evaluatorq.run`` span.

    Opt-in (``evaluatorq(..., single_trace=True)``). Without it no run span
    exists and every ``orq.job`` is its own root, so an N-row evaluation lands
    as N separate traces. With it, all rows share one trace.

    Named ``evaluatorq.run`` rather than ``orq.run`` because it brackets an
    ``evaluatorq()`` call specifically — red teaming and simulation already
    open their own equivalents (``Evaluatorq - Red Teaming`` /
    ``Evaluatorq - Agent Simulation``).

    Yields:
        The span if tracing is enabled, None otherwise.
    """
    # Delegates to the shared helper rather than repeating the open/status/error
    # dance: `with_span` also catches BaseException (so a cancelled run is still
    # marked failed), stamps `error.type`, and won't clobber an ERROR the body
    # set deliberately.
    from evaluatorq.common.tracing import with_span

    async with with_span(
        'evaluatorq.run',
        {
            'orq.trace_type': options.trace_type,
            'orq.run_id': options.run_id,
            'orq.run_name': options.run_name,
            # Same key the red-team / simulation roots stamp, so one query
            # finds the root of any evaluatorq run whatever the surface.
            'orq.evaluatorq_run_id': options.run_id,
        },
        parent_context=options.parent_context,
    ) as span:
        yield span


@asynccontextmanager
async def with_job_span(
    options: JobSpanOptions,
) -> AsyncGenerator[Span | None, None]:
    """
    Execute code within an orq.job span.
    Job spans are independent roots, or children of a parent context if provided.

    Args:
        options: Job span configuration

    Yields:
        The span if tracing is enabled, None otherwise

    Example:
        ```python
        async with with_job_span(JobSpanOptions(run_id="abc", row_index=0)) as span:
            # Your job code here
            pass
        ```
    """
    from evaluatorq.common.tracing import with_span

    attributes: dict[str, Any] = {
        'orq.trace_type': options.trace_type,
        'orq.run_id': options.run_id,
        'orq.row_index': options.row_index,
    }
    if options.job_name:
        attributes['orq.job_name'] = options.job_name

    async with with_span(
        'orq.job',
        attributes,
        parent_context=options.parent_context,
    ) as span:
        yield span


@asynccontextmanager
async def with_evaluation_span(
    options: EvaluationSpanOptions,
) -> AsyncGenerator[Span | None, None]:
    """
    Execute code within an orq.evaluation span.
    Evaluation spans are children of the job span.

    Args:
        options: Evaluation span configuration

    Yields:
        The span if tracing is enabled, None otherwise

    Example:
        ```python
        async with with_evaluation_span(EvaluationSpanOptions(
            run_id="abc",
            evaluator_name="string-contains"
        )) as span:
            # Your evaluator code here
            pass
        ```
    """
    from evaluatorq.common.tracing import with_span

    async with with_span(
        # Named so concurrent spans are distinguishable, mirroring `chat {model}`.
        f'orq.evaluation {options.evaluator_name}',
        {
            'orq.run_id': options.run_id,
            'orq.evaluator_name': options.evaluator_name,
        },
    ) as span:
        _set_evaluator_identity(span, evaluator_name=options.evaluator_name, evaluator_type=options.evaluator_type)
        yield span


def _set_evaluator_identity(span: Span | None, *, evaluator_name: str | None, evaluator_type: str | None) -> None:
    """Classify the span before execution so failed scorers retain their input panel."""
    if span is None or evaluator_type is None:
        return
    span.set_attribute('orq.span_type', 'span.evaluator')
    span.set_attribute('orq.evaluator.type', evaluator_type)
    if evaluator_name is not None:
        span.set_attribute('gen_ai.evaluation.name', evaluator_name)
        span.set_attribute('orq.evaluator.key', evaluator_name)


def set_evaluation_attributes(
    span: Span | None,
    score: str | float | bool | dict[str, Any] | EvaluationResultCell | None,  # noqa: FBT001
    *,
    explanation: str | None = None,
    pass_: bool | None = None,
    evaluator_name: str | None = None,
    evaluator_type: str | None = None,
    raw_output: dict[str, Any] | None = None,
    evaluated_span: Span | None = None,
) -> None:
    """
    Set evaluation result attributes on a span.

    Args:
        span: The span to set attributes on (can be None)
        score: The evaluation score
        explanation: Optional explanation of the score
        pass_: Optional pass/fail status
        evaluator_name: Name of the evaluator (used for the gen_ai.evaluation.* block)
        raw_output: Optional detailed scorer result for the trace Output panel.
        evaluator_type: Opt-in evaluator kind, matching Orq's EvaluatorType enum
            (e.g. "llm_eval", "python_eval"). When provided, the flat
            gen_ai.evaluation.* / orq.evaluator.* attributes plus the
            orq.evaluation.output verdict payload the Orq trace UI classifies +
            renders evaluator spans from are additionally emitted.
            When ``None`` (the default), only the legacy orq.score/explanation/pass
            attributes are written.
        evaluated_span: The span whose output was scored (the ``orq.job`` span).
            When given with ``evaluator_type``, it also receives a standard
            ``gen_ai.evaluation.result`` event carrying the gen_ai.evaluation.*
            attributes, as the OTel GenAI conventions place it.
    """
    from evaluatorq.common.tracing import EVALUATION_OUTPUT_KEYS, record_span_payload

    if span is None:
        return

    # The Orq trace ingestion (genai_traces.go extractTypedAttributes) DROPS any
    # dynamic string attribute whose value exceeds 512 chars — it doesn't
    # truncate, it skips the whole attribute. So cap the span copy of the
    # explanation here or a long one silently vanishes from the trace UI. The
    # full, untruncated text still reaches the uploaded experiment via the
    # EvaluationResult (this only touches the span mirror).
    full_explanation = explanation
    if explanation is not None and len(explanation) > _SPAN_TEXT_MAX_CHARS:
        explanation = explanation[: _SPAN_TEXT_MAX_CHARS - 1] + '…'

    if score is not None:
        span.set_attribute(
            'orq.score',
            json.dumps(score.model_dump())
            if isinstance(score, EvaluationResultCell)
            else json.dumps(score)
            if isinstance(score, dict)
            else str(score),
        )
    if explanation is not None:
        span.set_attribute('orq.explanation', explanation)
    if pass_ is not None:
        span.set_attribute('orq.pass', pass_)

    # Additive evaluator-span emission — opt-in via evaluator_type. The Orq trace
    # UI (ClickHouse/Go ingestion) classifies + shows evaluator detail only from
    # these FLAT attributes; nested JSON is dropped. orq.workspace_id is injected
    # server-side, so it is intentionally not set here.
    if evaluator_type is None:
        return
    _set_evaluator_identity(span, evaluator_name=evaluator_name, evaluator_type=evaluator_type)

    # orq.evaluator.output_type drives how the UI formats the score (see
    # deriveEvaluatorResult in orquesta-web). bool is an int subclass, so it
    # must be tested before the numeric branch.
    if isinstance(score, bool):
        output_type, score_value = 'boolean', float(score)
    elif isinstance(score, (int, float)):
        output_type, score_value = 'number', float(score)
    else:
        output_type, score_value = 'string', None
    span.set_attribute('orq.evaluator.output_type', output_type)
    if score_value is not None:
        span.set_attribute('gen_ai.evaluation.score.value', score_value)
    if explanation is not None:
        span.set_attribute('gen_ai.evaluation.explanation', explanation)
    label = _score_label(score, pass_, evaluator_name)
    # A bool score with no pass_ is its own verdict, as Orq's graders treat it.
    passed = pass_ if pass_ is not None else score if isinstance(score, bool) else None
    if passed is not None:
        # Not in the GenAI conventions, but Orq stores, alerts and scores
        # boolean verdicts from gen_ai.evaluation.passed with no fallback.
        span.set_attribute('gen_ai.evaluation.passed', passed)
    if label is not None:
        span.set_attribute('gen_ai.evaluation.score.label', label)
        span.set_attribute('orq.evaluator.score.label', label)

    if evaluator_name is not None:
        scorer_error = _scorer_error_type(raw_output, evaluator_name)
        add_evaluation_result_event(
            evaluated_span,
            name=evaluator_name,
            score_value=None if scorer_error else score_value,
            label=None if scorer_error else label,
            explanation=None if scorer_error else explanation,
            error_type=scorer_error,
        )

    # The evaluator span's "Output" panel renders orq.evaluation.output verbatim.
    # Ingestion routes this key to blob storage rather than the 512-char-capped
    # typed attribute maps, so the untruncated explanation belongs here.
    verdict: dict[str, Any] = {
        'passed': pass_,
        'value': score.model_dump() if isinstance(score, EvaluationResultCell) else score,
        'type': evaluator_type,
    }
    if full_explanation is not None:
        verdict['reason'] = full_explanation
    record_span_payload(
        span,
        key=EVALUATION_OUTPUT_KEYS,
        payload=verdict,
        optional_fields={'raw_output': raw_output} if raw_output is not None else None,
    )


def add_evaluation_result_event(
    evaluated_span: Span | None,
    *,
    name: str,
    score_value: float | None = None,
    label: str | None = None,
    explanation: str | None = None,
    error_type: str | None = None,
) -> None:
    """Add the OTel GenAI ``gen_ai.evaluation.result`` event to the evaluated span.

    With ``error_type`` the event carries only the name and ``error.type``, as the conventions
    prescribe for an evaluation that did not produce a score. A ``None`` span emits nothing.
    """
    if evaluated_span is None:
        return
    event: dict[str, str | float] = {'gen_ai.evaluation.name': name}
    if error_type is not None:
        event['error.type'] = error_type
    else:
        if score_value is not None:
            event['gen_ai.evaluation.score.value'] = score_value
        if label is not None:
            event['gen_ai.evaluation.score.label'] = label
        if explanation is not None:
            event['gen_ai.evaluation.explanation'] = explanation
    evaluated_span.add_event(_EVALUATION_RESULT_EVENT, attributes=event)


def _scorer_error_type(raw_output: dict[str, Any] | None, evaluator_name: str) -> str | None:
    """Return ``error.type`` when a scorer returned an inconclusive result instead of raising.

    Scorers mark that by putting ``EVAL_ERROR_RAW_OUTPUT_KEY`` in ``raw_output``. Its value is
    an error payload whose ``error_type`` names the cause; a non-empty string of at most
    ``_ERROR_TYPE_MAX_CHARS`` there, or as the value itself, is used as is. Anything else
    becomes ``'evaluation_error'`` with a warning.
    """
    if not raw_output or EVAL_ERROR_RAW_OUTPUT_KEY not in raw_output:
        return None
    payload = raw_output[EVAL_ERROR_RAW_OUTPUT_KEY]
    candidate = payload.get('error_type') if isinstance(payload, dict) else payload
    if isinstance(candidate, str) and candidate.strip() and len(candidate) <= _ERROR_TYPE_MAX_CHARS:
        return candidate
    logger.warning(
        'evaluator {!r} returned an error result without a usable error_type ({!r}); '
        'the trace event reports error.type={!r}',
        evaluator_name,
        candidate,
        _FALLBACK_ERROR_TYPE,
    )
    return _FALLBACK_ERROR_TYPE


def _score_label(
    score: str | float | bool | dict[str, Any] | EvaluationResultCell | None,  # noqa: FBT001
    pass_: bool | None,  # noqa: FBT001
    evaluator_name: str | None,
) -> str | None:
    """Return the gen_ai.evaluation.score.label for a verdict, or ``None``.

    Mirrors Orq's own graders (graders/client.go ``evaluationScore``): a non-empty string
    score is its own label even when ``pass_`` is set, cut to ``_SCORE_LABEL_MAX_CHARS``
    with a warning so the cut is never silent. Otherwise ``pass_``, or a bool score
    without one, maps to ``pass``/``fail``.
    """
    if not isinstance(score, str) or not score:
        verdict = pass_ if pass_ is not None else score if isinstance(score, bool) else None
        if verdict is None:
            return None
        return 'pass' if verdict else 'fail'
    if len(score) <= _SCORE_LABEL_MAX_CHARS:
        return score
    logger.warning(
        'evaluator {!r} returned a {}-character string score; the trace label is cut to {} characters '
        '(the full value is kept in the result and in orq.evaluation.output)',
        evaluator_name,
        len(score),
        _SCORE_LABEL_MAX_CHARS,
    )
    return score[: _SCORE_LABEL_MAX_CHARS - 1] + '…'


def set_job_name_attribute(span: Span | None, job_name: str) -> None:
    """
    Set the job name attribute on a span after job execution.

    Args:
        span: The span to set the attribute on (can be None)
        job_name: The name of the job
    """
    if span is None:
        return
    span.set_attribute('orq.job_name', job_name)
