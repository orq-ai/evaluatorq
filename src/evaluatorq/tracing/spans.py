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

from evaluatorq.types import EvaluationResultCell

from .setup import get_tracer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from opentelemetry.trace import Span

# Orq's trace ingestion drops (not truncates) any dynamic string span attribute
# longer than this (genai_traces.go: dynamicAttributeMaxStringLength). Keep the
# span copy of evaluator explanations at or under it so they always render.
_SPAN_TEXT_MAX_CHARS = 512


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


@asynccontextmanager
async def with_run_span(  # noqa: RUF029
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
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    try:
        from opentelemetry import context as otel_context
        from opentelemetry.trace import SpanKind, Status, StatusCode

        parent_ctx = options.parent_context or otel_context.get_current()

        with tracer.start_as_current_span(
            'evaluatorq.run',
            context=parent_ctx,
            kind=SpanKind.INTERNAL,
            attributes={
                'orq.trace_type': options.trace_type,
                'orq.run_id': options.run_id,
                'orq.run_name': options.run_name,
                # Same key the red-team / simulation roots stamp, so one query
                # finds the root of any evaluatorq run whatever the surface.
                'orq.evaluatorq_run_id': options.run_id,
            },
        ) as span:
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    except ImportError:
        # OTEL not available, run without span
        yield None


@asynccontextmanager
async def with_job_span(  # noqa: RUF029
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
        async with with_job_span(JobSpanOptions(run_id="abc", row_index=0)) as span:
            # Your job code here
            pass
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    try:
        from opentelemetry import context as otel_context
        from opentelemetry.trace import SpanKind, Status, StatusCode

        # Use parent context if provided, otherwise use active context
        parent_ctx = options.parent_context or otel_context.get_current()

        attributes: dict[str, Any] = {
            'orq.trace_type': options.trace_type,
            'orq.run_id': options.run_id,
            'orq.row_index': options.row_index,
        }
        if options.job_name:
            attributes['orq.job_name'] = options.job_name

        with tracer.start_as_current_span(
            'orq.job',
            context=parent_ctx,
            kind=SpanKind.INTERNAL,
            attributes=attributes,
        ) as span:
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    except ImportError:
        # OTEL not available, run without span
        yield None


@asynccontextmanager
async def with_evaluation_span(  # noqa: RUF029
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
        async with with_evaluation_span(EvaluationSpanOptions(
            run_id="abc",
            evaluator_name="string-contains"
        )) as span:
            # Your evaluator code here
            pass
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    try:
        from opentelemetry.trace import SpanKind, Status, StatusCode

        with tracer.start_as_current_span(
            # Include the evaluator name so concurrent evaluator spans are
            # distinguishable in the trace tree / UI (mirrors `chat {model}`)
            # rather than N identical `orq.evaluation` rows.
            f'orq.evaluation {options.evaluator_name}',
            kind=SpanKind.INTERNAL,
            attributes={
                'orq.run_id': options.run_id,
                'orq.evaluator_name': options.evaluator_name,
            },
        ) as span:
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

    except ImportError:
        # OTEL not available, run without span
        yield None


def set_evaluation_attributes(
    span: Span | None,
    score: str | float | bool | dict[str, Any] | EvaluationResultCell,  # noqa: FBT001
    *,
    explanation: str | None = None,
    pass_: bool | None = None,
    evaluator_name: str | None = None,
    evaluator_type: str | None = None,
) -> None:
    """
    Set evaluation result attributes on a span.

    Args:
        span: The span to set attributes on (can be None)
        score: The evaluation score
        explanation: Optional explanation of the score
        pass_: Optional pass/fail status
        evaluator_name: Name of the evaluator (used for the gen_ai.evaluation.* block)
        evaluator_type: Opt-in evaluator kind, matching Orq's EvaluatorType enum
            (e.g. "llm_eval", "python_eval"). When provided, the flat
            gen_ai.evaluation.* / orq.evaluator.* attributes plus the
            orq.evaluation.output verdict payload the Orq trace UI classifies +
            renders evaluator spans from are additionally emitted.
            When ``None`` (the default), only the legacy orq.score/explanation/pass
            attributes are written, so red-team spans are unchanged.
    """
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
    span.set_attribute('orq.span_type', 'span.evaluator')
    span.set_attribute('orq.evaluator.type', evaluator_type)
    if evaluator_name is not None:
        span.set_attribute('gen_ai.evaluation.name', evaluator_name)
        span.set_attribute('orq.evaluator.key', evaluator_name)

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
    if pass_ is not None:
        span.set_attribute('gen_ai.evaluation.passed', pass_)
        # Orq's own graders emit pass/fail here, not true/false.
        label = 'pass' if pass_ else 'fail'
        span.set_attribute('gen_ai.evaluation.score.label', label)
        span.set_attribute('orq.evaluator.score.label', label)

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
    span.set_attribute('orq.evaluation.output', json.dumps(verdict, default=str))


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
