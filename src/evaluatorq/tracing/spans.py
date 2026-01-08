"""
Span creation utilities for OpenTelemetry instrumentation.

Span hierarchy:
- orq.evaluation_run (root or child of parent context)
  └── orq.job (per job per data point)
      ├── [User's instrumented code becomes child spans]
      └── orq.evaluation (per evaluator - child of its job)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .setup import get_tracer

if TYPE_CHECKING:
    from opentelemetry.trace import Span


@dataclass
class JobSpanOptions:
    """Options for creating a job span."""

    run_id: str
    row_index: int
    job_name: str | None = None
    parent_context: Any | None = None


@dataclass
class EvaluationSpanOptions:
    """Options for creating an evaluation span."""

    run_id: str
    evaluator_name: str


@asynccontextmanager
async def with_job_span(
    options: JobSpanOptions,
) -> AsyncGenerator["Span | None", None]:
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
            "orq.run_id": options.run_id,
            "orq.row_index": options.row_index,
        }
        if options.job_name:
            attributes["orq.job_name"] = options.job_name

        with tracer.start_as_current_span(
            "orq.job",
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
async def with_evaluation_span(
    options: EvaluationSpanOptions,
) -> AsyncGenerator["Span | None", None]:
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
            "orq.evaluation",
            kind=SpanKind.INTERNAL,
            attributes={
                "orq.run_id": options.run_id,
                "orq.evaluator_name": options.evaluator_name,
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
    span: "Span | None",
    score: str | int | float | bool,
    explanation: str | None = None,
    pass_: bool | None = None,
) -> None:
    """
    Set evaluation result attributes on a span.

    Args:
        span: The span to set attributes on (can be None)
        score: The evaluation score
        explanation: Optional explanation of the score
        pass_: Optional pass/fail status
    """
    if span is None:
        return

    span.set_attribute("orq.score", str(score))
    if explanation is not None:
        span.set_attribute("orq.explanation", explanation)
    if pass_ is not None:
        span.set_attribute("orq.pass", pass_)


def set_job_name_attribute(span: "Span | None", job_name: str) -> None:
    """
    Set the job name attribute on a span after job execution.

    Args:
        span: The span to set the attribute on (can be None)
        job_name: The name of the job
    """
    if span is None:
        return
    span.set_attribute("orq.job_name", job_name)
