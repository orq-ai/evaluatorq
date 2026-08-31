"""Red teaming span utilities for OpenTelemetry instrumentation.

Domain-specific span builders (with_redteam_span, with_llm_span,
annotate_current_span) live here. Generic recording utilities are imported from
evaluatorq.common.tracing.

Span hierarchy:

    dynamic/hybrid: pipeline → context/datapoint work → job → attack
                    → target_call/attack_turn → evaluation
                    → recommendations/executive_summary
    static:         pipeline → job → attack → target_call → evaluation
                    → recommendations/executive_summary

The evaluation span is the evaluatorq framework's ``orq.evaluation`` evaluator
span; the OWASP scorer annotates it in place (via ``annotate_current_span``)
rather than nesting a separate ``security_evaluation`` child under it.

Static is intentionally single-shot: it has no multi-turn ``attack_turn`` or
adversarial-generation spans. Dynamic and hybrid runs add their context/datapoint
work and may include target calls both before and during multi-turn attacks.

The trees above are lossy: leaf LLM spans (capability_classification,
strategy_planning, adversarial_generation) are elided, and ``memory_cleanup``
is emitted as a direct child of ``pipeline`` after all attacks complete.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from evaluatorq.common.tracing import set_span_attrs
from evaluatorq.common.tracing import with_llm_span as _common_with_llm_span
from evaluatorq.tracing.setup import get_tracer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from opentelemetry.trace import Span

    from evaluatorq.contracts import JuryResult


def set_jury_span_attrs(span: Span | None, jury: JuryResult | None) -> None:
    """Record panel-of-judges reliability on the evaluator span.

    Emits the flat ``orq.redteam.jury.*`` attributes (dotted keys the Orq trace
    UI nests back into an object) *and* a JSON ``metadata`` attribute so the
    breakdown surfaces as a visible field on the span rather than only living in
    the raw attribute list. No-op when no jury ran.
    """
    if span is None or jury is None:
        return
    set_span_attrs(
        span,
        {
            'orq.redteam.jury.judges_configured': jury.judges_configured,
            'orq.redteam.jury.judges_succeeded': jury.judges_succeeded,
            'orq.redteam.jury.judges_failed': jury.judges_failed,
            'orq.redteam.jury.replacements_used': jury.replacements_used,
            'orq.redteam.jury.raw_agreement': jury.raw_agreement,
            'orq.redteam.jury.tie': jury.tie,
            'orq.redteam.jury.inconclusive': jury.inconclusive,
            # Visible metadata field (JSON — OTel attrs can't hold nested objects).
            'metadata': json.dumps({
                'jury': {
                    'judges_configured': jury.judges_configured,
                    'judges_succeeded': jury.judges_succeeded,
                    'judges_failed': jury.judges_failed,
                    'replacements_used': jury.replacements_used,
                    'raw_agreement': jury.raw_agreement,
                    'tie': jury.tie,
                    'inconclusive': jury.inconclusive,
                }
            }),
        },
    )


@asynccontextmanager
async def with_redteam_span(  # noqa: RUF029
    name: str,
    attributes: dict[str, Any] | None = None,
    parent_context: Any | None = None,
) -> AsyncGenerator[Span | None, None]:
    """Execute code within a red teaming span (SpanKind.INTERNAL).

    Exceptions propagate and are recorded on the span with ERROR status.

    Yields:
        The span when tracing is enabled, None otherwise.
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    try:
        from opentelemetry import context as otel_context
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except ImportError:
        yield None
        return

    ctx = parent_context or otel_context.get_current()

    with tracer.start_as_current_span(
        name,
        context=ctx,
        kind=SpanKind.INTERNAL,
        attributes=attributes or {},
    ) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except BaseException as e:
            span.set_attribute('error.type', type(e).__name__)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


@asynccontextmanager
async def annotate_current_span(  # noqa: RUF029
    attributes: dict[str, Any] | None = None,
) -> AsyncGenerator[Span | None, None]:
    """Tag the *current* span (no new child) and yield it.

    Used by evaluator scorers: the evaluatorq framework already runs them inside
    the ``orq.evaluation`` evaluator span (via ``start_as_current_span``), which
    carries the verdict/score/explanation. Annotating that span directly avoids a
    redundant ``orq.redteam.security_evaluation`` layer between it and the judge's
    own LLM span (``responses`` by default, ``chat`` on fallback).

    Yields:
        The current span when tracing is enabled, None otherwise.
    """
    try:
        from opentelemetry.trace import get_current_span
    except ImportError:
        yield None
        return
    span = get_current_span()
    set_span_attrs(span, attributes or {})
    yield span


@asynccontextmanager
async def with_llm_span(
    *,
    model: str,
    operation: str = 'chat',
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    input_messages: list[Any] | None = None,
    attributes: dict[str, Any] | None = None,
    parent_context: Any | None = None,
) -> AsyncGenerator[Span | None, None]:
    """Execute code within a GenAI LLM span (SpanKind.CLIENT).

    Delegates to ``evaluatorq.common.tracing.with_llm_span`` after mapping the
    redteam-specific ``orq.redteam.llm_purpose`` key onto the neutral
    ``orq.llm.purpose`` key so cross-domain purpose queries include redteam spans.

    Yields:
        The active span, or None when tracing is disabled.
    """
    # Map the redteam-domain key onto the neutral key before passing attributes
    # to common. This is the only redteam-specific behaviour; common must not do it.
    resolved_attrs: dict[str, Any] = dict(attributes or {})
    redteam_purpose = resolved_attrs.get('orq.redteam.llm_purpose')
    if redteam_purpose is not None and 'orq.llm.purpose' not in resolved_attrs:
        resolved_attrs['orq.llm.purpose'] = redteam_purpose

    async with _common_with_llm_span(
        model=model,
        operation=operation,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        input_messages=input_messages,
        attributes=resolved_attrs,
        parent_context=parent_context,
    ) as span:
        yield span
