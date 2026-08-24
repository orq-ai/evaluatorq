"""OTel span helpers for the agent simulation module.

with_simulation_span is domain-specific; with_llm_span is a thin wrapper over
the canonical builder in evaluatorq.common.tracing.

Span hierarchy:
    Evaluatorq - Agent Simulation (root)
      ├── orq.simulation.first_message_generation (once, whole persona x scenario sweep)
      ├── orq.simulation.run (per datapoint)
      │   ├── orq.simulation.first_message_generation (only when not pre-generated)
      │   └── orq.simulation.turn (per turn)
      │       ├── orq.simulation.target_call
      │       ├── orq.simulation.judge_evaluation
      │       └── orq.simulation.user_simulator_call
      └── chat/responses {model}  (LLM client spans, GenAI semconv)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from loguru import logger

from evaluatorq.common.tracing import with_llm_span as _common_with_llm_span
from evaluatorq.contracts import content_to_text
from evaluatorq.tracing.setup import get_tracer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from opentelemetry.trace import Span

    from evaluatorq.common.tracing import AttrMap, AttrValue
    from evaluatorq.contracts import ContentPart

_otel_import_warned = False


def span_message_text(content: str | list[ContentPart] | None) -> str:
    """Flatten one message's content for a span attribute.

    Delegates to the canonical `content_to_text` (CLAUDE.md: never ``str()`` a
    ``str | list[ContentPart]`` — it lands on the span as a Python repr). The
    Responses path legitimately carries image/file parts, which that helper
    refuses; tracing must never break a run, so those degrade to a placeholder
    naming the part types, with a warning, rather than raising.

    Lives here rather than beside either caller so the agent LLM spans and the
    runner's target-call span cannot drift into two different renderings of the
    same message.
    """
    try:
        return content_to_text(content)
    except NotImplementedError:
        parts = content if isinstance(content, list) else []
        types = ', '.join(sorted({str(p.type) for p in parts})) or 'unknown'
        logger.warning(
            'Span input: non-text content part(s) ({}) recorded as a placeholder; the span shows no text for them.',
            types,
        )
        return f'<non-text content: {types}>'


@asynccontextmanager
async def with_simulation_span(  # noqa: RUF029
    name: str,
    attributes: AttrMap | None = None,
) -> AsyncGenerator[Span | None, None]:
    """Execute a block within a simulation span (SpanKind.INTERNAL).

    Records exceptions (including asyncio.CancelledError) and sets span status.

    Yields:
            The active span, or None when tracing is disabled.
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    try:
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except ImportError as exc:
        global _otel_import_warned
        if not _otel_import_warned:
            logger.warning('OpenTelemetry import failed; tracing disabled: %s', exc)
            _otel_import_warned = True
        yield None
        return

    clean_attrs: dict[str, AttrValue] = {k: v for k, v in (attributes or {}).items() if v is not None}

    with tracer.start_as_current_span(
        name,
        kind=SpanKind.INTERNAL,
        attributes=clean_attrs,
    ) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except BaseException as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.set_attribute('error.type', type(e).__name__)
            raise


@asynccontextmanager
async def with_llm_span(
    *,
    model: str,
    operation: str = 'chat',
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    purpose: str | None = None,
) -> AsyncGenerator[Span | None, None]:
    """Execute a block within a GenAI LLM span (SpanKind.CLIENT).

    Delegates to ``evaluatorq.common.tracing.with_llm_span`` after mapping
    ``purpose`` onto the neutral ``orq.llm.purpose`` key and the legacy
    ``orq.simulation.llm_purpose`` kept for dashboard back-compat.

    Yields:
        The active span, or None when tracing is disabled.
    """
    attributes = {'orq.llm.purpose': purpose, 'orq.simulation.llm_purpose': purpose} if purpose else None
    async with _common_with_llm_span(
        model=model,
        operation=operation,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        attributes=attributes,
    ) as span:
        yield span
