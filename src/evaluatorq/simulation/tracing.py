"""OpenTelemetry tracing utilities for the agent simulation module.

Provides span creation helpers that mirror the TypeScript simulation module's
tracing patterns. All functions gracefully degrade to no-ops when tracing is
not enabled.

Span hierarchy:
    orq.simulation.pipeline (root)
      ├── orq.simulation.run (per datapoint)
      │   ├── orq.simulation.first_message_generation
      │   └── orq.simulation.turn (per turn)
      │       ├── orq.simulation.target_call
      │       ├── orq.simulation.judge_evaluation
      │       └── orq.simulation.user_simulator_call
      └── chat/responses {model}  (LLM client spans, GenAI semconv)
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from evaluatorq.tracing.setup import get_tracer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from opentelemetry.trace import Span


AttrValue = str | int | float | bool
AttrMap = dict[str, AttrValue | None]

# Max content length per message to avoid oversized spans (matches TS).
MAX_CONTENT_LEN = 2000


# ---------------------------------------------------------------------------
# Internal span: orq.simulation.*
# ---------------------------------------------------------------------------


@asynccontextmanager
async def with_simulation_span(  # noqa: RUF029
    name: str,
    attributes: AttrMap | None = None,
) -> AsyncGenerator[Span | None, None]:
    """Execute a block within a simulation span (SpanKind.INTERNAL).

    Yields ``None`` when tracing is not enabled. Records exceptions and sets
    span status automatically.
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    try:
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except ImportError:
        yield None
        return

    clean_attrs: dict[str, AttrValue] = {
        k: v for k, v in (attributes or {}).items() if v is not None
    }

    with tracer.start_as_current_span(
        name,
        kind=SpanKind.INTERNAL,
        attributes=clean_attrs,
    ) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except BaseException as e:
            # Catch BaseException so asyncio.CancelledError (used by
            # asyncio.wait_for / timeouts) is recorded on the span instead
            # of leaving it with UNSET status. KeyboardInterrupt / SystemExit
            # are also surfaced — desirable for trace visibility.
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.set_attribute("error.type", type(e).__name__)
            raise


# ---------------------------------------------------------------------------
# LLM span: GenAI semantic conventions (SpanKind.CLIENT)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def with_llm_span(  # noqa: RUF029
    *,
    model: str,
    operation: str = "chat",
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    purpose: str | None = None,
) -> AsyncGenerator[Span | None, None]:
    """Execute a block within a GenAI LLM span (SpanKind.CLIENT).

    Follows OTel GenAI semantic conventions for client inference spans. Span
    name is ``"{operation} {model}"``. Use ``operation="chat"`` for Chat
    Completions and ``operation="responses"`` for the OpenAI Responses API.
    """
    tracer = get_tracer()
    if tracer is None:
        yield None
        return

    try:
        from opentelemetry.trace import SpanKind, Status, StatusCode
    except ImportError:
        yield None
        return

    resolved_provider = provider or _derive_provider(model)
    span_name = f"{operation} {model}"

    attrs: dict[str, AttrValue] = {
        "gen_ai.operation.name": operation,
        "gen_ai.system": resolved_provider,
        "gen_ai.provider.name": resolved_provider,
        "gen_ai.request.model": model,
    }
    if temperature is not None:
        attrs["gen_ai.request.temperature"] = temperature
    if max_tokens is not None:
        attrs["gen_ai.request.max_tokens"] = max_tokens
    if purpose:
        attrs["orq.simulation.llm_purpose"] = purpose

    with tracer.start_as_current_span(
        span_name,
        kind=SpanKind.CLIENT,
        attributes=attrs,
    ) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except BaseException as e:
            # See with_simulation_span for why BaseException (asyncio
            # cancellation visibility on timeout).
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.set_attribute("error.type", type(e).__name__)
            raise


# ---------------------------------------------------------------------------
# Token usage recording
# ---------------------------------------------------------------------------


def record_token_usage(
    span: Span | None,
    *,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
) -> None:
    """Record token usage attributes on a span.

    Sets both OTel GenAI names and bare attribute keys for platform
    compatibility (matches TS dual-naming convention).
    """
    if span is None:
        return

    prompt = prompt_tokens or 0
    completion = completion_tokens or 0
    total = total_tokens if total_tokens is not None else prompt + completion

    span.set_attribute("gen_ai.usage.input_tokens", prompt)
    span.set_attribute("gen_ai.usage.output_tokens", completion)
    span.set_attribute("gen_ai.usage.total_tokens", total)

    if cache_read_input_tokens is not None:
        span.set_attribute(
            "gen_ai.usage.cache_read.input_tokens", cache_read_input_tokens
        )
    if cache_creation_input_tokens is not None:
        span.set_attribute(
            "gen_ai.usage.cache_creation.input_tokens", cache_creation_input_tokens
        )

    # Aliases for platform compatibility
    span.set_attribute("gen_ai.usage.prompt_tokens", prompt)
    span.set_attribute("gen_ai.usage.completion_tokens", completion)
    span.set_attribute("prompt_tokens", prompt)
    span.set_attribute("completion_tokens", completion)
    span.set_attribute("input_tokens", prompt)
    span.set_attribute("output_tokens", completion)
    span.set_attribute("total_tokens", total)


# ---------------------------------------------------------------------------
# Message recording (gen_ai.input/output.messages)
# ---------------------------------------------------------------------------


def _truncate(text: str) -> str:
    if len(text) <= MAX_CONTENT_LEN:
        return text
    return f"{text[:MAX_CONTENT_LEN]}…"


def _serialize_messages(messages: list[dict[str, Any]]) -> str:
    return json.dumps(
        [
            {
                "role": str(m.get("role", "")),
                "content": _truncate(str(m.get("content") or "")),
            }
            for m in messages
        ]
    )


def _capture_message_content() -> bool:
    """Honor ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` env flag.

    Deliberate deviation from the OTel GenAI semconv: the spec classifies
    ``gen_ai.input.messages`` and ``gen_ai.output.messages`` as opt-in
    (default ``false``) due to PII risk. We default to ``True`` to match
    the TypeScript implementation (RES-595) so the Orq dashboard's input/
    output panels keep rendering. Set
    ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false`` to opt out
    if traces will be exported to a third-party backend.
    """
    flag = os.environ.get("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT")
    if flag is None:
        return True
    return flag.lower() == "true" or flag == "1"


def record_llm_input(
    span: Span | None, messages: list[dict[str, str]]
) -> None:
    """Record LLM input messages on a span.

    Sets both ``gen_ai.input.messages`` (OTel GenAI convention) and ``input``
    (platform fallback). Suppressed when
    ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false``.
    """
    if span is None or not messages:
        return
    if not _capture_message_content():
        return

    serialized = _serialize_messages(messages)
    span.set_attribute("gen_ai.input.messages", serialized)
    span.set_attribute("input", serialized)


def record_llm_output(span: Span | None, output: str) -> None:
    """Record a single LLM output string on a span."""
    if span is None or not output:
        return
    if not _capture_message_content():
        return

    serialized = _serialize_messages([{"role": "assistant", "content": output}])
    span.set_attribute("gen_ai.output.messages", serialized)
    span.set_attribute("output", serialized)


def record_llm_response(span: Span | None, response: Any) -> None:
    """Record LLM response attributes on a span from an OpenAI response.

    Handles both Chat Completions and Responses API objects (duck-typed).
    Sets ``gen_ai.output.messages``, ``output``, token usage, finish reasons,
    and response metadata.
    """
    if span is None:
        return

    response_id = getattr(response, "id", None)
    if response_id:
        span.set_attribute("gen_ai.response.id", response_id)
    response_model = getattr(response, "model", None)
    if response_model:
        span.set_attribute("gen_ai.response.model", response_model)

    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt = getattr(usage, "prompt_tokens", None)
        if prompt is None:
            prompt = getattr(usage, "input_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        if completion is None:
            completion = getattr(usage, "output_tokens", None)
        total = getattr(usage, "total_tokens", None)
        details = getattr(usage, "prompt_tokens_details", None)
        cache_read = getattr(details, "cached_tokens", None) if details else None
        record_token_usage(
            span,
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            cache_read_input_tokens=cache_read,
        )

    if _capture_message_content():
        output_messages: list[dict[str, str]] = []
        choices = getattr(response, "choices", None)
        if choices:
            for choice in choices:
                message = getattr(choice, "message", None)
                content = getattr(message, "content", None) if message else None
                if content:
                    role = getattr(message, "role", None) or "assistant"
                    output_messages.append({"role": role, "content": content})
        else:
            # Responses API: response.output is a list of typed items.
            # Prefer the SDK helper response.output_text when available,
            # else descend into item.content[*].text for "message" items
            # and fall back to a flat .text attribute for older shapes.
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str) and output_text:
                output_messages.append(
                    {"role": "assistant", "content": output_text}
                )
            else:
                output_items = getattr(response, "output", None) or []
                parts: list[str] = []
                for item in output_items:
                    content = getattr(item, "content", None)
                    if content:
                        for part in content:
                            text = getattr(part, "text", None)
                            if isinstance(text, str) and text:
                                parts.append(text)
                    else:
                        text = getattr(item, "text", None)
                        if isinstance(text, str) and text:
                            parts.append(text)
                joined = "".join(parts)
                if joined:
                    output_messages.append(
                        {"role": "assistant", "content": joined}
                    )

        if output_messages:
            serialized = _serialize_messages(output_messages)
            span.set_attribute("gen_ai.output.messages", serialized)
            span.set_attribute("output", serialized)

    finish_reasons: list[str] = []
    finish_choices = getattr(response, "choices", None)
    if finish_choices:
        for choice in finish_choices:
            reason = getattr(choice, "finish_reason", None)
            if reason:
                finish_reasons.append(reason)
    if finish_reasons:
        span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)


# ---------------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------------


def set_span_attrs(span: Span | None, attrs: AttrMap) -> None:
    """Batch set multiple attributes on a span. Skips ``None`` values."""
    if span is None:
        return
    for key, value in attrs.items():
        if value is not None:
            span.set_attribute(key, value)


async def get_trace_context_headers() -> dict[str, str]:  # noqa: RUF029
    """Get W3C trace context headers for the current active span.

    Returns an empty dict when tracing is not available. Used to propagate
    trace context into outgoing HTTP requests so the router can create child
    spans under the current simulation span.
    """
    try:
        from opentelemetry import context, propagate
    except ImportError:
        return {}

    headers: dict[str, str] = {}
    propagate.inject(headers, context=context.get_current())
    return headers


# ---------------------------------------------------------------------------
# Provider derivation
# ---------------------------------------------------------------------------

# OTel GenAI semconv ``gen_ai.system`` enum aliases. The router uses prefixes
# like "azure/" that don't map 1:1 to the spec — translate the known ones.
_PROVIDER_ALIASES: dict[str, str] = {
    "azure": "azure.ai.openai",
}


def _derive_provider(model: str) -> str:
    if "/" in model:
        prefix = model.split("/", 1)[0]
        return _PROVIDER_ALIASES.get(prefix, prefix)
    return "openai"
