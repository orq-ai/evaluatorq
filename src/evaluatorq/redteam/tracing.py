"""Red teaming span utilities for OpenTelemetry instrumentation.

Span hierarchy:
- orq.redteam.pipeline (root or child of parent context)      [runner.py]
  +-- orq.redteam.context_retrieval                            [runner.py]
  +-- orq.redteam.datapoint_generation                         [runner.py]
  |   +-- orq.redteam.capability_classification                [strategy_planner.py]
  |   |   +-- chat (llm_purpose=classify_tools)                [capability_classifier.py]
  |   |   +-- chat (llm_purpose=infer_resources)               [capability_classifier.py]
  |   +-- orq.redteam.strategy_planning                        [strategy_planner.py]
  |       +-- chat (llm_purpose=generate_strategies)           [objective_generator.py]
  +-- orq.job (framework)                                      [processings.py]
  |   +-- orq.redteam.attack                                   [pipeline.py]
  |   |   +-- orq.redteam.target_call                          [pipeline.py] (single-turn template)
  |   |   |   +-- agent <key> (llm_purpose=target)             [orq.py] (ORQ agent targets)
  |   |   |   +-- chat (llm_purpose=target)                    [openai.py] (OpenAI model targets)
  |   |   +-- orq.redteam.attack_turn x N                      [orchestrator.py]
  |   |       +-- orq.redteam.adversarial_generation           [orchestrator.py]
  |   |       |   +-- chat (llm_purpose=adversarial)           [orchestrator.py]
  |   |       +-- orq.redteam.target_call                      [orchestrator.py]
  |   |           +-- agent <key> (llm_purpose=target)         [orq.py] (ORQ agent targets)
  |   |           +-- chat (llm_purpose=target)                [openai.py] (OpenAI model targets)
  |   +-- orq.evaluation (framework)                           [processings.py]
  |       +-- orq.redteam.security_evaluation                  [pipeline.py]
  |           +-- chat (llm_purpose=evaluation)                [evaluator.py]
  +-- orq.redteam.memory_cleanup                               [runner.py]

LLM spans use ``SpanKind.CLIENT`` with span name ``"{operation} {model}"``
(e.g. ``"chat gpt-5-mini"``) per OTel GenAI semantic conventions and carry
``gen_ai.*`` attributes. ORQ agent target calls use ``SpanKind.INTERNAL``
with span name ``"agent {key}"`` so the platform classifies them as agent
spans (cumulative usage across internal LLM calls).
The semantic purpose is recorded in ``orq.redteam.llm_purpose``.
All other red teaming spans use ``SpanKind.INTERNAL``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from evaluatorq.tracing.setup import get_tracer

if TYPE_CHECKING:
    from opentelemetry.trace import Span


@asynccontextmanager
async def with_redteam_span(
    name: str,
    attributes: dict[str, Any] | None = None,
    parent_context: Any | None = None,
) -> AsyncGenerator["Span | None", None]:
    """Execute code within a red teaming span (``SpanKind.INTERNAL``).

    Yields the span when tracing is enabled, ``None`` otherwise.
    Exceptions propagate and are recorded on the span with ERROR status.

    Args:
        name: Span name (e.g. ``"orq.redteam.pipeline"``).
        attributes: Initial span attributes set at creation time.
        parent_context: Explicit parent OTEL context. Falls back to the
            current active context when ``None``.
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
        except Exception as e:
            span.set_attribute("error.type", type(e).__name__)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


@asynccontextmanager
async def with_llm_span(
    *,
    model: str,
    operation: str = "chat",
    provider: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    input_messages: list[Any] | None = None,
    attributes: dict[str, Any] | None = None,
    parent_context: Any | None = None,
) -> AsyncGenerator["Span | None", None]:
    """Execute code within a GenAI LLM span (``SpanKind.CLIENT``).

    Follows `OTel GenAI semantic conventions`_ for client inference spans.
    The span name is auto-derived as ``"{operation} {model}"`` per spec.

    The span is created with ``gen_ai.*`` request attributes. After the LLM
    call completes, callers should use :func:`record_llm_response` to set
    response-side attributes (tokens, finish reason, output content).

    Args:
        model: Model identifier as sent in the request
            (maps to ``gen_ai.request.model``).
        operation: GenAI operation name (default ``"chat"``).
        provider: GenAI provider name. Auto-derived from *model* when
            ``None`` (e.g. ``"azure/gpt-5-mini"`` → ``"azure"``).
        temperature: Sampling temperature if set.
        max_tokens: Max output tokens if set.
        input_messages: Chat messages sent to the model. Serialized to
            ``gen_ai.input.messages`` as a JSON string.
        attributes: Extra span attributes merged after GenAI attributes.
        parent_context: Explicit parent OTEL context.

    .. _OTel GenAI semantic conventions:
       https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
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

    resolved_provider = provider or _derive_provider(model)
    span_name = f"{operation} {model}"

    genai_attrs: dict[str, Any] = {
        "gen_ai.operation.name": operation,
        "gen_ai.system": resolved_provider,
        "gen_ai.provider.name": resolved_provider,
        "gen_ai.request.model": model,
    }
    if temperature is not None:
        genai_attrs["gen_ai.request.temperature"] = float(temperature)
    if max_tokens is not None:
        genai_attrs["gen_ai.request.max_tokens"] = max_tokens
    if input_messages is not None:
        serialized = json.dumps(
            _sanitize_messages(input_messages), ensure_ascii=False,
        )
        genai_attrs["gen_ai.input.messages"] = serialized
        genai_attrs["input"] = serialized

    if attributes:
        genai_attrs.update(attributes)

    with tracer.start_as_current_span(
        span_name,
        context=ctx,
        kind=SpanKind.CLIENT,
        attributes=genai_attrs,
    ) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_attribute("error.type", type(e).__name__)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def record_llm_response(
    span: "Span | None",
    response: Any,
    *,
    output_content: str | None = None,
) -> None:
    """Record GenAI response attributes on an LLM span.

    Extracts token usage, finish reasons, response model, and response ID
    from an OpenAI-compatible chat completion response and sets the
    corresponding ``gen_ai.*`` attributes.

    Args:
        span: The LLM span (may be ``None`` when tracing is off).
        response: OpenAI-compatible chat completion response object.
        output_content: Optional output text to record as
            ``gen_ai.output.messages``.
    """
    if span is None:
        return

    # Response metadata
    response_id = getattr(response, 'id', None)
    response_model = getattr(response, 'model', None)
    if response_id:
        span.set_attribute("gen_ai.response.id", response_id)
    if response_model:
        span.set_attribute("gen_ai.response.model", response_model)

    # Finish reasons
    choices = getattr(response, 'choices', None) or []
    finish_reasons: list[str] = [
        str(getattr(c, 'finish_reason', ''))
        for c in choices
        if getattr(c, 'finish_reason', None)
    ]
    if finish_reasons:
        span.set_attribute("gen_ai.response.finish_reasons", finish_reasons)

    # Token usage — set both OTel GenAI names and OpenAI-style names
    # so the ORQ platform can pick up whichever it checks first.
    usage = getattr(response, 'usage', None)
    if usage is not None:
        input_tokens = int(getattr(usage, 'prompt_tokens', 0) or 0)
        output_tokens = int(getattr(usage, 'completion_tokens', 0) or 0)
        raw_total = getattr(usage, 'total_tokens', None)
        total = int(raw_total) if raw_total else (input_tokens + output_tokens)
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        span.set_attribute("gen_ai.usage.prompt_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.completion_tokens", output_tokens)
        span.set_attribute("gen_ai.usage.total_tokens", total)
        # Bare keys for platform GenericAdapter extraction
        span.set_attribute("prompt_tokens", input_tokens)
        span.set_attribute("completion_tokens", output_tokens)
        span.set_attribute("input_tokens", input_tokens)
        span.set_attribute("output_tokens", output_tokens)
        span.set_attribute("total_tokens", total)

        # Cached / detailed token breakdowns
        prompt_details = getattr(usage, 'prompt_tokens_details', None)
        if prompt_details is not None:
            cached = getattr(prompt_details, 'cached_tokens', None)
            if cached is not None:
                span.set_attribute(
                    "gen_ai.usage.prompt_tokens_details.cached_tokens",
                    int(cached),
                )
        completion_details = getattr(usage, 'completion_tokens_details', None)
        if completion_details is not None:
            reasoning = getattr(completion_details, 'reasoning_tokens', None)
            if reasoning is not None:
                span.set_attribute(
                    "gen_ai.usage.completion_tokens_details.reasoning_tokens",
                    int(reasoning),
                )

    # Output content
    if output_content is not None:
        serialized = json.dumps(
            [{"role": "assistant", "content": output_content}], ensure_ascii=False,
        )
        span.set_attribute("gen_ai.output.messages", serialized)
        span.set_attribute("output", serialized)


def set_span_attrs(span: "Span | None", attrs: dict[str, Any]) -> None:
    """Set attributes on a span. Safe no-op when *span* is ``None``."""
    if span is None:
        return
    for key, value in attrs.items():
        if value is not None:
            span.set_attribute(key, value)


def record_token_usage(
    span: "Span | None",
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    calls: int = 0,
) -> None:
    """Record token usage attributes on a span. Safe no-op when *span* is ``None``."""
    if span is None:
        return
    computed_total = total_tokens if total_tokens else (prompt_tokens + completion_tokens)
    # OTel GenAI semantic convention attributes
    span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
    span.set_attribute("gen_ai.usage.prompt_tokens", prompt_tokens)
    span.set_attribute("gen_ai.usage.completion_tokens", completion_tokens)
    span.set_attribute("gen_ai.usage.total_tokens", computed_total)
    # Bare keys for platform GenericAdapter extraction
    span.set_attribute("prompt_tokens", prompt_tokens)
    span.set_attribute("completion_tokens", completion_tokens)
    span.set_attribute("input_tokens", prompt_tokens)
    span.set_attribute("output_tokens", completion_tokens)
    span.set_attribute("total_tokens", computed_total)


def _derive_provider(model: str) -> str:
    """Derive ``gen_ai.provider.name`` from a model string.

    Handles ``provider/model`` patterns (e.g. ``"azure/gpt-5-mini"`` →
    ``"azure"``). Falls back to ``"openai"`` when no prefix is present.
    """
    if '/' in model:
        return model.split('/', 1)[0]
    return "openai"


def _sanitize_messages(
    messages: list[Any],
) -> list[dict[str, str]]:
    """Produce a JSON-safe list of {role, content} dicts.

    Truncates very long content to keep span attributes bounded.
    """
    _MAX_CONTENT_LEN = 2000
    sanitized: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get('role', ''))
        content = str(msg.get('content', ''))
        if len(content) > _MAX_CONTENT_LEN:
            content = content[:_MAX_CONTENT_LEN] + '... [truncated]'
        sanitized.append({'role': role, 'content': content})
    return sanitized
