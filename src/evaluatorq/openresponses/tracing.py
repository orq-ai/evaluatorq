"""OTel span helpers for the OpenResponses runtime.

Provides with_llm_span for the Responses API call path (a thin wrapper over
common/tracing.py's builder) and record_openresponses_request/response helpers
that record the full Responses API payload alongside the gen_ai.* attributes.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.tracing import (
    capture_message_content,
    record_llm_response,
    truncate_for_span,
)
from evaluatorq.common.tracing import with_llm_span as _common_with_llm_span
from evaluatorq.openresponses.otel_messages import items_to_input_messages, items_to_output_messages

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from opentelemetry.trace import Span


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


def record_openresponses_request(span: Span | None, payload: dict[str, Any]) -> None:
    """Record a Responses API request with both generic and Orq-specific attrs."""
    if span is None:
        return
    model = payload.get('model')
    if model:
        span.set_attribute('gen_ai.request.model', str(model))
    max_output_tokens = payload.get('max_output_tokens')
    if isinstance(max_output_tokens, int):
        span.set_attribute('gen_ai.request.max_tokens', max_output_tokens)
    reasoning = payload.get('reasoning')
    effort = reasoning.get('effort') if isinstance(reasoning, dict) else None
    if effort:
        # Unconditional: without it two runs at different efforts are indistinguishable
        # when EVALUATORQ_CAPTURE_MESSAGE_CONTENT=false.
        span.set_attribute('gen_ai.request.reasoning_effort', str(effort))
    if not capture_message_content():
        return
    input_items = payload.get('input') or []
    # gen_ai.* gets the parts shape (see otel_messages): a reader that keys on
    # `role` drops the role-less function_call / reasoning items otherwise.
    serialized_messages = truncate_for_span(
        json.dumps(items_to_input_messages(input_items), ensure_ascii=False, default=str)
    )
    span.set_attribute('gen_ai.input.messages', serialized_messages)
    span.set_attribute('input', serialized_messages)
    # Raw items, verbatim, under the key Orq's own gateway uses for them.
    span.set_attribute(
        'openresponses.input',
        truncate_for_span(json.dumps(input_items, ensure_ascii=False, default=str)),
    )
    instructions = payload.get('instructions')
    if instructions:
        span.set_attribute('gen_ai.system_instructions', truncate_for_span(instructions))
        span.set_attribute('openresponses.instructions', truncate_for_span(instructions))
    span.set_attribute(
        'orq.openresponses.request',
        truncate_for_span(json.dumps(payload, ensure_ascii=False, default=str)),
    )


def record_openresponses_response(span: Span | None, response: Any) -> None:
    """Record a Responses API response with standard gen_ai.* attributes."""
    if span is None:
        return
    record_llm_response(span, response)
    try:
        # warnings=False: Response.output is a non-discriminated union, so a
        # partially-constructed item makes Pydantic try every member and warn
        # once each. This is a best-effort diagnostic dump — silence the spam.
        payload = response.model_dump(mode='json', warnings=False) if hasattr(response, 'model_dump') else response
    except Exception as exc:
        logger.debug(
            'record_openresponses_response: model_dump failed ({}); falling back to repr',
            exc,
        )
        # Wrap in a dict so json.dumps produces {"repr": "..."} rather than
        # double-encoding a bare string into '"..."' (extra outer quotes).
        payload = {'repr': repr(response)}
    if capture_message_content():
        span.set_attribute(
            'orq.openresponses.response',
            truncate_for_span(json.dumps(payload, ensure_ascii=False, default=str)),
        )
        output_items = payload.get('output') if isinstance(payload, dict) else None
        if output_items is not None:
            span.set_attribute(
                'openresponses.output',
                truncate_for_span(json.dumps(output_items, ensure_ascii=False, default=str)),
            )
            # Overwrites the flat {role, content} pair record_llm_response just
            # wrote: same messages, parts shape, tool calls intact.
            finish_reason = str(payload.get('status') or '') if isinstance(payload, dict) else ''
            serialized_output = truncate_for_span(
                json.dumps(
                    items_to_output_messages(output_items, finish_reason),
                    ensure_ascii=False,
                    default=str,
                )
            )
            span.set_attribute('gen_ai.output.messages', serialized_output)
            span.set_attribute('output', serialized_output)
