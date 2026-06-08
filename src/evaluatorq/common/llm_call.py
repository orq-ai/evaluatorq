"""Domain-neutral chat-completion mechanic shared by the redteam judge and the
simulation BaseAgent.

Owns ONLY: params assembly, input/response span recording, W3C trace-header
injection, the timed ``create`` call, and token-usage extraction. Does NOT own the
span (caller opens its own domain ``with_llm_span`` and passes it in), retry (caller
wraps with ``with_retry`` if desired), or parsing/result-shaping.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from evaluatorq.common.tracing import (
    get_trace_context_headers,
    record_llm_input,
    record_llm_response,
)
from evaluatorq.contracts import TokenUsage

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion
    from opentelemetry.trace import Span


async def execute_chat_completion(
    *,
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    span: Span | None,
    timeout_s: float,
    temperature: float | None = None,
    max_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
    inject_trace_headers: bool = True,
    extra_kwargs: dict[str, Any] | None = None,
) -> tuple[ChatCompletion, TokenUsage | None]:
    """Execute one Chat Completions call. Records input/response on ``span``.

    Returns the raw response and the token-usage delta (or None). Exceptions
    propagate — the caller owns retry and error policy.
    """
    params: dict[str, Any] = {'model': model, 'messages': messages}
    if temperature is not None:
        params['temperature'] = temperature
    if max_tokens is not None:
        params['max_tokens'] = max_tokens
    if max_completion_tokens is not None:
        params['max_completion_tokens'] = max_completion_tokens
    if tools:  # truthiness (not `is not None`) for parity with BaseAgent
        params['tools'] = tools
        params['tool_choice'] = 'auto'
    if response_format is not None:
        params['response_format'] = response_format
    if extra_kwargs:
        params.update(extra_kwargs)

    record_llm_input(span, messages)

    if inject_trace_headers:
        headers = await get_trace_context_headers()
        if headers:
            # Merge, don't overwrite: a caller may have supplied extra_headers via
            # extra_kwargs. Trace headers win on conflict.
            existing = params.get('extra_headers') or {}
            params['extra_headers'] = {**existing, **headers}

    response = await asyncio.wait_for(client.chat.completions.create(**params), timeout=timeout_s)
    record_llm_response(span, response)
    return response, TokenUsage.from_completion(response)
