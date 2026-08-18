"""Domain-neutral chat-completion mechanic shared by the redteam judge and the
simulation BaseAgent.

Owns ONLY: params assembly, input/response span recording, W3C trace-header
injection, the timed ``create`` call, and token-usage extraction. Does NOT own the
span (caller opens its own domain ``with_llm_span`` and passes it in), retry (caller
wraps with ``with_retry`` if desired), or parsing/result-shaping.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from openai import BadRequestError

from evaluatorq.common.llm_limit import llm_slot
from evaluatorq.common.model_catalogue import price_usage
from evaluatorq.common.thread_context import pipeline_metadata
from evaluatorq.common.tracing import (
    get_trace_context_headers,
    record_llm_input,
    record_llm_response,
)
from evaluatorq.contracts import TokenUsage

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion, ParsedChatCompletion
    from opentelemetry.trace import Span
    from pydantic import BaseModel

logger = logging.getLogger(__name__)

# (model, has_tools) pairs that 400 on `reasoning_effort` (e.g. gpt-5.4-mini on
# /v1/chat/completions rejects it only when function tools are present).
# Populated on the first rejection so we strip the param up front thereafter
# instead of re-paying a 400 + retry — and an orphaned error trace — on every
# subsequent call. Keyed on has_tools too: a tools-only rejection must not strip
# reasoning_effort from a later tool-free call that would accept it.
# ponytail: process-lifetime set, fine for a CLI run; not persisted across processes.
_REASONING_EFFORT_REJECTORS: set[tuple[str, bool]] = set()

# Same idea for the Responses API, where the reasoning param is the nested
# `reasoning={'effort': ...}` block rather than the flat `reasoning_effort`.
# Kept as a separate memo so the two param shapes never cross-strip.
_RESPONSES_REASONING_REJECTORS: set[tuple[str, bool]] = set()


def reset_reasoning_rejectors() -> None:
    """Clear the process-lifetime rejection memos; exists for test isolation."""
    _REASONING_EFFORT_REJECTORS.clear()
    _RESPONSES_REASONING_REJECTORS.clear()


def _reasoning_key(model: str, params: dict[str, Any]) -> tuple[str, bool]:
    """Memo key: the model paired with whether this call carries function tools."""
    return (model, bool(params.get('tools')))


def _strip_known_rejected_reasoning(model: str, params: dict[str, Any]) -> None:
    """Drop `reasoning_effort` before the call if this (model, has_tools) rejected it."""
    if _reasoning_key(model, params) in _REASONING_EFFORT_REJECTORS:
        params.pop('reasoning_effort', None)


def _is_reasoning_effort_rejection(params: dict[str, Any], exc: BadRequestError) -> bool:
    """True if `exc` is the reasoning_effort-unsupported 400 for this call."""
    err_body = str(getattr(exc, 'body', None) or getattr(exc, 'message', '') or '').lower()
    return 'reasoning_effort' in params and 'reasoning' in err_body


def strip_known_rejected_responses_reasoning(model: str, params: dict[str, Any]) -> None:
    """Drop the Responses `reasoning` block up front if this model already rejected it.

    Mirrors ``_strip_known_rejected_reasoning`` for the Responses API so a
    non-reasoning model (e.g. gpt-4o-mini) pays the 400 + retry once per process
    instead of on every judge / user-simulator call.
    """
    if _reasoning_key(model, params) in _RESPONSES_REASONING_REJECTORS:
        params.pop('reasoning', None)


def is_responses_reasoning_rejection(params: dict[str, Any], exc: BadRequestError) -> bool:
    """True if ``exc`` is the reasoning-unsupported 400 for this Responses call."""
    err_body = str(getattr(exc, 'body', None) or getattr(exc, 'message', '') or '').lower()
    return 'reasoning' in params and 'reasoning' in err_body


def remember_responses_reasoning_rejection(model: str, params: dict[str, Any]) -> None:
    """Memoize that ``model`` (with this tool shape) rejects the Responses reasoning block."""
    _RESPONSES_REASONING_REJECTORS.add(_reasoning_key(model, params))


def apply_pipeline_metadata(params: dict[str, Any]) -> None:
    """Tag the invocation with the active run surface + run id via ``metadata``.

    Mutates ``params`` in place; no-op when no run is bound. Caller-supplied
    metadata (via ``extra_kwargs``) wins on key conflict. Public: direct
    ``create()`` sites that build their own kwargs dict (structured output,
    first-message generation) call this instead of routing through
    `execute_chat_completion`.

    Sent regardless of endpoint. ``metadata`` is a first-class field on OpenAI's
    own Chat Completions / Responses APIs, so it is safe off-Orq — unlike the
    router-only ``thread`` body param, which callers still gate on
    ``client_routes_through_orq``. Only set when non-empty: an explicit
    ``metadata=None`` would serialize as ``"metadata": null`` rather than being
    stripped from the body.
    """
    md = pipeline_metadata()
    if md:
        params['metadata'] = {**md, **(params.get('metadata') or {})}


async def apply_trace_headers(params: dict[str, Any]) -> None:
    """Merge the active trace context into ``params['extra_headers']`` in place.

    Merge, don't overwrite: a caller may have supplied extra_headers via
    extra_kwargs. Trace headers win on conflict. No-op when no trace is bound.
    """
    headers = await get_trace_context_headers()
    if headers:
        params['extra_headers'] = {**(params.get('extra_headers') or {}), **headers}


async def _bounded_call(request: Awaitable[Any], timeout_s: float) -> Any:
    """Await one provider request holding an LLM slot.

    ``request`` must be an un-awaited coroutine so no work starts before the slot is
    held. The timeout runs inside the slot, so queueing never counts against it.
    """
    async with llm_slot():
        return await asyncio.wait_for(request, timeout=timeout_s)


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

    _strip_known_rejected_reasoning(model, params)
    apply_pipeline_metadata(params)

    record_llm_input(span, messages)

    if inject_trace_headers:
        await apply_trace_headers(params)

    try:
        response = await _bounded_call(client.chat.completions.create(**params), timeout_s)
    except BadRequestError as exc:
        # "where possible": endpoints that don't support reasoning_effort 400 on
        # it — drop the param and retry once rather than failing the call. Gate on
        # the error body so an unrelated 400 (bad tools schema, context length, …)
        # isn't silently masked by a reasoning-stripped retry. Remember the model
        # so later calls strip the param up front (see _strip_known_rejected_reasoning).
        if not _is_reasoning_effort_rejection(params, exc):
            raise
        _REASONING_EFFORT_REJECTORS.add(_reasoning_key(model, params))
        logger.warning('Model %s rejected reasoning_effort; dropping it and retrying once', model)
        params.pop('reasoning_effort', None)
        response = await _bounded_call(client.chat.completions.create(**params), timeout_s)
    record_llm_response(span, response)
    # The Orq router prices Responses but not Chat Completions, so usage here carries
    # tokens only; price it from the model catalogue (RES-1295).
    return response, await price_usage(TokenUsage.from_completion(response), model, client)


async def execute_chat_parse(
    *,
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    span: Span | None,
    timeout_s: float,
    response_model: type[BaseModel],
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    inject_trace_headers: bool = True,
    extra_kwargs: dict[str, Any] | None = None,
) -> tuple[ParsedChatCompletion[Any], TokenUsage | None]:
    """Execute one structured Chat Completions call via ``.parse``.

    Mirrors `execute_chat_completion` (span recording, trace headers,
    reasoning_effort drop-retry) but routes through ``client.chat.completions.parse``
    with a Pydantic ``response_model``. The parsed object is available on
    ``response.choices[0].message.parsed`` (or ``.refusal``).
    """
    params: dict[str, Any] = {'model': model, 'messages': messages, 'response_format': response_model}
    if temperature is not None:
        params['temperature'] = temperature
    if max_completion_tokens is not None:
        params['max_completion_tokens'] = max_completion_tokens
    if extra_kwargs:
        params.update(extra_kwargs)

    _strip_known_rejected_reasoning(model, params)
    apply_pipeline_metadata(params)

    record_llm_input(span, messages)

    if inject_trace_headers:
        await apply_trace_headers(params)

    try:
        response = await _bounded_call(client.chat.completions.parse(**params), timeout_s)
    except BadRequestError as exc:
        if not _is_reasoning_effort_rejection(params, exc):
            raise
        _REASONING_EFFORT_REJECTORS.add(_reasoning_key(model, params))
        logger.warning('Model %s rejected reasoning_effort; dropping it and retrying once', model)
        params.pop('reasoning_effort', None)
        response = await _bounded_call(client.chat.completions.parse(**params), timeout_s)
    record_llm_response(span, response)
    return response, await price_usage(TokenUsage.from_completion(response), model, client)


async def execute_response(
    *,
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    span: Span | None,
    timeout_s: float,
    response_model: type[BaseModel] | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    inject_trace_headers: bool = True,
    extra_kwargs: dict[str, Any] | None = None,
) -> tuple[Any, TokenUsage | None]:
    """Execute one Responses API call — ``.parse`` with a ``response_model``, else ``.create``.

    The Responses counterpart of `execute_chat_completion`. Preferred for
    judges because the Orq router prices this endpoint and not Chat Completions:
    ``usage`` comes back with ``input_cost``/``output_cost``/``total_cost``
    already filled in, so the ``price_usage`` call below is a no-op on the Orq
    path and only covers a non-Orq endpoint or an unpriced model (RES-1295).

    ``messages`` are passed straight through as Responses ``input`` items; the
    system entry rides along as a message rather than as ``instructions``, which
    keeps the judge prompt assembly identical across both endpoints.
    """
    params: dict[str, Any] = {'model': model, 'input': messages}
    if temperature is not None:
        params['temperature'] = temperature
    if max_output_tokens is not None:
        params['max_output_tokens'] = max_output_tokens
    if extra_kwargs:
        params.update(extra_kwargs)

    strip_known_rejected_responses_reasoning(model, params)
    apply_pipeline_metadata(params)

    record_llm_input(span, messages)

    if inject_trace_headers:
        await apply_trace_headers(params)

    async def _call() -> Any:
        if response_model is not None:
            return await _bounded_call(client.responses.parse(text_format=response_model, **params), timeout_s)
        return await _bounded_call(client.responses.create(**params), timeout_s)

    try:
        response = await _call()
    except BadRequestError as exc:
        # Same drop-and-retry-once contract as the chat path, for the Responses
        # `reasoning` block. An unrelated 400 propagates to the caller.
        if not is_responses_reasoning_rejection(params, exc):
            raise
        remember_responses_reasoning_rejection(model, params)
        logger.warning('Model %s rejected the reasoning block; dropping it and retrying once', model)
        params.pop('reasoning', None)
        response = await _call()
    record_llm_response(span, response)
    # Priced by the router; price_usage is a no-op unless it came back unpriced
    # (a non-Orq endpoint, or a model the router does not price).
    return response, await price_usage(TokenUsage.extract(getattr(response, 'usage', None), calls=1), model, client)
