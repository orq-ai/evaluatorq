"""Generic OTel span-recording utilities shared by all evaluatorq domains.

``with_llm_span`` lives here and is the canonical implementation for all domains.
Domain tracing modules (redteam/tracing.py, etc.) delegate to this function
rather than duplicating the body. This module must not import from redteam,
simulation, or openresponses.
"""

from __future__ import annotations

import functools
import json
import os
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.common.fields import get_field as _field
from evaluatorq.tracing.setup import get_tracer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from opentelemetry.trace import Span

    from evaluatorq.contracts import Usage

AttrValue = str | int | float | bool
AttrMap = dict[str, AttrValue | None]

_UNSET_CALLS: Any = object()

_TRUNCATION_MARKER = '... [truncated]'
# Canonical/recommended cap when truncation is opted into. NOT the default —
# truncation is off by default (full content captured). See _default_span_max_text_chars.
_RECOMMENDED_SPAN_MAX_TEXT_CHARS = 8192


@functools.lru_cache(maxsize=1)
def _default_span_max_text_chars() -> int | None:
    """Read EVALUATORQ_SPAN_MAX_TEXT_CHARS once.

    **Defaults to None (capture all — no truncation).** Set a positive integer
    (e.g. 8192) to cap span text at that many characters. ``-1`` / ``0`` /
    unset all mean "capture all".

    Call _default_span_max_text_chars.cache_clear() in tests after changing the env var.
    """
    raw = os.getenv('EVALUATORQ_SPAN_MAX_TEXT_CHARS')
    if raw is None or raw == '':
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            'EVALUATORQ_SPAN_MAX_TEXT_CHARS={!r} is not a valid int; capturing all (no truncation)',
            raw,
        )
        return None
    # Any non-positive value (including -1) disables truncation.
    return value if value > 0 else None


def truncate_for_span(text: object, *, max_chars: int | None = None) -> str:
    """Truncate text for span attribute storage.

    Defaults to the EVALUATORQ_SPAN_MAX_TEXT_CHARS env var, which itself
    **defaults to no truncation (capture all)**. A non-positive ``max_chars``
    (None, 0, or negative) means "capture all". A positive ``max_chars`` caps
    the output, which never exceeds it (the marker is reserved within the budget).
    """
    if isinstance(text, str):
        s = text
    else:
        try:
            s = str(text)
        except Exception as e:  # pragma: no cover
            s = f'<unrepresentable {type(text).__name__}: {e}>'
    if max_chars is None:
        max_chars = _default_span_max_text_chars()
    if max_chars is None or max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    marker_len = len(_TRUNCATION_MARKER)
    if max_chars <= marker_len:
        return _TRUNCATION_MARKER[:max_chars]
    return s[: max_chars - marker_len] + _TRUNCATION_MARKER


def capture_message_content() -> bool:
    """Whether to write LLM message text (prompts + responses) onto spans.

    Controlled by the ``EVALUATORQ_CAPTURE_MESSAGE_CONTENT`` env var.

    **Defaults to True** so the Orq dashboard's input/output panels render out of
    the box. Set it to ``"false"`` / ``"0"`` to keep raw message content out of
    traces (e.g. when exporting spans to a third-party backend, or to avoid
    capturing PII) while still recording token usage, model, and latency.

    Gates ``gen_ai.input.messages``, ``gen_ai.output.messages``, the bare
    ``input`` / ``output`` keys, and the ``orq.openresponses.request/response``
    payloads. Public so domain span builders (redteam/openresponses) can gate
    input-message capture too.
    """
    flag = os.environ.get('EVALUATORQ_CAPTURE_MESSAGE_CONTENT')
    if flag is None:
        return True
    return flag.lower() == 'true' or flag == '1'


# Back-compat private alias (pre-existing callers/tests import the underscored name).
_capture_message_content = capture_message_content


def _serialize_messages(messages: list[dict[str, Any]]) -> str:
    return json.dumps(
        [
            {
                'role': str(m.get('role', '') if isinstance(m, dict) else getattr(m, 'role', '')),
                'content': truncate_for_span(
                    m.get('content', '') if isinstance(m, dict) else getattr(m, 'content', '')
                ),
            }
            for m in messages
        ],
        ensure_ascii=False,
    )


def _serialize_tool_call_content(tool_calls: list[dict[str, str]]) -> str:
    return json.dumps({'tool_calls': tool_calls}, ensure_ascii=False)


def _extract_chat_tool_call_payloads(tool_calls: Any) -> list[dict[str, str]]:
    payloads: list[dict[str, str]] = []
    for tool_call in tool_calls or []:
        function = _field(tool_call, 'function')
        name = _field(function, 'name') or _field(tool_call, 'name')
        arguments = _field(function, 'arguments') or _field(tool_call, 'arguments')
        payload: dict[str, str] = {}
        if name:
            payload['name'] = str(name)
        if arguments is not None:
            payload['arguments'] = str(arguments)
        if payload:
            payloads.append(payload)
    return payloads


def _extract_response_tool_call_payloads(output_items: list[Any]) -> list[dict[str, str]]:
    payloads: list[dict[str, str]] = []
    for item in output_items:
        call_id = _field(item, 'call_id')
        name = _field(item, 'name')
        arguments = _field(item, 'arguments')
        if call_id or name or arguments is not None:
            payload: dict[str, str] = {}
            if call_id:
                payload['call_id'] = str(call_id)
            if name:
                payload['name'] = str(name)
            if arguments is not None:
                payload['arguments'] = str(arguments)
            payloads.append(payload)
    return payloads


def _extract_output_messages(response: Any) -> list[dict[str, str]]:
    """Extract output message dicts from Chat Completions or Responses API shape."""
    output_messages: list[dict[str, str]] = []
    choices = _field(response, 'choices')
    if choices:
        for choice in choices:
            message = _field(choice, 'message')
            content = _field(message, 'content') if message else None
            if content:
                role = _field(message, 'role') or 'assistant'
                output_messages.append({'role': str(role), 'content': str(content)})
                continue
            tool_payloads = _extract_chat_tool_call_payloads(_field(message, 'tool_calls') if message else None)
            if tool_payloads:
                role = _field(message, 'role') or 'assistant'
                output_messages.append({
                    'role': str(role),
                    'content': _serialize_tool_call_content(tool_payloads),
                })
    else:
        output_text = _field(response, 'output_text')
        if isinstance(output_text, str) and output_text:
            output_messages.append({'role': 'assistant', 'content': output_text})
        else:
            output_items = _field(response, 'output') or []
            parts: list[str] = []
            for item in output_items:
                content = _field(item, 'content')
                if content:
                    for part in content:
                        text = _field(part, 'text')
                        if isinstance(text, str) and text:
                            parts.append(text)
                else:
                    text = _field(item, 'text')
                    if isinstance(text, str) and text:
                        parts.append(text)
            joined = ''.join(parts)
            if joined:
                output_messages.append({'role': 'assistant', 'content': joined})
            else:
                tool_payloads = _extract_response_tool_call_payloads(output_items)
                if tool_payloads:
                    output_messages.append({
                        'role': 'assistant',
                        'content': _serialize_tool_call_content(tool_payloads),
                    })
    return output_messages


def set_span_attrs(span: Span | None, attrs: AttrMap) -> None:
    """Batch-set span attributes. Skips None values. Safe no-op when span is None."""
    if span is None:
        return
    for key, value in attrs.items():
        if value is not None:
            span.set_attribute(key, value)


def set_span_error(span: Span | None, message: str) -> None:
    """Mark a span as failed without raising. Safe no-op when span is None.

    For swallowed failures — the code recovered, but the span should not read
    as OK in a trace viewer.
    """
    if span is None:
        return
    try:
        from opentelemetry.trace import Status, StatusCode
    except ImportError:  # pragma: no cover - OTel always present when a span exists
        return
    span.set_status(Status(StatusCode.ERROR, message))


def record_token_usage(
    span: Span | None,
    *,
    usage: Usage | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    calls: int | None = _UNSET_CALLS,
    cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
    input_cost: float | None = None,
    output_cost: float | None = None,
    total_cost: float | None = None,
    cost_source: str | None = None,
) -> None:
    """Record token usage on a span. Safe no-op when span is None.

    One canonical attribute name per number, all under the OTel GenAI
    ``gen_ai.usage.*`` namespace. The former ``prompt_tokens`` /
    ``completion_tokens`` spellings and the bare (un-namespaced) keys are
    deliberately not emitted: they carried the same values under three names
    each, which triples attribute volume and lets two consumers disagree about
    which key is authoritative. Also records the provider-reported cost
    breakdown (Orq Responses v3) when present.

    ``usage`` accepts a `evaluatorq.contracts.Usage` and expands it into
    the individual parameters; explicitly-passed parameters win over it.

    ``cost_source`` (``'provider'`` | ``'catalogue'`` | ``'mixed'``) is the
    provenance of ``total_cost`` and is emitted as ``gen_ai.usage.cost_source``
    beside it. Defaults to ``usage.cost_source`` when a `Usage` is supplied, so a
    client-side catalogue estimate cannot reach a trace viewer labelled as billed.
    Only ever set alongside a cost: a provenance attribute on a span with no cost
    would describe a number that is not there.
    """
    if span is None:
        return
    if usage is not None:
        prompt_tokens = prompt_tokens if prompt_tokens is not None else usage.input_tokens
        completion_tokens = completion_tokens if completion_tokens is not None else usage.output_tokens
        total_tokens = total_tokens if total_tokens is not None else usage.total_tokens
        # An explicitly-passed `calls` (including 0) wins over usage.calls;
        # only fall back to usage.calls when the caller left it unset. A
        # falsy check here (`calls or usage.calls`) would let usage.calls
        # (which Usage.extract defaults to 1) clobber an explicit calls=0.
        if calls is _UNSET_CALLS:
            calls = usage.calls
        # Zero-valued details are treated as absent so spans don't grow
        # cache/reasoning attributes for providers that never report them.
        cached_tokens = cached_tokens if cached_tokens is not None else (usage.cached_tokens or None)
        reasoning_tokens = reasoning_tokens if reasoning_tokens is not None else (usage.reasoning_tokens or None)
        if cache_creation_input_tokens is None and usage.cache_creation_tokens:
            cache_creation_input_tokens = usage.cache_creation_tokens
        input_cost = input_cost if input_cost is not None else usage.input_cost
        output_cost = output_cost if output_cost is not None else usage.output_cost
        total_cost = total_cost if total_cost is not None else usage.total_cost
        cost_source = cost_source if cost_source is not None else usage.cost_source
    if calls is _UNSET_CALLS:
        calls = 0
    prompt = prompt_tokens if prompt_tokens is not None else 0
    completion = completion_tokens if completion_tokens is not None else 0
    total = total_tokens if total_tokens is not None else prompt + completion
    span.set_attribute('gen_ai.usage.input_tokens', prompt)
    span.set_attribute('gen_ai.usage.output_tokens', completion)
    span.set_attribute('gen_ai.usage.total_tokens', total)
    if calls:
        span.set_attribute('gen_ai.usage.calls', calls)
    cache_read = cached_tokens if cached_tokens is not None else cache_read_input_tokens
    if cache_read is not None:
        span.set_attribute('gen_ai.usage.cache_read.input_tokens', cache_read)
    if reasoning_tokens is not None:
        # This spelling (not the flat completion_tokens_details.* one) is what the
        # platform's generic OTel adapter reads — see extractCommonUsage in
        # orquesta-web apps/traces-api/.../utils/adapter-patterns.ts.
        span.set_attribute('gen_ai.usage.reasoning.output_tokens', reasoning_tokens)
    if cache_creation_input_tokens is not None:
        span.set_attribute('gen_ai.usage.cache_creation.input_tokens', cache_creation_input_tokens)
    # Provider-reported cost breakdown (USD). Only set when reported — a $0
    # attribute on a span whose cost is unknown would read as "free".
    if input_cost is not None:
        span.set_attribute('gen_ai.usage.input_cost', input_cost)
    if output_cost is not None:
        span.set_attribute('gen_ai.usage.output_cost', output_cost)
    if total_cost is not None:
        span.set_attribute('gen_ai.usage.total_cost', total_cost)
        span.set_attribute('gen_ai.usage.cost', total_cost)
        if cost_source is not None:
            span.set_attribute('gen_ai.usage.cost_source', cost_source)
        else:
            # A cost with no provenance: either the caller passed a bare number
            # rather than a Usage, or it passed a Usage carrying a cost with
            # priced_calls=0. The trace UI can then only show the figure, not
            # whether it was billed. No in-`src` caller can reach this branch, so
            # the warning cannot cry wolf — if it ever fires it is exactly the
            # defect this provenance plumbing exists to prevent.
            logger.warning('record_token_usage: cost {} recorded without provenance (no cost_source)', total_cost)


def record_llm_response(
    span: Span | None,
    response: Any,
    *,
    output_content: str | None = None,
) -> None:
    """Record LLM response attributes on a span.

    Superset of both former impls: duck-typed (_field handles dicts + objects),
    handles Chat Completions and Responses API shapes, honors the PII capture
    gate, accepts an optional output_content override for backward compat with
    redteam callers that pass the output string explicitly.
    """
    if span is None:
        return

    response_id = _field(response, 'id')
    if response_id:
        span.set_attribute('gen_ai.response.id', response_id)
    response_model = _field(response, 'model')
    if response_model:
        span.set_attribute('gen_ai.response.model', response_model)

    usage = _field(response, 'usage')
    if usage is not None:
        # Lazy import: contracts transitively imports openresponses, which this
        # module must not pull in at import time.
        from evaluatorq.contracts import Usage

        parsed = Usage.extract(usage, calls=1)
        if parsed is not None:
            record_token_usage(span, usage=parsed, calls=0)
        else:
            # Usage was present but empty/unparseable (Usage.extract returned
            # None). Still record the zero token attributes the hand-rolled
            # parser used to set, rather than emitting nothing.
            record_token_usage(span, calls=0)

    if _capture_message_content():
        if output_content is not None:
            serialized = json.dumps(
                [{'role': 'assistant', 'content': truncate_for_span(output_content)}],
                ensure_ascii=False,
            )
            span.set_attribute('gen_ai.output.messages', serialized)
            span.set_attribute('output', serialized)
        else:
            output_messages = _extract_output_messages(response)
            if output_messages:
                serialized = _serialize_messages(output_messages)
                span.set_attribute('gen_ai.output.messages', serialized)
                span.set_attribute('output', serialized)

    finish_reasons: list[str] = []
    choices = _field(response, 'choices')
    if choices:
        for choice in choices:
            reason = _field(choice, 'finish_reason')
            if reason:
                finish_reasons.append(reason)
    else:
        status = _field(response, 'status')
        if isinstance(status, str) and status:
            finish_reasons.append(status)
    if finish_reasons:
        span.set_attribute('gen_ai.response.finish_reasons', finish_reasons)


def record_llm_input(span: Span | None, messages: list[dict[str, Any]]) -> None:
    """Record LLM input messages. Suppressed when capture gate is off."""
    if span is None or not messages:
        return
    if not _capture_message_content():
        return
    serialized = _serialize_messages(messages)
    span.set_attribute('gen_ai.input.messages', serialized)
    span.set_attribute('input', serialized)


def record_llm_output(span: Span | None, output: str) -> None:
    """Record a single LLM output string. Suppressed when capture gate is off."""
    if span is None or not output:
        return
    if not _capture_message_content():
        return
    serialized = _serialize_messages([{'role': 'assistant', 'content': output}])
    span.set_attribute('gen_ai.output.messages', serialized)
    span.set_attribute('output', serialized)


def _derive_provider(model: str) -> str:
    if '/' in model:
        return model.split('/', 1)[0]
    return 'openai'


@asynccontextmanager
async def with_llm_span(  # noqa: RUF029
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
    """Execute code within a generic GenAI LLM span.

    Yields:
        The active span when tracing is enabled, otherwise None.
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

    # `is not None`, not truthiness: Context subclasses dict, so a legitimately
    # empty (root) context is falsy and would be silently swapped for ambient.
    ctx = parent_context if parent_context is not None else otel_context.get_current()
    resolved_provider = provider or _derive_provider(model)
    span_name = f'{operation} {model}'

    genai_attrs: dict[str, Any] = {
        'gen_ai.operation.name': operation,
        'gen_ai.system': resolved_provider,
        'gen_ai.provider.name': resolved_provider,
        'gen_ai.request.model': model,
    }
    if temperature is not None:
        genai_attrs['gen_ai.request.temperature'] = float(temperature)
    if max_tokens is not None:
        genai_attrs['gen_ai.request.max_tokens'] = max_tokens
    if input_messages is not None:
        recordable = [
            {
                'role': str(m.get('role', '') if isinstance(m, dict) else getattr(m, 'role', '')),
                'content': truncate_for_span(
                    m.get('content', '') if isinstance(m, dict) else getattr(m, 'content', '')
                ),
            }
            for m in input_messages
        ]
        if capture_message_content():
            serialized = json.dumps(recordable, ensure_ascii=False)
            genai_attrs['gen_ai.input.messages'] = serialized
            genai_attrs['input'] = serialized
    if attributes:
        genai_attrs.update(attributes)
    # Domain-specific key mapping (e.g. orq.redteam.llm_purpose -> orq.llm.purpose)
    # is the responsibility of the domain's own tracing wrapper before delegating here.
    # Common only handles the neutral orq.llm.purpose key set by callers via attributes.

    with tracer.start_as_current_span(
        span_name,
        context=ctx,
        kind=SpanKind.CLIENT,
        attributes=genai_attrs,
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
async def with_span(  # noqa: RUF029
    name: str,
    attributes: AttrMap | None = None,
    *,
    parent_context: Any | None = None,
) -> AsyncGenerator[Span | None, None]:
    """Execute code within a generic INTERNAL span (not an LLM call span).

    The neutral counterpart to `with_llm_span` for orchestration spans
    that group work (e.g. a jury deliberation and its per-judge children).
    ``get_tracer()``-gated, so it is a zero-cost no-op when tracing is off.

    ``parent_context`` pins the span's parent explicitly, which matters when
    the caller opens several children under one parent via ``asyncio.gather``:
    the shared context is captured once and passed in, so parenting does not
    depend on gather scheduling order.

    Yields:
        The active span when tracing is enabled, otherwise None.
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

    # `is not None`, not truthiness: Context subclasses dict, so a legitimately
    # empty (root) context is falsy and would be silently swapped for ambient.
    ctx = parent_context if parent_context is not None else otel_context.get_current()
    clean_attrs = {k: v for k, v in (attributes or {}).items() if v is not None}

    with tracer.start_as_current_span(
        name,
        context=ctx,
        kind=SpanKind.INTERNAL,
        attributes=clean_attrs,
    ) as span:
        try:
            yield span
            # Don't clobber an ERROR the body set deliberately (set_span_error):
            # OK is final in the OTel spec and would hide a swallowed failure.
            # `status` is on the SDK's Span, not the API protocol — read it
            # defensively so a non-SDK span just takes the OK path.
            status = getattr(span, 'status', None)
            if getattr(status, 'status_code', None) is not StatusCode.ERROR:
                span.set_status(Status(StatusCode.OK))
        except BaseException as e:
            span.set_attribute('error.type', type(e).__name__)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def current_otel_context() -> Any | None:
    """Return the active OTel context, or None when OTel is unavailable.

    Used to capture a parent span's context for passing to ``with_span`` /
    ``with_llm_span`` children created concurrently (see ``parent_context``).
    """
    try:
        from opentelemetry import context as otel_context
    except ImportError:
        return None
    return otel_context.get_current()


async def get_trace_context_headers() -> dict[str, str]:  # noqa: RUF029
    """Return W3C trace context headers for the current active span.

    Empty dict when OTel is not available. Used to propagate trace context
    into outgoing HTTP requests.
    """
    try:
        from opentelemetry import context, propagate
    except ImportError:
        return {}
    headers: dict[str, str] = {}
    propagate.inject(headers, context=context.get_current())
    return headers
