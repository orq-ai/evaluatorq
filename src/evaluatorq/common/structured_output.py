"""Structured-output helper with json_object fallback, shared across domains.

Tries ``client.chat.completions.parse()`` first (schema-enforced structured
output). When the model doesn't support it the API returns 400; we fall back to
``response_format={"type": "json_object"}`` and return the raw content for the
caller to parse (fence-tolerant parsing lives in ``common.extract_json``).

Lives in ``common`` rather than ``simulation`` so both the simulation and
red-team report code can reuse one copy (RES-822). It delegates to the canonical
``common.tracing.with_llm_span``; a domain that needs its own span attributes
passes them through ``attributes``.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar, cast

from openai import APIStatusError, AsyncOpenAI, LengthFinishReasonError
from pydantic import BaseModel

from evaluatorq.common.llm_call import apply_pipeline_metadata
from evaluatorq.common.retry import with_retry
from evaluatorq.common.tracing import get_trace_context_headers, record_llm_response, with_llm_span

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

# Structural request fields a caller's extra_kwargs may not replace: they are
# owned by this helper, and letting extra_kwargs swap them out would silently
# break the call it rides on (e.g. replacing the response_format schema this
# helper exists to enforce). Same contract as LLMCallConfig.completion_params.
# extra_body is deliberately NOT reserved here: it is the documented carrier
# for provider options like the Orq router retry body.
_STRUCTURAL_KEYS = frozenset({'model', 'messages', 'response_format'})


async def generate_structured(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_format: type[T],
    max_tokens: int,
    label: str,
    temperature: float | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> tuple[T | None, str]:
    """Generate a chat completion with structured output, falling back to json_object.

    Returns ``(parsed_model, "")`` when structured output succeeds, or
    ``(None, raw_content)`` when the model doesn't support it and we fall back to
    json_object mode. The caller parses the raw content itself (typically via
    ``extract_json_from_response`` + ``model_validate_json``).

    ``temperature`` is sent only when not ``None`` (some callers deliberately let
    the provider default stand). ``extra_kwargs`` is merged LAST into both the
    ``parse()`` and the fallback ``create()`` call via a single dict splat, so a
    caller can carry provider options (``extra_body`` for the Orq router retry,
    a reasoning-model ``temperature`` override, user ``llm_kwargs``) and those
    win over the base fields without ever tripping a "multiple values for
    keyword" error. Structural fields (``model``, ``messages``,
    ``response_format``) are reserved and raise ``ValueError`` — an
    ``extra_kwargs`` entry silently replacing the schema would defeat the
    helper.
    """
    reserved = _STRUCTURAL_KEYS & (extra_kwargs or {}).keys()
    if reserved:
        raise ValueError(
            f'extra_kwargs cannot override structural request field(s) {sorted(reserved)}; '
            'these are owned by generate_structured.'
        )

    # Cast once — the OpenAI SDK accepts dict literals at runtime; the TypedDict
    # union just doesn't type-narrow from dict[str, Any].
    typed_messages = cast('Any', messages)

    async with with_llm_span(
        model=model,
        operation='chat',
        temperature=temperature,
        max_tokens=max_tokens,
        input_messages=messages,
        attributes={'orq.llm.purpose': label},
    ) as span:
        trace_headers = await get_trace_context_headers()

        def _params(response_kwargs: dict[str, Any]) -> dict[str, Any]:
            base: dict[str, Any] = {
                'model': model,
                'messages': typed_messages,
                # max_completion_tokens, not max_tokens: OpenAI rejects
                # max_tokens outright for the o-series and gpt-5 families, and
                # every other chat call in the repo already sends this key.
                'max_completion_tokens': max_tokens,
                **response_kwargs,
            }
            if temperature is not None:
                base['temperature'] = temperature
            # extra_kwargs first: a caller's provider options (extra_body, a
            # reasoning-model temperature override, user llm_kwargs) win over the
            # base fields without a "multiple values for keyword" error.
            if extra_kwargs:
                base.update(extra_kwargs)
            # Trace headers are applied LAST so the active span's traceparent
            # propagates even when a caller passed its own headers (merge, not
            # replace, so other caller headers survive). Run metadata fills in
            # defaults only: apply_pipeline_metadata spreads any caller-supplied
            # metadata (merged from extra_kwargs above) last, so a caller key
            # wins on conflict — by that helper's contract.
            if trace_headers:
                base['extra_headers'] = {**(base.get('extra_headers') or {}), **trace_headers}
            apply_pipeline_metadata(base)
            return base

        # 1. Try structured output via parse().
        try:
            parse_params = _params({'response_format': response_format})
            response = await with_retry(
                lambda: client.chat.completions.parse(**parse_params),
                label=label,
            )
            record_llm_response(span, response)
            message = response.choices[0].message
            refusal = getattr(message, 'refusal', None)
            if refusal:
                raise RuntimeError(f'{label}: model refused to generate: {refusal}')
            parsed = message.parsed
            if parsed is not None:
                return parsed, ''
            logger.debug('%s: parse() returned None, falling back to json_object', label)
        except APIStatusError as e:
            if e.status_code != 400:
                raise
            # Only fall back if this looks like a schema-support issue.
            err_body = str(getattr(e, 'body', None) or getattr(e, 'message', '') or '').lower()
            schema_keywords = ('structured', 'response_format', 'json_schema', 'not supported')
            if not any(kw in err_body for kw in schema_keywords):
                raise
            logger.warning('%s: structured output not supported by model, falling back to json_object', label)
            if span is not None:
                fallback = True
                span.set_attribute('orq.structured_output.fallback', fallback)
        except LengthFinishReasonError as exc:
            # Length-truncated structured output is unusable — the JSON is cut
            # off mid-string. Falling back to json_object would truncate at the
            # same budget, so fail loudly with an actionable message instead.
            logger.exception('%s: structured output truncated at the token limit (max_tokens=%s)', label, max_tokens)
            raise RuntimeError(
                f'{label}: the model hit the token limit (max_tokens={max_tokens}) and the '
                f'structured output was truncated, so the result is unusable. Raise the budget '
                f'via EVALUATORQ_LLM_MAX_TOKENS and retry.'
            ) from exc

        # 2. Fallback: json_object mode.
        fallback_params = _params({'response_format': {'type': 'json_object'}})
        fallback_response = await with_retry(
            lambda: client.chat.completions.create(**fallback_params),  # pyright: ignore[reportUnknownLambdaType]
            label=f'{label} (fallback)',
        )
        record_llm_response(span, fallback_response)
        content = fallback_response.choices[0].message.content if fallback_response.choices else ''
        return None, content or ''
