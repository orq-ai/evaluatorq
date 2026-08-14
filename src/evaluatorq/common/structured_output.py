"""Structured-output helper with a non-strict json_schema fallback, shared across domains.

Tries ``client.chat.completions.parse()`` first (strict schema-enforced structured
output). When the model doesn't support it the API returns 400; we fall back to a
plain ``create()`` carrying the same schema as a **non-strict**
``response_format={"type": "json_schema", ...}`` and return the raw content for the
caller to parse (fence-tolerant parsing lives in ``common.extract_json``).

The fallback sends the schema rather than a bare ``json_object`` because most
providers that reject ``.parse()`` reject *strict* mode — the schema itself is
still honoured, and it is the only thing telling the model which keys to emit.
``json_object`` asks for "some JSON" and leaves field names to chance, which is
what made the fence-tolerant parsing necessary in the first place. A provider
that rejects the schema form outright degrades to ``json_object`` on a second
400, so nothing that worked before stops working.

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
# helper exists to enforce). This mirrors the INTENT of
# LLMCallConfig.completion_params, but the key set is intentionally narrower:
# that guard also reserves extra_body, whereas here extra_body is deliberately
# NOT reserved because it is the documented carrier for provider options like
# the Orq router retry body.
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
    """Generate a chat completion with structured output, falling back to a non-strict schema.

    Returns ``(parsed_model, "")`` when strict structured output succeeds, or
    ``(None, raw_content)`` when the model rejects it and we fall back to a plain
    ``create()`` carrying the same schema non-strictly (and, if the provider
    rejects that too, to bare ``json_object``). The caller parses the raw content
    itself (typically via ``extract_json_from_response`` + ``model_validate_json``).

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

    On a length-truncated structured response this raises ``RuntimeError``
    rather than falling back (a same-budget retry would truncate
    again). "Loud" is scoped to this helper: it surfaces a specific, actionable
    reason instead of returning cut-off JSON. Both report call sites still wrap
    the call in a broad ``except`` and skip that one item, so a truncation
    degrades a single section — but now with a clear log line naming the budget,
    not the silent drop this migration set out to remove.
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
                # The literal True is the span attribute's value, not a boolean flag.
                span.set_attribute('orq.structured_output.fallback', True)  # noqa: FBT003
        except LengthFinishReasonError as exc:
            # Length-truncated structured output is unusable — the JSON is cut
            # off mid-string. Falling back would truncate at the
            # same budget, so fail loudly with an actionable message instead.
            logger.exception('%s: structured output truncated at the token limit (max_tokens=%s)', label, max_tokens)
            raise RuntimeError(
                f'{label}: the model hit the token limit (max_tokens={max_tokens}) and the '
                f'structured output was truncated, so the result is unusable. Raise the max_tokens '
                f'budget passed to this call and retry.'
            ) from exc

        # 2. Fallback: the same schema, non-strict, via plain create().
        schema_format: dict[str, Any] = {
            'type': 'json_schema',
            'json_schema': {
                'name': response_format.__name__,
                # Non-strict on purpose: strict mode is what .parse() already
                # tried and what the provider just rejected. The schema still
                # names the fields; only the enforcement is relaxed.
                'strict': False,
                'schema': response_format.model_json_schema(),
            },
        }
        try:
            fallback_response = await with_retry(
                lambda: client.chat.completions.create(**_params({'response_format': schema_format})),  # pyright: ignore[reportUnknownLambdaType]
                label=f'{label} (json_schema fallback)',
            )
        except APIStatusError as e:
            if e.status_code != 400:
                raise
            # A provider that rejects the schema form outright still gets the
            # old behaviour rather than losing the call entirely.
            logger.warning('%s: json_schema not accepted either, falling back to json_object', label)
            fallback_response = await with_retry(
                lambda: client.chat.completions.create(**_params({'response_format': {'type': 'json_object'}})),  # pyright: ignore[reportUnknownLambdaType]
                label=f'{label} (json_object fallback)',
            )
        record_llm_response(span, fallback_response)
        if not fallback_response.choices:
            return None, ''
        choice = fallback_response.choices[0]
        if choice.finish_reason == 'length':
            # Same defect as the parse() leg above, which the SDK raises for us:
            # cut-off JSON parses as a validation error two frames away, or worse,
            # extract_json_from_response salvages a truncated object and the caller
            # scores a half-answer. Fail with the same actionable message.
            logger.error('%s: fallback output truncated at the token limit (max_tokens=%s)', label, max_tokens)
            raise RuntimeError(
                f'{label}: the model hit the token limit (max_tokens={max_tokens}) and the '
                f'fallback output was truncated, so the result is unusable. Raise the max_tokens '
                f'budget passed to this call and retry.'
            )
        return None, choice.message.content or ''
