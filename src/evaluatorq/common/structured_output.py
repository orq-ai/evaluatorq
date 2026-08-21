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

``api='responses'`` puts ``client.responses.parse(text_format=...)`` in front of
that ladder, which the whole simulation pipeline uses so a run's spans are
uniformly ``responses ...``. It is a leg, not a replacement: a provider without
the endpoint, or one that names the schema form unsupported, falls through to
everything described above.

Lives in ``common`` rather than ``simulation`` so both the simulation and
red-team report code can reuse one copy (RES-822). It delegates to the canonical
``common.tracing.with_llm_span``; a domain that needs its own span attributes
passes them through ``attributes``.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TypeVar, cast

from openai import APIStatusError, AsyncOpenAI, LengthFinishReasonError
from pydantic import BaseModel, ValidationError

from evaluatorq.common.llm_call import apply_pipeline_metadata, execute_response
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
# Keyed by API because the two legs own different field names for the same
# structural role. An api='responses' call may still fall through to the chat
# legs, so its reserved set is the union — a key that is safe on the endpoint
# actually reached is not safe on the one it degrades to.
_STRUCTURAL_KEYS = frozenset({'model', 'messages', 'response_format'})
_STRUCTURAL_KEYS_RESPONSES = frozenset({'model', 'input', 'text_format', 'max_output_tokens'})
_STRUCTURAL_KEYS_BY_API = {
    'chat_completions': _STRUCTURAL_KEYS,
    'responses': _STRUCTURAL_KEYS | _STRUCTURAL_KEYS_RESPONSES,
}

# A provider that cannot do schema-enforced output says so in the error body.
# A bare status code does not: a 400 is far more often a bad parameter, an
# over-length context or a content-policy rejection, and degrading on those
# masks the real cause and re-bills the same broken request on the other leg.
_SCHEMA_KEYWORDS = ('structured', 'response_format', 'json_schema', 'text_format', 'not supported')


def _looks_like_schema_rejection(exc: APIStatusError) -> bool:
    err_body = str(getattr(exc, 'body', None) or getattr(exc, 'message', '') or '').lower()
    return any(kw in err_body for kw in _SCHEMA_KEYWORDS)


def _first_responses_refusal(response: Any) -> str | None:
    """Return the first refusal in a Responses output, if present."""
    for item in getattr(response, 'output', None) or []:
        for part in getattr(item, 'content', None) or []:
            if getattr(part, 'type', None) == 'refusal':
                return getattr(part, 'refusal', '') or ''
    return None


def _looks_like_truncated_json(exc: ValidationError) -> bool:
    """Recognize the end-of-input validation error caused by cut-off JSON."""
    for error in exc.errors():
        if error.get('type') != 'json_invalid':
            continue
        context = error.get('ctx') or {}
        detail = str(context.get('error') or error.get('msg') or '').lower()
        if 'eof' in detail or 'end of input' in detail:
            return True
    return False


# The Responses leg's per-request ceiling. A batched generation asking for tens
# of items is a slow call by design, so this is well above LLMCallConfig's
# 90s default; without it the call has no bound at all, which is what the
# hand-rolled version this leg replaced had.
_STRUCTURED_TIMEOUT_S = 300.0


def _truncated_output_error(label: str, max_tokens: int, *, schema: str | None = None) -> RuntimeError:
    """One message for every way this helper can be handed a cut-off payload.

    Truncated structured output is unusable and unrecoverable: the JSON stops
    mid-string, and a retry at the same budget truncates in the same place. All
    three legs raise this rather than degrading, so the user gets the one action
    that works instead of a parse error several frames away.
    """
    detail = (
        'the structured output was truncated, so the result is unusable'
        if schema is None
        else f'the reply did not validate against {schema}, which is what a truncated payload looks like'
    )
    logger.error('%s: output truncated at the token limit (max_tokens=%s)', label, max_tokens)
    return RuntimeError(
        f'{label}: the model hit the token limit (max_tokens={max_tokens}) and {detail}. '
        f'Raise the max_tokens budget passed to this call and retry.'
    )


def token_budget_for_items(count: int, *, per_item: int, minimum: int) -> int:
    """Scale a ``max_tokens`` budget with how many items one call asks for.

    A batched structured call that asks for N items and carries a flat cap
    truncates once N grows, and truncated structured output is unrecoverable —
    both legs of ``generate_structured`` raise rather than retry at the same
    budget. Every caller that takes a caller-controlled count derives its budget
    here so a new one cannot reintroduce a flat cap.

    The result is deliberately unbounded at the top: clamping would trade a
    clear "raise the budget" error for silent truncation at whatever ceiling was
    picked. A count large enough to exceed the model's own limit fails with the
    provider naming the limit, which is the more useful error.
    """
    return max(minimum, per_item * count)


async def _generate_structured_via_responses(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_format: type[T],
    max_tokens: int,
    label: str,
    temperature: float | None,
    extra_kwargs: dict[str, Any] | None,
) -> T | None:
    """Structured output through the Responses API; ``None`` means "use the chat legs".

    Returns ``None`` — after a warning naming the cause — when the endpoint is
    absent (404), when a 400's body names the schema form as unsupported, or
    when the provider hands back nothing parsed. Those cases let the caller
    degrade to ``chat.completions`` rather than losing the call. Refusals and
    schema-validation failures raise instead: a refusal must not be retried on a
    second endpoint, and a validation error is not evidence of a capability gap.
    Any other 400 re-raises: a bad parameter or an over-length context is not a
    capability signal, and degrading on it would blame the provider and re-bill
    the same failure.
    Transport is ``common.llm_call.execute_response``, so this leg inherits the
    concurrency slot, the timeout, the reasoning-block drop-and-retry and the
    usage pricing rather than re-deriving them. Only the fallback policy lives
    here.

    A length-truncated response raises, matching the chat leg: a same-budget
    retry would truncate again. Truncation surfaces two ways —
    ``responses.parse`` validates the payload before returning, so a cut-off
    body raises a JSON ``ValidationError`` with an end-of-input error, and only
    a response truncated with no text at all (a reasoning model that spent the
    budget before answering) reaches the ``incomplete_details`` check.
    """
    async with with_llm_span(
        model=model,
        operation='responses',
        temperature=temperature,
        max_tokens=max_tokens,
        input_messages=messages,
        attributes={'orq.llm.purpose': label},
    ) as span:
        try:
            response, _usage = await with_retry(
                lambda: execute_response(
                    client=client,
                    model=model,
                    messages=messages,
                    span=span,
                    timeout_s=_STRUCTURED_TIMEOUT_S,
                    response_model=response_format,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    extra_kwargs=extra_kwargs,
                ),
                label=label,
            )
        except APIStatusError as e:
            if e.status_code == 404:
                cause = 'no Responses endpoint (HTTP 404)'
            elif e.status_code == 400 and _looks_like_schema_rejection(e):
                cause = 'Responses structured output rejected (HTTP 400)'
            else:
                raise
            logger.warning('%s: %s, falling back to chat.completions', label, cause)
            return None
        except ValidationError as exc:
            if _looks_like_truncated_json(exc):
                raise _truncated_output_error(label, max_tokens, schema=response_format.__name__) from exc
            logger.exception('%s: Responses output did not validate against %s', label, response_format.__name__)
            raise RuntimeError(
                f'{label}: the Responses output did not validate against {response_format.__name__}.'
            ) from exc

        if getattr(getattr(response, 'incomplete_details', None), 'reason', None) == 'max_output_tokens':
            raise _truncated_output_error(label, max_tokens)

        refusal = _first_responses_refusal(response)
        if refusal is not None:
            raise RuntimeError(f'{label}: model refused to generate: {refusal}')

        parsed = getattr(response, 'output_parsed', None)
        if parsed is None:
            logger.warning('%s: Responses returned no parsed output, falling back to chat.completions', label)
            return None
        return cast('T', parsed)


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
    api: Literal['chat_completions', 'responses'] = 'chat_completions',
) -> tuple[T | None, str]:
    """Generate structured output, falling back to a non-strict schema.

    ``api`` selects the endpoint tried first. The default ``chat_completions``
    runs the ladder below unchanged. ``responses`` tries
    ``client.responses.parse(text_format=...)`` first and falls through to that
    same ladder when the endpoint is absent, names the schema form unsupported,
    or returns nothing parsed — so this helper's contract does not depend on the
    provider implementing the Responses API. On the ``responses`` success path
    the returned raw-content string is always ``""``. Structural fields are
    reserved per endpoint, and an ``api='responses'`` call reserves both sets
    because it may still reach the chat legs.

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

    RES-1295: neither the ``parse()`` call below nor the json_object fallback
    extracts usage, so their tokens never reach any total. The 11 call sites
    across ``persona_generator.py``, ``scenario_generator.py``, ``traces.py``,
    and both ``recommendations.py`` modules track no usage today — adding a
    usage element to this function's return tuple would mean threading it
    through every one of them. See "What the totals do not include" in
    docs/guides/red-teaming.md.
    """
    reserved = _STRUCTURAL_KEYS_BY_API[api] & (extra_kwargs or {}).keys()
    if reserved:
        raise ValueError(
            f'extra_kwargs cannot override structural request field(s) {sorted(reserved)}; '
            'these are owned by generate_structured.'
        )

    # Cast once — the OpenAI SDK accepts dict literals at runtime; the TypedDict
    # union just doesn't type-narrow from dict[str, Any].
    typed_messages = cast('Any', messages)

    if api == 'responses':
        parsed_via_responses = await _generate_structured_via_responses(
            client,
            model=model,
            messages=messages,
            response_format=response_format,
            max_tokens=max_tokens,
            label=label,
            temperature=temperature,
            extra_kwargs=extra_kwargs,
        )
        if parsed_via_responses is not None:
            return parsed_via_responses, ''
        # The warning naming the cause is emitted by the helper; the chat legs
        # below are the fallback, so a provider without Responses structured
        # output still gets an answer.

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
            if not _looks_like_schema_rejection(e):
                raise
            logger.warning('%s: structured output not supported by model, falling back to json_object', label)
            if span is not None:
                # The literal True is the span attribute's value, not a boolean flag.
                span.set_attribute('orq.structured_output.fallback', True)  # noqa: FBT003
        except LengthFinishReasonError as exc:
            raise _truncated_output_error(label, max_tokens) from exc

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
            raise _truncated_output_error(label, max_tokens)
        return None, choice.message.content or ''
