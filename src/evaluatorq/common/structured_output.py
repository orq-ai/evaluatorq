"""Structured-output helper with a degrading four-rung chat ladder, shared across domains.

Each rung asks for the same schema in a way the previous rung's provider might
still support; the first one that yields something validating against the schema
wins. Every rung's text output goes through the canonical fence-tolerant parser
(``common.extract_json``) before validation, so a ```json-fenced payload comes back
parsed rather than raw — the raw string is still returned alongside for callers
with their own salvage.

===  ==========================  ================  ==================================
#    Rung                        Prompt            Continues the ladder on
===  ==========================  ================  ==================================
1    strict ``parse()``          caller's          schema-rejection 400, no ``parsed``
2    non-strict ``json_schema``  caller's          400, content fails validation
3    forced tool call            **+1 user turn**  400, no ``tool_calls``, invalid args
4    bare ``json_object``        caller's          — terminal
===  ==========================  ================  ==================================

Rung 2 keeps the schema rather than dropping straight to ``json_object`` because
most providers that reject ``.parse()`` reject *strict* mode — the schema itself
is still honoured, and it is the only thing telling the model which keys to emit.

Rung 3 is stricter than the two around it, which is why it sits below the rungs
that preserve the prompt and above the one that abandons the schema: forcing a
named tool leaves the model no prose channel at all, and function calling is a
different provider capability than ``response_format``, so a model that 400s on a
JSON schema often still supports it. It is the only rung that edits the prompt
(one appended user turn), and it is skipped when the caller supplied their own
``tools``.

Rung 4 asks for "some JSON" and leaves the field names to chance, which is what
made fence-tolerant parsing necessary in the first place — hence last.

Truncation and refusals never continue the ladder: a same-budget retry truncates
in the same place, and a refusal must not be re-asked on another rung.

``api='responses'`` puts a raw ``client.responses.create()`` call carrying the
same strict JSON schema in front of that ladder, which the whole simulation
pipeline uses so a run's spans are uniformly ``responses ...``. It is a leg,
not a replacement: a provider without the endpoint, or one that names the
schema form unsupported, falls through to everything described above.

Lives in ``common`` rather than ``simulation`` so both the simulation and
red-team report code can reuse one copy (RES-822). It delegates to the canonical
``common.tracing.with_llm_span``; a domain that needs its own span attributes
passes them through ``attributes``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from openai import APIStatusError, AsyncOpenAI, LengthFinishReasonError, pydantic_function_tool
from pydantic import BaseModel, ValidationError

from evaluatorq.common.extract_json import extract_json_from_response
from evaluatorq.common.llm_call import apply_pipeline_metadata, execute_response
from evaluatorq.common.responses import first_responses_refusal, parse_responses_response, responses_stop_reason
from evaluatorq.common.retry import with_retry
from evaluatorq.common.tracing import get_trace_context_headers, record_llm_response, with_llm_span

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from opentelemetry.trace import Span

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

# The chat ladder's rungs, in order. Recorded on the span so a run says which
# one paid rather than only that *some* fallback did.
_LEG_PARSE = 'parse'
_LEG_JSON_SCHEMA = 'json_schema'
_LEG_TOOL = 'tools'
_LEG_JSON_OBJECT = 'json_object'

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
_STRUCTURAL_KEYS_RESPONSES = frozenset({'model', 'input', 'text', 'text_format', 'max_output_tokens'})
_STRUCTURAL_KEYS_BY_API = {
    'chat_completions': _STRUCTURAL_KEYS,
    'responses': _STRUCTURAL_KEYS | _STRUCTURAL_KEYS_RESPONSES,
}

# A provider that cannot do schema-enforced output says so in the error body.
# A bare status code does not: a 400 is far more often a bad parameter, an
# over-length context or a content-policy rejection, and degrading on those
# masks the real cause and re-bills the same broken request on the other leg.
_SCHEMA_KEYWORDS = (
    'response_format',
    'json_schema',
    'text_format',
    'text.format',
    'structured output',
)


def _looks_like_schema_rejection(exc: APIStatusError) -> bool:
    err_body = str(getattr(exc, 'body', None) or getattr(exc, 'message', '') or '').lower()
    return any(kw in err_body for kw in _SCHEMA_KEYWORDS)


# The Responses leg's per-request ceiling. A batched generation asking for tens
# of items is a slow call by design, so this is well above LLMCallConfig's
# 90s default; without it the call has no bound at all, which is what the
# hand-rolled version this leg replaced had.
_STRUCTURED_TIMEOUT_S = 300.0


def _truncated_output_error(label: str, max_tokens: int) -> RuntimeError:
    """Return the actionable error for a provider-reported cut-off payload.

    Truncated structured output is unusable and unrecoverable: the JSON stops
    mid-string, and a retry at the same budget truncates in the same place. Every
    rung raises this rather than degrading, so the user gets the one action that
    works instead of a parse error several frames away.
    """
    logger.error('%s: output truncated at the token limit (max_tokens=%s)', label, max_tokens)
    return RuntimeError(
        f'{label}: the model hit the token limit (max_tokens={max_tokens}) and the structured output is unusable. '
        f'Raise the max_tokens budget passed to this call and retry.'
    )


def token_budget_for_items(count: int, *, per_item: int, minimum: int) -> int:
    """Scale a ``max_tokens`` budget with how many items one call asks for.

    A batched structured call that asks for N items and carries a flat cap
    truncates once N grows, and truncated structured output is unrecoverable —
    every rung of ``generate_structured`` raises rather than retrying at the same
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
    retry would truncate again. The raw response is inspected before parsing so
    this decision uses the provider's completion metadata, never the shape of
    the returned JSON.
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
                    response_text_format=response_format,
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
        if responses_stop_reason(response) == 'length':
            raise _truncated_output_error(label, max_tokens)

        refusal = first_responses_refusal(response)
        if refusal is not None:
            raise RuntimeError(f'{label}: model refused to generate: {refusal}')

        raw_content = getattr(response, 'output_text', '') or ''
        if not raw_content:
            logger.warning('%s: Responses returned no output, falling back to chat.completions', label)
            return None

        try:
            parsed_response = parse_responses_response(
                response,
                response_format,
                input_tools=(extra_kwargs or {}).get('tools'),
            )
        except ValidationError as exc:
            logger.exception('%s: Responses output did not validate against %s', label, response_format.__name__)
            raise RuntimeError(
                f'{label}: the Responses output did not validate against {response_format.__name__}.'
            ) from exc
        parsed = getattr(parsed_response, 'output_parsed', None)
        if parsed is None:
            logger.warning('%s: Responses returned no parsed output, falling back to chat.completions', label)
            return None
        return cast('T', parsed)


def _chat_params(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None,
    extra_kwargs: dict[str, Any] | None,
    trace_headers: dict[str, str] | None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the per-leg request-parameter factory for the chat legs.

    Returns a callable that takes the leg's own fields (``response_format``, or
    ``tools``/``tool_choice``/``messages``) and merges them over the shared base.
    Every chat leg goes through it, so the merge order is defined once:

    1. base (``model``, ``messages``, ``max_completion_tokens``, ``temperature``)
    2. the leg's own fields
    3. ``extra_kwargs`` — a caller's provider options (``extra_body``, a
       reasoning-model ``temperature`` override, user ``llm_kwargs``) win over
       the base fields without a "multiple values for keyword" error
    4. trace headers, merged (not replaced) into any caller ``extra_headers`` so
       the active span's traceparent propagates either way, then pipeline
       metadata, which fills in defaults only.

    Note ``extra_kwargs`` wins over the leg's own fields too: structural keys are
    already rejected by the caller, so what remains cannot displace the schema.
    """

    def _params(leg_kwargs: dict[str, Any]) -> dict[str, Any]:
        base: dict[str, Any] = {
            'model': model,
            # Cast — the OpenAI SDK accepts dict literals at runtime; the
            # TypedDict union just doesn't type-narrow from dict[str, Any].
            'messages': cast('Any', messages),
            # max_completion_tokens, not max_tokens: OpenAI rejects max_tokens
            # outright for the o-series and gpt-5 families, and every other chat
            # call in the repo already sends this key.
            'max_completion_tokens': max_tokens,
            **leg_kwargs,
        }
        if temperature is not None:
            base['temperature'] = temperature
        if extra_kwargs:
            base.update(extra_kwargs)
        if trace_headers:
            base['extra_headers'] = {**(base.get('extra_headers') or {}), **trace_headers}
        apply_pipeline_metadata(base)
        return base

    return _params


def _validate_content(content: str, *, response_format: type[T], label: str, leg: str) -> T | None:
    """Fence-tolerant parse + schema validation of one leg's text output.

    The single salvage implementation, shared by the ``json_schema`` leg, the
    tool leg's ``arguments`` string and the ``json_object`` leg — a provider
    degraded far enough down the ladder to answer in prose is exactly the one
    that wraps its JSON in a ```json fence. Returns ``None`` after a warning
    when nothing validates, which lets the caller try the next leg rather than
    handing back content no one checked.
    """
    if not content:
        return None
    try:
        return response_format.model_validate_json(extract_json_from_response(content))
    except (ValidationError, ValueError) as exc:
        logger.warning('%s: %s output did not validate against %s (%s)', label, leg, response_format.__name__, exc)
        return None


def _mark_leg(span: Span | None, leg: str) -> None:
    """Record which rung of the ladder answered.

    With four rungs a boolean cannot say which one paid, so the leg name is the
    real attribute. ``fallback`` is kept alongside it so dashboards built on the
    old boolean do not go blank.
    """
    if span is None:
        return
    span.set_attribute('orq.structured_output.leg', leg)
    span.set_attribute('orq.structured_output.fallback', leg != _LEG_PARSE)


async def _leg_strict_parse(
    client: AsyncOpenAI,
    params_for: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    response_format: type[T],
    label: str,
    span: Span | None,
    max_tokens: int,
) -> tuple[T | None, str]:
    """Rung 1: strict schema-enforced structured output via ``parse()``.

    Continues the ladder on a 400 whose body names the schema form, and on a
    response the SDK could not validate. Raises on a refusal (it must not be
    retried on another rung), on truncation, and on any other error.
    """
    try:
        response = await with_retry(
            lambda: client.chat.completions.parse(**params_for({'response_format': response_format})),  # pyright: ignore[reportUnknownLambdaType]
            label=label,
        )
    except APIStatusError as e:
        if e.status_code != 400 or not _looks_like_schema_rejection(e):
            raise
        logger.warning('%s: structured output not supported by model, trying the non-strict schema', label)
        return None, ''
    except LengthFinishReasonError as exc:
        raise _truncated_output_error(label, max_tokens) from exc
    record_llm_response(span, response)
    message = response.choices[0].message
    refusal = getattr(message, 'refusal', None)
    if refusal:
        raise RuntimeError(f'{label}: model refused to generate: {refusal}')
    parsed = message.parsed
    if parsed is None:
        logger.debug('%s: parse() returned None, trying the non-strict schema', label)
        return None, ''
    _mark_leg(span, _LEG_PARSE)
    return parsed, ''


async def _leg_json_schema(
    client: AsyncOpenAI,
    params_for: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    response_format: type[T],
    label: str,
    span: Span | None,
    max_tokens: int,
) -> tuple[T | None, str]:
    """Rung 2: the same schema, non-strict, via plain ``create()``.

    Non-strict on purpose: strict mode is what rung 1 already tried and what the
    provider just rejected. The schema still names the fields; only the
    enforcement is relaxed, and it is the only thing telling the model which keys
    to emit. Content that does not validate continues the ladder rather than
    returning here — the rungs below are stricter, not looser.
    """
    schema_format: dict[str, Any] = {
        'type': 'json_schema',
        'json_schema': {
            'name': response_format.__name__,
            'strict': False,
            'schema': response_format.model_json_schema(),
        },
    }
    try:
        response = await with_retry(
            lambda: client.chat.completions.create(**params_for({'response_format': schema_format})),  # pyright: ignore[reportUnknownLambdaType]
            label=f'{label} (json_schema fallback)',
        )
    except APIStatusError as e:
        if e.status_code != 400:
            raise
        logger.warning('%s: json_schema not accepted, trying a forced tool call', label)
        return None, ''
    return _content_result(
        response, response_format=response_format, label=label, span=span, max_tokens=max_tokens, leg=_LEG_JSON_SCHEMA
    )


async def _leg_forced_tool(
    client: AsyncOpenAI,
    params_for: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    response_format: type[T],
    messages: list[dict[str, Any]],
    label: str,
    span: Span | None,
    max_tokens: int,
    extra_kwargs: dict[str, Any] | None,
) -> tuple[T | None, str]:
    """Rung 3: force a named tool call carrying the same schema.

    Stricter than the two rungs below it: ``tool_choice`` naming the function
    makes emitting the schema the only legal output, so there is no prose channel
    at all — no fences, no preamble. It is a different provider capability than
    ``response_format``, which is the point: routed and local models that 400 on
    a JSON schema often do support function calling, and today those drop
    straight to bare ``json_object``.

    **This is the only rung that changes the prompt.** It appends one user turn
    telling the model the tool will be called, which rescues providers that
    accept ``tools`` but quietly downgrade a named ``tool_choice`` to auto. The
    turn is passed through ``params_for``, never appended to the caller's list,
    so it is visible in the span's recorded input rather than hidden — and if a
    model's answer differs between rungs, the prompt is one of the reasons.

    Skipped entirely when the caller supplied their own ``tools`` or
    ``tool_choice``: those are functional, and forcing ours would break the call
    this leg is only trying to salvage.
    """
    caller_kwargs = extra_kwargs or {}
    if 'tools' in caller_kwargs or 'tool_choice' in caller_kwargs:
        logger.debug('%s: caller supplied tools, skipping the forced tool call leg', label)
        return None, ''

    tool = pydantic_function_tool(response_format)
    tool_name = tool['function']['name']
    leg_kwargs: dict[str, Any] = {
        'tools': [tool],
        'tool_choice': {'type': 'function', 'function': {'name': tool_name}},
        'messages': cast(
            'Any',
            [
                *messages,
                {
                    'role': 'user',
                    'content': (
                        f'Respond by calling the `{tool_name}` tool with the requested fields. Do not reply with text.'
                    ),
                },
            ],
        ),
    }
    try:
        response = await with_retry(
            lambda: client.chat.completions.parse(**params_for(leg_kwargs)),  # pyright: ignore[reportUnknownLambdaType]
            label=f'{label} (forced tool call)',
        )
    except APIStatusError as e:
        if e.status_code != 400:
            raise
        logger.warning('%s: forced tool call not accepted, falling back to json_object', label)
        return None, ''
    except LengthFinishReasonError as exc:
        raise _truncated_output_error(label, max_tokens) from exc
    record_llm_response(span, response)
    if not response.choices:
        return None, ''
    message = response.choices[0].message
    refusal = getattr(message, 'refusal', None)
    if refusal:
        raise RuntimeError(f'{label}: model refused to generate: {refusal}')
    if response.choices[0].finish_reason == 'length':
        raise _truncated_output_error(label, max_tokens)

    tool_calls = message.tool_calls or []
    if not tool_calls:
        logger.warning('%s: forced tool call returned no tool_calls, falling back to json_object', label)
        return None, ''
    function = tool_calls[0].function
    # .parse() validates tool arguments against the same model for us; the raw
    # argument string is the fallback when the SDK could not (a non-OpenAI
    # provider whose tool_call shape the SDK does not narrow).
    parsed = cast('T | None', getattr(function, 'parsed_arguments', None))
    raw = function.arguments or ''
    if parsed is None:
        parsed = _validate_content(raw, response_format=response_format, label=label, leg=_LEG_TOOL)
    if parsed is None:
        return None, raw
    _mark_leg(span, _LEG_TOOL)
    return parsed, raw


async def _leg_json_object(
    client: AsyncOpenAI,
    params_for: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    response_format: type[T],
    label: str,
    span: Span | None,
    max_tokens: int,
) -> tuple[T | None, str]:
    """Rung 4: bare ``json_object``, the last rung.

    Asks for "some JSON" and leaves the field names to chance, which is what
    made fence-tolerant parsing necessary in the first place — so this is where
    the ladder ends, not where it starts. A provider that rejects even this
    raises, since there is nothing left to degrade to.
    """
    response = await with_retry(
        lambda: client.chat.completions.create(**params_for({'response_format': {'type': 'json_object'}})),  # pyright: ignore[reportUnknownLambdaType]
        label=f'{label} (json_object fallback)',
    )
    return _content_result(
        response, response_format=response_format, label=label, span=span, max_tokens=max_tokens, leg=_LEG_JSON_OBJECT
    )


def _content_result(
    response: Any,
    *,
    response_format: type[T],
    label: str,
    span: Span | None,
    max_tokens: int,
    leg: str,
) -> tuple[T | None, str]:
    """Turn a text-answering leg's response into ``(parsed | None, raw)``.

    Shared by the two ``create()`` rungs. Truncation raises here rather than
    degrading: the SDK raises ``LengthFinishReasonError`` for us on the
    ``parse()`` rungs but not on these, where a cut-off body comes back looking
    like ordinary content — ``extract_json_from_response`` would salvage half an
    object and the caller would score a half-answer.
    """
    record_llm_response(span, response)
    if not response.choices:
        return None, ''
    choice = response.choices[0]
    if choice.finish_reason == 'length':
        raise _truncated_output_error(label, max_tokens)
    content = choice.message.content or ''
    parsed = _validate_content(content, response_format=response_format, label=label, leg=leg)
    if parsed is not None:
        _mark_leg(span, leg)
    return parsed, content


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
    """Generate structured output, degrading through four chat rungs.

    ``api`` selects the endpoint tried first. The default ``chat_completions``
    runs the ladder described in the module docstring. ``responses`` puts a raw
    ``client.responses.create()`` call carrying the strict schema in front of it
    and falls through to that same ladder when the endpoint is absent, names the
    schema form unsupported, or returns no output — so this helper's contract
    does not depend on the provider implementing the Responses API. On the
    ``responses`` success path the returned raw-content string is always ``""``.
    Structural fields are reserved per endpoint, and an ``api='responses'`` call
    reserves both sets because it may still reach the chat legs.

    Returns ``(parsed_model, raw_content)`` from whichever rung answered — the
    raw string is ``""`` on the two rungs the SDK validates for us, and the
    model's own text on the rungs parsed here. ``(None, raw_content)`` means no
    rung produced anything that validated, with ``raw_content`` the last
    non-empty text seen so a caller still has something to log or salvage.
    ``parsed is None`` therefore means "nothing validated" uniformly.

    ``temperature`` is sent only when not ``None`` (some callers deliberately let
    the provider default stand). ``extra_kwargs`` is merged into every rung's
    params by ``_chat_params``, whose docstring states the order; structural
    fields (``model``, ``messages``, ``response_format``) are reserved and raise
    ``ValueError``, since an ``extra_kwargs`` entry silently replacing the schema
    would defeat the helper. A caller's own ``tools``/``tool_choice`` are not
    reserved — they skip the tool rung instead.

    On a length-truncated response this raises ``RuntimeError`` rather than
    falling back (a same-budget retry would truncate again). "Loud" is scoped to
    this helper: it surfaces a specific, actionable reason instead of returning
    cut-off JSON. Both report call sites still wrap the call in a broad
    ``except`` and skip that one item, so a truncation degrades a single section
    — but with a clear log line naming the budget, not a silent drop.

    RES-1295: none of the chat rungs extract usage, so their tokens never reach
    any total. The 11 call sites across ``persona_generator.py``,
    ``scenario_generator.py``, ``traces.py``, and both ``recommendations.py``
    modules track no usage today — adding a usage element to this function's
    return tuple would mean threading it through every one of them. See "What
    the totals do not include" in docs/guides/red-teaming.md.
    """
    reserved = _STRUCTURAL_KEYS_BY_API[api] & (extra_kwargs or {}).keys()
    if reserved:
        raise ValueError(
            f'extra_kwargs cannot override structural request field(s) {sorted(reserved)}; '
            'these are owned by generate_structured.'
        )

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
        params_for = _chat_params(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_kwargs=extra_kwargs,
            trace_headers=await get_trace_context_headers(),
        )
        common: dict[str, Any] = {
            'response_format': response_format,
            'label': label,
            'span': span,
            'max_tokens': max_tokens,
        }
        # Thunks, not coroutines: returning on rung 1 would leave three
        # never-awaited coroutine objects behind (and a RuntimeWarning each).
        legs: tuple[Callable[[], Awaitable[tuple[T | None, str]]], ...] = (
            lambda: _leg_strict_parse(client, params_for, **common),
            lambda: _leg_json_schema(client, params_for, **common),
            lambda: _leg_forced_tool(client, params_for, messages=messages, extra_kwargs=extra_kwargs, **common),
            lambda: _leg_json_object(client, params_for, **common),
        )
        last_raw = ''
        for leg in legs:
            parsed, raw = await leg()
            last_raw = raw or last_raw
            if parsed is not None:
                return parsed, raw
        return None, last_raw
