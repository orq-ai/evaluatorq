"""Structured-output helper with a degrading four-rung chat ladder, shared across domains.

Each rung asks for the same schema in a way the previous rung's provider might
still support; the first one that yields something validating against the schema
wins. Every rung's text output goes through ``common.extract_json`` before
validation, so a ```json-fenced payload comes back parsed rather than raw.

===  ==========================  ================  ==================================
#    Rung                        Prompt            Continues the ladder on
===  ==========================  ================  ==================================
1    strict ``parse()``          caller's          schema-rejection 400, no ``parsed``
2    non-strict ``json_schema``  caller's          400, content fails validation
3    forced tool call            **+1 user turn**  400, no ``tool_calls``, invalid args
4    bare ``json_object``        caller's          — terminal
===  ==========================  ================  ==================================

Rung 2 keeps the schema because most providers that reject ``.parse()`` reject
*strict* mode, and the schema is the only thing telling the model which keys to
emit. Rung 3 sits below the prompt-preserving rungs and above the schema-less
one: function calling is a different provider capability than
``response_format``, so a model that 400s on a JSON schema often still supports
it. It is the only rung that edits the prompt (one appended user turn) and is
skipped when the caller supplied their own ``tools``. Rung 4 asks for "some JSON"
and leaves the field names to chance — hence last.

Truncation and refusals never continue the ladder: a same-budget retry truncates
in the same place, and a refusal must not be re-asked on another rung.

``api='responses'`` puts a ``client.responses.create()`` call carrying the same
strict schema in front of the ladder. It is a leg, not a replacement: a provider
without the endpoint, or one that names the schema form unsupported, falls
through to the chat rungs.

Every rung is transported by ``common.llm_call``'s executors (concurrency slot,
per-request timeout, span recording, reserved-key guard, ``reasoning_effort``
drop-and-retry-once). ``with_retry`` around each rung is the only retry layer;
the client's SDK budget is disarmed once at the top of ``generate_structured``.
Only the fallback policy lives here.

One call can bill up to five provider requests, so `StructuredResult.usage` is
the **sum** over every rung that reached the provider, and a call that raises
carries the same total on the exception (`StructuredGenerationError`,
`usage_from_exception`). A rung whose usage block cannot be read counts as one
unpriced call after a warning, never as zero.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, NamedTuple, TypeVar, cast

from openai import APIStatusError, AsyncOpenAI, LengthFinishReasonError, pydantic_function_tool
from pydantic import BaseModel, ValidationError

from evaluatorq.common.extract_json import extract_json_from_response
from evaluatorq.common.llm_call import execute_chat_completion, execute_chat_parse, execute_response
from evaluatorq.common.model_catalogue import price_usage
from evaluatorq.common.responses import first_responses_refusal, parse_responses_response, responses_stop_reason
from evaluatorq.common.retry import with_retry, without_client_retries
from evaluatorq.common.tracing import with_llm_span
from evaluatorq.contracts import (
    _RESERVED_COMPLETION_KEYS,
    _RESERVED_RESPONSES_KEYS,
    LLMCallConfig,
    TokenUsage,
    check_reserved_keys,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.trace import Span

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

# The chat ladder's rungs, in order. Recorded on the span so a run says which
# one paid rather than only that *some* fallback did.
_LEG_PARSE = 'parse'
_LEG_JSON_SCHEMA = 'json_schema'
_LEG_TOOL = 'tools'
_LEG_JSON_OBJECT = 'json_object'


@dataclass(frozen=True)
class StructuredResult(Generic[T]):
    """What one ``generate_structured`` call produced, including what it cost.

    ``parsed`` is ``None`` when no rung produced anything that validated;
    ``raw`` is then the last non-empty text seen, for a caller with its own
    salvage. ``usage`` is the sum over every rung that reached the provider;
    ``None`` means no rung ever reached one.
    """

    parsed: T | None
    raw: str
    usage: TokenUsage | None = None


class StructuredGenerationError(RuntimeError):
    """A structured-generation failure that carries what the ladder already billed.

    Subclasses ``RuntimeError`` so callers matching on that keep working. Harvest
    the spend with `usage_from_exception` rather than reading the attribute —
    a provider error from the last rung propagates as itself, tagged the same way.
    """

    def __init__(self, message: str, *, usage: TokenUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


def _attach_usage(exc: BaseException, usage: TokenUsage | None) -> None:
    """Tag ``exc`` with the usage billed before it was raised, keeping its own type."""
    if usage is None:
        return
    try:
        exc.usage = usage  # pyright: ignore[reportAttributeAccessIssue]
    except AttributeError:  # pragma: no cover - defensive, no known such class
        logger.debug('could not attach structured-output usage to %s', type(exc).__name__)


def usage_from_exception(exc: BaseException) -> TokenUsage | None:
    """Spend the ladder billed before it raised, or ``None`` if it billed nothing.

    ``getattr`` rather than ``except StructuredGenerationError``: a provider error
    from the last rung propagates as **itself** so the caller keeps the status code.
    """
    usage = getattr(exc, 'usage', None)
    return usage if isinstance(usage, TokenUsage) else None


def sum_structured_usage(usages: Sequence[TokenUsage | None]) -> TokenUsage | None:
    """Add up `StructuredResult.usage` values, keeping ``None`` for "nothing was billed".

    Uses `Usage.__add__`, which carries ``calls`` and ``priced_calls`` through so
    an aggregate mixing priced and unpriced calls still answers `cost_is_partial`.
    """
    known = [u for u in usages if u is not None]
    if not known:
        return None
    total = known[0]
    for usage in known[1:]:
        total = total + usage
    return total


def log_structured_usage(usage: TokenUsage | None, *, phase: str) -> None:
    """Announce what a phase of structured generation cost.

    The call sites sit on paths with no report field for their usage (generators
    run before a run object exists, recommendation writers after the summary is
    final), so each phase logs its total here. Deliberately not recorded on the
    enclosing span: the child LLM spans already carry it, and the aggregate would
    double-count.
    """
    if usage is None:
        logger.info('%s: no usage reported by the provider', phase)
        return
    cost = f'${usage.total_cost:.4f}' if usage.total_cost is not None else 'cost unknown'
    if usage.cost_is_partial:
        cost += f' (priced for {usage.priced_calls} of {usage.calls} calls)'
    logger.info(
        '%s: %d tokens over %d LLM call(s), %s',
        phase,
        usage.total_tokens,
        usage.calls,
        cost,
    )


def _rung_usage(usage: TokenUsage | None, response: Any, *, label: str, leg: str) -> TokenUsage:
    """The ladder's unknown-usage policy: an unreadable block is one unpriced call, never $0.

    Pricing itself happens in the executor that made the call; summed against a
    rung that did report a cost, this makes `Usage.cost_is_partial` true.
    """
    if usage is None:
        logger.warning(
            '%s: the %s rung returned no readable usage block (%r); '
            'counting it as one unpriced call rather than as zero cost',
            label,
            leg,
            getattr(response, 'usage', None),
        )
        return TokenUsage(calls=1)
    return usage


async def _truncation_usage(exc: LengthFinishReasonError, call: _ChatLadderCall, *, leg: str) -> TokenUsage:
    """Harvest the billed usage off a raised ``LengthFinishReasonError``.

    The SDK attaches the ``ChatCompletion`` it refused to parse, usage block and
    all; the executor never returned, so pricing happens here instead.
    """
    completion = getattr(exc, 'completion', None)
    usage = await price_usage(TokenUsage.from_completion(completion), call.model, call.client)
    return _rung_usage(usage, completion, label=call.label, leg=leg)


# Structural request fields extra_kwargs may not replace, per endpoint asked
# for. An api='responses' call can fall through to the chat legs, so its set is
# the union: a key safe on one endpoint is not safe on the one it degrades to.
_STRUCTURAL_KEYS: dict[str, frozenset[str]] = {
    'chat_completions': _RESERVED_COMPLETION_KEYS | {'max_completion_tokens'},
    'responses': (
        _RESERVED_COMPLETION_KEYS
        | {'max_completion_tokens'}
        | _RESERVED_RESPONSES_KEYS
        | {'text_format', 'max_output_tokens'}
    ),
}

# A provider that cannot do schema-enforced output says so in the error body.
# A bare 400 is far more often a bad parameter, an over-length context or a
# content-policy rejection; degrading on those masks the cause and re-bills it.
_SCHEMA_KEYWORDS = (
    'response_format',
    'json_schema',
    'text_format',
    'text.format',
    'structured output',
)

# The forced-tool rung fails on a different capability, so it recognises a
# different vocabulary. Same rule: a bare 400 is not evidence.
_TOOL_KEYWORDS = (
    'tool_choice',
    'tools',
    'function call',
    'function_call',
)


def _looks_like_capability_rejection(exc: APIStatusError, keywords: tuple[str, ...]) -> bool:
    err_body = str(getattr(exc, 'body', None) or getattr(exc, 'message', '') or '').lower()
    return any(kw in err_body for kw in keywords)


def _looks_like_schema_rejection(exc: APIStatusError) -> bool:
    return _looks_like_capability_rejection(exc, _SCHEMA_KEYWORDS)


# Per-request ceiling. A batched generation asking for tens of items is a slow
# call by design, so this is well above LLMCallConfig's 90s default.
_STRUCTURED_TIMEOUT_S = 300.0


class Unset:
    """Sentinel for "this keyword was not passed".

    ``None`` cannot serve: it is a meaningful explicit value for every optional
    sampling keyword here (``temperature=None`` means "omit the parameter", which
    is what a reasoning-class model needs). Overloading it made
    ``generate_structured``'s documented precedence — an explicit keyword beats
    ``config`` — untrue for exactly the callers who most needed it.
    """

    def __repr__(self) -> str:
        return '<unset>'


UNSET = Unset()


class _CallSettings(NamedTuple):
    """The sampling knobs after the caller's keywords and ``config`` are folded."""

    temperature: float | None
    extra_kwargs: dict[str, Any] | None
    extra_body: dict[str, Any] | None
    reasoning_effort: str | None
    timeout_s: float


# `model` and `api` stay the call site's authority, so a config cannot redirect a call to another endpoint.
_CONSUMED_CONFIG_FIELDS = frozenset({
    'model',
    'client',
    'temperature',
    'extra_kwargs',
    'extra_body',
    'reasoning_effort',
    'timeout_ms',
})


def warn_unread_config_fields(config: LLMCallConfig | None, read: frozenset[str], *, caller: str) -> None:
    """Warn that ``caller`` will not read fields the caller set on ``config``.

    A call site that sizes its own budget, picks its own endpoint or owns its
    own retry loop takes those fields off the config. Dropping them without a
    word makes a config that did nothing look like a config that worked, so
    every such call site says which ones it ignored.
    """
    if config is None:
        return
    ignored = config.model_fields_set - read
    if ignored:
        logger.warning(
            '%s ignores llm_config %s — this call site owns those. Only %s are read.',
            caller,
            ', '.join(sorted(ignored)),
            ', '.join(sorted(read - {'model', 'client'})),
        )


def _fold_config(
    *,
    config: LLMCallConfig | None,
    temperature: float | Unset | None,
    extra_kwargs: dict[str, Any] | Unset | None,
    extra_body: dict[str, Any] | Unset | None,
    reasoning_effort: str | Unset | None,
    timeout_s: float | Unset,
) -> _CallSettings:
    """Explicit keyword beats ``config`` beats the call-site default.

    A keyword still at ``UNSET`` is one the caller never passed; anything else
    they meant, ``None`` included.
    """
    from_config = (
        config.set_values('temperature', 'extra_kwargs', 'extra_body', 'reasoning_effort') if config is not None else {}
    )
    warn_unread_config_fields(config, _CONSUMED_CONFIG_FIELDS, caller='generate_structured')
    if isinstance(timeout_s, Unset):
        resolved_timeout_s = config.timeout_s(_STRUCTURED_TIMEOUT_S) if config is not None else _STRUCTURED_TIMEOUT_S
    else:
        resolved_timeout_s = timeout_s
    return _CallSettings(
        temperature=from_config.get('temperature') if isinstance(temperature, Unset) else temperature,
        extra_kwargs=from_config.get('extra_kwargs') if isinstance(extra_kwargs, Unset) else extra_kwargs,
        extra_body=from_config.get('extra_body') if isinstance(extra_body, Unset) else extra_body,
        reasoning_effort=(
            from_config.get('reasoning_effort') if isinstance(reasoning_effort, Unset) else reasoning_effort
        ),
        timeout_s=resolved_timeout_s,
    )


def _truncated_output_error(
    label: str,
    max_tokens: int,
    usage: TokenUsage | None = None,
) -> StructuredGenerationError:
    """The actionable error for a provider-reported cut-off payload.

    Truncated structured output is unrecoverable — a retry at the same budget
    truncates in the same place — so every rung raises this rather than degrading.
    ``usage`` is the raising rung's own billed usage; `generate_structured`
    replaces it with the ladder total on the way out.
    """
    logger.error('%s: output truncated at the token limit (max_tokens=%s)', label, max_tokens)
    return StructuredGenerationError(
        f'{label}: the model hit the token limit (max_tokens={max_tokens}) and the structured output is unusable. '
        f'Raise the max_tokens budget passed to this call and retry.',
        usage=usage,
    )


def token_budget_for_items(count: int, *, per_item: int, minimum: int) -> int:
    """Scale a ``max_tokens`` budget with how many items one call asks for.

    A flat cap truncates once N grows, and truncated structured output raises
    rather than retrying. Deliberately unbounded at the top: a count that exceeds
    the model's own limit fails with the provider naming the limit, which beats
    silent truncation at a ceiling picked here.
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
    reasoning_effort: str | None = None,
    timeout_s: float = _STRUCTURED_TIMEOUT_S,
    extra_body: dict[str, Any] | None = None,
) -> StructuredResult[T]:
    """Structured output through the Responses API; ``parsed is None`` means "use the chat legs".

    Falls through (after a warning naming the cause) on a 404, on a 400 whose
    body names the schema form, and on a response with nothing parsed. Refusals,
    truncation and validation failures raise: none of them is a capability gap.
    Any other 400 re-raises for the same reason.

    ``usage`` is carried on the fall-throughs the provider answered; the two
    rejection fall-throughs return ``usage=None`` because a refused request never
    ran a model.
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
            response, raw_usage = await with_retry(
                lambda: execute_response(
                    client=client,
                    model=model,
                    messages=messages,
                    span=span,
                    timeout_s=timeout_s,
                    response_text_format=response_format,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    reasoning_effort=reasoning_effort,
                    extra_body=extra_body,
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
            return StructuredResult(None, '', None)
        usage = _rung_usage(raw_usage, response, label=label, leg='Responses')
        if responses_stop_reason(response) == 'length':
            raise _truncated_output_error(label, max_tokens, usage)

        refusal = first_responses_refusal(response)
        if refusal is not None:
            raise StructuredGenerationError(f'{label}: model refused to generate: {refusal}', usage=usage)

        raw_content = getattr(response, 'output_text', '') or ''
        if not raw_content:
            logger.warning('%s: Responses returned no output, falling back to chat.completions', label)
            return StructuredResult(None, '', usage)

        try:
            parsed_response = parse_responses_response(
                response,
                response_format,
                input_tools=(extra_kwargs or {}).get('tools'),
            )
        except ValidationError as exc:
            logger.exception('%s: Responses output did not validate against %s', label, response_format.__name__)
            raise StructuredGenerationError(
                f'{label}: the Responses output did not validate against {response_format.__name__}.',
                usage=usage,
            ) from exc
        parsed = getattr(parsed_response, 'output_parsed', None)
        if parsed is None:
            logger.warning('%s: Responses returned no parsed output, falling back to chat.completions', label)
            return StructuredResult(None, '', usage)
        return StructuredResult(cast('T', parsed), '', usage)


@dataclass(frozen=True)
class _ChatLadderCall:
    """The half of a chat rung's request that does not change between rungs.

    Each rung adds only its own fields (a ``response_format``, or
    ``tools``/``tool_choice`` and an edited message list). Merge order is the
    executors' (`execute_chat_completion` / `execute_chat_parse`): structural
    fields, then the rung's own, then ``extra_kwargs`` last so a caller's
    provider options win, then pipeline metadata and trace headers.

    ``span`` is the ladder's single chat span, shared by every rung; the rung
    that answered is recorded on it by `_mark_leg`.
    """

    client: AsyncOpenAI
    model: str
    messages: list[dict[str, Any]]
    max_tokens: int
    label: str
    span: Span | None
    temperature: float | None = None
    reasoning_effort: str | None = None
    timeout_s: float = _STRUCTURED_TIMEOUT_S
    extra_body: dict[str, Any] | None = None
    extra_kwargs: dict[str, Any] | None = None

    def executor_kwargs(self, **rung: Any) -> dict[str, Any]:
        """The shared executor arguments, with a rung's own fields layered on."""
        return {
            'client': self.client,
            'model': self.model,
            'messages': self.messages,
            'span': self.span,
            'timeout_s': self.timeout_s,
            'temperature': self.temperature,
            # max_completion_tokens, not max_tokens: OpenAI rejects max_tokens
            # outright for the o-series and gpt-5 families.
            'max_completion_tokens': self.max_tokens,
            'reasoning_effort': self.reasoning_effort,
            'extra_body': self.extra_body,
            'extra_kwargs': self.extra_kwargs,
            **rung,
        }


def _validate_content(content: str, *, response_format: type[T], label: str, leg: str) -> T | None:
    """Fence-tolerant parse + schema validation of one leg's text output.

    Shared by the ``json_schema`` leg, the tool leg's ``arguments`` string and
    the ``json_object`` leg. Returns ``None`` after a warning when nothing
    validates, so the caller tries the next leg. Empty content warns too, so a
    provider that answers with nothing does not look like one that answered
    off-schema.
    """
    if not content:
        logger.warning('%s: the %s rung returned empty content, continuing the ladder', label, leg)
        return None
    try:
        return response_format.model_validate_json(extract_json_from_response(content))
    except (ValidationError, ValueError) as exc:
        logger.warning('%s: %s output did not validate against %s (%s)', label, leg, response_format.__name__, exc)
        return None


def _mark_leg(span: Span | None, leg: str) -> None:
    """Record which rung answered. ``fallback`` is kept for dashboards built on the old boolean."""
    if span is None:
        return
    span.set_attribute('orq.structured_output.leg', leg)
    span.set_attribute('orq.structured_output.fallback', leg != _LEG_PARSE)


def _record_tool_nudge(span: Span | None, nudge: str) -> None:
    """Record rung 3's appended user turn under its own span attribute.

    `record_llm_input` **sets** ``gen_ai.input.messages`` on the shared span, so
    a rung 4 running after rung 3 overwrites the nudged list — exactly the run
    where a reader needs to see it. A dedicated attribute survives.
    """
    if span is None:
        return
    span.set_attribute('orq.structured_output.tool_nudge', nudge)


async def _parse_rung(
    call: _ChatLadderCall,
    *,
    leg: str,
    retry_label: str,
    rejection_keywords: tuple[str, ...],
    rejected: str,
    no_choices: str,
    **rung: Any,
) -> tuple[Any, TokenUsage | None]:
    """Run one ``parse()`` rung and apply the policy shared by rungs 1 and 3.

    Returns ``(message, usage)``. ``message`` is ``None`` when the ladder should
    continue — a 400 whose body names the capability (``usage`` then ``None``:
    nothing was billed), or a response with no ``choices``. Raises on a refusal,
    on truncation (with the billed usage harvested off the SDK's exception) and
    on any other error.
    """
    label = call.label
    try:
        response, raw_usage = await with_retry(
            lambda: execute_chat_parse(**call.executor_kwargs(**rung)),
            label=retry_label,
        )
    except APIStatusError as e:
        if e.status_code != 400 or not _looks_like_capability_rejection(e, rejection_keywords):
            raise
        logger.warning('%s: %s', label, rejected)
        return None, None
    except LengthFinishReasonError as exc:
        raise _truncated_output_error(
            label,
            call.max_tokens,
            await _truncation_usage(exc, call, leg=leg),
        ) from exc
    usage = _rung_usage(raw_usage, response, label=label, leg=leg)
    if not response.choices:
        logger.warning('%s: %s', label, no_choices)
        return None, usage
    message = response.choices[0].message
    refusal = getattr(message, 'refusal', None)
    if refusal:
        raise StructuredGenerationError(f'{label}: model refused to generate: {refusal}', usage=usage)
    return message, usage


async def _leg_strict_parse(call: _ChatLadderCall, *, response_format: type[T]) -> StructuredResult[T]:
    """Rung 1: strict schema-enforced structured output via ``parse()``."""
    message, usage = await _parse_rung(
        call,
        leg=_LEG_PARSE,
        retry_label=call.label,
        rejection_keywords=_SCHEMA_KEYWORDS,
        rejected='structured output not supported by model, trying the non-strict schema',
        no_choices='parse() returned no choices, trying the non-strict schema',
        response_model=response_format,
    )
    if message is None:
        return StructuredResult(None, '', usage)
    parsed = message.parsed
    if parsed is None:
        logger.debug('%s: parse() returned None, trying the non-strict schema', call.label)
        return StructuredResult(None, '', usage)
    _mark_leg(call.span, _LEG_PARSE)
    return StructuredResult(cast('T', parsed), '', usage)


async def _leg_json_schema(call: _ChatLadderCall, *, response_format: type[T]) -> StructuredResult[T]:
    """Rung 2: the same schema, non-strict, via plain ``create()``.

    Strict mode is what rung 1 just had rejected; the schema still names the
    fields. Content that does not validate continues the ladder.
    """
    label = call.label
    schema_format: dict[str, Any] = {
        'type': 'json_schema',
        'json_schema': {
            'name': response_format.__name__,
            'strict': False,
            'schema': response_format.model_json_schema(),
        },
    }
    try:
        response, raw_usage = await with_retry(
            lambda: execute_chat_completion(**call.executor_kwargs(response_format=schema_format)),
            label=f'{label} (json_schema fallback)',
        )
    except APIStatusError as e:
        if e.status_code != 400 or not _looks_like_schema_rejection(e):
            raise
        logger.warning('%s: json_schema not accepted, trying a forced tool call', label)
        return StructuredResult(None, '')
    return _content_result(
        response,
        raw_usage,
        call=call,
        response_format=response_format,
        leg=_LEG_JSON_SCHEMA,
    )


async def _leg_forced_tool(call: _ChatLadderCall, *, response_format: type[T]) -> StructuredResult[T]:
    """Rung 3: force a named tool call carrying the same schema.

    ``tool_choice`` naming the function leaves no prose channel at all, and
    function calling is a different capability than ``response_format``.

    **The only rung that changes the prompt.** It appends one user turn telling
    the model the tool will be called, which rescues providers that accept
    ``tools`` but quietly downgrade a named ``tool_choice`` to auto. Sent as this
    rung's own message list, never appended to the caller's, and recorded on the
    span by `_record_tool_nudge`. Skipped when the caller supplied their own
    ``tools``/``tool_choice``: forcing ours would break the call.
    """
    label = call.label
    caller_kwargs = call.extra_kwargs or {}
    if 'tools' in caller_kwargs or 'tool_choice' in caller_kwargs:
        logger.debug('%s: caller supplied tools, skipping the forced tool call leg', label)
        return StructuredResult(None, '')

    tool = pydantic_function_tool(response_format)
    tool_name = tool['function']['name']
    nudged_messages: list[dict[str, Any]] = [
        *call.messages,
        {
            'role': 'user',
            'content': (
                f'Respond by calling the `{tool_name}` tool with the requested fields. Do not reply with text.'
            ),
        },
    ]
    _record_tool_nudge(call.span, cast('str', nudged_messages[-1]['content']))
    # No response_model: the schema travels as the tool definition instead. The
    # SDK's TypedDict is a dict at runtime; widen it once for the executor.
    message, usage = await _parse_rung(
        call,
        leg=_LEG_TOOL,
        retry_label=f'{label} (forced tool call)',
        rejection_keywords=_SCHEMA_KEYWORDS + _TOOL_KEYWORDS,
        rejected='forced tool call not accepted, falling back to json_object',
        no_choices='forced tool call returned no choices, falling back to json_object',
        messages=nudged_messages,
        tools=[dict(tool)],
        tool_choice={'type': 'function', 'function': {'name': tool_name}},
    )
    if message is None:
        return StructuredResult(None, '', usage)
    tool_calls = message.tool_calls or []
    if not tool_calls:
        logger.warning('%s: forced tool call returned no tool_calls, falling back to json_object', label)
        return StructuredResult(None, '', usage)
    function = tool_calls[0].function
    # .parse() validates tool arguments against the same model for us; the raw
    # argument string is the fallback when the SDK could not.
    parsed = cast('T | None', getattr(function, 'parsed_arguments', None))
    raw = function.arguments or ''
    if parsed is None:
        parsed = _validate_content(raw, response_format=response_format, label=label, leg=_LEG_TOOL)
    if parsed is None:
        return StructuredResult(None, raw, usage)
    _mark_leg(call.span, _LEG_TOOL)
    return StructuredResult(parsed, raw, usage)


async def _leg_json_object(call: _ChatLadderCall, *, response_format: type[T]) -> StructuredResult[T]:
    """Rung 4: bare ``json_object``, the last rung. A provider that rejects even this raises."""
    response, raw_usage = await with_retry(
        lambda: execute_chat_completion(**call.executor_kwargs(response_format={'type': 'json_object'})),
        label=f'{call.label} (json_object fallback)',
    )
    return _content_result(
        response,
        raw_usage,
        call=call,
        response_format=response_format,
        leg=_LEG_JSON_OBJECT,
    )


def _content_result(
    response: Any,
    raw_usage: TokenUsage | None,
    *,
    call: _ChatLadderCall,
    response_format: type[T],
    leg: str,
) -> StructuredResult[T]:
    """Turn a text-answering leg's response into a `StructuredResult`.

    Shared by the two ``create()`` rungs. Truncation raises here: the SDK raises
    ``LengthFinishReasonError`` on the ``parse()`` rungs but not on these, where
    a cut-off body comes back looking like ordinary content.
    """
    usage = _rung_usage(raw_usage, response, label=call.label, leg=leg)
    if not response.choices:
        logger.warning('%s: the %s rung returned no choices, continuing the ladder', call.label, leg)
        return StructuredResult(None, '', usage)
    choice = response.choices[0]
    if choice.finish_reason == 'length':
        raise _truncated_output_error(call.label, call.max_tokens, usage)
    content = choice.message.content or ''
    parsed = _validate_content(content, response_format=response_format, label=call.label, leg=leg)
    if parsed is not None:
        _mark_leg(call.span, leg)
    return StructuredResult(parsed, content, usage)


async def generate_structured(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_format: type[T],
    max_tokens: int,
    label: str,
    temperature: float | Unset | None = UNSET,
    extra_kwargs: dict[str, Any] | Unset | None = UNSET,
    api: Literal['chat_completions', 'responses'] = 'chat_completions',
    reasoning_effort: str | Unset | None = UNSET,
    timeout_s: float | Unset = UNSET,
    extra_body: dict[str, Any] | Unset | None = UNSET,
    config: LLMCallConfig | None = None,
) -> StructuredResult[T]:
    """Generate structured output, degrading through the rungs in the module docstring.

    ``api`` selects the endpoint tried first; ``responses`` puts a Responses call
    in front of the chat ladder and falls through to it. On the ``responses``
    success path ``raw`` is always ``""``.

    Returns a `StructuredResult`. ``parsed is None`` means no rung produced
    anything that validated, with ``raw`` the last non-empty text seen. ``usage``
    is the sum over every rung that reached the provider; every exception leaving
    this function carries the same total as ``exc.usage``.

    ``reasoning_effort`` is sent only when truthy (flat on chat, as
    ``reasoning={'effort': ...}`` on Responses); a model that 400s on it is
    handled by the executors' drop-and-retry-once on both legs.

    ``timeout_s`` bounds **one request, not the call**: each rung is wrapped in
    `with_retry` (up to ``MAX_RETRY_ATTEMPTS`` = 5 attempts plus backoff), each
    attempt can issue a second request if the model rejects ``reasoning_effort``,
    and ``api='responses'`` adds a fifth rung. Worst case is on the order of
    *rungs x 5 x 2* requests — bound a whole call with an outer timeout.

    ``extra_kwargs`` is merged into every rung's params last, so a caller's
    provider options win. Structural fields (``_STRUCTURAL_KEYS[api]``) raise
    ``ValueError``; a caller's own ``tools``/``tool_choice`` are not reserved —
    they skip the tool rung instead.

    A length-truncated response raises `StructuredGenerationError` (a
    ``RuntimeError``) rather than falling back: a same-budget retry would
    truncate again.

    ``config`` supplies the sampling knobs a caller holds as one object rather
    than as five keywords: ``temperature``, ``extra_kwargs``, ``extra_body``,
    ``reasoning_effort`` and ``timeout_ms``. Only the fields the caller
    explicitly set on it are read (``model_fields_set``), and an explicit
    keyword here always wins — including ``temperature=None`` ("omit the
    parameter") and ``timeout_s=_STRUCTURED_TIMEOUT_S`` (the default, passed on
    purpose). That is why those keywords default to a private ``UNSET``
    sentinel rather than to ``None``: both spellings are real values a caller
    means, so neither can double as "said nothing".

    ``config.model`` and ``config.api`` are NOT read: this function's own
    ``model`` / ``api`` arguments stay the single authority, so a config cannot
    silently redirect a call to another endpoint.
    """
    temperature, extra_kwargs, extra_body, reasoning_effort, timeout_s = _fold_config(
        config=config,
        temperature=temperature,
        extra_kwargs=extra_kwargs,
        extra_body=extra_body,
        reasoning_effort=reasoning_effort,
        timeout_s=timeout_s,
    )
    # Every rung is wrapped in `with_retry`, so the SDK's own budget is disarmed
    # once here; otherwise five outer attempts over an SDK doing two retries is
    # fifteen requests per rung. `without_client_retries` clones rather than
    # mutates, so an injected client is untouched.
    client = without_client_retries(client)
    check_reserved_keys(extra_kwargs or {}, _STRUCTURAL_KEYS[api])
    # The router body travels in the dedicated `extra_body=` parameter; folding it
    # into extra_kwargs here would trip the reserved-key guard on every call.
    usages: list[TokenUsage | None] = []
    try:
        if api == 'responses':
            via_responses = await _generate_structured_via_responses(
                client,
                model=model,
                messages=messages,
                response_format=response_format,
                max_tokens=max_tokens,
                label=label,
                temperature=temperature,
                extra_kwargs=extra_kwargs,
                reasoning_effort=reasoning_effort,
                timeout_s=timeout_s,
                extra_body=extra_body,
            )
            usages.append(via_responses.usage)
            if via_responses.parsed is not None:
                return via_responses
            # The helper warned; its usage stays in `usages` because the leg
            # billed even though its payload was unusable.

        async with with_llm_span(
            model=model,
            operation='chat',
            temperature=temperature,
            max_tokens=max_tokens,
            input_messages=messages,
            attributes={'orq.llm.purpose': label},
        ) as span:
            call = _ChatLadderCall(
                client=client,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                label=label,
                span=span,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                timeout_s=timeout_s,
                extra_body=extra_body,
                extra_kwargs=extra_kwargs,
            )
            last_raw = ''
            for leg in (_leg_strict_parse, _leg_json_schema, _leg_forced_tool, _leg_json_object):
                result = await leg(call, response_format=response_format)
                usages.append(result.usage)
                last_raw = result.raw or last_raw
                if result.parsed is not None:
                    return StructuredResult(result.parsed, result.raw, sum_structured_usage(usages))
            logger.warning(
                '%s: all four chat rungs ran and none produced output validating against %s '
                '(%d provider call(s) billed); returning the last raw text for salvage',
                label,
                response_format.__name__,
                sum(1 for usage in usages if usage is not None),
            )
            return StructuredResult(None, last_raw, sum_structured_usage(usages))
    except Exception as exc:
        # The rungs below have already billed; fold what they spent onto the
        # exception (including the raising rung's own usage) so a caller's
        # `except` can still report it.
        _attach_usage(exc, sum_structured_usage([*usages, usage_from_exception(exc)]))
        raise
