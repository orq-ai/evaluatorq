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

Every rung — Responses leg and all four chat rungs — is transported by
``common.llm_call``'s executors, so the ladder inherits the concurrency slot,
the per-request timeout, span recording, trace-header injection, the
reserved-key guard and the ``reasoning_effort`` drop-and-retry-once rather than
re-deriving them, and the client's own SDK retry budget is disarmed once at the
top of ``generate_structured`` so the two retry layers cannot multiply. Only the
fallback policy lives here. (The chat rungs called
the SDK directly until RES-1295's follow-up; the visible symptom was that a
``reasoning_effort`` a model rejected killed the whole call on the chat legs
while the Responses leg self-healed.)

Because one call can bill up to five provider requests, the result carries the
usage of **all** of them (`StructuredResult.usage`), summed rather than taken
from the rung that answered — a rung that failed still billed (RES-1295). Rungs
are priced individually inside the executor that made them (through
``common.model_catalogue.price_usage``), and a rung whose usage block cannot be
read counts as one unpriced call after a warning, never as zero. A call that
raises carries the same total on the exception (`StructuredGenerationError`),
since the rungs below the raising one billed too.

Lives in ``common`` rather than ``simulation`` so both the simulation and
red-team report code can reuse one copy (RES-822). It delegates to the canonical
``common.tracing.with_llm_span``; a domain that needs its own span attributes
passes them through ``attributes``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast

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
    TokenUsage,
    check_reserved_keys,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

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

    A result object rather than a wider tuple: the ladder can bill up to five
    provider calls, so ``usage`` is a first-class part of the answer and reads
    better named than as a third positional slot at eleven call sites.

    ``parsed`` is ``None`` when no rung produced anything that validated;
    ``raw`` is then the last non-empty text seen, for a caller with its own
    salvage. ``usage`` is the **sum over every rung that reached the provider**,
    not just the one that answered — a call that burned rungs 1-3 before rung 4
    succeeded paid for all four. ``None`` means no rung ever reached a provider
    (e.g. every attempt was rejected before billing).

    A call that *raises* has no result object to carry usage on, so the same
    total is attached to the exception instead — see `StructuredGenerationError`.
    """

    parsed: T | None
    raw: str
    usage: TokenUsage | None = None


class StructuredGenerationError(RuntimeError):
    """A structured-generation failure that carries what the ladder already billed.

    The ladder can burn four or five provider calls before a rung raises
    (truncation, a refusal, a payload that does not validate). Those calls were
    billed, and before this type existed every one of them vanished on the way
    out: ``usages`` was folded into a `StructuredResult` only on the two
    ``return`` paths, so a run whose rungs 1-2 truncated and whose rung 3 raised
    reported "no usage reported by the provider".

    Subclasses ``RuntimeError`` because that is what these failures already
    raised — a caller matching on ``RuntimeError`` keeps working. Callers that
    want the spend harvest it defensively:

    ```python
    try:
        result = await generate_structured(...)
    except Exception as exc:
        usage = usage_from_exception(exc)
    ```

    Use `usage_from_exception` rather than reading the attribute directly — it is
    the one place the harvest rule is written down.
    """

    def __init__(self, message: str, *, usage: TokenUsage | None = None) -> None:
        super().__init__(message)
        self.usage = usage


def _attach_usage(exc: BaseException, usage: TokenUsage | None) -> None:
    """Tag ``exc`` with the usage billed before it was raised, in place.

    Used on exceptions this module did not create, so a provider error keeps its
    own type and status code while still carrying the spend. Best-effort: an
    exception class that refuses the attribute (``__slots__``) is left alone
    rather than being replaced by one that accepts it.
    """
    if usage is None:
        return
    try:
        exc.usage = usage  # pyright: ignore[reportAttributeAccessIssue]
    except AttributeError:  # pragma: no cover - defensive, no known such class
        logger.debug('could not attach structured-output usage to %s', type(exc).__name__)


def usage_from_exception(exc: BaseException) -> TokenUsage | None:
    """Spend the ladder billed before it raised, or ``None`` if it billed nothing.

    `generate_structured` can pay for four rungs before giving up, so a failed
    structured call is not a free one — dropping the total is how a run under-reports
    its own cost. `_attach_usage` tags the accumulated figure onto whatever propagates.

    ``getattr`` rather than ``except StructuredGenerationError``: a provider error
    (an `APIStatusError` from the last rung) propagates as **itself**, because
    masking a 429 behind a `RuntimeError` would cost the caller the status code.

    Harvest once per failed call. When the raise came from parsing *after* the
    caller already recorded that call's usage, the exception carries no total and
    this returns ``None`` — so it never double-counts.
    """
    usage = getattr(exc, 'usage', None)
    return usage if isinstance(usage, TokenUsage) else None


def sum_structured_usage(usages: Sequence[TokenUsage | None]) -> TokenUsage | None:
    """Add up `StructuredResult.usage` values, keeping ``None`` for "nothing was billed".

    The one adder for the call sites, so a phase that makes several structured
    calls reports one figure rather than each module re-implementing the
    ``None``-tolerant fold. Uses `Usage.__add__`, which carries ``calls`` and
    ``priced_calls`` through — so an aggregate that mixes priced and unpriced
    calls still answers `Usage.cost_is_partial` honestly.
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

    Every `generate_structured` call site sits on a path with **no report field
    for its usage**: the persona/scenario/trace generators run before a run
    object exists, and the recommendation writers run after the report summary
    has been finalized (RES-1295). The per-call numbers do reach the LLM spans,
    but a user reading the run log had no way to see this spend at all, so each
    phase logs its own total here rather than each module inventing a format.

    Deliberately not recorded on the enclosing span: `generate_structured`
    already records usage on the child LLM span it opens for every rung, and
    adding the aggregate to the parent would double-count it in any trace-level
    sum.
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
    """Apply the ladder's unknown-usage policy to one chat rung's usage.

    The pricing itself belongs to the executor that made the call
    (`execute_chat_completion` / `execute_chat_parse` both run the extracted
    counts through `price_usage`, because the Orq router prices Responses but
    not Chat Completions). What lives here is only the policy on top of it.

    A response whose ``usage`` block the executor could not read is counted as
    **one unpriced call** (``calls=1, priced_calls=0``), never dropped and never
    silently priced at $0: the rung reached the provider and was billed. Summed
    against a rung that did report a cost, that makes `Usage.cost_is_partial`
    true, which is how the rest of the package renders "this figure covers N of
    M calls". The cause is logged, since a shape this misses is a provider
    change worth seeing.
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

    ``openai._exceptions.LengthFinishReasonError`` carries the ``ChatCompletion``
    it refused to parse, usage block and all, so a truncated ``parse()`` rung is
    not an unbilled one: the provider generated ``max_tokens`` of output and
    charged for them. The executor that made the call never returned, so pricing
    could not happen there — it happens here instead, through the same
    `price_usage` the executors use, and the result goes through `_rung_usage` so
    a completion whose usage block is unreadable still counts as one unpriced
    call rather than as zero.
    """
    completion = getattr(exc, 'completion', None)
    usage = await price_usage(TokenUsage.from_completion(completion), call.model, call.client)
    return _rung_usage(usage, completion, label=call.label, leg=leg)


# Structural request fields extra_kwargs may not replace, derived from the
# contracts sets with `|` so a key added there reaches this ladder too.
# An api='responses' call can fall through to the chat legs, so its set is
# the union — a key safe on the endpoint asked for is not safe on the one it
# degrades to.
_STRUCTURAL_KEYS = _RESERVED_COMPLETION_KEYS | {'max_completion_tokens'}
_STRUCTURAL_KEYS_RESPONSES = _RESERVED_RESPONSES_KEYS | {'text_format', 'max_output_tokens'}
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


# The forced-tool rung fails on a different capability, so it recognises a
# different vocabulary. Same rule as above: a bare 400 is not evidence.
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


# The Responses leg's per-request ceiling. A batched generation asking for tens
# of items is a slow call by design, so this is well above LLMCallConfig's
# 90s default; without it the call has no bound at all, which is what the
# hand-rolled version this leg replaced had.
_STRUCTURED_TIMEOUT_S = 300.0


def _truncated_output_error(
    label: str,
    max_tokens: int,
    usage: TokenUsage | None = None,
) -> StructuredGenerationError:
    """Return the actionable error for a provider-reported cut-off payload.

    Truncated structured output is unusable and unrecoverable: the JSON stops
    mid-string, and a retry at the same budget truncates in the same place. Every
    rung raises this rather than degrading, so the user gets the one action that
    works instead of a parse error several frames away.

    ``usage`` is the raising rung's own billed usage. The rungs that see a
    response read it off that response; the ``parse()`` rungs, where the SDK
    raises ``LengthFinishReasonError`` instead of returning, read it off the
    ``completion`` the exception carries (`_truncation_usage`) — a truncated call
    is billed for every token it generated up to the cap, so dropping it would
    understate the most expensive failure the ladder has. ``None`` is left only
    for a caller that genuinely has nothing to report. `generate_structured`
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
    reasoning_effort: str | None = None,
    timeout_s: float = _STRUCTURED_TIMEOUT_S,
    extra_body: dict[str, Any] | None = None,
) -> StructuredResult[T]:
    """Structured output through the Responses API; ``parsed is None`` means "use the chat legs".

    Returns ``parsed=None`` — after a warning naming the cause — when the
    endpoint is absent (404), when a 400's body names the schema form as
    unsupported, or when the provider hands back nothing parsed. Those cases let the caller
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

    ``usage`` is carried on the fall-throughs the provider **answered** — an
    empty output or an unparsed one billed, and the chat ladder that runs next
    adds its own rungs on top of it. The two rejection fall-throughs (404, and
    the 400 naming the schema form) return ``usage=None`` instead: a request the
    provider refused never ran a model, so counting it as a billed call would
    invent spend.
    """
    async with with_llm_span(
        model=model,
        operation='responses',
        temperature=temperature,
        max_tokens=max_tokens,
        input_messages=messages,
        attributes={'orq.llm.purpose': label},
    ) as span:
        usage: TokenUsage | None = None
        try:
            response, usage = await with_retry(
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
            # A rejected request is not a billed one: there is no usage to carry.
            return StructuredResult(None, '', None)
        if usage is None:
            logger.warning(
                '%s: the Responses leg returned no readable usage block (%r); '
                'counting it as one unpriced call rather than as zero cost',
                label,
                getattr(response, 'usage', None),
            )
            usage = TokenUsage(calls=1)
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

    Every rung goes through `common.llm_call`'s executors rather than touching
    ``client.chat.completions`` itself, so this carries the arguments they take
    and each rung adds only its own (a ``response_format``, or
    ``tools``/``tool_choice`` and an edited message list). Bypassing the
    executors is what left the ladder without the ``reasoning_effort``
    drop-and-retry, the per-request timeout, the concurrency slot and the
    reserved-key guard — all of which the ``api='responses'`` leg one function
    away already had.

    Merge order is the executors' and is therefore defined once, for both
    endpoints, in `execute_chat_completion` / `execute_chat_parse`:

    1. the structural fields (``model``, ``messages``, ``max_completion_tokens``,
       ``temperature``, ``reasoning_effort`` when set)
    2. the rung's own fields
    3. ``extra_kwargs`` — a caller's provider options (a reasoning-model
       ``temperature`` override, user ``llm_kwargs``) applied **last**, so they
       win over both, having already been checked against the reserved set
    4. pipeline metadata (defaults only) and trace headers, merged into any
       caller ``extra_headers`` so the active span's traceparent propagates.

    ``span`` is the ladder's single chat span, shared by every rung: the ladder
    is one logical call, and the rung that answered is recorded on it by
    `_mark_leg`.
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
            # outright for the o-series and gpt-5 families, and every other chat
            # call in the repo already sends this key.
            'max_completion_tokens': self.max_tokens,
            'reasoning_effort': self.reasoning_effort,
            'extra_body': self.extra_body,
            'extra_kwargs': self.extra_kwargs,
            **rung,
        }


def _validate_content(content: str, *, response_format: type[T], label: str, leg: str) -> T | None:
    """Fence-tolerant parse + schema validation of one leg's text output.

    The single salvage implementation, shared by the ``json_schema`` leg, the
    tool leg's ``arguments`` string and the ``json_object`` leg — a provider
    degraded far enough down the ladder to answer in prose is exactly the one
    that wraps its JSON in a ```json fence. Returns ``None`` after a warning
    when nothing validates, which lets the caller try the next leg rather than
    handing back content no one checked. Empty content warns too: it is the same
    outcome for the same caller, and warning on one branch but not the neighbour
    made a provider that answers with nothing look like a provider that answered
    off-schema — one logged, the other silent.
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
    """Record which rung of the ladder answered.

    With four rungs a boolean cannot say which one paid, so the leg name is the
    real attribute. ``fallback`` is kept alongside it so dashboards built on the
    old boolean do not go blank.
    """
    if span is None:
        return
    span.set_attribute('orq.structured_output.leg', leg)
    span.set_attribute('orq.structured_output.fallback', leg != _LEG_PARSE)


def _record_tool_nudge(span: Span | None, nudge: str) -> None:
    """Record rung 3's appended user turn under its own span attribute.

    `common.tracing.record_llm_input` **sets** ``gen_ai.input.messages`` rather
    than appending, and every rung calls it on this ladder's one shared span, so
    a rung 4 that runs after rung 3 overwrites the nudged list and the nudge
    disappears — exactly the run where a reader needs to see it, since rung 3
    failing is why rung 4 ran at all. A dedicated attribute is written once by
    the only rung that edits the prompt and no later rung touches it.
    """
    if span is None:
        return
    span.set_attribute('orq.structured_output.tool_nudge', nudge)


async def _leg_strict_parse(call: _ChatLadderCall, *, response_format: type[T]) -> StructuredResult[T]:
    """Rung 1: strict schema-enforced structured output via ``parse()``.

    Continues the ladder on a 400 whose body names the schema form, on a response
    the SDK could not validate, and on a response with no ``choices`` at all —
    warning in the same shape the tool rung and `_content_result` do, rather than
    raising ``IndexError`` from under the ladder. Raises on a refusal (it must not
    be retried on another rung), on truncation, and on any other error.

    ``with_retry`` around `execute_chat_parse` is the **only** retry layer:
    `generate_structured` disarms the client's SDK budget before any rung runs,
    the executor deliberately does not retry ("the caller owns retry and error
    policy"), and its reasoning drop-and-retry-once is a different request, not
    a second backoff loop.
    """
    label = call.label
    try:
        response, raw_usage = await with_retry(
            lambda: execute_chat_parse(**call.executor_kwargs(response_model=response_format)),
            label=label,
        )
    except APIStatusError as e:
        if e.status_code != 400 or not _looks_like_schema_rejection(e):
            raise
        logger.warning('%s: structured output not supported by model, trying the non-strict schema', label)
        # Rejected before the model ran: nothing was billed, so no usage.
        return StructuredResult(None, '')
    except LengthFinishReasonError as exc:
        raise _truncated_output_error(
            label,
            call.max_tokens,
            await _truncation_usage(exc, call, leg=_LEG_PARSE),
        ) from exc
    usage = _rung_usage(raw_usage, response, label=label, leg=_LEG_PARSE)
    if not response.choices:
        logger.warning('%s: parse() returned no choices, trying the non-strict schema', label)
        return StructuredResult(None, '', usage)
    message = response.choices[0].message
    refusal = getattr(message, 'refusal', None)
    if refusal:
        raise StructuredGenerationError(f'{label}: model refused to generate: {refusal}', usage=usage)
    parsed = message.parsed
    if parsed is None:
        logger.debug('%s: parse() returned None, trying the non-strict schema', label)
        return StructuredResult(None, '', usage)
    _mark_leg(call.span, _LEG_PARSE)
    return StructuredResult(cast('T', parsed), '', usage)


async def _leg_json_schema(call: _ChatLadderCall, *, response_format: type[T]) -> StructuredResult[T]:
    """Rung 2: the same schema, non-strict, via plain ``create()``.

    Non-strict on purpose: strict mode is what rung 1 already tried and what the
    provider just rejected. The schema still names the fields; only the
    enforcement is relaxed, and it is the only thing telling the model which keys
    to emit. Content that does not validate continues the ladder rather than
    returning here — the rungs below are stricter, not looser.
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

    Stricter than the two rungs below it: ``tool_choice`` naming the function
    makes emitting the schema the only legal output, so there is no prose channel
    at all — no fences, no preamble. It is a different provider capability than
    ``response_format``, which is the point: routed and local models that 400 on
    a JSON schema often do support function calling, and today those drop
    straight to bare ``json_object``.

    **This is the only rung that changes the prompt.** It appends one user turn
    telling the model the tool will be called, which rescues providers that
    accept ``tools`` but quietly downgrade a named ``tool_choice`` to auto. The
    turn is sent as this rung's own message list, never appended to the caller's.
    If a model's answer differs between rungs, the prompt is one of the reasons,
    so the nudge is recorded on the ladder's span under
    ``orq.structured_output.tool_nudge`` by `_record_tool_nudge`. That attribute,
    not the recorded input, is the durable record: `execute_chat_parse` does
    record the messages it sends, but `record_llm_input` *sets*
    ``gen_ai.input.messages`` on the shared span, so a rung 4 running after this
    one overwrites the nudged list.

    Skipped entirely when the caller supplied their own ``tools`` or
    ``tool_choice``: those are functional, and forcing ours would break the call
    this leg is only trying to salvage.
    """
    label = call.label
    caller_kwargs = call.extra_kwargs or {}
    if 'tools' in caller_kwargs or 'tool_choice' in caller_kwargs:
        logger.debug('%s: caller supplied tools, skipping the forced tool call leg', label)
        return StructuredResult(None, '')

    tool = pydantic_function_tool(response_format)
    tool_name = tool['function']['name']
    # The SDK's TypedDict is a dict at runtime; the executor's signature takes
    # the plain shape, so widen it once here rather than casting at the call.
    tool_param: dict[str, Any] = dict(tool)
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
    try:
        response, raw_usage = await with_retry(
            lambda: execute_chat_parse(
                # No response_model: a provider that rejected the schema form is
                # exactly the one this rung works around, so the schema travels
                # as the tool definition instead.
                **call.executor_kwargs(
                    messages=nudged_messages,
                    tools=[tool_param],
                    tool_choice={'type': 'function', 'function': {'name': tool_name}},
                )
            ),
            label=f'{label} (forced tool call)',
        )
    except APIStatusError as e:
        if e.status_code != 400 or not _looks_like_capability_rejection(e, _SCHEMA_KEYWORDS + _TOOL_KEYWORDS):
            raise
        logger.warning('%s: forced tool call not accepted, falling back to json_object', label)
        return StructuredResult(None, '')
    except LengthFinishReasonError as exc:
        raise _truncated_output_error(
            label,
            call.max_tokens,
            await _truncation_usage(exc, call, leg=_LEG_TOOL),
        ) from exc
    usage = _rung_usage(raw_usage, response, label=label, leg=_LEG_TOOL)
    if not response.choices:
        logger.warning('%s: forced tool call returned no choices, falling back to json_object', label)
        return StructuredResult(None, '', usage)
    message = response.choices[0].message
    refusal = getattr(message, 'refusal', None)
    if refusal:
        raise StructuredGenerationError(f'{label}: model refused to generate: {refusal}', usage=usage)
    if response.choices[0].finish_reason == 'length':
        raise _truncated_output_error(label, call.max_tokens, usage)

    tool_calls = message.tool_calls or []
    if not tool_calls:
        logger.warning('%s: forced tool call returned no tool_calls, falling back to json_object', label)
        return StructuredResult(None, '', usage)
    function = tool_calls[0].function
    # .parse() validates tool arguments against the same model for us; the raw
    # argument string is the fallback when the SDK could not (a non-OpenAI
    # provider whose tool_call shape the SDK does not narrow).
    parsed = cast('T | None', getattr(function, 'parsed_arguments', None))
    raw = function.arguments or ''
    if parsed is None:
        parsed = _validate_content(raw, response_format=response_format, label=label, leg=_LEG_TOOL)
    if parsed is None:
        return StructuredResult(None, raw, usage)
    _mark_leg(call.span, _LEG_TOOL)
    return StructuredResult(parsed, raw, usage)


async def _leg_json_object(call: _ChatLadderCall, *, response_format: type[T]) -> StructuredResult[T]:
    """Rung 4: bare ``json_object``, the last rung.

    Asks for "some JSON" and leaves the field names to chance, which is what
    made fence-tolerant parsing necessary in the first place — so this is where
    the ladder ends, not where it starts. A provider that rejects even this
    raises, since there is nothing left to degrade to.
    """
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

    Shared by the two ``create()`` rungs. Truncation raises here rather than
    degrading: the SDK raises ``LengthFinishReasonError`` for us on the
    ``parse()`` rungs but not on these, where a cut-off body comes back looking
    like ordinary content — ``extract_json_from_response`` would salvage half an
    object and the caller would score a half-answer.
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
    temperature: float | None = None,
    extra_kwargs: dict[str, Any] | None = None,
    api: Literal['chat_completions', 'responses'] = 'chat_completions',
    reasoning_effort: str | None = None,
    timeout_s: float = _STRUCTURED_TIMEOUT_S,
    extra_body: dict[str, Any] | None = None,
) -> StructuredResult[T]:
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

    Returns a `StructuredResult`. ``parsed`` comes from whichever rung
    answered; ``raw`` is ``""`` on the rungs the SDK validates for us and the
    model's own text on the rungs parsed here. ``parsed is None`` means no rung
    produced anything that validated, with ``raw`` the last non-empty text seen
    so a caller still has something to log or salvage.

    ``usage`` is the sum over **every rung that reached the provider**, not just
    the one that answered: a call that burned rungs 1-3 before rung 4 succeeded
    paid for all four, and reporting only the last would understate the spend
    (RES-1295). Each rung is priced individually inside the executor that made
    it (`price_usage` refuses to price an aggregate), so a rung the catalogue
    does not cover contributes tokens without a cost and shows up as
    ``priced_calls < calls``. A rung whose usage block cannot be read is counted
    as one unpriced call after a warning, never as zero.

    ``usage is None`` on a **returned** result means no rung ever reached a
    provider. On a **raised** one it means every rung was rejected before
    billing — a truncation, a refusal and a validation failure all carry the
    spend of the rung that raised, the ``parse()`` rungs included, since
    ``LengthFinishReasonError`` hands back the completion it refused to parse
    (`_truncation_usage`). Nothing is lost on the way out either: every exception
    leaving this function carries the ladder total as ``exc.usage`` (see
    `StructuredGenerationError`), because a ladder that truncated on rungs 1-2
    and raised on rung 3 still billed three calls that used to vanish with the
    frame.

    ``reasoning_effort`` is sent only when truthy, as a flat ``reasoning_effort``
    field on the chat legs and as ``reasoning={'effort': ...}`` on the Responses
    leg (mirroring `LLMCallConfig.request_params`) — a
    caller wanting it must be on Chat Completions or the Responses API, since
    only those two shapes are rendered here. A model that 400s on it is handled
    by the executors' drop-and-retry-once on **both** legs, so an effort set on
    a non-reasoning model degrades the parameter, not the call.

    ``timeout_s`` bounds each provider request, on the Responses leg and on
    every chat rung alike — all of them go through `common.llm_call`, which
    takes it. Defaults to 300s — generous relative to `LLMCallConfig`'s 90s
    default because a batched generation asking for tens of items is a slow call
    by design. Note it bounds **one request, not the call**: each rung is
    wrapped in `with_retry` (up to ``MAX_RETRY_ATTEMPTS`` = 5 attempts, plus its
    backoff waits), each attempt can issue a second request if the model rejects
    ``reasoning_effort``, and ``api='responses'`` adds a fifth rung in front of
    the four. The worst case is on the order of *rungs x 5 x 2* requests — about
    fifty timeouts' worth on the Responses path, not four — so bound a whole call
    with an outer timeout rather than with this. Disarming the client's SDK
    retries does not change that figure: the outer ``with_retry`` attempts are
    the dominant term, not the SDK's.

    ``temperature`` is sent only when not ``None`` (some callers deliberately let
    the provider default stand). ``extra_kwargs`` is merged into every rung's
    params last, so a caller's provider options win over the rung's own fields;
    `_ChatLadderCall`'s docstring states the full order. Structural fields —
    ``_STRUCTURAL_KEYS_BY_API[api]``, wider than the chat trio and different per
    endpoint — are reserved and raise ``ValueError``, since an ``extra_kwargs``
    entry silently replacing the schema or the token budget would defeat the
    helper. A caller's own ``tools``/``tool_choice`` are not reserved — they skip
    the tool rung instead.

    On a length-truncated response this raises ``RuntimeError`` (specifically
    `StructuredGenerationError`) rather than falling back (a same-budget retry
    would truncate again). "Loud" is scoped to this helper: it surfaces a
    specific, actionable reason instead of returning cut-off JSON. Both report
    call sites still wrap the call in a broad ``except`` and skip that one item,
    so a truncation degrades a single section — but with a clear log line naming
    the budget, not a silent drop.

    """
    # Every rung is wrapped in `with_retry`, so the SDK's own budget is disarmed
    # here — once, for the Responses leg and all four chat rungs — rather than
    # per rung. Without it the two layers multiply on evaluatorq's own default
    # path: `resolve_llm_client` builds clients with `max_retries=2` and
    # `redteam/runner.py` builds one with `max_retries=retry_count`, so five
    # outer attempts over an SDK doing two retries is fifteen requests per rung.
    # `without_client_retries` clones rather than mutates, so an injected client
    # is untouched, and it returns a client with no integer budget (a test
    # double) unchanged. Mirrors `common.judge`, which disarms the same way one
    # line before its own ladder.
    client = without_client_retries(client)
    usages: list[TokenUsage | None] = []
    try:
        return await _generate_structured(
            client,
            usages,
            model=model,
            messages=messages,
            response_format=response_format,
            max_tokens=max_tokens,
            label=label,
            temperature=temperature,
            extra_kwargs=extra_kwargs,
            api=api,
            reasoning_effort=reasoning_effort,
            timeout_s=timeout_s,
            extra_body=extra_body,
        )
    except Exception as exc:
        # Fold every rung's spend onto the exception so a caller's `except` can still
        # report it, instead of it dying with the frame.
        _attach_usage(exc, sum_structured_usage([*usages, usage_from_exception(exc)]))
        raise


async def _generate_structured(
    client: AsyncOpenAI,
    usages: list[TokenUsage | None],
    *,
    model: str,
    messages: list[dict[str, Any]],
    response_format: type[T],
    max_tokens: int,
    label: str,
    temperature: float | None,
    extra_kwargs: dict[str, Any] | None,
    api: Literal['chat_completions', 'responses'],
    reasoning_effort: str | None,
    timeout_s: float,
    extra_body: dict[str, Any] | None,
) -> StructuredResult[T]:
    """The ladder itself; `generate_structured` owns the docstring and the usage boundary.

    ``usages`` is passed in rather than owned here so the caller still holds
    every rung's spend when this function leaves by raising.
    """
    check_reserved_keys(extra_kwargs or {}, _STRUCTURAL_KEYS_BY_API[api])
    # The router body travels in the dedicated `extra_body=` parameter; folding it
    # into extra_kwargs here would trip the reserved-key guard on every call.
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
        # The helper warns; usage stays in `usages` because the leg billed even
        # though its payload was unusable.

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
        # Thunks, not coroutines: returning on rung 1 would leave three
        # never-awaited coroutine objects behind (and a RuntimeWarning each).
        legs: tuple[Callable[[], Awaitable[StructuredResult[T]]], ...] = (
            lambda: _leg_strict_parse(call, response_format=response_format),
            lambda: _leg_json_schema(call, response_format=response_format),
            lambda: _leg_forced_tool(call, response_format=response_format),
            lambda: _leg_json_object(call, response_format=response_format),
        )
        last_raw = ''
        for leg in legs:
            result = await leg()
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
