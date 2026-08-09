"""Shared types used across evaluatorq subpackages."""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime  # noqa: TC003 — runtime-required by pydantic manifest models
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, TypeAlias

from loguru import logger
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_serializer, model_validator
from typing_extensions import TypedDict

from evaluatorq.openresponses.convert_models import (
    InputFileContent,
    InputImageContent,
    InputTextContent,
)

# A single part of multi-modal message content (Responses-API input shapes).
# Tagged on ``type`` (mirroring ``OutputMessage``) so Pydantic dispatches on the
# discriminator rather than try-each-member matching.
ContentPart: TypeAlias = Annotated[
    InputTextContent | InputImageContent | InputFileContent,
    Field(discriminator='type'),
]


def _content_part_to_chat_block(part: ContentPart) -> dict[str, Any]:
    """Render a Responses-style content part as an OpenAI chat-completions block.

    Images and files use the chat-completions shapes (``image_url`` / ``file``),
    which differ from the Responses ``input_image`` / ``input_file`` wire shapes.
    """
    if isinstance(part, InputTextContent):
        return {'type': 'text', 'text': part.text}
    if isinstance(part, InputImageContent):
        if part.image_url is None:
            # A file_id-only image is a Responses-API shape; chat-completions has
            # no slot for it, so emitting a block without a url would be silently
            # invalid. Fail loudly instead.
            raise NotImplementedError('file_id-backed images are Responses-API only; use OrqResponsesTarget.')
        return {'type': 'image_url', 'image_url': {'url': part.image_url, 'detail': part.detail}}
    if not isinstance(part, InputFileContent):
        raise NotImplementedError(f'Unsupported content part: {part.type!r}')
    if part.file_id is None and part.file_data is None:
        # file_url/mime_type are intentionally Responses-only — the chat-completions
        # file block has no slot for them. Without file_id or file_data there is no
        # representable content, so fail loudly rather than emit an empty block.
        raise NotImplementedError('file_url-backed files are Responses-API only; use OrqResponsesTarget.')
    file_obj: dict[str, Any] = {}
    if part.file_id is not None:
        file_obj['file_id'] = part.file_id
    if part.file_data is not None:
        file_obj['file_data'] = part.file_data
    if part.filename is not None:
        file_obj['filename'] = part.filename
    return {'type': 'file', 'file': file_obj}


def _render_chat_content(content: str | list[ContentPart] | None) -> str | list[dict[str, Any]]:
    """Render message content for the chat-completions API: a plain string, or a
    list of content blocks for multi-modal content."""
    if isinstance(content, list):
        return [_content_part_to_chat_block(p) for p in content]
    return content or ''


def content_to_text(content: str | list[ContentPart] | None) -> str:
    """Flatten message content to plain text.

    For targets that accept only text: a string (or ``None``) passes through; a
    multi-part list is concatenated from its text parts (no separator), but a
    non-text part (image or file) raises :class:`NotImplementedError` with a clear
    message rather than silently dropping content.
    """
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    out: list[str] = []
    for part in content:
        if isinstance(part, InputTextContent):
            out.append(part.text)
        else:
            raise NotImplementedError(
                f'This target accepts only text content; got a {part.type!r} part. '
                'Use a Responses-API target (OrqResponsesTarget) for image/file content.'
            )
    return ''.join(out)


if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """String enum compatible with Python 3.10."""


if TYPE_CHECKING:
    from openai import AsyncOpenAI

    _Client: TypeAlias = AsyncOpenAI | None
else:
    _Client = Any

# ---------------------------------------------------------------------------
# Pipeline defaults (mirrored from redteam.contracts for shared use)
# ---------------------------------------------------------------------------

DEFAULT_PIPELINE_MODEL: str = 'gpt-5-mini'
DEFAULT_TARGET_MAX_TOKENS: int = 5000
DEFAULT_TARGET_TIMEOUT_MS: int = 240_000


# ---------------------------------------------------------------------------
# LLMCallConfig
# ---------------------------------------------------------------------------


# Request fields extra_kwargs must never replace: they are structural (what
# call is being made), not tunable sampling/provider options.
_RESERVED_COMPLETION_KEYS = frozenset({'model', 'messages', 'response_format', 'extra_body'})


class LLMCallConfig(BaseModel):
    """Per-role LLM call configuration.

    One instance per pipeline role (attacker, evaluator). Controls model
    selection, API type, temperature, token limits, timeout, extra kwargs, and
    optionally an explicit pre-configured client.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = DEFAULT_PIPELINE_MODEL
    api: Literal['chat_completions', 'responses'] = 'chat_completions'
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=DEFAULT_TARGET_MAX_TOKENS, gt=0)
    timeout_ms: int = Field(default=90_000, gt=0)
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)
    client: _Client = None

    def completion_params(self, **params: Any) -> dict[str, Any]:
        """Merged kwargs for a chat-completions call: sampling fields first,
        then call-site params, then ``extra_kwargs`` last so user keys override.

        Never splat ``extra_kwargs`` next to explicit ``temperature=`` /
        ``max_completion_tokens=`` keywords: a user routing those keys through
        ``extra_kwargs`` (the documented pre-refactor way to tune them) turns
        every call into ``TypeError: got multiple values for keyword argument``.
        The override order also doubles as the escape hatch for reasoning-class
        models that reject a lowered temperature: ``extra_kwargs={'temperature': 1}``.

        Structural request fields are reserved: ``extra_kwargs`` tunes sampling
        and provider options, and letting it silently replace ``model`` /
        ``messages`` / ``response_format`` / ``extra_body`` would break the
        call it rides on (e.g. dropping a required JSON response format).
        """
        reserved = _RESERVED_COMPLETION_KEYS & self.extra_kwargs.keys()
        if reserved:
            raise ValueError(
                f'extra_kwargs cannot override structural request field(s) {sorted(reserved)}; '
                'these are owned by the call site.'
            )
        return {
            'temperature': self.temperature,
            'max_completion_tokens': self.max_tokens,
            **params,
            **self.extra_kwargs,
        }


# ---------------------------------------------------------------------------
# Output message item types — re-exported from openresponses
# ---------------------------------------------------------------------------
# AgentResponse output items align with the OpenResponses intermediate data
# format.  The type discriminators (output_text, function_call, reasoning)
# match OpenResponses wire format, making downstream conversion to
# ResponseResource straightforward.

from evaluatorq.common.fields import get_field as _gf
from evaluatorq.openresponses.convert_models import (
    FunctionCall as ToolCallOutputItem,
)
from evaluatorq.openresponses.convert_models import (
    OutputTextContent as TextOutputItem,
)


class ReasoningOutputItem(BaseModel):
    """A reasoning / thinking step in the agent output message list.

    Captures chain-of-thought or intermediate reasoning produced by the model
    (e.g. OpenAI o-series extended thinking, or agent intermediate steps).
    Aligns with the OpenResponses ``reasoning`` item type.
    """

    model_config = ConfigDict(frozen=True)

    type: Literal['reasoning'] = 'reasoning'
    text: str
    id: str | None = None


OutputMessage = Annotated[
    TextOutputItem | ToolCallOutputItem | ReasoningOutputItem,
    Field(discriminator='type'),
]


# ---------------------------------------------------------------------------
# Shared message + token-usage types (RES-596)
# ---------------------------------------------------------------------------
# These were previously duplicated across simulation/types.py and
# redteam/contracts.py. The redteam variants are the supersets — Message
# supports tool calls and the "tool" role; TokenUsage carries an extra
# ``calls`` field plus ``from_completion()``. Both are re-exported from the
# subpackage modules so legacy import paths keep working.


class FunctionCall(BaseModel):
    """Function call details in OpenAI tool call format."""

    name: str = Field(description='Function name to call')
    arguments: str = Field(description='JSON string of function arguments')


class StrategyToolCall(BaseModel):
    """Tool call in assistant message (OpenAI format)."""

    id: str = Field(description='Unique tool call ID (e.g., call_abc123)')
    type: Literal['function'] = Field(default='function', description='Tool type')
    function: FunctionCall = Field(description='Function call details')
    # Responses-API item id (e.g. ``fc_abc123``), distinct from ``id`` (which maps to
    # the chat-completions / Responses ``call_id``). Preserved through transcript
    # replay so :class:`OpenAIAgentTarget` can echo the original ``function_call``
    # item back to the Responses API on subsequent turns. ``None`` for tool calls
    # that did not originate from a Responses-API turn.
    item_id: str | None = Field(default=None, description='Responses-API function_call item id (fc_*)')


class Message(BaseModel):
    """Single message in conversation history (OpenAI format with tool support).

    Supports both simple messages and tool calls:
    - Simple: ``{"role": "user", "content": "Hello"}``
    - Tool call: ``{"role": "assistant", "tool_calls": [...]}``
    - Tool response: ``{"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}``
    """

    role: Literal['user', 'assistant', 'tool', 'system', 'developer']
    content: str | list[ContentPart] | None = Field(
        default=None,
        description=(
            'Message content. A plain string, or a list of multi-modal content parts '
            '(text/image/file). Required for user/system, optional for assistant with tool_calls.'
        ),
    )

    # Tool call fields (OpenAI format)
    tool_calls: list[StrategyToolCall] | None = Field(default=None, description='Tool calls made by assistant')
    tool_call_id: str | None = Field(
        default=None,
        description='ID linking tool response to call (for role=tool)',
    )
    name: str | None = Field(default=None, description='Function name (for role=tool)')

    def to_chat_completion(self) -> dict[str, Any]:
        """Render this message as an OpenAI chat-completions message dict.

        Preserves tool-call structure that a naive ``{"role", "content"}`` flatten
        would drop: an assistant message's ``tool_calls`` and a ``tool`` row's
        ``tool_call_id``/``name``. Used by stateless targets to replay a transcript
        (built by ``turns_to_messages``) without losing multi-turn tool context.
        """
        if self.role == 'tool':
            # A tool message carries a result keyed to tool_call_id; tool_calls belong
            # on assistant messages, so any present here are malformed and not emitted.
            param: dict[str, Any] = {
                'role': 'tool',
                'tool_call_id': self.tool_call_id or '',
                'content': self.content or '',
            }
            if self.name:
                param['name'] = self.name
            return param
        if self.role == 'assistant' and self.tool_calls:
            # content may be None on a pure tool-call turn; OpenAI accepts null.
            return {
                'role': 'assistant',
                'content': self.content,
                'tool_calls': [
                    {
                        'id': tc.id,
                        'type': 'function',
                        'function': {'name': tc.function.name, 'arguments': tc.function.arguments},
                    }
                    for tc in self.tool_calls
                ],
            }
        return {'role': self.role, 'content': _render_chat_content(self.content)}


def _usage_get(usage: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a usage object or dict, returning ``default`` if absent."""
    if isinstance(usage, dict):
        return usage.get(key, default)
    return getattr(usage, key, default)


def _usage_first_int(usage: Any, keys: tuple[str, ...]) -> int:
    """First numeric value across ``keys``, coerced to int (0 if none present).

    Only genuine ``int``/``float`` values count; ``None`` and non-numeric stand-ins
    (e.g. a bare ``MagicMock`` attribute) are skipped so the next alias is tried.
    """
    for key in keys:
        val = _usage_get(usage, key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return int(val)
    return 0


def _usage_detail_int(usage: Any, containers: tuple[str, ...], keys: tuple[str, ...]) -> int:
    """Read a nested detail token count (e.g. input_tokens_details.cached_tokens).

    Tries each container name and, within it, each key — covering responses
    (``input_tokens_details.cached_tokens``) and langchain
    (``input_token_details.cache_read``) shapes alike.
    """
    for container in containers:
        detail = _usage_get(usage, container)
        if detail is None:
            continue
        val = _usage_first_int(detail, keys)
        if val:
            return val
    return 0


def _usage_first_float(usage: Any, keys: tuple[str, ...]) -> float | None:
    """First numeric value across ``keys`` coerced to float, or None if absent."""
    for key in keys:
        val = _usage_get(usage, key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    return None


def _clamped_cost(usage: Any, keys: tuple[str, ...]) -> float | None:
    """First cost value across ``keys``, clamped at 0.

    A provider may report a negative cost (credit/adjustment), which would
    otherwise trip the ge=0 field constraint and crash the whole extraction path
    for an otherwise-valid billed call. Log it so a genuine provider/accounting
    bug is not silently rewritten to $0.
    """
    raw = _usage_first_float(usage, keys)
    if raw is not None and raw < 0:
        logger.warning('Usage.extract: provider reported negative cost {!r} for {}; clamping to 0.0', raw, keys[0])
        return 0.0
    return raw


class Usage(BaseModel):
    """Token usage and cost for an LLM call or aggregation of calls.

    Field naming follows the OpenTelemetry GenAI semconv / OpenResponses standard
    (``input_tokens``/``output_tokens`` rather than ``prompt``/``completion``).
    ``cached_tokens`` is a subset of ``input_tokens`` and ``reasoning_tokens`` a
    subset of ``output_tokens``, so the reconciliation invariant is
    ``total_tokens == input_tokens + output_tokens`` (with cached ≤ input,
    reasoning ≤ output). ``total_tokens`` is stored as provided by the upstream
    provider and never overridden.

    Cost fields mirror the Orq Responses v3 usage shape (``input_cost``/
    ``output_cost``/``total_cost``, USD). ``None`` means the provider did not
    report that component — never fabricated as 0.

    Back-compat: the legacy ``prompt_tokens``/``completion_tokens``/``cost_usd``
    names are accepted on construction and exposed as read-only properties so
    call sites that have not migrated keep working. ``TokenUsage`` remains as a
    deprecated alias for this class.
    """

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    input_tokens: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices('input_tokens', 'prompt_tokens'),
        description='Input/prompt tokens (includes cached_tokens)',
    )
    output_tokens: int = Field(
        default=0,
        ge=0,
        validation_alias=AliasChoices('output_tokens', 'completion_tokens'),
        description='Output/completion tokens (includes reasoning_tokens)',
    )
    total_tokens: int = Field(default=0, ge=0, description='Total tokens as reported by the provider')
    cached_tokens: int = Field(default=0, ge=0, description='Cached input tokens (subset of input_tokens)')
    cache_creation_tokens: int = Field(
        default=0,
        ge=0,
        description='Cache-write input tokens (subset of input_tokens for OpenAI-shaped payloads; reported separately by Anthropic)',
    )
    cache_creation_1h_tokens: int = Field(
        default=0, ge=0, description='Cache-write tokens at the 1h TTL tier (subset of cache_creation_tokens)'
    )
    cache_creation_5m_tokens: int = Field(
        default=0, ge=0, description='Cache-write tokens at the 5m TTL tier (subset of cache_creation_tokens)'
    )
    reasoning_tokens: int = Field(default=0, ge=0, description='Reasoning output tokens (subset of output_tokens)')
    input_cost: float | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices('input_cost', 'prompt_cost'),
        description='Input cost in USD when the provider reports it, else None',
    )
    output_cost: float | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices('output_cost', 'completion_cost'),
        description='Output cost in USD when the provider reports it, else None',
    )
    total_cost: float | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices('total_cost', 'cost_usd', 'cost'),
        description='Total cost in USD when the provider reports it, else None',
    )
    calls: int = Field(default=0, ge=0, description='Number of LLM API calls')
    priced_calls: int = Field(
        default=0,
        ge=0,
        description='How many of `calls` reported a cost. Below `calls` means the cost fields are a lower bound.',
    )

    if TYPE_CHECKING:
        # The legacy ``prompt_tokens``/``completion_tokens`` names are accepted at
        # runtime via the validation aliases above, but a static type checker only
        # sees the declared field names. Declare the constructor explicitly so call
        # sites passing either spelling type-check cleanly.
        def __init__(  # pyright: ignore[reportMissingSuperCall]
            self,
            *,
            input_tokens: int = ...,
            output_tokens: int = ...,
            prompt_tokens: int = ...,
            completion_tokens: int = ...,
            total_tokens: int = ...,
            cached_tokens: int = ...,
            cache_creation_tokens: int = ...,
            cache_creation_1h_tokens: int = ...,
            cache_creation_5m_tokens: int = ...,
            reasoning_tokens: int = ...,
            input_cost: float | None = ...,
            prompt_cost: float | None = ...,
            output_cost: float | None = ...,
            completion_cost: float | None = ...,
            total_cost: float | None = ...,
            cost_usd: float | None = ...,
            calls: int = ...,
            priced_calls: int = ...,
        ) -> None: ...

    @property
    def prompt_tokens(self) -> int:
        """Deprecated alias for :attr:`input_tokens`."""
        return self.input_tokens

    @property
    def completion_tokens(self) -> int:
        """Deprecated alias for :attr:`output_tokens`."""
        return self.output_tokens

    @property
    def cost_usd(self) -> float | None:
        """Deprecated alias for :attr:`total_cost`."""
        return self.total_cost

    @property
    def cost_is_partial(self) -> bool:
        """True when a cost is known but some calls in this aggregate had none.

        The cost fields are then a *lower bound*, not a total — ``_combine_cost``
        adds a known cost to an unknown one by treating the unknown as 0.0, so the
        result looks authoritative but under-counts. Render "from N of M calls"
        next to any cost when this is set.

        ``priced_calls == 0`` alongside a known cost cannot happen for a freshly
        built aggregate (``extract`` prices all-or-nothing, ``__add__`` sums), so it
        only occurs in reports saved before this field existed — treated as "not
        tracked" rather than "nothing was priced", so old reports stay unannotated.
        """
        return self.total_cost is not None and 0 < self.priced_calls < self.calls

    def with_calls(self, calls: int) -> Usage:
        """Stamp the call count, keeping :attr:`priced_calls` consistent.

        ``from_openresponses`` parses with ``calls=0`` and leaves the real count to
        the caller that knows the call was billed. A bare
        ``model_copy(update={'calls': 1})`` would leave ``priced_calls`` at 0, which
        :attr:`cost_is_partial` reads as legacy untracked data — use this instead.
        """
        return self.model_copy(update={'calls': calls, 'priced_calls': calls if self.total_cost is not None else 0})

    @model_serializer(mode='wrap')
    def _serialize(self, handler: Any) -> dict[str, Any]:
        """Emit the legacy ``prompt_tokens``/``completion_tokens``/``cost_usd`` keys
        alongside the standard names so dict/JSON consumers that have not migrated
        keep working.

        Note: the legacy aliases are injected unconditionally, so ``model_dump`` with
        ``include=``/``exclude=`` does not filter them out — field-level filtering is
        not supported on this model."""
        data = handler(self)
        data['prompt_tokens'] = self.input_tokens
        data['completion_tokens'] = self.output_tokens
        data['cost_usd'] = self.total_cost
        return data

    @classmethod
    def extract(cls, usage: Any, *, calls: int = 1) -> Usage | None:
        """Map any provider usage shape (object or dict) into a Usage.

        Handles chat-completions (``prompt``/``completion_tokens`` + ``*_details``),
        responses (``input``/``output_tokens`` + ``*_tokens_details``), vercel
        camelCase, and langgraph ``usage_metadata``. A single fallback rule:
        ``total_tokens`` is trusted when > 0, otherwise ``input + output``.

        Returns ``None`` when the payload is present but yields no positive token
        counts and no cost — i.e. an unparseable/empty usage block. This mirrors
        ``from_openresponses``: a present-but-empty payload must not be recorded as
        a confident "0 tokens, 1 call", which would undercount tokens on a genuinely
        billed call while still incrementing the call count.
        """
        if usage is None:
            return None
        inp = _usage_first_int(usage, ('input_tokens', 'prompt_tokens', 'promptTokens', 'prompt'))
        out = _usage_first_int(usage, ('output_tokens', 'completion_tokens', 'completionTokens', 'completion'))
        # Provider total is trusted only when > 0; otherwise fall back to input+output.
        reported_total = _usage_first_int(usage, ('total_tokens', 'totalTokens', 'total'))
        total = reported_total if reported_total > 0 else inp + out
        # Anthropic reports cache reads as a flat top-level `cache_read_input_tokens`
        # (anthropic.types.Usage), not nested in a details object like Orq/OpenAI.
        cached = _usage_detail_int(
            usage,
            ('input_tokens_details', 'prompt_tokens_details', 'input_token_details'),
            ('cached_tokens', 'cache_read'),
        ) or _usage_first_int(usage, ('cache_read_input_tokens',))
        # Anthropic prices a 1h cache write above a 5m one, so the tier split is
        # billing-relevant even though Orq folds both into the flat input_cost.
        # Orq nests the tiers under input_tokens_details; Anthropic puts them in a
        # `cache_creation` object under its own `ephemeral_*` names
        # (anthropic.types.CacheCreation).
        cache_creation_1h = _usage_detail_int(
            usage,
            ('input_tokens_details', 'prompt_tokens_details', 'input_token_details', 'cache_creation'),
            ('cache_creation_1h_tokens', 'ephemeral_1h_input_tokens'),
        ) or _usage_first_int(usage, ('cache_creation_1h_tokens',))
        cache_creation_5m = _usage_detail_int(
            usage,
            ('input_tokens_details', 'prompt_tokens_details', 'input_token_details', 'cache_creation'),
            ('cache_creation_5m_tokens', 'ephemeral_5m_input_tokens'),
        ) or _usage_first_int(usage, ('cache_creation_5m_tokens',))
        # Both Anthropic and Orq v3 do report a pre-summed total, so the tier sum is
        # only a last-resort fallback for a payload that carries the split alone —
        # without it those cache-write tokens would vanish from cache_creation_tokens.
        cache_creation = (
            _usage_detail_int(
                usage,
                ('input_tokens_details', 'prompt_tokens_details', 'input_token_details'),
                ('cache_creation_tokens', 'cache_creation_input_tokens'),
            )
            or _usage_first_int(usage, ('cache_creation_input_tokens',))
            or cache_creation_1h + cache_creation_5m
        )
        reasoning = _usage_detail_int(
            usage,
            ('output_tokens_details', 'completion_tokens_details', 'output_token_details'),
            ('reasoning_tokens', 'reasoning'),
        )
        input_cost = _clamped_cost(usage, ('input_cost', 'prompt_cost'))
        output_cost = _clamped_cost(usage, ('output_cost', 'completion_cost'))
        cost = _clamped_cost(usage, ('total_cost', 'cost_usd', 'cost'))
        if cost is None and (input_cost is not None or output_cost is not None):
            cost = (input_cost or 0.0) + (output_cost or 0.0)
        # A present-but-empty/unparseable payload yields no signal: don't fabricate a
        # billed-zero record. A genuine zero-token call is kept only if it carries
        # some signal — cached tokens or a cost (e.g. a cached-only / minimum charge).
        if total <= 0 and cached <= 0 and cost is None:
            return None
        # Honour a calls count already carried by the payload (e.g. an aggregated
        # Usage dump); fall back to the per-call default otherwise.
        raw_calls = _usage_get(usage, 'calls')
        resolved_calls = (
            int(raw_calls) if isinstance(raw_calls, (int, float)) and not isinstance(raw_calls, bool) else calls
        )
        # Same deal for priced_calls: an aggregated dump carries its own count,
        # a raw provider payload prices all-or-nothing on whether cost showed up.
        raw_priced = _usage_get(usage, 'priced_calls')
        resolved_priced = (
            int(raw_priced)
            if isinstance(raw_priced, (int, float)) and not isinstance(raw_priced, bool)
            else (resolved_calls if cost is not None else 0)
        )
        return cls(
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
            cached_tokens=cached,
            cache_creation_tokens=cache_creation,
            cache_creation_1h_tokens=cache_creation_1h,
            cache_creation_5m_tokens=cache_creation_5m,
            reasoning_tokens=reasoning,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=cost,
            calls=resolved_calls,
            priced_calls=min(resolved_priced, resolved_calls),
        )

    @classmethod
    def from_completion(cls, response: Any) -> Usage | None:
        """Extract token usage from an OpenAI-compatible completion response."""
        return cls.extract(getattr(response, 'usage', None), calls=1)

    @staticmethod
    def _combine_cost(a: float | None, b: float | None, *, sign: int = 1) -> float | None:
        """Cost stays "unknown" (None) only when BOTH sides are unknown; if either
        side carries a known cost, treat the missing side as 0.0 rather than
        poisoning the result to None (which would discard a real known cost on the
        first turn a cost-bearing provider appears)."""
        if a is None and b is None:
            return None
        return max((a or 0.0) + sign * (b or 0.0), 0.0)

    def __add__(self, other: Usage | None) -> Usage:
        if other is None:
            return self.model_copy()
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_creation_tokens=self.cache_creation_tokens + other.cache_creation_tokens,
            cache_creation_1h_tokens=self.cache_creation_1h_tokens + other.cache_creation_1h_tokens,
            cache_creation_5m_tokens=self.cache_creation_5m_tokens + other.cache_creation_5m_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            input_cost=self._combine_cost(self.input_cost, other.input_cost),
            output_cost=self._combine_cost(self.output_cost, other.output_cost),
            total_cost=self._combine_cost(self.total_cost, other.total_cost),
            calls=self.calls + other.calls,
            priced_calls=self.priced_calls + other.priced_calls,
        )

    def __radd__(self, other: object) -> Usage:
        """Support sum() which starts with integer 0, and reflected addition."""
        if isinstance(other, int) and other == 0:
            return self.model_copy()
        if isinstance(other, Usage):
            return other.__add__(self)
        return NotImplemented

    def __sub__(self, other: Usage | None) -> Usage:
        """Component-wise difference, clamped at 0 — used for per-turn deltas."""
        if other is None:
            return self.model_copy()
        return Usage(
            input_tokens=max(self.input_tokens - other.input_tokens, 0),
            output_tokens=max(self.output_tokens - other.output_tokens, 0),
            total_tokens=max(self.total_tokens - other.total_tokens, 0),
            cached_tokens=max(self.cached_tokens - other.cached_tokens, 0),
            cache_creation_tokens=max(self.cache_creation_tokens - other.cache_creation_tokens, 0),
            cache_creation_1h_tokens=max(self.cache_creation_1h_tokens - other.cache_creation_1h_tokens, 0),
            cache_creation_5m_tokens=max(self.cache_creation_5m_tokens - other.cache_creation_5m_tokens, 0),
            reasoning_tokens=max(self.reasoning_tokens - other.reasoning_tokens, 0),
            input_cost=self._combine_cost(self.input_cost, other.input_cost, sign=-1),
            output_cost=self._combine_cost(self.output_cost, other.output_cost, sign=-1),
            total_cost=self._combine_cost(self.total_cost, other.total_cost, sign=-1),
            calls=max(self.calls - other.calls, 0),
            priced_calls=max(self.priced_calls - other.priced_calls, 0),
        )


# Deprecated alias — the class gained cost fields, so the name widened to Usage.
TokenUsage = Usage


# ---------------------------------------------------------------------------
# Generic jury contracts
# ---------------------------------------------------------------------------


# Key under which scorer implementations stash the serialized JuryResult inside
# EvaluationResult.raw_output, and under which converters lift it back onto typed
# result models. Shared so writers/readers cannot drift.
JURY_RAW_OUTPUT_KEY = 'jury'


class JuryVote(BaseModel):
    """A single judge's aggregate vote within a jury.

    ``abstained`` is a first-class successful non-verdict: the judge returned
    cleanly but declined to choose a label. It is distinct from ``success=False``
    failures and is excluded from decisive tallies.
    """

    model_config = ConfigDict(frozen=True)

    model: str = Field(description='Judge model ID')
    replacement: bool = Field(default=False, description='True if this judge stood in for a failed configured judge')
    success: bool = Field(description='True if the judge completed without a mechanical failure')
    abstained: bool = Field(default=False, description='True when the judge explicitly declined to choose a verdict')
    value: bool | float | str | None = Field(default=None, description='Judge verdict')
    explanation: str = Field(default='', description='Explanation from a representative pass')
    error: str | None = Field(default=None, description='Failure reason when success is False')
    repetitions: list[bool | float | str | None] = Field(
        default_factory=list,
        description='Raw per-repetition verdicts in call order; None includes abstain/error passes',
    )
    repetitions_failed: int = Field(
        default=0,
        ge=0,
        description='Count of repetitions that raised an error (non-decisive due to error); independent of success/abstained',
    )

    @model_validator(mode='after')
    def _check_vote_consistency(self) -> JuryVote:
        if self.success and self.value is None and not self.abstained:
            raise ValueError('successful JuryVote must have a non-null value unless abstained=True')
        if self.abstained and self.value is not None:
            raise ValueError('abstained JuryVote must have a null value')
        if not self.success and self.value is not None:
            raise ValueError('failed JuryVote must have a null value')
        if not self.success and self.abstained:
            raise ValueError('failed JuryVote cannot also be abstained')
        return self


class JuryStats(BaseModel):
    """Distribution statistics for judge verdicts."""

    model_config = ConfigDict(frozen=True)

    mean: float = Field(description='Mean judge verdict value')
    std: float = Field(ge=0.0, description='Population standard deviation of judge verdict values')


class JuryResult(BaseModel):
    """Aggregated panel-of-judges result."""

    model_config = ConfigDict(frozen=True)

    judges_configured: int = Field(ge=0, description='Number of configured non-replacement judges')
    judges_succeeded: int = Field(ge=0, description='Number of judges that produced a decisive non-abstain verdict')
    judges_failed: int = Field(
        ge=0,
        description=(
            'Number of configured judges that failed mechanically. Replacement-vote outcomes are stored in votes.'
        ),
    )
    replacements_used: int = Field(default=0, ge=0, description='Number of replacement judges invoked')
    tie: bool = Field(default=False, description='True when decisive votes tied and a caller tie policy was applied')
    inconclusive: bool = Field(
        default=False,
        description='True when too few decisive judges produced a verdict; stats/raw_agreement are None',
    )
    votes: list[JuryVote] = Field(default_factory=list, description='Per-judge votes')
    stats: JuryStats | None = Field(default=None, description='Verdict distribution; None when inconclusive')
    raw_agreement: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description='Raw fraction of decisive judges in the largest verdict bloc; None when inconclusive',
    )

    @model_validator(mode='after')
    def _check_jury_consistency(self) -> JuryResult:
        decisive_successes = sum(1 for v in self.votes if v.success and not v.abstained)
        if self.judges_succeeded != decisive_successes:
            raise ValueError(
                f'judges_succeeded ({self.judges_succeeded}) does not match decisive successful votes '
                f'({decisive_successes})'
            )
        if self.judges_failed > self.judges_configured:
            raise ValueError(
                f'judges_failed ({self.judges_failed}) exceeds judges_configured ({self.judges_configured})'
            )
        replacement_votes = sum(1 for v in self.votes if v.replacement)
        if self.replacements_used != replacement_votes:
            raise ValueError(
                f'replacements_used ({self.replacements_used}) does not match replacement votes ({replacement_votes})'
            )
        if len(self.votes) != self.judges_configured + self.replacements_used:
            raise ValueError(
                f'vote count ({len(self.votes)}) != judges_configured + replacements_used '
                f'({self.judges_configured} + {self.replacements_used})'
            )
        if self.tie and self.inconclusive:
            raise ValueError('jury result cannot be both tie and inconclusive')
        if self.inconclusive and (self.stats is not None or self.raw_agreement is not None):
            raise ValueError('inconclusive jury result must not include stats/raw_agreement')
        return self


# ---------------------------------------------------------------------------
# AgentResponseError
# ---------------------------------------------------------------------------


class AgentResponseError(BaseModel):
    """A per-response error marker on :class:`AgentResponse`.

    Set when a target (or simulation agent) failed to produce a real response,
    e.g. a timeout or backend exception. The orchestrator uses its presence to
    exclude the turn from the transcript replayed to the target. ``message`` is
    the same human text surfaced in ``AgentResponse.text``; ``code`` is an
    optional provider/mapped code.

    ``error_type`` is a coarse, open-vocabulary string — consumers key off the
    error's *presence*, not its value, so the field is intentionally not an
    enum. The orchestrator sets ``"timeout"`` on the timeout path and otherwise
    classifies the failure via
    :func:`evaluatorq.common.target_call.classify_error_type`, which yields one
    of ``content_filter``, ``rate_limit``, ``timeout``, ``network_error``,
    ``server_error``, ``client_error``, or ``target_error`` (fallback for an
    unmatched error). Other producers may set their own values.

    This is the leaf, per-response error. The whole-run rollup is
    :class:`evaluatorq.redteam.contracts.RunError`.
    """

    model_config = ConfigDict(frozen=True)

    message: str
    error_type: str
    code: str | None = None


# ---------------------------------------------------------------------------
# ResponseTrace
# ---------------------------------------------------------------------------


class ResponseTrace(BaseModel):
    """Orq observability handles for one target-agent response.

    Persisted per successful response so a conversation/attack retains the exact
    trace (and span) of every target turn; the dashboard links to the last one.
    Either field may be None when the target isn't routed through Orq.
    """

    trace_id: str | None = None
    span_id: str | None = None


# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------


class AgentResponse(BaseModel):
    """Structured response from a target agent as an ordered list of output messages.

    Each item in ``output`` is a :class:`TextOutputItem`, :class:`ToolCallOutputItem`,
    or :class:`ReasoningOutputItem`, preserving the order in which they were produced.
    Item ``type`` discriminators align with the OpenResponses intermediate data format
    (``output_text``, ``function_call``, ``reasoning``).

    Accessors:
        ``.text``       — all ``TextOutputItem`` contents concatenated, or ``""`` if none
        ``.tool_calls`` — list of :class:`ToolCallOutputItem` filtered from
        :attr:`output` in order
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    output: list[OutputMessage] = Field(default_factory=list)
    usage: TokenUsage | None = None
    model: str | None = None
    response_id: str | None = None
    # Orq trace id for this call (both the router and agents endpoints read
    # `response.telemetry.trace_id` from the response body). Links this
    # response back to the trace in Orq observability. None when not routed
    # through Orq.
    trace_id: str | None = None
    # Orq span id for this call (``response.telemetry.span_id`` when the router
    # includes it). Pinpoints the exact span within the trace. None when not
    # routed through Orq or the router omits it.
    span_id: str | None = None
    finish_reason: str | None = None
    error: AgentResponseError | None = None

    if TYPE_CHECKING:

        def __init__(  # pyright: ignore[reportMissingSuperCall]
            self,
            *,
            output: list[OutputMessage] | None = None,
            text: str | None = None,
            usage: TokenUsage | None = None,
            model: str | None = None,
            response_id: str | None = None,
            trace_id: str | None = None,
            span_id: str | None = None,
            finish_reason: str | None = None,
            error: AgentResponseError | None = None,
        ) -> None: ...

    @model_validator(mode='before')
    @classmethod
    def _accept_text_shorthand(cls, data: Any) -> Any:
        # Preserve legacy ``AgentResponse(text="...")`` construction.
        if isinstance(data, dict) and 'text' in data:
            payload = dict(data)
            text = payload.pop('text')
            if payload.get('output') is not None:
                raise ValueError('AgentResponse accepts either output= or text=, not both')
            payload['output'] = [TextOutputItem(text=text, annotations=[])] if text is not None else []
            return payload
        return data

    @property
    def text(self) -> str:
        """Concatenate all text output items into a single string."""
        return ''.join(item.text for item in self.output if isinstance(item, TextOutputItem))

    @property
    def tool_calls(self) -> list[ToolCallOutputItem]:
        """Return the tool call items from ``.output`` in order."""
        return [item for item in self.output if isinstance(item, ToolCallOutputItem)]

    @classmethod
    def from_openresponses(cls, response: Any) -> AgentResponse:
        """Build a full AgentResponse from a Responses API response object.

        Single parse path for the OpenResponses wire format. Populates
        ``output`` + ``usage`` + ``model`` + ``finish_reason`` + ``response_id``.

        ``usage`` is ``None`` when the response carries no ``usage`` block —
        callers must distinguish "no usage reported" from "zero tokens used" so
        cost reports stay honest. ``usage.calls`` is intentionally left at 0:
        this is a pure parse; per-call-site accounting (``calls=1`` / ``+1``) is
        applied by callers to the returned object's ``usage``.
        """
        items: list[OutputMessage] = []
        for item in _gf(response, 'output') or []:
            item_type = _gf(item, 'type')
            if item_type == 'message':
                items.extend(
                    TextOutputItem(
                        type='output_text',
                        text=_gf(part, 'text'),
                        annotations=[],
                        logprobs=[],
                    )
                    for part in _gf(item, 'content') or []
                    if _gf(part, 'type') == 'output_text' and _gf(part, 'text')
                )
            # 'function_call' is the standard Responses shape; the Orq router
            # instead types agent tool calls as 'orq:<tool_name>' (e.g.
            # 'orq:query_knowledge_base'). Treat both as tool calls so the
            # agent's tool activity is captured rather than silently dropped.
            elif item_type == 'function_call' or (isinstance(item_type, str) and item_type.startswith('orq:')):
                raw_args = _gf(item, 'arguments') or '{}'
                call_id = _gf(item, 'call_id') or _gf(item, 'id') or ''
                result = _gf(item, 'result')
                if result is None:
                    result = _gf(item, 'output')
                name = _gf(item, 'name') or (item_type.split(':', 1)[1] if item_type.startswith('orq:') else '')
                items.append(
                    ToolCallOutputItem(
                        type='function_call',
                        name=str(name),
                        call_id=str(call_id),
                        arguments=raw_args if isinstance(raw_args, str) else json.dumps(raw_args),
                        result=str(result) if result is not None else None,
                    )
                )
            elif item_type == 'reasoning':
                pass  # o1/o3/o4-mini reasoning steps intentionally excluded
            else:
                logger.warning('AgentResponse.from_openresponses: skipping unknown item type={!r}', item_type)

        usage_obj = _gf(response, 'usage')
        if usage_obj is None:
            logger.warning(
                'AgentResponse.from_openresponses: response.usage is None; usage left '
                'None so cost reports do not record fake-zero usage for billed calls'
            )
            usage = None
        else:
            # calls=0 here; the caller patches +1 once it knows this was a billed call.
            # extract() carries cached_tokens / reasoning_tokens that the old hand-roll dropped.
            usage = TokenUsage.extract(usage_obj, calls=0)

        model = _gf(response, 'model')
        status = _gf(response, 'status')
        response_id = _gf(response, 'id')
        return cls(
            output=items,
            usage=usage,
            model=model if isinstance(model, str) else None,
            finish_reason=status if isinstance(status, str) else None,
            response_id=response_id if isinstance(response_id, str) else None,
        )


# ---------------------------------------------------------------------------
# Agent context models
# ---------------------------------------------------------------------------


class ToolInfo(BaseModel):
    """Information about an agent's tool."""

    name: str = Field(description='Tool name/identifier')
    description: str | None = Field(default=None, description='Tool description')
    parameters: dict[str, Any] | None = Field(default=None, description='Tool parameter schema')
    action_type: str | None = Field(default=None, description='Tool action type (e.g. function, http, mcp)')


class MemoryStoreInfo(BaseModel):
    """Information about an agent's memory store."""

    id: str = Field(description='Memory store ID')
    key: str | None = Field(default=None, description='Memory store key/slug')
    description: str | None = Field(default=None, description='Memory store description')


class KnowledgeBaseInfo(BaseModel):
    """Information about an agent's knowledge base."""

    id: str = Field(description='Knowledge base ID')
    key: str | None = Field(default=None, description='Knowledge base key/slug')
    name: str | None = Field(default=None, description='Knowledge base name')
    description: str | None = Field(default=None, description='Knowledge base description')


def _coerce_text(value: Any) -> str | None:
    """Normalize one provider-supplied text field to ``str | None``.

    ``str`` passes through untouched. ``bool``/``int`` are stringified — a
    wrong-typed but real value is worth more than a dropped one. Everything
    else is treated as absent, which covers ``None`` and the SDK placeholder
    objects (orq-ai-sdk parks ``Unset()`` in unset optional fields, so a
    ``getattr(obj, name, None)`` default never fires and the sentinel reaches
    us instead of ``None``).
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int)):
        return str(value)
    return None


class AgentContext(BaseModel):
    """Extended agent information for context-aware attacks.

    Describes the target agent's configuration and capabilities.

    Text fields carry one of three gates, applied by ``_normalize_text_fields``:

    * ``key`` — strict. Never coerced: it identifies *which* agent is under
      attack, so a bad value must fail loudly rather than resolve to something
      plausible.
    * ``system_prompt`` / ``instructions`` — always a ``str``, empty when not
      supplied. These two are *mutually exclusive*: agents carry
      ``instructions``, deployments carry ``system_prompt``, and a given target
      fills one and leaves the other empty. Neither is ever ``None``, so
      consumers can read them without a ``None`` guard; the empty one simply
      falls through an ``instructions or system_prompt`` chain. Both empty is
      also valid — a bring-your-own target we cannot introspect.
    * everything else — ``str | None``. Genuinely absent for some target kinds:
      a direct/OpenAI target has no orq ``id`` or ``workspace_id``, a
      bring-your-own ``AgentTarget`` has no ``display_name``/``description``/
      ``model`` we can introspect.
    """

    _REQUIRED_TEXT_FIELDS: ClassVar[tuple[str, ...]] = ('system_prompt', 'instructions')
    _LITERAL_FIELDS: ClassVar[dict[str, frozenset[str]]] = {
        'target_kind': frozenset({'agent', 'deployment', 'direct', 'openai'}),
        'agent_type': frozenset({'internal', 'a2a'}),
        'engine': frozenset({'text', 'jinja', 'mustache'}),
    }
    _OPTIONAL_TEXT_FIELDS: ClassVar[tuple[str, ...]] = (
        'id',
        'workspace_id',
        'display_name',
        'description',
        'model',
        'version',
    )

    key: str = Field(description='Agent identifier key')
    id: str | None = Field(default=None, description='Entity UUID (orq agent/deployment id), for deep-linking')
    workspace_id: str | None = Field(default=None, description='Orq workspace id, for deep-linking')
    target_kind: Literal['agent', 'deployment', 'direct', 'openai'] | None = Field(
        default=None,
        description="Target kind: 'agent' | 'deployment' | 'direct' | 'openai'. Gates the Studio deep-link.",
    )
    display_name: str | None = Field(default=None, description='Agent display name')
    description: str | None = Field(default=None, description='Agent description')
    system_prompt: str = Field(
        default='',
        description="Deployment system prompt; empty when not supplied. Exclusive with 'instructions'.",
    )
    instructions: str = Field(
        default='',
        description="Agent behavioral instructions; empty when not supplied. Exclusive with 'system_prompt'.",
    )
    tools: list[ToolInfo] = Field(default_factory=list, description='Available tools')
    memory_stores: list[MemoryStoreInfo] = Field(default_factory=list, description='Memory stores')
    knowledge_bases: list[KnowledgeBaseInfo] = Field(default_factory=list, description='Knowledge bases')
    model: str | None = Field(default=None, description='Primary model')
    version: str | None = Field(
        default=None,
        description=(
            'Target configuration version at scan time, for reproducibility: results only '
            'describe the config that was attacked. None when the provider does not expose one.'
        ),
    )
    skills: list[str] = Field(
        default_factory=list,
        description='Skill keys attached to the agent — a capability surface alongside tools',
    )
    agent_type: Literal['internal', 'a2a'] | None = Field(
        default=None,
        description=(
            "Provider-declared agent type: 'a2a' agents delegate to other agents. Distinct from "
            "'target_kind', which records how *we* reach the target. None when not exposed."
        ),
    )
    engine: Literal['text', 'jinja', 'mustache'] | None = Field(
        default=None,
        description='Prompt template engine; a templated agent has a different injection surface than plain text',
    )
    is_multi_agent: bool = Field(
        default=False,
        description=(
            'Whether the target orchestrates, delegates to, or hands off to other agents '
            '(a multi-agent system). Inferred during capability classification. Gates '
            'multi-agent-only attack categories (ASI07 Inter-Agent Communication, '
            'ASI08 Cascading Failures), which are skipped for single-agent targets. '
            'Defaults to False (conservative: do not run multi-agent attacks unless confirmed).'
        ),
    )

    @model_validator(mode='before')
    @classmethod
    def _normalize_text_fields(cls, data: Any) -> Any:
        """Coerce provider-supplied text fields before validation.

        Context retrieval is enrichment — it feeds tool and memory-store names
        into attack-strategy prompts — so a metadata field of the wrong type
        must not be able to fail the whole run. ``key`` is deliberately left
        out: it is identity, not metadata.

        The required-text fields are normalized independently, never against
        each other: ``system_prompt`` and ``instructions`` are mutually
        exclusive, so a target supplying one and omitting the other is the
        normal case and must not be treated as missing data.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for name in cls._OPTIONAL_TEXT_FIELDS:
            if name in data:
                data[name] = _coerce_text(data[name])
        for name in cls._REQUIRED_TEXT_FIELDS:
            if name in data:
                data[name] = _coerce_text(data[name]) or ''
        for name, allowed in cls._LITERAL_FIELDS.items():
            if name in data:
                value = data[name]
                # Unknown members are dropped rather than raised on: providers add
                # enum values faster than we can follow, and an unrecognized one is
                # not worth failing an entire scan over.
                data[name] = value if isinstance(value, str) and value in allowed else None
        if 'skills' in data:
            value = data['skills']
            data['skills'] = [s for s in value if isinstance(s, str)] if isinstance(value, list) else []
        return data

    @property
    def has_tools(self) -> bool:
        """Check if agent has any tools configured."""
        return len(self.tools) > 0

    @property
    def has_memory(self) -> bool:
        """Check if agent has any memory stores configured."""
        return len(self.memory_stores) > 0

    @property
    def has_knowledge(self) -> bool:
        """Check if agent has any knowledge bases configured."""
        return len(self.knowledge_bases) > 0

    def get_tool_names(self) -> list[str]:
        """Get list of tool names."""
        return [t.name for t in self.tools]


# ---------------------------------------------------------------------------
# AgentTarget ABC
# ---------------------------------------------------------------------------


class AgentTarget(ABC):
    """Abstract base class for agent targets that can receive messages.

    Subclasses implement ``respond`` (the canonical message-based interface)
    and ``new``. ``respond`` is the sole response method; callers own the
    conversation transcript.
    Targets that back a server-side memory store override ``get_agent_context``
    (self-describing), ``cleanup_memory`` (release created entities), and
    ``map_error`` (provider-specific error codes); stateless targets inherit
    the safe defaults. ``memory_entity_id`` is an instance attribute (set in
    ``__init__``) so subclasses can mutate it without shadowing a class default.
    """

    def __init__(self, memory_entity_id: str | None = None) -> None:
        self.memory_entity_id = memory_entity_id
        # Seeded (constructor arg / plain assignment) vs auto-minted: subclasses
        # whose ``new()`` distinguishes the two (ORQAgentTarget,
        # OrqResponsesTarget) preserve seeded ids across clones and re-mint
        # minted ones. ``mint_memory_entity_id`` is the non-seeding write path.
        self._memory_entity_seeded = memory_entity_id is not None

    def mint_memory_entity_id(self, value: str) -> None:
        """Install an auto-minted memory entity id without marking it seeded.

        Plain assignment to ``memory_entity_id`` counts as an explicit seed
        (clone-preserving); backends minting a fallback scope use this instead
        so every clone keeps re-minting its own isolated entity.
        """
        self.memory_entity_id = value
        self._memory_entity_seeded = False

    @abstractmethod
    async def respond(self, messages: list[Message]) -> AgentResponse:
        """Send a list of messages; return the response.

        Contract notes for implementers and callers:
        - ``Message`` carries the full chat shape (tool calls, tool results),
          but most targets consume only ``role`` + ``content`` and treat the
          tool-call fields as advisory. Do not rely on a target round-tripping
          ``tool_calls`` / ``tool_call_id`` / ``name`` unless its docstring says so.
        - Some targets that hold server-side conversation state (e.g.
          ``ORQAgentTarget``) require ``messages[-1].role == "user"`` and forward
          only that last turn; they raise ``ValueError`` otherwise.
        """
        ...

    @abstractmethod
    def new(self) -> AgentTarget:
        """Return a fresh independent instance for a new attack."""
        ...

    async def get_agent_context(self) -> AgentContext:
        """Default: minimal context. Override for platform-backed targets."""
        return AgentContext(key=getattr(self, 'agent_key', 'unknown'))

    async def cleanup_memory(self, ctx: AgentContext, entity_ids: list[str]) -> None:
        """Release any memory entities this target created. Default: no-op (stateless)."""
        return

    def map_error(self, exc: Exception) -> tuple[str, str] | None:
        """Map an exception to a provider-specific ``(code, message)``.

        Return ``None`` to defer to the backend's default mapping. Stateless
        targets inherit this no-opinion default.
        """
        return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


ReportSectionKind = Literal[
    # Shared
    'summary',
    'token_usage',
    'individual_results',
    'agent_context',
    # Simulation-specific
    'overview',
    'failures_first',
    'failure_mode',
    'persona_scenario_heatmap',
    'score_distribution',
    'turn_quality_timeline',
    'persona_breakdown',
    'scenario_breakdown',
    'judge_verdicts',
    'turn_metrics',
    'evaluator_scores',
    'errors',
    'recommendations',
    # Red-team-specific
    'focus_areas',
    'vulnerability_breakdown',
    'category_breakdown',
    'technique_breakdown',
    'delivery_breakdown',
    'error_analysis',
    'attack_heatmap',
    'agent_comparison',
    'agent_disagreements',
    'framework_breakdown',
    'turn_scope_breakdown',
    'turn_depth_analysis',
    'source_distribution',
    'severity_definitions',
    'methodology',
    # Pairwise-specific
    'pairwise_consensus',
    'pairwise_judges',
    'pairwise_comparisons',
]
"""Closed set of known report section kinds across simulation, red-team and pairwise."""


@dataclass(frozen=True)
class ReportSection:
    """Renderer-agnostic section of a report.

    Section builders (in ``redteam.reports`` / ``simulation.reports``) emit a
    list of these; renderers (in ``evaluatorq.common.reports``) consume them and
    dispatch by ``kind`` to a module-specific render function.

    Frozen so the ``kind`` / ``title`` identity is fixed once built. ``data``
    is itself mutable — red-team's HTML renderer enriches the summary
    section's data dict with values derived from the full report object,
    which would be awkward to thread through the section builders.

    Attributes:
        kind: Machine-readable section identifier (e.g. ``"summary"``).
        title: Human-readable section title.
        data: Free-form dict of section data consumed by renderers.
    """

    kind: ReportSectionKind
    title: str
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Run lifecycle manifest
# ---------------------------------------------------------------------------
# Shared by red teaming and agent simulation to track a run's stage + status on
# disk while it executes (the report only lands on completion). Behaviour lives
# in ``evaluatorq.common.run_manifest``; the data models live here per the
# "all shared cross-surface data models in contracts.py" convention.


class ManifestSurface(StrEnum):
    """Which runner produced a manifest."""

    SIM = 'sim'
    REDTEAM = 'redteam'


class ManifestStatus(StrEnum):
    """Terminal / in-flight status of a run or one of its stages."""

    RUNNING = 'running'
    COMPLETED = 'completed'
    ERROR = 'error'
    CANCELLED = 'cancelled'


# Constant so writer sites and readers agree on the one summary key both surfaces
# populate — a typo here can't silently read back as 0.
RUN_SUMMARY_TOTAL_KEY = 'total_results'


class RunSummary(TypedDict, total=False):
    """Compact headline stats stashed on ``RunManifest.summary`` at completion.

    All keys optional (``total=False``). ``total_results`` is the one key both
    surfaces populate (the run-list row's case count); the remaining keys are
    surface-specific extras the ``runs`` tables / dashboard cards read.
    """

    total_results: int
    # Red-team extras
    pipeline: str
    total_attacks: int
    vulnerability_rate: float
    resistance_rate: float
    tested_agents: list[str]
    # Sim extras
    mode: str
    target_kind: str
    scorer_averages: dict[str, float]


class StageRecord(BaseModel):
    """One pipeline stage's own status + timing within a run.

    Note: a stage's ``status`` is only ever ``running`` / ``completed`` /
    ``error`` — ``cancelled`` is a run-level terminal state and is never applied
    to an individual stage.
    """

    name: str
    target: str | None = None
    status: ManifestStatus = ManifestStatus.RUNNING
    started_at: datetime
    ended_at: datetime | None = None

    @property
    def duration_seconds(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()


class RunManifest(BaseModel):
    """Lifecycle record for a single run. One file per run, keyed by run_id."""

    run_id: str
    surface: ManifestSurface
    run_name: str
    status: ManifestStatus = ManifestStatus.RUNNING
    stage: str | None = None  # name of the current / most-recent stage
    stages: list[StageRecord] = Field(default_factory=list)
    started_at: datetime
    updated_at: datetime
    ended_at: datetime | None = None  # set when the run reaches a terminal status
    error: str | None = None
    report_path: str | None = None
    summary: RunSummary | None = Field(
        default=None,
        description=(
            'Compact headline stats captured at completion so a run-list row can be built '
            'without reading the full report. Small scalars / short lists only — never full results.'
        ),
    )

    @property
    def duration_seconds(self) -> float | None:
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()


__all__ = [
    'DEFAULT_PIPELINE_MODEL',
    'DEFAULT_TARGET_MAX_TOKENS',
    'DEFAULT_TARGET_TIMEOUT_MS',
    'JURY_RAW_OUTPUT_KEY',
    'RUN_SUMMARY_TOTAL_KEY',
    'AgentContext',
    'AgentResponse',
    'AgentResponseError',
    'AgentTarget',
    'ContentPart',
    'FunctionCall',
    'JuryResult',
    'JuryStats',
    'JuryVote',
    'KnowledgeBaseInfo',
    'LLMCallConfig',
    'ManifestStatus',
    'ManifestSurface',
    'MemoryStoreInfo',
    'Message',
    'OutputMessage',
    'ReasoningOutputItem',
    'ReportSection',
    'ReportSectionKind',
    'RunManifest',
    'RunSummary',
    'StageRecord',
    'StrategyToolCall',
    'TextOutputItem',
    'TokenUsage',
    'ToolCallOutputItem',
    'ToolInfo',
    'Usage',
    'content_to_text',
]
