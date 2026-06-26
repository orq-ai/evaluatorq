"""Shared types used across evaluatorq subpackages."""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias

from loguru import logger
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_serializer, model_validator

from evaluatorq.openresponses.convert_models import (
    InputFileContent,
    InputImageContent,
    InputTextContent,
)

# A single part of multi-modal message content (Responses-API input shapes).
ContentPart: TypeAlias = InputTextContent | InputImageContent | InputFileContent


def _content_part_to_chat_block(part: ContentPart) -> dict[str, Any]:
    """Render a Responses-style content part as an OpenAI chat-completions block.

    Images and files use the chat-completions shapes (``image_url`` / ``file``),
    which differ from the Responses ``input_image`` / ``input_file`` wire shapes.
    """
    if isinstance(part, InputTextContent):
        return {'type': 'text', 'text': part.text}
    if isinstance(part, InputImageContent):
        image_url: dict[str, Any] = {}
        if part.image_url is not None:
            image_url['url'] = part.image_url
        image_url['detail'] = part.detail
        return {'type': 'image_url', 'image_url': image_url}
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
    multi-part list is joined from its text parts, but a non-text part (image or
    file) raises :class:`NotImplementedError` with a clear message rather than
    silently dropping content.
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


class TokenUsage(BaseModel):
    """Token usage and cost for an LLM call or aggregation of calls.

    Field naming follows the OpenTelemetry GenAI semconv / OpenResponses standard
    (``input_tokens``/``output_tokens`` rather than ``prompt``/``completion``).
    ``cached_tokens`` is a subset of ``input_tokens`` and ``reasoning_tokens`` a
    subset of ``output_tokens``, so the reconciliation invariant is
    ``total_tokens == input_tokens + output_tokens`` (with cached ≤ input,
    reasoning ≤ output). ``total_tokens`` is stored as provided by the upstream
    provider and never overridden.

    Back-compat: the legacy ``prompt_tokens``/``completion_tokens`` names are
    accepted on construction and exposed as read-only properties so call sites
    that have not migrated keep working.
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
    reasoning_tokens: int = Field(default=0, ge=0, description='Reasoning output tokens (subset of output_tokens)')
    cost_usd: float | None = Field(
        default=None, ge=0, description='Cost in USD when the provider reports it, else None'
    )
    calls: int = Field(default=0, ge=0, description='Number of LLM API calls')

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
            reasoning_tokens: int = ...,
            cost_usd: float | None = ...,
            calls: int = ...,
        ) -> None: ...

    @property
    def prompt_tokens(self) -> int:
        """Deprecated alias for :attr:`input_tokens`."""
        return self.input_tokens

    @property
    def completion_tokens(self) -> int:
        """Deprecated alias for :attr:`output_tokens`."""
        return self.output_tokens

    @model_serializer(mode='wrap')
    def _serialize(self, handler: Any) -> dict[str, Any]:
        """Emit the legacy ``prompt_tokens``/``completion_tokens`` keys alongside the
        standard names so dict/JSON consumers that have not migrated keep working.

        Note: the legacy aliases are injected unconditionally, so ``model_dump`` with
        ``include=``/``exclude=`` does not filter them out — field-level filtering is
        not supported on this model."""
        data = handler(self)
        data['prompt_tokens'] = self.input_tokens
        data['completion_tokens'] = self.output_tokens
        return data

    @classmethod
    def extract(cls, usage: Any, *, calls: int = 1) -> TokenUsage | None:
        """Map any provider usage shape (object or dict) into a TokenUsage.

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
        cached = _usage_detail_int(
            usage,
            ('input_tokens_details', 'prompt_tokens_details', 'input_token_details'),
            ('cached_tokens', 'cache_read'),
        )
        reasoning = _usage_detail_int(
            usage,
            ('output_tokens_details', 'completion_tokens_details', 'output_token_details'),
            ('reasoning_tokens', 'reasoning'),
        )
        # Clamp at 0: a provider may report a negative cost (credit/adjustment),
        # which would otherwise trip the cost_usd ge=0 constraint and crash the
        # whole extraction path for an otherwise-valid billed call. Log it so a
        # genuine provider/accounting bug is not silently rewritten to $0.
        cost_raw = _usage_first_float(usage, ('cost_usd', 'cost', 'total_cost'))
        if cost_raw is not None and cost_raw < 0:
            logger.warning('TokenUsage.extract: provider reported negative cost {!r}; clamping to 0.0', cost_raw)
        cost = max(cost_raw, 0.0) if cost_raw is not None else None
        # A present-but-empty/unparseable payload yields no signal: don't fabricate a
        # billed-zero record. A genuine zero-token call is kept only if it carries
        # some signal — cached tokens or a cost (e.g. a cached-only / minimum charge).
        if total <= 0 and cached <= 0 and cost is None:
            return None
        # Honour a calls count already carried by the payload (e.g. an aggregated
        # TokenUsage dump); fall back to the per-call default otherwise.
        raw_calls = _usage_get(usage, 'calls')
        resolved_calls = (
            int(raw_calls) if isinstance(raw_calls, (int, float)) and not isinstance(raw_calls, bool) else calls
        )
        return cls(
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
            cached_tokens=cached,
            reasoning_tokens=reasoning,
            cost_usd=cost,
            calls=resolved_calls,
        )

    @classmethod
    def from_completion(cls, response: Any) -> TokenUsage | None:
        """Extract token usage from an OpenAI-compatible completion response."""
        return cls.extract(getattr(response, 'usage', None), calls=1)

    def __add__(self, other: TokenUsage | None) -> TokenUsage:
        if other is None:
            return self.model_copy()
        cost = self.cost_usd if other.cost_usd is None else (self.cost_usd or 0.0) + other.cost_usd
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cost_usd=cost,
            calls=self.calls + other.calls,
        )

    def __radd__(self, other: object) -> TokenUsage:
        """Support sum() which starts with integer 0, and reflected addition."""
        if isinstance(other, int) and other == 0:
            return self.model_copy()
        if isinstance(other, TokenUsage):
            return other.__add__(self)
        return NotImplemented

    def __sub__(self, other: TokenUsage | None) -> TokenUsage:
        """Component-wise difference, clamped at 0 — used for per-turn deltas."""
        if other is None:
            return self.model_copy()
        # Cost stays "unknown" (None) only when BOTH sides are unknown; if either
        # side carries a known cost, treat the missing side as 0.0 rather than
        # poisoning the delta to None (which would discard a real known cost on the
        # first turn a cost-bearing provider appears). Mirrors __add__'s asymmetric
        # "known + None = known" handling.
        cost = (
            None
            if self.cost_usd is None and other.cost_usd is None
            else max((self.cost_usd or 0.0) - (other.cost_usd or 0.0), 0.0)
        )
        return TokenUsage(
            input_tokens=max(self.input_tokens - other.input_tokens, 0),
            output_tokens=max(self.output_tokens - other.output_tokens, 0),
            total_tokens=max(self.total_tokens - other.total_tokens, 0),
            cached_tokens=max(self.cached_tokens - other.cached_tokens, 0),
            reasoning_tokens=max(self.reasoning_tokens - other.reasoning_tokens, 0),
            cost_usd=cost,
            calls=max(self.calls - other.calls, 0),
        )


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
    :func:`evaluatorq.redteam.contracts.classify_error_type`, which yields one
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
# AgentResponse
# ---------------------------------------------------------------------------


class AgentResponse(BaseModel):
    """Structured response from a target agent as an ordered list of output messages.

    Each item in ``output`` is a :class:`TextOutputItem`, :class:`ToolCallOutputItem`,
    or :class:`ReasoningOutputItem`, preserving the order in which they were produced.
    Item ``type`` discriminators align with the OpenResponses intermediate data format
    (``output_text``, ``function_call``, ``reasoning``).

    Accessors:
        ``.text``       — last ``TextOutputItem`` content, or ``""`` if none
        ``.tool_calls`` — list of :class:`ToolCallOutputItem` filtered from
        :attr:`output` in order
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    output: list[OutputMessage] = Field(default_factory=list)
    usage: TokenUsage | None = None
    model: str | None = None
    response_id: str | None = None
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
            elif item_type == 'function_call':
                raw_args = _gf(item, 'arguments') or '{}'
                call_id = _gf(item, 'call_id') or _gf(item, 'id') or ''
                result = _gf(item, 'result')
                items.append(
                    ToolCallOutputItem(
                        type='function_call',
                        name=str(_gf(item, 'name') or ''),
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


class AgentContext(BaseModel):
    """Extended agent information for context-aware attacks.

    Describes the target agent's configuration and capabilities.
    """

    key: str = Field(description='Agent identifier key')
    display_name: str | None = Field(default=None, description='Agent display name')
    description: str | None = Field(default=None, description='Agent description')
    system_prompt: str | None = Field(default=None, description='Agent system prompt (used by deployments)')
    instructions: str | None = Field(default=None, description='Agent behavioral instructions')
    tools: list[ToolInfo] = Field(default_factory=list, description='Available tools')
    memory_stores: list[MemoryStoreInfo] = Field(default_factory=list, description='Memory stores')
    knowledge_bases: list[KnowledgeBaseInfo] = Field(default_factory=list, description='Knowledge bases')
    model: str | None = Field(default=None, description='Primary model')
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
]
"""Closed set of known report section kinds across simulation and red-team."""


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


__all__ = [
    'DEFAULT_PIPELINE_MODEL',
    'DEFAULT_TARGET_MAX_TOKENS',
    'DEFAULT_TARGET_TIMEOUT_MS',
    'JURY_RAW_OUTPUT_KEY',
    'AgentContext',
    'AgentResponse',
    'AgentResponseError',
    'AgentTarget',
    'FunctionCall',
    'JuryResult',
    'JuryStats',
    'JuryVote',
    'KnowledgeBaseInfo',
    'LLMCallConfig',
    'MemoryStoreInfo',
    'Message',
    'OutputMessage',
    'ReasoningOutputItem',
    'ReportSection',
    'ReportSectionKind',
    'StrategyToolCall',
    'TextOutputItem',
    'TokenUsage',
    'ToolCallOutputItem',
    'ToolInfo',
]
