"""Shared types used across evaluatorq subpackages."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """String enum compatible with Python 3.10."""

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    _ClientT: TypeAlias = "AsyncOpenAI | None"
else:
    _ClientT = Any

# ---------------------------------------------------------------------------
# Pipeline defaults (mirrored from redteam.contracts for shared use)
# ---------------------------------------------------------------------------

DEFAULT_PIPELINE_MODEL: str = "gpt-5-mini"
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
    api: Literal["chat_completions", "responses"] = "chat_completions"
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=DEFAULT_TARGET_MAX_TOKENS, gt=0)
    timeout_ms: int = Field(default=90_000, gt=0)
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)
    client: _ClientT = None


# ---------------------------------------------------------------------------
# Output message item types — re-exported from openresponses
# ---------------------------------------------------------------------------
# AgentResponse output items align with the OpenResponses intermediate data
# format.  The type discriminators (output_text, function_call, reasoning)
# match OpenResponses wire format, making downstream conversion to
# ResponseResource straightforward.

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

    type: Literal["reasoning"] = "reasoning"
    text: str
    id: str | None = None


OutputMessage = Annotated[
    Union[TextOutputItem, ToolCallOutputItem, ReasoningOutputItem],
    Field(discriminator="type"),
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

    name: str = Field(description="Function name to call")
    arguments: str = Field(description="JSON string of function arguments")


class StrategyToolCall(BaseModel):
    """Tool call in assistant message (OpenAI format)."""

    id: str = Field(description="Unique tool call ID (e.g., call_abc123)")
    type: Literal["function"] = Field(default="function", description="Tool type")
    function: FunctionCall = Field(description="Function call details")


class Message(BaseModel):
    """Single message in conversation history (OpenAI format with tool support).

    Supports both simple messages and tool calls:
    - Simple: ``{"role": "user", "content": "Hello"}``
    - Tool call: ``{"role": "assistant", "tool_calls": [...]}``
    - Tool response: ``{"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}``
    """

    role: Literal["user", "assistant", "tool", "system"]
    content: str | None = Field(
        default=None,
        description="Message content (required for user/system, optional for assistant with tool_calls)",
    )

    # Tool call fields (OpenAI format)
    tool_calls: list[StrategyToolCall] | None = Field(
        default=None, description="Tool calls made by assistant"
    )
    tool_call_id: str | None = Field(
        default=None,
        description="ID linking tool response to call (for role=tool)",
    )
    name: str | None = Field(default=None, description="Function name (for role=tool)")


class TokenUsage(BaseModel):
    """Token usage and cost for an LLM call or aggregation of calls.

    ``total_tokens`` is stored as provided by the upstream provider and is
    never overridden. This preserves provider-specific token breakdowns
    (cached_tokens, reasoning_tokens, audio tokens, etc.) that would be
    silently corrupted by a normalisation step.
    """

    model_config = ConfigDict(frozen=True)

    total_tokens: int = Field(default=0, ge=0, description="Total tokens used as reported by the provider")
    prompt_tokens: int = Field(default=0, ge=0, description="Prompt/input tokens")
    completion_tokens: int = Field(default=0, ge=0, description="Completion/output tokens")
    calls: int = Field(default=0, ge=0, description="Number of LLM API calls")

    @classmethod
    def from_completion(cls, response: Any) -> "TokenUsage | None":
        """Extract token usage from an OpenAI-compatible completion response."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion = int(getattr(usage, "completion_tokens", 0) or 0)
        # Mirror the > 0 guard used by other integrations: a provider-reported
        # total of 0 (or absent) falls back to prompt+completion rather than
        # propagating the zero, which would silently under-count totals on
        # responses where prompt+completion are non-zero.
        raw_total = getattr(usage, "total_tokens", None)
        total = int(raw_total) if raw_total is not None and int(raw_total) > 0 else prompt + completion
        return cls(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total, calls=1)

    def __add__(self, other: "TokenUsage | None") -> "TokenUsage":
        if other is None:
            return self.model_copy()
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            calls=self.calls + other.calls,
        )

    def __radd__(self, other: object) -> "TokenUsage":
        """Support sum() which starts with integer 0, and reflected addition."""
        if isinstance(other, int) and other == 0:
            return self.model_copy()
        if isinstance(other, TokenUsage):
            return other.__add__(self)
        return NotImplemented


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

    if TYPE_CHECKING:

        def __init__(  # pyright: ignore[reportMissingSuperCall]
            self,
            *,
            output: list[OutputMessage] | None = None,
            text: str | None = None,
            usage: "TokenUsage | None" = None,
            model: str | None = None,
            response_id: str | None = None,
            finish_reason: str | None = None,
        ) -> None: ...

    @model_validator(mode="before")
    @classmethod
    def _accept_text_shorthand(cls, data: Any) -> Any:
        # Preserve legacy ``AgentResponse(text="...")`` construction.
        if isinstance(data, dict) and "text" in data:
            payload = dict(data)
            text = payload.pop("text")
            if payload.get("output") is not None:
                raise ValueError("AgentResponse accepts either output= or text=, not both")
            payload["output"] = (
                [TextOutputItem(text=text, annotations=[])] if text is not None else []
            )
            return payload
        return data

    @property
    def text(self) -> str:
        """Concatenate all text output items into a single string."""
        return "".join(item.text for item in self.output if isinstance(item, TextOutputItem))

    @property
    def tool_calls(self) -> list[ToolCallOutputItem]:
        """Return the tool call items from ``.output`` in order."""
        return [item for item in self.output if isinstance(item, ToolCallOutputItem)]


__all__ = [
    "DEFAULT_PIPELINE_MODEL",
    "DEFAULT_TARGET_MAX_TOKENS",
    "DEFAULT_TARGET_TIMEOUT_MS",
    "AgentResponse",
    "FunctionCall",
    "LLMCallConfig",
    "Message",
    "OutputMessage",
    "ReasoningOutputItem",
    "StrategyToolCall",
    "TextOutputItem",
    "TokenUsage",
    "ToolCallOutputItem",
]
