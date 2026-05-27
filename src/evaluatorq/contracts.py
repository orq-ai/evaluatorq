"""Shared types used across evaluatorq subpackages."""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        """String enum compatible with Python 3.10."""

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    _ClientT: TypeAlias = AsyncOpenAI | None
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
    TextOutputItem | ToolCallOutputItem | ReasoningOutputItem,
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
    def from_completion(cls, response: Any) -> TokenUsage | None:
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

    def __add__(self, other: TokenUsage | None) -> TokenUsage:
        if other is None:
            return self.model_copy()
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            calls=self.calls + other.calls,
        )

    def __radd__(self, other: object) -> TokenUsage:
        """Support sum() which starts with integer 0, and reflected addition."""
        if isinstance(other, int) and other == 0:
            return self.model_copy()
        if isinstance(other, TokenUsage):
            return other.__add__(self)
        return NotImplemented


# ---------------------------------------------------------------------------
# AgentResponseError
# ---------------------------------------------------------------------------


class AgentResponseError(BaseModel):
    """A per-response error marker on :class:`AgentResponse`.

    Set when a target (or simulation agent) failed to produce a real response,
    e.g. a timeout or backend exception. The orchestrator uses its presence to
    exclude the turn from the transcript replayed to the target. ``message`` is
    the same human text surfaced in ``AgentResponse.text``; ``error_type`` is a
    coarse kind ("timeout" | "exception" | "content_filter" | "target_error" | ...); ``code`` is
    an optional provider/mapped code.

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


# ---------------------------------------------------------------------------
# Agent context models
# ---------------------------------------------------------------------------


class ToolInfo(BaseModel):
    """Information about an agent's tool."""

    name: str = Field(description='Tool name/identifier')
    description: str | None = Field(default=None, description='Tool description')
    parameters: dict[str, Any] | None = Field(default=None, description='Tool parameter schema')


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
    and ``new``. ``send_prompt`` is a concrete shim retained for single-prompt
    callers — it wraps the prompt in one user message and calls ``respond``.
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

    async def send_prompt(self, prompt: str) -> AgentResponse:
        """Back-compat shim: wrap a single prompt in one user message and call ``respond``."""
        return await self.respond([Message(role="user", content=prompt)])

    async def get_agent_context(self) -> AgentContext:
        """Default: minimal context. Override for platform-backed targets."""
        return AgentContext(key=getattr(self, "agent_key", "unknown"))

    async def cleanup_memory(self, ctx: AgentContext, entity_ids: list[str]) -> None:
        """Release any memory entities this target created. Default: no-op (stateless)."""
        return

    def map_error(self, exc: Exception) -> tuple[str, str] | None:
        """Map an exception to a provider-specific ``(code, message)``.

        Return ``None`` to defer to the backend's default mapping. Stateless
        targets inherit this no-opinion default.
        """
        return None


__all__ = [
    "DEFAULT_PIPELINE_MODEL",
    "DEFAULT_TARGET_MAX_TOKENS",
    "DEFAULT_TARGET_TIMEOUT_MS",
    "AgentContext",
    "AgentResponse",
    "AgentResponseError",
    "AgentTarget",
    "FunctionCall",
    "KnowledgeBaseInfo",
    "LLMCallConfig",
    "MemoryStoreInfo",
    "Message",
    "OutputMessage",
    "ReasoningOutputItem",
    "StrategyToolCall",
    "TextOutputItem",
    "TokenUsage",
    "ToolCallOutputItem",
    "ToolInfo",
]
