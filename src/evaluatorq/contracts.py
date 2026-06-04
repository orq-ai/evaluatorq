"""Shared types used across evaluatorq subpackages."""

from __future__ import annotations

import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias

from loguru import logger
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
    # Responses-API item id (e.g. ``fc_abc123``), distinct from ``id`` (which maps to
    # the chat-completions / Responses ``call_id``). Preserved through transcript
    # replay so :class:`OpenAIAgentTarget` can echo the original ``function_call``
    # item back to the Responses API on subsequent turns. ``None`` for tool calls
    # that did not originate from a Responses-API turn.
    item_id: str | None = Field(default=None, description="Responses-API function_call item id (fc_*)")


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


def message_to_chat_param(m: Message) -> dict[str, Any]:
    """Render a :class:`Message` as an OpenAI chat-completions message dict.

    Preserves tool-call structure that a naive ``{"role", "content"}`` flatten
    would drop: an assistant message's ``tool_calls`` and a ``tool`` row's
    ``tool_call_id``/``name``. Used by stateless targets to replay a transcript
    (built by ``turns_to_messages``) without losing multi-turn tool context.
    """
    if m.role == "tool":
        # A tool message carries a result keyed to tool_call_id; tool_calls belong
        # on assistant messages, so any present here are malformed and not emitted.
        param: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": m.tool_call_id or "",
            "content": m.content or "",
        }
        if m.name:
            param["name"] = m.name
        return param
    if m.role == "assistant" and m.tool_calls:
        # content may be None on a pure tool-call turn; OpenAI accepts null.
        return {
            "role": "assistant",
            "content": m.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in m.tool_calls
            ],
        }
    return {"role": m.role, "content": m.content or ""}


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
        for item in _gf(response, "output") or []:
            item_type = _gf(item, "type")
            if item_type == "message":
                for part in _gf(item, "content") or []:
                    if _gf(part, "type") == "output_text" and _gf(part, "text"):
                        items.append(
                            TextOutputItem(
                                type="output_text",
                                text=_gf(part, "text"),
                                annotations=[],
                                logprobs=[],
                            )
                        )
            elif item_type == "function_call":
                raw_args = _gf(item, "arguments") or "{}"
                call_id = _gf(item, "call_id") or _gf(item, "id") or ""
                result = _gf(item, "result")
                items.append(
                    ToolCallOutputItem(
                        type="function_call",
                        name=str(_gf(item, "name") or ""),
                        call_id=str(call_id),
                        arguments=raw_args if isinstance(raw_args, str) else json.dumps(raw_args),
                        result=str(result) if result is not None else None,
                    )
                )
            elif item_type == "reasoning":
                pass  # o1/o3/o4-mini reasoning steps intentionally excluded
            else:
                logger.warning("AgentResponse.from_openresponses: skipping unknown item type={!r}", item_type)

        usage_obj = _gf(response, "usage")
        if usage_obj is None:
            logger.warning(
                "AgentResponse.from_openresponses: response.usage is None; usage left "
                "None so cost reports do not record fake-zero usage for billed calls"
            )
            usage = None
        else:
            input_toks = int(_gf(usage_obj, "input_tokens", 0) or 0)
            output_toks = int(_gf(usage_obj, "output_tokens", 0) or 0)
            usage = TokenUsage(
                prompt_tokens=input_toks,
                completion_tokens=output_toks,
                total_tokens=input_toks + output_toks,
            )

        model = _gf(response, "model")
        status = _gf(response, "status")
        response_id = _gf(response, "id")
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


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


ReportSectionKind = Literal[
    # Shared
    "summary",
    "token_usage",
    "individual_results",
    "agent_context",
    # Simulation-specific
    "overview",
    "failures_first",
    "failure_mode",
    "persona_scenario_heatmap",
    "score_distribution",
    "turn_quality_timeline",
    "persona_breakdown",
    "scenario_breakdown",
    "judge_verdicts",
    "turn_metrics",
    "evaluator_scores",
    "errors",
    # Red-team-specific
    "focus_areas",
    "vulnerability_breakdown",
    "category_breakdown",
    "technique_breakdown",
    "delivery_breakdown",
    "error_analysis",
    "attack_heatmap",
    "agent_comparison",
    "agent_disagreements",
    "framework_breakdown",
    "turn_scope_breakdown",
    "turn_depth_analysis",
    "source_distribution",
    "severity_definitions",
    "methodology",
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
    "ReportSection",
    "ReportSectionKind",
    "StrategyToolCall",
    "TextOutputItem",
    "TokenUsage",
    "ToolCallOutputItem",
    "ToolInfo",
    "message_to_chat_param",
]
