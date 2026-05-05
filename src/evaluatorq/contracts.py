"""Shared types used across evaluatorq subpackages."""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    temperature: float = 1.0
    max_tokens: int = DEFAULT_TARGET_MAX_TOKENS
    timeout_ms: int = 90_000
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)
    client: Any = None  # AsyncOpenAI | None — Any avoids import cycle


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


OutputMessage = TextOutputItem | ToolCallOutputItem | ReasoningOutputItem


# ---------------------------------------------------------------------------
# ExecutedToolCall
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutedToolCall:
    """Backward-compatible view of an executed tool call.

    ``AgentResponse.tool_calls`` derives this from ``ToolCallOutputItem`` and
    intentionally drops OpenResponses-only metadata such as ``id``. Consumers
    that need full-fidelity output ordering or IDs should inspect
    ``AgentResponse.output`` directly.
    """

    name: str
    arguments: dict[str, Any]
    result: str | None = None


# ---------------------------------------------------------------------------
# AgentResponse
# ---------------------------------------------------------------------------


def _validate_output_messages(output: list[OutputMessage]) -> None:
    if not isinstance(output, list):
        raise TypeError("AgentResponse.output must be a list of output message items")
    for index, item in enumerate(output):
        if not isinstance(item, (TextOutputItem, ToolCallOutputItem, ReasoningOutputItem)):
            raise TypeError(
                "AgentResponse.output items must be TextOutputItem (OutputTextContent), "
                f"ToolCallOutputItem (FunctionCall), or ReasoningOutputItem; item {index} "
                f"was {type(item).__name__}"
            )


@dataclass(frozen=True, init=False)
class AgentResponse:
    """Structured response from a target agent as an ordered list of output messages.

    Each item in ``output`` is a :class:`TextOutputItem`, :class:`ToolCallOutputItem`,
    or :class:`ReasoningOutputItem`, preserving the order in which they were produced.
    Item ``type`` discriminators align with the OpenResponses intermediate data format
    (``output_text``, ``function_call``, ``reasoning``).

    This unified representation lets evaluators reason about tool call order and
    the interleaving of tool calls, reasoning steps, and text responses — something
    not possible when text, tool calls, and intermediate steps are stored separately.

    Backward-compatible accessors:
        ``.text``       — last ``TextOutputItem`` content, or ``""`` if none
        ``.tool_calls`` — lossy :class:`ExecutedToolCall` view extracted from
        tool call items; this intentionally omits ``ToolCallOutputItem.id``
    """

    output: list[OutputMessage] = field(default_factory=list)

    def __init__(
        self,
        output: list[OutputMessage] | None = None,
        *,
        text: str | None = None,
    ) -> None:
        """Create a response from ordered output items or legacy ``text=``.

        ``text=`` is kept for older tests/integrations that constructed
        ``AgentResponse(text="...")`` before ordered output items existed.
        """
        if output is not None and text is not None:
            raise ValueError("AgentResponse accepts either output= or text=, not both")
        items: list[OutputMessage]
        if output is None:
            items = [TextOutputItem(text=text, annotations=[])] if text is not None else []
        else:
            _validate_output_messages(output)
            items = output
        object.__setattr__(self, "output", items)

    @property
    def text(self) -> str:
        """Return the last text output item's content (backward-compatible)."""
        texts = [item.text for item in self.output if isinstance(item, TextOutputItem)]
        return texts[-1] if texts else ""

    @property
    def tool_calls(self) -> list[ExecutedToolCall]:
        """Return a lossy backward-compatible view of tool call output items.

        ``ToolCallOutputItem.id`` is not represented on ``ExecutedToolCall``.
        Use ``.output`` when preserving OpenResponses IDs or item ordering
        matters.

        ``ToolCallOutputItem.arguments`` is stored as a JSON string (OpenResponses
        wire format); this property parses it back to a dict for ``ExecutedToolCall``.
        """
        result: list[ExecutedToolCall] = []
        for item in self.output:
            if not isinstance(item, ToolCallOutputItem):
                continue
            raw_args = item.arguments
            if isinstance(raw_args, str):
                try:
                    args: dict[str, Any] = _json.loads(raw_args)
                    if not isinstance(args, dict):
                        args = {}
                except (ValueError, TypeError):
                    args = {}
            else:
                args = raw_args if isinstance(raw_args, dict) else {}
            result.append(ExecutedToolCall(name=item.name, arguments=args, result=item.result))
        return result


__all__ = [
    "DEFAULT_PIPELINE_MODEL",
    "DEFAULT_TARGET_MAX_TOKENS",
    "DEFAULT_TARGET_TIMEOUT_MS",
    "AgentResponse",
    "ExecutedToolCall",
    "LLMCallConfig",
    "OutputMessage",
    "ReasoningOutputItem",
    "TextOutputItem",
    "ToolCallOutputItem",
]
