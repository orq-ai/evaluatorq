"""Shared types used across evaluatorq subpackages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, TypeAlias, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    ``usage`` holds a :class:`evaluatorq.redteam.contracts.TokenUsage` instance
    when available. The field is typed ``Any | None`` at runtime to avoid a
    circular import: ``redteam.contracts`` imports this module, so
    ``TokenUsage`` cannot be imported here unconditionally. Static type
    checkers see the correct annotation via the ``TYPE_CHECKING`` stub below.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    output: list[OutputMessage] = Field(default_factory=list)
    usage: Any | None = None
    model: str | None = None
    response_id: str | None = None
    finish_reason: str | None = None

    if TYPE_CHECKING:
        from evaluatorq.redteam.contracts import TokenUsage

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
    "LLMCallConfig",
    "OutputMessage",
    "ReasoningOutputItem",
    "TextOutputItem",
    "ToolCallOutputItem",
]

