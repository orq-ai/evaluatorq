"""Pydantic models for convert.py - subset that avoids discriminator issues.

This module contains only the models needed for the LangChain convert.py module,
extracted from models.py to avoid the problematic discriminated unions that
cause runtime import errors.
"""

from __future__ import annotations

import json
import uuid
from enum import Enum
from typing import Annotated, Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator


class MessageRole(Enum):
    user = "user"
    assistant = "assistant"
    system = "system"
    developer = "developer"


class MessageStatus(Enum):
    in_progress = "in_progress"
    completed = "completed"
    incomplete = "incomplete"


class FunctionCallStatus(Enum):
    """Status for function calls and function call outputs."""
    in_progress = "in_progress"
    completed = "completed"
    incomplete = "incomplete"


# Alias for backwards compatibility - both function calls and outputs use the same status values
FunctionCallOutputStatusEnum = FunctionCallStatus


class InputTextContent(BaseModel):
    type: Annotated[
        Literal["input_text"],
        Field(description="The type of the input item. Always `input_text`."),
    ]
    text: Annotated[str, Field(description="The text input to the model.")]


class _OmitNoneContent(BaseModel):
    """Content part that drops None-valued optional fields on serialization.

    The Orq Responses API treats an unset optional content field as *absent*,
    not as an explicit ``null`` — sending ``null`` is rejected. So a content
    part serializes only the fields actually provided.
    """

    @model_serializer(mode="wrap")
    def _omit_none(self, handler: Any) -> dict[str, Any]:
        return {k: v for k, v in handler(self).items() if v is not None}


class InputImageContent(_OmitNoneContent):
    type: Annotated[
        Literal["input_image"],
        Field(description="The type of the input item. Always `input_image`."),
    ]
    image_url: Annotated[str, Field(description="The URL of the image (https or data: URI).")]
    file_id: str | None = Field(default=None, description="The ID of an uploaded file to use as the image.")
    detail: Literal["auto", "low", "high"] = Field(
        default="auto", description="The detail level the model uses to process the image."
    )


class InputFileContent(_OmitNoneContent):
    type: Annotated[
        Literal["input_file"],
        Field(description="The type of the input item. Always `input_file`."),
    ]
    file_id: str | None = Field(default=None, description="The ID of an uploaded file.")
    file_data: str | None = Field(default=None, description="The base64-encoded contents of the file.")
    file_url: str | None = Field(default=None, description="The URL of the file.")
    filename: str | None = Field(default=None, description="The name of the file.")

    @model_validator(mode="after")
    def _require_one_source(self) -> InputFileContent:
        # Orq's ResponsesInputFileSchema is a union of three single-source
        # variants — exactly one of file_data / file_id / file_url is required.
        # An empty file part is rejected server-side, so fail fast here instead.
        sources = (self.file_data, self.file_id, self.file_url)
        if sum(s is not None for s in sources) != 1:
            raise ValueError(
                "InputFileContent requires exactly one of file_data, file_id, or file_url"
            )
        return self


class UrlCitationBody(BaseModel):
    type: Annotated[
        Literal["url_citation"],
        Field(description="The type of the URL citation. Always `url_citation`."),
    ]
    url: Annotated[str, Field(description="The URL of the web resource.")]
    start_index: Annotated[
        int,
        Field(
            description="The index of the first character of the URL citation in the message."
        ),
    ]
    end_index: Annotated[
        int,
        Field(
            description="The index of the last character of the URL citation in the message."
        ),
    ]
    title: Annotated[str, Field(description="The title of the web resource.")]


class TopLogProb(BaseModel):
    token: str
    logprob: float
    bytes: list[int]


class LogProb(BaseModel):
    token: str
    logprob: float
    bytes: list[int]
    top_logprobs: list[TopLogProb]


class OutputTextContent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["output_text"] = Field(
        default="output_text",
        description="The type of the output text. Always `output_text`.",
    )
    text: Annotated[str, Field(description="The text output from the model.")]
    annotations: Annotated[
        list[UrlCitationBody], Field(default_factory=list, description="The annotations of the text output.")
    ]
    logprobs: list[LogProb] = Field(default_factory=list)


class FunctionCall(BaseModel):
    """OpenResponses wire-format function call (Responses API spec) — distinct from contracts.FunctionCall (internal canonical, chat-completions shape: name/args only)."""

    model_config = ConfigDict(frozen=True)

    type: Literal["function_call"] = Field(
        default="function_call",
        description="The type of the item. Always `function_call`.",
    )
    id: str = Field(
        default_factory=lambda: f"fc_{uuid.uuid4().hex}",
        description="The unique ID of the function call item.",
    )
    call_id: str = Field(
        default_factory=lambda: f"call_{uuid.uuid4().hex}",
        description="The unique ID of the function tool call that was generated.",
    )
    name: Annotated[str, Field(description="The name of the function that was called.")]
    arguments: Annotated[
        str, Field(description="The arguments JSON string that was generated.")
    ]
    status: FunctionCallStatus = FunctionCallStatus.completed
    result: str | None = Field(
        default=None,
        description="The result returned by the tool call, if available.",
    )

    @field_validator("arguments", mode="before")
    @classmethod
    def _serialize_arguments(cls, v: Any) -> str:
        """Accept dict or str for arguments; serialize dict to JSON string."""
        if isinstance(v, str):
            return v
        try:
            return json.dumps(v, default=str)
        except (TypeError, ValueError) as e:
            logger.warning(
                "FunctionCall._serialize_arguments: arguments not JSON-serializable, "
                "falling back to repr (value={!r}, err={})",
                v,
                e,
            )
            return str(v)

    @property
    def arguments_dict(self) -> dict[str, Any]:
        """Parse the JSON ``arguments`` string into a dict.

        Returns an empty dict if the JSON is malformed or not an object —
        callers should not rely on the parsed shape for adversary-controlled
        tool calls.
        """
        try:
            parsed = json.loads(self.arguments)
        except (ValueError, TypeError) as e:
            logger.warning(
                "FunctionCall.arguments_dict: arguments not valid JSON (err={}, raw={!r})",
                e,
                self.arguments[:200],
            )
            return {}
        if not isinstance(parsed, dict):
            logger.warning(
                "FunctionCall.arguments_dict: parsed JSON is not an object (type={}, raw={!r})",
                type(parsed).__name__,
                self.arguments[:200],
            )
            return {}
        return parsed


class FunctionCallOutput(BaseModel):
    type: Annotated[
        Literal["function_call_output"],
        Field(
            description="The type of the function tool call output. Always `function_call_output`."
        ),
    ]
    id: Annotated[
        str,
        Field(
            description="The unique ID of the function tool call output. Populated when this item is returned via API."
        ),
    ]
    call_id: Annotated[
        str,
        Field(
            description="The unique ID of the function tool call generated by the model."
        ),
    ]
    output: str
    status: FunctionCallOutputStatusEnum


class Message(BaseModel):
    """OpenResponses wire-format message (Responses API spec: type/id/status/role) — distinct from contracts.Message (internal canonical, chat-completions shape)."""

    type: Annotated[
        Literal["message"],
        Field(description="The type of the message. Always set to `message`."),
    ]
    id: Annotated[str, Field(description="The unique ID of the message.")]
    status: MessageStatus
    role: MessageRole
    content: Annotated[
        list[InputTextContent | InputImageContent | InputFileContent | OutputTextContent],
        Field(description="The content of the message"),
    ]


class FunctionTool(BaseModel):
    type: Annotated[
        Literal["function"],
        Field(description="The type of the function tool. Always `function`."),
    ]
    name: Annotated[str, Field(description="The name of the function to call.")]
    description: str | None
    parameters: dict[str, Any] | None
    strict: bool | None


class InputTokensDetails(BaseModel):
    cached_tokens: Annotated[
        int,
        Field(description="The number of input tokens that were served from cache."),
    ]


class OutputTokensDetails(BaseModel):
    reasoning_tokens: Annotated[
        int,
        Field(
            description="The number of output tokens that were attributed to reasoning."
        ),
    ]


class Usage(BaseModel):
    """OpenResponses wire-format token usage (Responses API spec: input/output/total + detail fields) — distinct from contracts.TokenUsage (internal canonical, carries calls count)."""

    input_tokens: Annotated[
        int,
        Field(
            description="The number of input tokens that were used to generate the response."
        ),
    ]
    output_tokens: Annotated[
        int,
        Field(
            description="The number of output tokens that were generated by the model."
        ),
    ]
    total_tokens: Annotated[
        int, Field(description="The total number of tokens that were used.")
    ]
    input_tokens_details: InputTokensDetails
    output_tokens_details: OutputTokensDetails


class IncompleteDetails(BaseModel):
    reason: Annotated[
        str, Field(description="The reason the response could not be completed.")
    ]
