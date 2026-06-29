"""Conversion functions from SimulationResult to OpenResponses format."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from evaluatorq.contracts import content_to_text
from evaluatorq.openresponses.convert_models import (
    FunctionCall,
    FunctionCallOutput,
    FunctionCallStatus,
    IncompleteDetails,
    InputFileContent,
    InputImageContent,
    InputTextContent,
    InputTokensDetails,
    Message,
    MessageRole,
    MessageStatus,
    OutputTextContent,
    OutputTokensDetails,
    Usage,
)

# Message.content accepts any content part; simulation only emits text parts, but
# the variable annotation must match the (invariant) list element union.
_ContentList = list[InputTextContent | InputImageContent | InputFileContent | OutputTextContent]

if TYPE_CHECKING:
    from evaluatorq.simulation.types import SimulationResult


def _generate_item_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def to_open_responses(
    result: SimulationResult,
    model: str = "simulation",
) -> dict[str, Any]:
    """Convert a SimulationResult to OpenResponses format.

    Mapping:
    - messages with role "user"      -> input[] as Message with input_text content
    - messages with role "assistant"  -> output[] as Message with output_text content
    - messages with role "system"     -> input[] as Message with input_text content
    - token_usage                     -> Usage
    - terminated_by                   -> status
    - goal_achieved, rules_broken, criteria_results, turn_metrics -> metadata
    """
    now = int(time.time())

    input_items: list[dict[str, Any]] = []
    output_items: list[dict[str, Any]] = []

    for msg in result.messages:
        if msg.role in ("user", "system"):
            in_content: _ContentList = [
                InputTextContent(type="input_text", text=content_to_text(msg.content))
            ]
            message = Message(
                type="message",
                id=_generate_item_id("msg"),
                role=MessageRole(msg.role),
                status=MessageStatus.completed,
                content=in_content,
            )
            input_items.append(message.model_dump(mode="json"))
        elif msg.role == "assistant":
            # An assistant turn can carry text and/or tool_calls. Emit the text
            # message when there is content, then a function_call item per call
            # (separate Responses output items). A tool-only turn skips the empty
            # text message. Mirrors the langchain integration's mapping.
            if msg.content:
                out_content: _ContentList = [
                    OutputTextContent(text=content_to_text(msg.content), annotations=[])
                ]
                message = Message(
                    type="message",
                    id=_generate_item_id("msg"),
                    role=MessageRole.assistant,
                    status=MessageStatus.completed,
                    content=out_content,
                )
                output_items.append(message.model_dump(mode="json"))
            for tc in msg.tool_calls or []:
                function_call = FunctionCall(
                    type="function_call",
                    id=tc.item_id or _generate_item_id("fc"),
                    call_id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                    status=FunctionCallStatus.completed,
                )
                output_items.append(function_call.model_dump(mode="json"))
        elif msg.role == "tool":
            function_call_output = FunctionCallOutput(
                type="function_call_output",
                id=_generate_item_id("fco"),
                call_id=msg.tool_call_id or "",
                output=content_to_text(msg.content),
                status=FunctionCallStatus.completed,
            )
            output_items.append(function_call_output.model_dump(mode="json"))

    # Map terminated_by to status
    if result.terminated_by.value == "judge":
        status = "completed"
    elif result.terminated_by.value == "error":
        status = "failed"
    else:
        status = "incomplete"

    incomplete_details = (
        IncompleteDetails(
            reason=f"{result.terminated_by.value}: {result.reason}"
        ).model_dump(mode="json")
        if status == "incomplete"
        else None
    )

    # Build usage from token_usage
    usage_data = None
    if result.token_usage.total_tokens > 0:
        usage_data = Usage(
            input_tokens=result.token_usage.prompt_tokens,
            output_tokens=result.token_usage.completion_tokens,
            total_tokens=result.token_usage.total_tokens,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        ).model_dump(mode="json")

    metadata: dict[str, Any] = {
        "framework": "simulation",
        "goal_achieved": result.goal_achieved,
        "goal_completion_score": result.goal_completion_score,
        "terminated_by": result.terminated_by.value,
        "reason": result.reason,
        "turn_count": result.turn_count,
        "rules_broken": result.rules_broken,
    }
    if result.criteria_results:
        metadata["criteria_results"] = result.criteria_results
    if result.turn_metrics:
        metadata["turn_metrics"] = [
            tm.model_dump(mode="json") for tm in result.turn_metrics
        ]

    return {
        "id": _generate_item_id("resp"),
        "object": "response",
        "created_at": now,
        "completed_at": now if status == "completed" else None,
        "status": status,
        "incomplete_details": incomplete_details,
        "model": model,
        "previous_response_id": None,
        "instructions": None,
        "input": input_items,
        "output": output_items,
        "error": {"message": result.reason}
        if result.terminated_by.value == "error"
        else None,
        "tools": [],
        "tool_choice": "auto",
        "truncation": "disabled",
        "parallel_tool_calls": False,
        "text": {"format": {"type": "text"}},
        "top_p": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "top_logprobs": 0,
        "temperature": 1,
        "reasoning": None,
        "user": None,
        "usage": usage_data,
        "max_output_tokens": None,
        "max_tool_calls": None,
        "store": False,
        "background": False,
        "service_tier": "default",
        "metadata": metadata,
        "safety_identifier": None,
        "prompt_cache_key": None,
    }
