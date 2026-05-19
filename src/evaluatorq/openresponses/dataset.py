"""OpenResponses dataset helpers for red team samples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evaluatorq.redteam.contracts import RedTeamInput, RedTeamSample, StaticDataset, Turn


def user_input_item(content: str) -> dict[str, Any]:
    """Build an OpenResponses input entry for a user message."""
    return {"role": "user", "content": content}


def assistant_input_item(content: str) -> dict[str, Any]:
    """Build an OpenResponses input entry for an assistant message."""
    return {"role": "assistant", "content": content}


def system_input_item(content: str) -> dict[str, Any]:
    """Build an OpenResponses input entry for a system message."""
    return {"role": "system", "content": content}


def _get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def build_openresponses_request(
    *,
    model: str,
    prompt: str | None = None,
    conversation: list[dict[str, Any]] | None = None,
    instructions: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Package a prompt or conversation into the Responses API request shape."""
    input_items: list[dict[str, Any]] = list(conversation) if conversation else []
    if prompt is not None:
        input_items.append(user_input_item(prompt))
    if not input_items:
        raise ValueError(
            "build_openresponses_request: requires at least one of prompt or non-empty conversation"
        )

    payload: dict[str, Any] = {"model": model, "input": input_items}
    if instructions is not None:
        payload["instructions"] = instructions
    if extra:
        payload.update(extra)
    return payload


def _text_from_output_item(item: Any) -> str:
    item_type = _get_field(item, "type")
    if item_type in ("output_text", "input_text"):
        return str(_get_field(item, "text", "") or "")
    if item_type == "message":
        return "".join(_text_from_output_item(part) for part in _get_field(item, "content") or [])
    return ""


def _assistant_text(response: Any) -> str:
    text = _get_field(response, "text")
    if isinstance(text, str) and text:
        return text
    output = _get_field(response, "output")
    if output is not None:
        return "".join(_text_from_output_item(item) for item in output)
    return ""


def _assistant_message_item(item: Any) -> dict[str, Any] | None:
    text = _text_from_output_item(item)
    if text:
        return assistant_input_item(text)
    return None


def _tool_call_item(item: Any) -> dict[str, Any] | None:
    if _get_field(item, "type") != "function_call":
        return None

    name = _get_field(item, "name")
    arguments = _get_field(item, "arguments")
    call_id = _get_field(item, "call_id") or _get_field(item, "id")
    result = _get_field(item, "result")

    payload: dict[str, Any] = {
        "type": "function_call",
        "name": str(name or ""),
        "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments or {}),
    }
    if call_id:
        payload["call_id"] = str(call_id)
    if result is not None:
        payload["result"] = result
    return payload


def _assistant_items_from_output(response: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in _get_field(response, "output") or []:
        input_item = _tool_call_item(item) or _assistant_message_item(item)
        if input_item is not None:
            items.append(input_item)
    return items


def _assistant_tool_call_items(response: Any) -> list[dict[str, Any]]:
    tool_calls = [_tool_call_item(item) for item in _get_field(response, "tool_calls") or []]
    return [item for item in tool_calls if item is not None]


def append_assistant_turn(input_array: list[dict[str, Any]], response: Any) -> list[dict[str, Any]]:
    """Append assistant text and tool calls from a response to an OpenResponses input array."""
    output_items = _assistant_items_from_output(response)
    if output_items:
        input_array.extend(output_items)
        return input_array

    text = _assistant_text(response)
    if text:
        input_array.append(assistant_input_item(text))
    input_array.extend(_assistant_tool_call_items(response))
    return input_array


def append_user_followup(input_array: list[dict[str, Any]], prompt: str) -> list[dict[str, Any]]:
    """Append a follow-up user message to an OpenResponses input array."""
    input_array.append(user_input_item(prompt))
    return input_array


def turns_to_openresponses_input(
    turns: list[Turn],
    *,
    include_final_assistant: bool = True,
) -> list[dict[str, Any]]:
    """Convert redteam turns to an OpenResponses input array."""
    out: list[dict[str, Any]] = []
    for idx, turn in enumerate(turns):
        out.append(user_input_item(turn.attacker.generated_prompt))
        if idx == len(turns) - 1 and not include_final_assistant:
            continue
        append_assistant_turn(out, turn.target)
    return out


def orchestrator_result_to_openresponses_input(turns: list[Turn]) -> list[dict[str, Any]]:
    """Alias for callers that hold an orchestrator result turn list."""
    return turns_to_openresponses_input(turns)


def messages_from_openresponses_input(input_array: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenResponses input items to redteam chat message dicts."""
    out: list[dict[str, Any]] = []
    for item in input_array:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        if role in ("user", "assistant", "system"):
            content = item.get("content")
            if content is None:
                continue
            if isinstance(content, list):
                text = "".join(
                    str(block.get("text") or "") if isinstance(block, dict) else ""
                    for block in content
                )
            else:
                text = str(content)
            out.append({"role": role, "content": text})
            continue

        if item.get("type") == "message":
            text = "".join(_text_from_output_item(block) for block in item.get("content") or [])
            if text:
                out.append({"role": item.get("role", "assistant"), "content": text})
    return out


def load_openresponses_dataset(path: str | Path) -> StaticDataset:
    """Load a static redteam dataset authored in OpenResponses input format."""
    from evaluatorq.redteam.contracts import RedTeamInput, StaticDataset

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    if p.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSONL on line {line_no} of {p}: {e}") from e
    else:
        loaded = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "samples" in loaded:
            rows = loaded["samples"]
        elif isinstance(loaded, list):
            rows = loaded
        else:
            raise ValueError(
                f"Unexpected dataset shape in {p}: expected a top-level list "
                'or {"samples": [...]} object'
            )

    samples = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Dataset row {idx} is not an object: {row!r}")  # noqa: TRY004
        meta = row.get("input")
        if not isinstance(meta, dict):
            raise ValueError(f"Dataset row {idx} is missing the 'input' metadata object")  # noqa: TRY004
        conv = row.get("openresponses_input")
        if conv is None:
            conv = row.get("input_messages")
        if not isinstance(conv, list):
            raise ValueError(  # noqa: TRY004
                f"Dataset row {idx} is missing 'openresponses_input' (or legacy "
                "'input_messages') - expected a list of OpenResponses input items"
            )
        try:
            rt_input = RedTeamInput(**meta)
        except Exception as exc:
            raise ValueError(f"Dataset row {idx}: 'input' metadata failed validation: {exc}") from exc
        samples.append(redteam_sample_from_openresponses(input=rt_input, openresponses_input=conv))
    return StaticDataset(samples=samples)


def redteam_sample_from_openresponses(
    *,
    input: RedTeamInput,  # noqa: A002
    openresponses_input: list[dict[str, Any]],
) -> RedTeamSample:
    """Build a RedTeamSample from OpenResponses input items."""
    from evaluatorq.redteam.contracts import Message, RedTeamSample

    messages_data = messages_from_openresponses_input(openresponses_input)
    if not messages_data:
        raise ValueError(
            "redteam_sample_from_openresponses: openresponses_input produced no messages - "
            "the array must contain at least one role/content item or message block"
        )
    return RedTeamSample(input=input, messages=[Message(**m) for m in messages_data])


__all__ = [
    "append_assistant_turn",
    "append_user_followup",
    "assistant_input_item",
    "build_openresponses_request",
    "load_openresponses_dataset",
    "messages_from_openresponses_input",
    "orchestrator_result_to_openresponses_input",
    "redteam_sample_from_openresponses",
    "system_input_item",
    "turns_to_openresponses_input",
    "user_input_item",
]
