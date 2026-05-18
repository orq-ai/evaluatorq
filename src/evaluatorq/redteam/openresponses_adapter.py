"""OpenResponses format adaptation for the red teaming pipeline (RES-540).

Provides the format-translation layer between the red teaming pipeline's
internal canonical state (``OrchestratorResult`` / ``Turn`` / ``AgentResponse``
— OpenResponses-shaped output items, OpenAI chat-completions wire format)
and the OpenResponses request/response wire format that platform agents
and deployments speak natively.

Touchpoints covered (per RES-540):

1. **Input format adaptation** — :func:`build_openresponses_request` packages
   a model id + an attack prompt (and any prior turns) into the OpenResponses
   request shape ``{"model": "...", "input": [...]}``.

2. **Multi-turn conversation support** — :func:`append_assistant_turn` and
   :func:`append_user_followup` grow the ``input`` array turn-by-turn so the
   adversarial agent can maintain conversation state in OpenResponses
   format. :func:`turns_to_openresponses_input` reconstructs the array from
   the orchestrator's canonical ``Turn`` record.

3. **Response parsing** — re-exports :func:`extract_assistant_text` and
   :func:`extract_tool_calls` from :mod:`evaluatorq.redteam.parsing` so
   judges / safety classifiers consume the OpenResponses output uniformly.

4. **Trace integration** — :func:`record_openresponses_request` and
   :func:`record_openresponses_response` set ``gen_ai.*`` attributes on an
   existing LLM span using the OpenResponses shape (an ``input`` array
   instead of OpenAI ``messages``; an ``output`` array of typed items).

This module is purely additive — the v1 ``send_prompt(str)`` path is
untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.contracts import (
    AgentResponse,
    TextOutputItem,
    ToolCallOutputItem,
)
from evaluatorq.redteam.parsing import (
    extract_assistant_text,
    extract_reasoning,
    extract_tool_calls,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from evaluatorq.redteam.contracts import RedTeamInput, RedTeamSample, StaticDataset, Turn


# ---------------------------------------------------------------------------
# Input item helpers (OpenResponses wire shape)
# ---------------------------------------------------------------------------


def user_input_item(content: str) -> dict[str, Any]:
    """Build an OpenResponses ``input`` array entry for a user message."""
    return {"role": "user", "content": content}


def assistant_input_item(content: str) -> dict[str, Any]:
    """Build an OpenResponses ``input`` array entry for an assistant reply.

    Multi-turn red teaming needs the prior assistant response in the input
    array so the target agent (or adversarial replay) sees the full
    conversation. We use the simple ``{"role": "assistant", "content": "..."}``
    form which the OpenResponses backend treats as a prior message.
    """
    return {"role": "assistant", "content": content}


def system_input_item(content: str) -> dict[str, Any]:
    """Build an OpenResponses ``input`` array entry for a system / instruction message."""
    return {"role": "system", "content": content}


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------


def build_openresponses_request(
    *,
    model: str,
    prompt: str | None = None,
    conversation: list[dict[str, Any]] | None = None,
    instructions: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Package an attack prompt into the OpenResponses request shape.

    ``{"model": "<agent-id>", "input": [{"role": "user", "content": "..."}], ...}``

    Args:
        model: Agent id / model identifier sent as ``model``.
        prompt: New user prompt. Appended to ``conversation`` if both are set,
            or used as the sole user input when ``conversation`` is ``None``.
            May be ``None`` if ``conversation`` already contains the desired
            input.
        conversation: Prior turns as an OpenResponses input array. Items may
            be ``user``/``assistant`` simple dicts (use the helpers above) or
            full OpenResponses items (``message``, ``function_call``,
            ``function_call_output``, etc.) — passed through verbatim.
        instructions: Optional system instructions field. Mapped to the
            top-level ``instructions`` key per OpenResponses spec.
        extra: Additional top-level fields merged into the request
            (``tools``, ``tool_choice``, ``temperature``, ``store``, etc.).

    Returns:
        A JSON-serializable dict in OpenResponses request shape.
    """
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


# ---------------------------------------------------------------------------
# Multi-turn state
# ---------------------------------------------------------------------------


def append_assistant_turn(
    input_array: list[dict[str, Any]],
    response: Any,
) -> list[dict[str, Any]]:
    """Append an assistant turn (and any tool calls) to an OpenResponses input array.

    ``response`` may be an :class:`AgentResponse`, an OpenResponses
    ``ResponseResource`` dict, or any value :func:`extract_assistant_text`
    accepts. The visible assistant text is appended as a single
    ``{"role": "assistant", "content": ...}`` entry; function calls are
    appended as ``{"type": "function_call", ...}`` items so a follow-up
    turn round-trips the full conversation.

    Returns the same list (mutated in place) for chaining.
    """
    text = extract_assistant_text(response)
    if text:
        input_array.append(assistant_input_item(text))
    for call in extract_tool_calls(response):
        item: dict[str, Any] = {
            "type": "function_call",
            "name": call["name"],
            "arguments": call["arguments"],
        }
        call_id = call.get("call_id")
        if call_id:
            item["call_id"] = call_id
        input_array.append(item)
    return input_array


def append_user_followup(
    input_array: list[dict[str, Any]],
    prompt: str,
) -> list[dict[str, Any]]:
    """Append a follow-up user attack to an OpenResponses input array."""
    input_array.append(user_input_item(prompt))
    return input_array


def turns_to_openresponses_input(
    turns: list[Turn],
    *,
    include_final_assistant: bool = True,
) -> list[dict[str, Any]]:
    """Convert the orchestrator's canonical ``turns`` list to an OpenResponses input array.

    Each :class:`Turn` becomes a pair of items:
    - ``{"role": "user", "content": <attacker prompt>}``
    - assistant text + function calls (via :func:`append_assistant_turn`)

    Args:
        turns: Per-turn record from :attr:`OrchestratorResult.turns`.
        include_final_assistant: When ``False``, omits the assistant items
            from the last turn — useful when the caller is about to issue
            the next attacker prompt and wants the array to end on the
            last assistant request that needs answering.
    """
    out: list[dict[str, Any]] = []
    for idx, turn in enumerate(turns):
        out.append(user_input_item(turn.attacker.generated_prompt))
        is_last = idx == len(turns) - 1
        if is_last and not include_final_assistant:
            continue
        append_assistant_turn(out, turn.target)
    return out


# ---------------------------------------------------------------------------
# Response normalization
# ---------------------------------------------------------------------------


def agent_response_from_openresponses(response: dict[str, Any]) -> AgentResponse:
    """Convert a raw OpenResponses ``ResponseResource`` dict to an :class:`AgentResponse`.

    Walks ``response["output"]``, lifting ``output_text`` content blocks from
    ``message`` items into :class:`TextOutputItem` and ``function_call``
    items into :class:`ToolCallOutputItem`. Reasoning items and other shapes
    are skipped (use :func:`extract_reasoning` if needed).

    Reads ``model``, ``id`` (→ ``response_id``), and ``status`` (→
    ``finish_reason``) from the resource when present.
    """
    items: list[TextOutputItem | ToolCallOutputItem] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            text = "".join(
                block.get("text", "")
                for block in (item.get("content") or [])
                if isinstance(block, dict) and block.get("type") == "output_text"
            )
            if text:
                items.append(TextOutputItem(text=text, annotations=[]))
        elif item_type == "function_call":
            items.append(ToolCallOutputItem(
                name=str(item.get("name", "")),
                arguments=str(item.get("arguments", "")),
                call_id=str(item.get("call_id") or item.get("id") or ""),
            ))
    return AgentResponse(
        output=list(items),
        model=response.get("model"),
        response_id=response.get("id"),
        finish_reason=response.get("status"),
    )


# ---------------------------------------------------------------------------
# Reverse converters (AgentResponse / Turn → OpenResponses wire format)
# ---------------------------------------------------------------------------


def agent_response_to_openresponses(response: AgentResponse) -> dict[str, Any]:
    """Convert an :class:`AgentResponse` to a ``ResponseResource``-shaped dict.

    Symmetric with :func:`agent_response_from_openresponses`. Useful for
    serializing executed attack turns back to OpenResponses wire format for
    trace replay, dataset capture, or cross-system handoff.

    The returned dict carries:
    - ``object``: always ``"response"``
    - ``id`` / ``model``: from the corresponding ``AgentResponse`` fields
        (omitted when ``None``)
    - ``status``: from ``finish_reason`` (omitted when ``None``)
    - ``output``: list of ``message`` and ``function_call`` items aggregated
        from the response's output. Consecutive text items collapse into a
        single ``message`` (matching how the OpenResponses API returns
        them).
    - ``usage``: ``UsageDict``-shaped block when ``response.usage`` is set.
    """
    output: list[dict[str, Any]] = []
    text_buffer: list[str] = []

    def _flush_text() -> None:
        if text_buffer:
            output.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "".join(text_buffer)}],
            })
            text_buffer.clear()

    for item in response.output:
        if isinstance(item, TextOutputItem):
            text_buffer.append(item.text)
        elif isinstance(item, ToolCallOutputItem):
            _flush_text()
            call_item: dict[str, Any] = {
                "type": "function_call",
                "name": item.name,
                "arguments": item.arguments,
                "call_id": item.call_id,
            }
            output.append(call_item)
        # ReasoningOutputItem is intentionally dropped — the OpenResponses
        # API only emits reasoning items from reasoning models. Round-tripping
        # would require a stable id we don't necessarily have.
    _flush_text()

    payload: dict[str, Any] = {"object": "response", "output": output}
    if response.response_id is not None:
        payload["id"] = response.response_id
    if response.model is not None:
        payload["model"] = response.model
    if response.finish_reason is not None:
        payload["status"] = response.finish_reason

    usage = response.usage
    if usage is not None:
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total = int(getattr(usage, "total_tokens", 0) or 0) or (input_tokens + output_tokens)
        payload["usage"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total,
        }
    return payload


def orchestrator_result_to_openresponses_input(
    turns: list[Turn],
) -> list[dict[str, Any]]:
    """Alias for :func:`turns_to_openresponses_input` for symmetric naming.

    Reads more naturally at call sites that already have an
    ``OrchestratorResult`` in hand.
    """
    return turns_to_openresponses_input(turns)


# ---------------------------------------------------------------------------
# Dataset adapters
# ---------------------------------------------------------------------------


def messages_from_openresponses_input(
    input_array: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert an OpenResponses ``input`` array into chat-completions message dicts.

    Used when authoring static red teaming datasets in OpenResponses format —
    the existing :class:`RedTeamSample` schema stores attack conversations as a
    ``list[Message]`` in OpenAI chat-completions format. For ``role``/``content``
    items the two formats are equivalent; tool / function-call items are
    flattened into role-keyed messages so they fit the schema.

    Returns a list of dicts that can be passed directly to ``Message(**dict)``.
    """
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
                # OpenResponses sometimes stores content as a list of text blocks.
                text = "".join(
                    block.get("text", "") if isinstance(block, dict) else ""
                    for block in content
                )
                out.append({"role": role, "content": text})
            else:
                out.append({"role": role, "content": str(content)})
            continue
        item_type = item.get("type")
        if item_type == "message":
            msg_role = item.get("role", "assistant")
            text_parts: list[str] = []
            for block in item.get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in ("output_text", "input_text"):
                    text = block.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            if text_parts:
                out.append({"role": msg_role, "content": "".join(text_parts)})
            continue
        # function_call / function_call_output items are dropped — the chat
        # completions Message schema requires tool_call_id linking which we
        # cannot reliably synthesize without the assistant call that produced
        # them. Callers needing tool-call fidelity should use AgentResponse /
        # OrchestratorResult.turns directly.
    return out


def load_openresponses_dataset(
    path: str | Path,
) -> StaticDataset:
    """Load a static red teaming dataset stored in OpenResponses format.

    Accepts a JSON file containing either a top-level ``{"samples": [...]}``
    object or a bare list. Each sample row must carry:

    - ``input``: an object compatible with :class:`RedTeamInput` (carries
      attack metadata — vulnerability, severity, source, etc.).
    - ``openresponses_input`` (or legacy ``input_messages``): the attack
      conversation as an OpenResponses input array.

    Returns a :class:`StaticDataset` whose ``samples`` carry the converted
    :class:`RedTeamSample` instances.

    Also accepts JSONL (one object per line) when the file extension is
    ``.jsonl``.

    Raises ``FileNotFoundError`` for missing files and ``ValueError`` for
    malformed rows.
    """
    from evaluatorq.redteam.contracts import RedTeamInput, StaticDataset

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    rows: list[dict[str, Any]]
    if p.suffix.lower() == ".jsonl":
        rows = []
        # Explicit utf-8 — on Windows / non-UTF-8 locales the platform default
        # can corrupt or fail on dataset files containing non-ASCII content.
        for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Malformed JSONL on line {line_no} of {p}: {e}"
                ) from e
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
            raise ValueError(  # noqa: TRY004
                f"Dataset row {idx} is missing the 'input' metadata object"
            )
        conv = row.get("openresponses_input")
        if conv is None:
            conv = row.get("input_messages")
        if not isinstance(conv, list):
            raise ValueError(  # noqa: TRY004
                f"Dataset row {idx} is missing 'openresponses_input' (or legacy "
                "'input_messages') — expected a list of OpenResponses input items"
            )
        samples.append(redteam_sample_from_openresponses(
            input=RedTeamInput(**meta),
            openresponses_input=conv,
        ))
    return StaticDataset(samples=samples)


def redteam_sample_from_openresponses(
    *,
    input: RedTeamInput,  # noqa: A002 — public kwarg name, matches RedTeamSample.input
    openresponses_input: list[dict[str, Any]],
) -> RedTeamSample:
    """Build a :class:`RedTeamSample` from an OpenResponses input array.

    Lets dataset authors store the attack conversation in OpenResponses wire
    format and convert to the existing schema at load time. The ``input``
    parameter carries the attack metadata (vulnerability, severity, source);
    ``openresponses_input`` is the conversation in OpenResponses format.

    Raises ``ValueError`` if no messages could be extracted from the input
    array (``RedTeamSample.messages`` requires at least one).
    """
    from evaluatorq.redteam.contracts import Message, RedTeamSample
    messages_data = messages_from_openresponses_input(openresponses_input)
    if not messages_data:
        raise ValueError(
            "redteam_sample_from_openresponses: openresponses_input produced no messages — "
            "the array must contain at least one role/content item or message block"
        )
    return RedTeamSample(input=input, messages=[Message(**m) for m in messages_data])


# ---------------------------------------------------------------------------
# Tracing helpers
# ---------------------------------------------------------------------------


def record_openresponses_request(
    span: Span | None,
    payload: dict[str, Any],
) -> None:
    """Record an OpenResponses request payload on an existing LLM span.

    Sets ``gen_ai.input.messages`` to the JSON-serialized ``input`` array so
    the platform's GenericAdapter and OTel-aware viewers display the same
    conversation shape they show for chat-completions spans, while keeping
    a copy of the full payload under ``orq.openresponses.request`` for
    debugging.
    """
    if span is None:
        return
    from evaluatorq.redteam.tracing import truncate_for_span

    input_items = payload.get("input") or []
    span.set_attribute(
        "gen_ai.input.messages",
        truncate_for_span(json.dumps(input_items, ensure_ascii=False, default=str)),
    )
    span.set_attribute(
        "orq.openresponses.request",
        truncate_for_span(json.dumps(payload, ensure_ascii=False, default=str)),
    )
    model = payload.get("model")
    if model:
        span.set_attribute("gen_ai.request.model", str(model))


def record_openresponses_response(
    span: Span | None,
    response: Any,
) -> None:
    """Record an OpenResponses response on an existing LLM span.

    Accepts a ``ResponseResource`` dict, an :class:`AgentResponse`, or any
    value :func:`extract_assistant_text` understands. Sets:

    - ``gen_ai.output.messages`` — JSON array with a single ``assistant``
      role row carrying the extracted text (matches the chat-completions
      convention so platform viewers render uniformly).
    - ``orq.openresponses.response`` — full JSON of the response when it is
      a dict, otherwise the model-dumped form for ``AgentResponse``.
    - ``gen_ai.response.id`` / ``gen_ai.response.model`` when present.
    - Token-usage attributes when ``response["usage"]`` is a dict in
      OpenResponses ``UsageDict`` shape (``input_tokens``, ``output_tokens``,
      ``total_tokens``).
    """
    if span is None:
        return
    from evaluatorq.redteam.tracing import truncate_for_span

    text = extract_assistant_text(response)
    output_messages = [{"role": "assistant", "content": truncate_for_span(text)}]
    span.set_attribute(
        "gen_ai.output.messages",
        json.dumps(output_messages, ensure_ascii=False),
    )

    if isinstance(response, dict):
        span.set_attribute(
            "orq.openresponses.response",
            truncate_for_span(json.dumps(response, ensure_ascii=False, default=str)),
        )
        resp_id = response.get("id")
        if resp_id:
            span.set_attribute("gen_ai.response.id", str(resp_id))
        model = response.get("model")
        if model:
            span.set_attribute("gen_ai.response.model", str(model))
        usage = response.get("usage")
        if isinstance(usage, dict):
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            total = int(usage.get("total_tokens", 0) or 0) or (input_tokens + output_tokens)
            _record_usage_attrs(span, input_tokens, output_tokens, total)
    elif isinstance(response, AgentResponse):
        try:
            dumped = response.model_dump(mode="json")
        except Exception as exc:  # tracing must not raise; log + fall back
            logger.debug(
                "record_openresponses_response: model_dump failed ({}); "
                "falling back to str repr of output items",
                exc,
            )
            dumped = {"output": [str(item) for item in response.output]}
        span.set_attribute(
            "orq.openresponses.response",
            truncate_for_span(json.dumps(dumped, ensure_ascii=False, default=str)),
        )
        if response.response_id:
            span.set_attribute("gen_ai.response.id", response.response_id)
        if response.model:
            span.set_attribute("gen_ai.response.model", response.model)
        agent_usage = response.usage
        if agent_usage is not None:
            input_tokens = int(getattr(agent_usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(agent_usage, "completion_tokens", 0) or 0)
            total = int(getattr(agent_usage, "total_tokens", 0) or 0) or (input_tokens + output_tokens)
            _record_usage_attrs(span, input_tokens, output_tokens, total)


def _record_usage_attrs(
    span: Span, input_tokens: int, output_tokens: int, total: int,
) -> None:
    """Mirror token usage to both ``gen_ai.usage.*`` and bare keys.

    The bare keys (``input_tokens``, ``output_tokens``, ``total_tokens``)
    are read by the platform GenericAdapter for span extraction; the
    ``gen_ai.usage.*`` keys follow OTel GenAI semantic conventions. Setting
    both keeps OpenResponses spans consistent with the chat-completions
    spans emitted by :mod:`evaluatorq.redteam.tracing`.
    """
    span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
    span.set_attribute("gen_ai.usage.total_tokens", total)
    span.set_attribute("input_tokens", input_tokens)
    span.set_attribute("output_tokens", output_tokens)
    span.set_attribute("total_tokens", total)


__all__ = [
    "agent_response_from_openresponses",
    "agent_response_to_openresponses",
    "append_assistant_turn",
    "append_user_followup",
    "assistant_input_item",
    "build_openresponses_request",
    "extract_assistant_text",
    "extract_reasoning",
    "extract_tool_calls",
    "load_openresponses_dataset",
    "messages_from_openresponses_input",
    "orchestrator_result_to_openresponses_input",
    "record_openresponses_request",
    "record_openresponses_response",
    "redteam_sample_from_openresponses",
    "system_input_item",
    "turns_to_openresponses_input",
    "user_input_item",
]
