"""Shared OWASP evaluator prompt renderer.

Single source of truth for substituting the three standard placeholders
used by both the dynamic (adaptive.evaluator) and static (evaluatorq_bridge)
scoring paths.

  {{input.all_messages}}   — sanitized JSON of the conversation messages
  {{output.tool_calls}}    — sanitized JSON of tool-call records
  {{output.response}}      — the agent's text response

All adversary-controlled substitutions are neutralized with
:func:`_sanitize_placeholders` before embedding to prevent cross-expansion
attacks (a crafted tool-call name containing ``{{output.response}}`` must not
cause the subsequent replace to expand that placeholder a second time).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from evaluatorq.common.messages import coerce_content_text

if TYPE_CHECKING:
    from evaluatorq.redteam.contracts import ToolCallOutputItem


def render_owasp_evaluator_prompt(
    prompt: str,
    *,
    messages: list[dict[str, Any]] | list[Any],
    response: str,
    tool_calls: list[ToolCallOutputItem] | None,
) -> str:
    """Substitute ``{{input.all_messages}}``, ``{{output.tool_calls}}``, and
    ``{{output.response}}`` in *prompt* with injection-safe content.

    Trusted internal data (``{{input.all_messages}}``) is substituted first so
    that a crafted response or tool-call field containing literal placeholder
    strings cannot be expanded by a later ``.replace()`` call.  Adversary-
    controlled values are sanitized with :func:`_sanitize_placeholders` before
    embedding.

    Args:
        prompt:     The evaluator prompt template string.
        messages:   Conversation messages — either plain ``dict`` objects
                    (``{"role": ..., "content": ...}``) or ``Message`` model
                    instances.  Only ``role`` and ``content`` are forwarded.
        response:   The agent's text response (adversary-influenced).
        tool_calls: Structured tool call records, or ``None`` / empty list.
                    Each item must expose ``.name``, ``.arguments_dict``,
                    ``.result``, and ``.id`` attributes
                    (:class:`~evaluatorq.redteam.contracts.ToolCallOutputItem`).

    Returns:
        The rendered prompt string with all three placeholders replaced.
        If a placeholder is absent from the template the replace is a no-op;
        the prompt is returned unchanged for that placeholder.
    """
    # 1. {{input.all_messages}} — serialize + sanitize first so later
    #    adversary-controlled substitutions cannot re-expand this placeholder.
    prompt = prompt.replace(
        "{{input.all_messages}}",
        _sanitize_placeholders(
            json.dumps(_serialize_messages(messages), indent=2)
        ),
    )

    # 2. {{output.tool_calls}} — adversary-influenced; sanitize before embed.
    tool_calls_payload = [
        {
            "name": tc.name,
            "arguments": tc.arguments_dict,
            "result": tc.result,
            "id": tc.id,
        }
        for tc in (tool_calls or [])
    ]
    prompt = prompt.replace(
        "{{output.tool_calls}}",
        _sanitize_placeholders(
            json.dumps(tool_calls_payload, indent=2, default=str)
        ),
    )

    # 3. {{output.response}} — adversary-influenced; no extra sanitization
    #    needed because it is the *last* substitution (nothing to re-expand).
    return prompt.replace("{{output.response}}", response or "")


def _serialize_messages(messages: list[dict[str, Any]] | list[Any]) -> list[dict[str, Any]]:
    """Normalize messages to plain role/content dicts for prompt interpolation."""
    serialized: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, dict):
            serialized.append({"role": str(msg.get("role", "")), "content": coerce_content_text(msg.get("content"))})
            continue
        serialized.append({"role": str(msg.role), "content": coerce_content_text(msg.content)})
    return serialized


def _sanitize_placeholders(text: str) -> str:
    """Neutralize template placeholder markers in adversary-controlled content.

    Replaces ``{{`` with ``{ {`` so that crafted tool call names or argument
    values containing placeholder strings (e.g. ``{{output.response}}``) cannot
    be expanded by a subsequent ``.replace()`` call in the evaluator prompt
    pipeline.
    """
    return text.replace("{{", "{ {")
