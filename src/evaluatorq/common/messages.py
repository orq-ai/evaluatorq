"""Shared helpers for normalizing chat-message content."""

from __future__ import annotations

from typing import Any


def coerce_content_text(content: Any) -> str:
    """Flatten message content to a plain text string.

    Multi-part content (e.g. tool/result messages shaped like
    ``[{"type": "text", "text": "..."}]``) surfaces the joined text rather than a
    Python ``repr`` of the list. ``None`` becomes ``""``; plain strings (and anything
    else) pass through ``str``.

    Unlike :func:`evaluatorq.contracts.content_to_text`, this best-effort helper
    does not raise on non-text parts (it is used in report/transcript rendering).
    Image and file parts are surfaced as a ``[image]`` / ``[file]`` placeholder so
    they are visibly accounted for rather than silently dropped.
    """
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            # Both the chat-completions ("text") and Responses ("input_text")
            # shapes carry their text under a "text" key.
            if part_type in ("text", "input_text"):
                texts.append(part.get("text", "") if isinstance(part, dict) else getattr(part, "text", ""))
            elif part_type in ("image_url", "input_image"):
                texts.append("[image]")
            elif part_type in ("file", "input_file"):
                texts.append("[file]")
        return "\n".join(texts)
    return str(content or "")
