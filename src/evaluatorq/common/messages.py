"""Shared helpers for normalizing chat-message content."""

from __future__ import annotations

from typing import Any


def coerce_content_text(content: Any) -> str:
    """Flatten message content to a plain text string.

    Multi-part content (e.g. tool/result messages shaped like
    ``[{"type": "text", "text": "..."}]``) surfaces the joined text rather than a
    Python ``repr`` of the list. ``None`` becomes ``""``; plain strings (and anything
    else) pass through ``str``.
    """
    if isinstance(content, list):
        texts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                # Both the chat-completions ("text") and Responses ("input_text")
                # shapes carry their text under a "text" key.
                if part.get("type") in ("text", "input_text"):
                    texts.append(part.get("text", ""))
            elif getattr(part, "type", None) == "input_text":
                # An InputTextContent pydantic content part.
                texts.append(getattr(part, "text", ""))
        return "\n".join(texts)
    return str(content or "")
