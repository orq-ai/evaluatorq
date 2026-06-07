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
        return "\n".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return str(content or "")
