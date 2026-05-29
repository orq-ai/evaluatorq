"""Input sanitization utilities for prompt injection prevention."""

from __future__ import annotations

import re


def delimit(text: str) -> str:
    """Wrap user-controlled text in delimiters to prevent prompt injection.

    Uses XML-like data tags to clearly separate user content from
    system instructions in LLM prompts. The closing tag in the input
    is escaped to prevent breakout.
    """
    sanitized = text.replace("&", "&amp;")
    sanitized = re.sub(r"<data>", "&lt;data&gt;", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"</data>", "&lt;/data&gt;", sanitized, flags=re.IGNORECASE)
    return f"<data>{sanitized}</data>"
