"""Shared sanitization utilities for prompt injection prevention.

Two complementary functions are provided:

* ``delimit(text, tag="data")`` — wraps user-controlled text in XML-like
  boundary tags (``<tag>…</tag>``) *and* escapes any occurrences of the
  chosen tag inside the text so the boundary cannot be broken.  Use this
  whenever you embed untrusted strings into an LLM prompt.

* ``xml_escape(text)`` — character-level XML escaping (``&``, ``<``, ``>``).
  Use this when you need to embed text inside XML tags that
  are already controlled by the caller (e.g. building an XML payload where
  the tag structure is fixed but the content is untrusted).
"""

from __future__ import annotations

import re

import xml.sax.saxutils


def delimit(text: str, *, tag: str = "data") -> str:
    """Wrap user-controlled text in delimiters to prevent prompt injection.

    Uses XML-like boundary tags to clearly separate user content from
    system instructions in LLM prompts.  Both the opening and closing
    tag inside the input are escaped (case-insensitive) to prevent
    breakout.

    Args:
        text: The untrusted text to wrap.
        tag: Tag name to use for the boundary (default ``"data"``).

    Returns:
        The text wrapped in ``<tag>…</tag>`` with internal occurrences
        of the tag escaped.
    """
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", tag):
        raise ValueError(f"Invalid tag name: {tag!r}")
    sanitized = text.replace("&", "&amp;")
    sanitized = re.sub(
        rf"<{re.escape(tag)}>", f"&lt;{tag}&gt;", sanitized, flags=re.IGNORECASE
    )
    sanitized = re.sub(
        rf"</{re.escape(tag)}>", f"&lt;/{tag}&gt;", sanitized, flags=re.IGNORECASE
    )
    return f"<{tag}>{sanitized}</{tag}>"


def xml_escape(text: str) -> str:
    """Escape text for safe embedding inside XML tags.

    Wraps :func:`xml.sax.saxutils.escape` for convenience and
    discoverability.
    """
    return xml.sax.saxutils.escape(text)
