"""Backend-internal exception extraction helpers.

Module-private. Used by ``ORQBackend.map_error`` and ``OpenAIBackend.map_error``.
"""

from __future__ import annotations

import re


def extract_status_code(exc: Exception) -> int | None:
    """Extract HTTP-like status code from structured exception fields or text."""
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int) and 100 <= status_code <= 599:
        return status_code

    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int) and 100 <= value <= 599:
            return value

    text = str(exc)
    patterns = [
        r"\bstatus(?:_code)?\s*[=:]\s*(\d{3})\b",
        r"\bHTTP\s*(\d{3})\b",
        r"\bcode\s*[=:]\s*(\d{3})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        code = int(match.group(1))
        if 100 <= code <= 599:
            return code
    return None


def extract_provider_error_code(exc: Exception) -> str | None:
    """Extract provider-specific symbolic error code if present."""
    for attr in ("code", "error_code", "type"):
        value = getattr(exc, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error") if isinstance(body.get("error"), dict) else body
        for key in ("code", "type", "error_code"):
            value = error.get(key) if isinstance(error, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip().lower()

    # Text-based fallback only — patterns may match Python type annotations or
    # generic words in non-provider exceptions (e.g. "TypeError: type=<class 'str'>"),
    # yielding misleading `orq.code.<name>` codes. Structured attribute checks above
    # cover all production SDK errors; this is a best-effort last resort.
    text = str(exc)
    patterns = [
        r'\b(?:error_)?code\s*[=:]\s*["\']?([a-z0-9_.-]+)["\']?',
        r'\btype\s*[=:]\s*["\']?([a-z0-9_.-]+)["\']?',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().lower()
    return None
