"""Backend-internal exception extraction helpers.

Module-private. Used by ``ORQBackend.map_error`` and ``OpenAIBackend.map_error``.

Status extraction deliberately lives in ``common.target_call`` and is re-exported
here: the retry boundary and ``map_error`` must classify the same exception the
same way, or a status the report shows as a 4xx gets retried anyway.
"""

from __future__ import annotations

import re

from evaluatorq.common.llm_errors import error_body
from evaluatorq.common.target_call import extract_status_code

__all__ = ['extract_provider_error_code', 'extract_status_code']


def extract_provider_error_code(exc: Exception) -> str | None:
    """Extract provider-specific symbolic error code if present."""
    for attr in ('code', 'error_code', 'type'):
        value = getattr(exc, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()

    error = error_body(exc)
    if error is not None:
        for key in ('code', 'type', 'error_code'):
            value = error.get(key)
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
