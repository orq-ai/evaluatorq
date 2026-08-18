"""JSON extraction utilities for parsing LLM responses.

Canonical fence-tolerant parser for model output — do not write another
code-fence regex. ``common.structured_output`` hands its ``json_object``
fallback content here.
"""

from __future__ import annotations

import json
import re

_JSON_BLOCK_PATTERN = re.compile(
    r'```(?P<language>[^\r\n`]*)\r?\n?(?P<body>[\s\S]*?)\n?```',
    re.IGNORECASE,
)


def coerce_str_list(value: object) -> object:
    """``BeforeValidator`` for ``list[str]`` fields; tolerates messy fallback JSON.

    Attached to the model field, so it runs on BOTH validation paths: it is a
    no-op on well-formed structured (``parse()``) output, and it matters on the
    ``json_object`` fallback, which runs on exactly the models most likely to
    emit a stray ``1`` or ``null`` in a string array — a ``ValidationError``
    there drops the whole report section. Stringify items and drop falsy ones
    instead, matching the pre-RES-822 tolerance; non-list values pass through
    for pydantic to reject normally.
    """
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return value


def coerce_str(value: object) -> object:
    """``BeforeValidator`` for ``str`` fields; runs on both the ``parse()`` and
    the fallback path (no-op on clean output). Stringifies scalars (``None``
    becomes ``''``) instead of failing validation on messy fallback JSON."""
    if value is None:
        return ''
    if isinstance(value, (int, float, bool)):
        return str(value)
    return value


def extract_json_from_response(content: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks.

    Robust extraction that handles:
    - ``json ... `` blocks
    - `` ... `` blocks (no language specifier)
    - Plain JSON arrays or objects (no code block)
    - Multiple code blocks (returns first one)
    """
    if not content:
        return ''

    # Try every fenced block, preferring a tagged JSON block. Providers can put
    # an illustrative Python block before their actual JSON response.
    matches = list(_JSON_BLOCK_PATTERN.finditer(content))
    ordered_matches = sorted(matches, key=lambda match: match.group('language').strip().lower() != 'json')
    for match in ordered_matches:
        candidate = match.group('body').strip()
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            continue

    # No code block found — extract the outermost JSON structure. Whichever
    # opener appears FIRST wins: trying arrays unconditionally before objects
    # would pull the inner array out of '{"recommendations": [...]}' and hand
    # the caller a list where it validates an object.
    first_obj = content.find('{')
    first_arr = content.find('[')
    object_starts_first = first_obj != -1 and (first_arr == -1 or first_obj < first_arr)
    order = ('{}', '[]') if object_starts_first else ('[]', '{}')
    for open_ch, close_ch in order:
        extracted = _extract_balanced(content, open_ch, close_ch)
        if extracted is not None:
            return extracted

    # Fallback: return trimmed content as-is
    return content.strip()


def _extract_balanced_from(content: str, open_ch: str, close_ch: str, start_idx: int) -> str | None:
    """Find the outermost balanced pair starting at start_idx.

    Respects JSON string literals to avoid counting brackets inside strings.
    """
    depth = 0
    in_string = False
    escaped = False

    for i in range(start_idx, len(content)):
        ch = content[i]

        if escaped:
            escaped = False
            continue

        if ch == '\\' and in_string:
            escaped = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return content[start_idx : i + 1]

    return None


def _extract_balanced(content: str, open_ch: str, close_ch: str) -> str | None:
    """Find the outermost balanced pair of characters.

    Tries each candidate occurrence and returns the first that parses as valid JSON.
    Falls back to the first balanced extraction if none parse.
    """
    first_match: str | None = None
    search_from = 0

    while search_from < len(content):
        idx = content.find(open_ch, search_from)
        if idx == -1:
            break

        candidate = _extract_balanced_from(content, open_ch, close_ch, idx)
        if candidate is None:
            search_from = idx + 1
            continue

        if first_match is None:
            first_match = candidate

        try:
            json.loads(candidate)
            return candidate  # Valid JSON — use it
        except (json.JSONDecodeError, ValueError):
            pass

        search_from = idx + 1

    return first_match
