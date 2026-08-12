"""JSON extraction utilities for parsing LLM responses."""

from __future__ import annotations

import json
import re

_JSON_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', re.IGNORECASE)


def coerce_str_list(value: object) -> object:
    """``BeforeValidator`` for ``list[str]`` fields parsed from fallback JSON.

    The ``json_object`` fallback runs on exactly the models most likely to emit
    a stray ``1`` or ``null`` in a string array, and a ``ValidationError`` there
    drops the whole report section. Stringify items and drop falsy ones instead,
    matching the pre-RES-822 tolerance; non-list values pass through for
    pydantic to reject normally.
    """
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return value


def coerce_str(value: object) -> object:
    """``BeforeValidator`` for ``str`` fields parsed from fallback JSON:
    stringify scalars (``None`` becomes ``''``) instead of failing validation."""
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

    # Try to extract from code block using regex
    match = _JSON_BLOCK_PATTERN.search(content)
    if match and match.group(1):
        return match.group(1).strip()

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
