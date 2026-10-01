"""Helpers shared by the format converters: text joining, ISO times, tool-call arguments, final metrics, media."""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from evaluatorq.common.messages import content_part_text

if TYPE_CHECKING:
    from collections.abc import Iterable

    from evaluatorq.formats.atif import AtifContentPart, AtifFinalMetrics, AtifStep

RAW_ARGUMENTS_EXTRA_KEY = 'evaluatorq.raw_arguments'
COMPACTION_EXTRA_KEY = 'evaluatorq.compaction'
_FRACTION = re.compile(r'(?<=\d{2}:\d{2}:\d{2})\.(\d+)')


def join_text(texts: Iterable[str]) -> str:
    """Join the text parts of one message with newlines, skipping empty parts (the one rule for every converter)."""
    return '\n'.join(text for text in texts if text)


def part_text(part: Any, where: str) -> str:
    """The text of a Responses text part, read by `content_part_text`; a part with no string text is warned, not repr-ed."""
    text = content_part_text(part)
    if text is None or (not text and not any(isinstance(part.get(key), str) for key in ('text', 'refusal', 'content'))):
        kind = part.get('type') if isinstance(part, dict) else type(part).__name__
        logger.warning('{} content part of type {!r} carries no text; dropping it.', where, kind)
        return ''
    return text


def compaction_extra(compactions: list[dict[str, Any]]) -> dict[str, Any]:
    """Step `extra` of a compaction system step: ATIF's `context_management` marker plus the raw compaction data."""
    return {'context_management': {'type': 'compaction', 'boundary': 'replace'}, COMPACTION_EXTRA_KEY: compactions}


def parse_iso(value: str) -> datetime:
    """Parse an ISO 8601 string on any supported Python: `Z` suffix, and fractions of any length (cut to 6 digits).

    Raises:
        ValueError: `value` is not an ISO 8601 date or time.
    """
    text = value.strip()
    if text[-1:] in ('Z', 'z'):
        text = text[:-1] + '+00:00'
    text = _FRACTION.sub(lambda match: '.' + match.group(1)[:6].ljust(6, '0'), text, count=1)
    return datetime.fromisoformat(text)


def atif_tool_arguments(raw: Any, name: object) -> tuple[dict[str, Any], str | None]:
    """Return ATIF arguments and separate raw provenance, avoiding collisions with user JSON objects."""
    if isinstance(raw, dict):
        return cast('dict[str, Any]', raw), None
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            return cast('dict[str, Any]', decoded), None
    logger.warning(
        'Tool call {!r} arguments are not a JSON object; preserving their original text in ATIF extra.', name
    )
    return {}, raw if isinstance(raw, str) else json.dumps(raw, default=str)


def json_arguments_text(arguments: dict[str, Any]) -> str:
    """Serialize an ATIF JSON argument object without interpreting any keys as raw-text sentinels."""
    try:
        return json.dumps(arguments, separators=(',', ':'), sort_keys=True)
    except (TypeError, ValueError) as exc:
        logger.warning('Tool call arguments are not JSON-encodable ({}); writing unencodable values as text.', exc)
        return json.dumps(arguments, separators=(',', ':'), sort_keys=True, default=str)


def final_metrics(steps: list[AtifStep]) -> AtifFinalMetrics | None:
    """Sum the steps' own metrics (never any span or response total), or None when no step is measured."""
    from evaluatorq.formats.atif import AtifFinalMetrics  # atif imports this module for parse_iso

    measured = [step.metrics for step in steps if step.metrics is not None]
    if not measured:
        return None

    def total(values: list[Any]) -> Any:
        present = [value for value in values if value is not None]
        return sum(present) if present else None

    return AtifFinalMetrics(
        total_prompt_tokens=total([m.prompt_tokens for m in measured]),
        total_completion_tokens=total([m.completion_tokens for m in measured]),
        total_cached_tokens=total([m.cached_tokens for m in measured]),
        total_cost_usd=total([m.cost_usd for m in measured]),
        total_steps=len(steps),
    )


def media_marker(part: AtifContentPart, target: str) -> str:
    """Render an ATIF image or audio part as `[type: path]` text, warning that `target` cannot hold it."""
    path = part.source.path if part.source is not None else ''
    logger.warning('ATIF {} part at {!r} has no {} equivalent; rendering a marker.', part.type, path, target)
    return f'[{part.type}: {path}]'


def atif_content_text(content: str | list[AtifContentPart], target: str) -> str:
    """Flatten ATIF content to text with `join_text`; image and audio parts become warned `[type: path]` markers."""
    if isinstance(content, str):
        return content
    return join_text((part.text or '') if part.type == 'text' else media_marker(part, target) for part in content)
