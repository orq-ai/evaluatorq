"""Helpers shared by the ATIF converters: tool-call arguments, final metrics and ATIF content flattening."""

from __future__ import annotations

import json
from typing import Any, cast

from loguru import logger

from evaluatorq.contracts import InputTextContent, content_to_text
from evaluatorq.formats.atif import AtifContentPart, AtifFinalMetrics, AtifStep

RAW_ARGUMENTS_KEY = '_raw'


def tool_arguments(raw: Any, name: object) -> dict[str, Any]:
    """Read tool-call arguments as an ATIF dict; anything but a JSON object is kept as `{"_raw": text}` (warned)."""
    if isinstance(raw, dict):
        return cast('dict[str, Any]', raw)
    decoded: Any = None
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            decoded = None
    if isinstance(decoded, dict):
        return cast('dict[str, Any]', decoded)
    logger.warning('Tool call {!r} arguments are not a JSON object; keeping them as {{"_raw": ...}}.', name)
    return {RAW_ARGUMENTS_KEY: raw if isinstance(raw, str) else json.dumps(raw, default=str)}


def arguments_text(arguments: dict[str, Any]) -> str:
    """Render ATIF arguments as a JSON string; `{"_raw": s}` is written back as `s`."""
    raw = arguments.get(RAW_ARGUMENTS_KEY)
    if len(arguments) == 1 and isinstance(raw, str):
        return raw
    return json.dumps(arguments, separators=(',', ':'), sort_keys=True)


def final_metrics(steps: list[AtifStep]) -> AtifFinalMetrics | None:
    """Sum the steps' own metrics (never any span or response total), or None when no step is measured."""
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
    """Flatten ATIF content to text; image and audio parts become warned `[type: path]` markers."""
    if isinstance(content, str):
        return content
    texts = [part.text or '' if part.type == 'text' else media_marker(part, target) for part in content]
    return content_to_text([InputTextContent(type='input_text', text=text) for text in texts])
