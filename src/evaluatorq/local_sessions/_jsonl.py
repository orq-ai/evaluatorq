"""Bounded, malformed-tolerant JSONL reading. `OSError` propagates; callers wrap it."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.formats._shared import parse_iso
from evaluatorq.local_sessions.models import SessionLoadError

if TYPE_CHECKING:
    from pathlib import Path


def _decode(lines: list[bytes]) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    skipped = 0
    for raw in lines:
        line = raw.decode('utf-8', errors='replace').strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            skipped += 1
            continue
        if isinstance(record, dict):
            records.append(record)
        else:
            skipped += 1
    return records, skipped


def read_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    """All dict records plus the number of malformed lines; an unterminated bad last line is an in-progress write."""
    lines = path.read_bytes().split(b'\n')
    # The final element is whatever follows the last newline: empty for a complete file.
    tail = lines.pop()
    records, skipped = _decode(lines)
    if tail.strip():
        try:
            last = json.loads(tail.decode('utf-8', errors='replace'))
        except ValueError:
            return records, skipped
        if isinstance(last, dict):
            records.append(last)
        else:
            skipped += 1
    return records, skipped


def read_head(path: Path, *, max_bytes: int) -> list[dict[str, Any]]:
    """Complete records from the first `max_bytes`; a trailing partial line is dropped."""
    with path.open('rb') as handle:
        data = handle.read(max_bytes)
    if not data:
        return []
    cut = len(data) == max_bytes
    lines = data.split(b'\n')
    if cut:
        # The last element is a cut-off line. Without a cut, an unterminated last line is kept when it parses and
        # silently ignored when it does not (an in-progress write).
        lines.pop()
    if cut and not lines:
        raise SessionLoadError(f'{path}: first record exceeds {max_bytes} bytes')
    return _decode(lines)[0]


def read_tail(path: Path, *, max_bytes: int) -> list[dict[str, Any]]:
    """Complete records from the last `max_bytes`; a leading partial line is dropped."""
    with path.open('rb') as handle:
        size = handle.seek(0, 2)
        start = max(0, size - max_bytes)
        handle.seek(start)
        data = handle.read()
    lines = data.split(b'\n')
    if start > 0:
        lines.pop(0)
    return _decode(lines)[0]


def warn_skipped(path: Path, skipped: int) -> None:
    if skipped > 0:
        logger.warning('Skipped {} malformed lines in {}', skipped, path)


def timestamp_or(value: object, default: datetime) -> datetime:
    """An aware datetime from an ISO string; the default for a missing or malformed value."""
    if not isinstance(value, str) or not value:
        return default
    try:
        parsed = parse_iso(value)
    except ValueError:
        return default
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
