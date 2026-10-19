"""Bounded, malformed-tolerant JSONL reading. `OSError` propagates; callers wrap it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from evaluatorq.formats._shared import parse_iso
from evaluatorq.local_sessions.models import ParsedSession, SessionLoadError, SessionSource, SessionSummary

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

HEAD_BYTES = 262_144
TAIL_BYTES = 262_144
TEXT_CHARS = 200

Records = list[dict[str, Any]]


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
        # The first record alone exceeds the budget: nothing complete to return; the tail decides.
        return []
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


def last_timestamp(tail: Records, fallback: datetime) -> datetime:
    """The newest `timestamp` string in `tail`, else `fallback`."""
    raw = next((record['timestamp'] for record in reversed(tail) if isinstance(record.get('timestamp'), str)), '')
    return timestamp_or(raw, fallback)


def unreadable(path: Path, exc: BaseException) -> SessionLoadError:
    """Names the exception type only: its text can hold session content."""
    return SessionLoadError(f'{path}: unreadable session ({type(exc).__name__})')


@dataclass(frozen=True)
class SummaryParts:
    """What a reader extracts from a session's head and tail; `summarize_session` adds the file facts."""

    source: SessionSource
    session_id: str
    title: str
    project_dir: str
    started: object
    first_prompt: str
    agent_version: str


def summarize_session(path: Path, extract: Callable[[Records, Records], SummaryParts | None]) -> SessionSummary | None:
    """Read head and tail once and let `extract` decide; `None` means readable but not a main session."""
    try:
        head = read_head(path, max_bytes=HEAD_BYTES)
        tail = read_tail(path, max_bytes=TAIL_BYTES)
        stat = path.stat()
        parts = extract(head, tail)
        if parts is None:
            return None
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        return SessionSummary(
            source=parts.source,
            path=path.absolute(),
            session_id=parts.session_id or path.stem,
            title=parts.title[:TEXT_CHARS],
            project_dir=parts.project_dir,
            started_at=timestamp_or(parts.started, mtime),
            updated_at=last_timestamp(tail, mtime),
            size_bytes=stat.st_size,
            first_prompt=parts.first_prompt.strip()[:TEXT_CHARS],
            agent_version=parts.agent_version,
        )
    except SessionLoadError:
        raise
    except Exception as exc:
        raise unreadable(path, exc) from exc


def parse_session(
    path: Path,
    summary: SessionSummary,
    map_records: Callable[[Records], tuple[list[dict[str, Any]], str, int | None]],
) -> ParsedSession:
    """Read every record of an already summarized session and map it to Responses items."""
    try:
        records, skipped = read_records(path)
        warn_skipped(path, skipped)
        converted, model, total_tokens = map_records(records)
    except SessionLoadError:
        raise
    except Exception as exc:
        raise unreadable(path, exc) from exc
    if not converted:
        raise SessionLoadError(f'{path}: no conversation items')
    return ParsedSession(
        summary=summary, items=converted, model=model, total_tokens=total_tokens, skipped_lines=skipped
    )
