"""Bounded search over local sessions, safe selection, and snapshot export. Stdlib + pydantic + loguru only."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import TypeAdapter

from evaluatorq.local_sessions.documents import session_document
from evaluatorq.local_sessions.items import blocks_text
from evaluatorq.local_sessions.models import (
    FAMILY_OF,
    MAX_SELECTED_SESSIONS,
    SEARCH_DEADLINE_SECONDS,
    SESSION_FAMILIES,
    SessionLoadError,
    SessionQuery,
    SessionRef,
    SessionSearchResult,
    SessionSummary,
)
from evaluatorq.local_sessions.readers import READERS
from evaluatorq.local_sessions.roots import session_roots

if TYPE_CHECKING:
    from collections.abc import Sequence

    from evaluatorq.common.trace_document import TraceDocument
    from evaluatorq.local_sessions.models import ParsedSession, SessionFamily
    from evaluatorq.local_sessions.readers import SessionReader

_CACHE_MAX_ENTRIES = 50_000
_CHUNK_SIZE = 1024 * 1024

_CacheKey = tuple[str, int, int]
_summary_cache: OrderedDict[_CacheKey, SessionSummary | None] = OrderedDict()
_summary_cache_lock = threading.Lock()


def clear_summary_cache() -> None:
    with _summary_cache_lock:
        _summary_cache.clear()


def _cache_get(key: _CacheKey) -> tuple[bool, SessionSummary | None]:
    with _summary_cache_lock:
        if key in _summary_cache:
            _summary_cache.move_to_end(key)
            return True, _summary_cache[key]
    return False, None


def _cache_put(key: _CacheKey, value: SessionSummary | None) -> None:
    with _summary_cache_lock:
        _summary_cache[key] = value
        _summary_cache.move_to_end(key)
        while len(_summary_cache) > _CACHE_MAX_ENTRIES:
            _summary_cache.popitem(last=False)


def _real(path: str | Path) -> Path:
    return Path(os.path.normcase(os.path.realpath(path)))


@lru_cache(maxsize=4096)
def _worktree_main_repo(directory: str) -> str | None:
    """The main repository of a linked git worktree at `directory`, else None."""
    try:
        text = (Path(directory) / '.git').read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None
    first = text.splitlines()[0] if text else ''
    if not first.startswith('gitdir:'):
        return None
    gitdir = Path(first[len('gitdir:') :].strip())
    if not gitdir.is_absolute():
        gitdir = Path(directory) / gitdir
    if gitdir.parent.name != 'worktrees' or gitdir.parent.parent.name != '.git':
        return None
    return str(gitdir.parent.parent.parent)


def project_matches(project_dir: str, base: str) -> bool:
    """True when `project_dir` equals or lies under `base`, or is a linked git worktree of a repo that does.

    A `base` that is itself a linked worktree stands for its main repository, so
    `--project-dir .` run inside a worktree also finds the sibling worktrees.
    """
    if not project_dir or not base:
        return False
    base_path = _real(base)
    base_repo = _worktree_main_repo(str(base_path))
    if base_repo is not None:
        base_path = _real(base_repo)
    directory = _real(project_dir)
    if directory.is_relative_to(base_path):
        return True
    repo = _worktree_main_repo(str(directory))
    return repo is not None and _real(repo).is_relative_to(base_path)


_Prefilter = Callable[[Path, Callable[[], bool]], bool | None]


def _ascii_prefilter(text: str) -> _Prefilter:
    """Raw-bytes scan: `bytes.lower` folds ASCII only, which is all an ASCII needle contains."""
    lowered = text.lower()
    needles = tuple({lowered.encode(), json.dumps(lowered)[1:-1].encode()})
    overlap = max(len(needle) for needle in needles) - 1

    def scan(path: Path, stop: Callable[[], bool]) -> bool | None:
        carry = b''
        with path.open('rb') as raw:
            while chunk := raw.read(_CHUNK_SIZE):
                if stop():
                    return None
                window = carry + chunk.lower()
                if any(needle in window for needle in needles):
                    return True
                carry = window[-overlap:] if overlap > 0 else b''
        return False

    return scan


def _char_alternatives(char: str) -> list[str]:
    """Every spelling of `char` that a JSON line can hold, in either case."""
    forms = {char, char.lower(), char.upper()}
    spellings = {form for form in forms if len(form) == 1}
    spellings |= {json.dumps(form)[1:-1] for form in spellings}
    spellings |= {json.dumps(form, ensure_ascii=False)[1:-1] for form in spellings}
    return sorted(spellings, key=len, reverse=True)


def _unicode_prefilter(text: str) -> _Prefilter:
    """Per-character alternation (literal or JSON-escaped, any case), so non-ASCII case never causes a miss."""
    groups = [_char_alternatives(char) for char in text.lower()]
    pattern = re.compile(
        ''.join(f'(?:{"|".join(re.escape(alternative) for alternative in group)})' for group in groups),
        re.IGNORECASE,
    )
    overlap = sum(len(group[0]) for group in groups) - 1

    def scan(path: Path, stop: Callable[[], bool]) -> bool | None:
        carry = ''
        with path.open(encoding='utf-8', errors='replace') as handle:
            while chunk := handle.read(_CHUNK_SIZE):
                if stop():
                    return None
                window = carry + chunk
                if pattern.search(window):
                    return True
                carry = window[-overlap:] if overlap > 0 else ''
        return False

    return scan


def _prefilter(text: str) -> _Prefilter:
    return _ascii_prefilter(text) if text.isascii() else _unicode_prefilter(text)


def _message_texts(parsed: ParsedSession) -> list[str]:
    texts: list[str] = []
    for item in parsed.items:
        kind = item.get('type')
        if kind == 'message':
            texts.append(blocks_text(item.get('content')))
        elif kind == 'reasoning':
            texts.append(blocks_text(item.get('summary')))
    return texts


def _confirm_text(reader: SessionReader, path: Path, summary: SessionSummary, needle: str) -> bool:
    parsed = reader.parse(path, summary)
    return any(needle in text.lower() for text in _message_texts(parsed))


def _in_window(summary: SessionSummary, query: SessionQuery) -> bool:
    return (query.start is None or summary.updated_at >= query.start) and (
        query.end is None or summary.started_at < query.end
    )


def _summary_for(reader: SessionReader, path: Path, key: _CacheKey) -> SessionSummary | None:
    hit, cached = _cache_get(key)
    if hit:
        return cached
    summary = reader.summarize(path)
    _cache_put(key, summary)
    return summary


def _collect_candidates(
    query: SessionQuery, stopped: Callable[[], bool]
) -> tuple[list[tuple[float, _CacheKey, Path, SessionReader]], bool]:
    """Session files of the requested families, newest first, skipping files untouched since `query.start`.

    Returns the files found and whether the walk finished; `stopped` is checked before each file, so a deadline or
    a cancel ends the walk with what was found so far.
    """
    families: list[SessionFamily] = [
        family for family in SESSION_FAMILIES if any(FAMILY_OF[source] == family for source in query.sources)
    ]
    start_ts = query.start.astimezone(timezone.utc).timestamp() if query.start is not None else None
    candidates: list[tuple[float, _CacheKey, Path, SessionReader]] = []
    for family in families:
        reader = READERS[family]
        for root in session_roots(family):
            for path in reader.iter_files(root):
                if stopped():
                    candidates.sort(key=itemgetter(0), reverse=True)
                    return candidates, False
                try:
                    stat = path.stat()
                except OSError as exc:
                    logger.warning('Skipping unreadable session file {}: {}', path, type(exc).__name__)
                    continue
                if start_ts is not None and stat.st_mtime < start_ts:
                    continue
                candidates.append((stat.st_mtime, (str(path), stat.st_mtime_ns, stat.st_size), path, reader))
    candidates.sort(key=itemgetter(0), reverse=True)
    return candidates, True


def search_sessions(
    query: SessionQuery,
    *,
    deadline_seconds: float = SEARCH_DEADLINE_SECONDS,
    cancelled: Callable[[], bool] | None = None,
) -> SessionSearchResult:
    """Scan session files newest first, stopping at `query.limit` matches, the deadline, or a cancel.

    The deadline and the cancel also bound the file walk; a walk cut short reports `complete=False`, with
    `candidate_files` counting the files found so far.
    """
    deadline = time.monotonic() + deadline_seconds
    _worktree_main_repo.cache_clear()

    def stopped() -> bool:
        return time.monotonic() >= deadline or (cancelled is not None and cancelled())

    candidates, complete = _collect_candidates(query, stopped)
    if not complete:
        logger.warning('Session file walk stopped after finding {} files', len(candidates))

    prefilter = _prefilter(query.text) if query.text else None
    lowered = query.text.lower() if query.text else ''
    matches: list[SessionSummary] = []
    scanned = 0
    for _mtime, key, path, reader in candidates if complete else ():
        if len(matches) >= query.limit:
            break
        if stopped():
            complete = False
            logger.warning('Session search stopped after {} of {} files', scanned, len(candidates))
            break
        scanned += 1
        try:
            summary = _summary_for(reader, path, key)
            if summary is None or summary.source not in query.sources or not _in_window(summary, query):
                continue
            if query.project_dir and not project_matches(summary.project_dir, query.project_dir):
                continue
            if prefilter is not None:
                may_contain = prefilter(path, stopped)
                if may_contain is None:
                    complete = False
                    logger.warning('Session search stopped after {} of {} files', scanned - 1, len(candidates))
                    break
                if not may_contain or not _confirm_text(reader, path, summary, lowered):
                    continue
        except (SessionLoadError, OSError, ValueError) as exc:
            logger.warning('Skipping session file {}: {}', path, type(exc).__name__)
            continue
        matches.append(summary)
    matches.sort(key=lambda summary: summary.updated_at, reverse=True)
    return SessionSearchResult(
        sessions=tuple(matches), scanned_files=scanned, candidate_files=len(candidates), complete=complete
    )


def _resolve(ref: SessionRef) -> tuple[Path, SessionSummary]:
    try:
        path = Path(os.path.realpath(ref.path))
        reader = READERS[FAMILY_OF[ref.source]]
        for root in session_roots(FAMILY_OF[ref.source]):
            real_root = Path(os.path.realpath(root))
            if path.is_relative_to(real_root) and reader.is_session_path(path, real_root):
                break
        else:
            raise SessionLoadError(f'{ref.path}: not a main session file inside the {ref.source} session folders')
        if not path.is_file():
            raise SessionLoadError(f'{ref.path}: not a file')
        summary = reader.summarize(path)
    except SessionLoadError:
        raise
    except (OSError, ValueError) as exc:
        raise SessionLoadError(f'{ref.path}: {type(exc).__name__}') from exc
    if summary is None:
        raise SessionLoadError(f'{ref.path}: not a main session')
    if summary.source != ref.source:
        raise SessionLoadError(f'{ref.path}: source mismatch, the file is a {summary.source} session')
    return path, summary


def resolve_session_ref(ref: SessionRef) -> SessionSummary:
    """Accept a ref only when it is a main session file inside a session root of its own source family."""
    return _resolve(ref)[1]


def load_session_document(ref: SessionRef) -> TraceDocument:
    path, summary = _resolve(ref)
    try:
        return session_document(READERS[FAMILY_OF[ref.source]].parse(path, summary))
    except SessionLoadError:
        raise
    except (OSError, ValueError) as exc:
        raise SessionLoadError(f'{ref.path}: {type(exc).__name__}') from exc


def _document_end(document: TraceDocument) -> datetime:
    return datetime.fromisoformat(str(document.metadata.capture_metadata['end']))


class SnapshotTooLarge(Exception):  # noqa: N818 - reads as a condition, like the issue it reports
    """The snapshot passed the `max_bytes` budget of `build_session_snapshot`."""


@dataclass(frozen=True)
class FrozenSnapshot:
    """An Insights snapshot as its final JSON bytes, ready to write, plus what was and was not included."""

    data: bytes
    n_sessions: int
    failed: list[tuple[SessionRef, str]]


_METADATA_JSON = TypeAdapter(dict[str, Any])
_SNAPSHOT_HEAD = b'{"traces":[],"documents":['
_SNAPSHOT_MIDDLE = b'],"capture_metadata":'


def build_session_snapshot(refs: Sequence[SessionRef], *, max_bytes: int | None = None) -> FrozenSnapshot:
    """Freeze the selected sessions into snapshot JSON, byte-for-byte what `Snapshot(...).model_dump_json()` writes.

    Each document is serialized once and only its bytes are kept. With `max_bytes`, raises `SnapshotTooLarge` as
    soon as the serialized documents pass it, so loading stops early; memory holds at most the budget plus one
    document. The final size, envelope included, is checked against `max_bytes` once, before returning.

    A failed session's reason is the `SessionLoadError` text, which the readers build from the file path plus a
    reason or an exception type name, never from session content.
    """
    if len(refs) > MAX_SELECTED_SESSIONS:
        raise SessionLoadError(f'at most {MAX_SELECTED_SESSIONS} sessions can be selected, got {len(refs)}')
    by_id: dict[str, tuple[datetime, bytes]] = {}
    total = 0
    failed: list[tuple[SessionRef, str]] = []
    duplicates = 0
    for ref in refs:
        try:
            document = load_session_document(ref)
        except SessionLoadError as exc:
            failed.append((ref, str(exc)))
            logger.warning('Skipping local session {}: {}', ref.path, type(exc).__name__)
            continue
        trace_id = document.metadata.trace_id
        ended = _document_end(document)
        existing = by_id.get(trace_id)
        if existing is not None:
            duplicates += 1
            if ended <= existing[0]:
                continue
            total -= len(existing[1])
        serialized = document.model_dump_json().encode()
        by_id[trace_id] = (ended, serialized)
        total += len(serialized)
        if max_bytes is not None and total > max_bytes:
            raise SnapshotTooLarge
    if duplicates:
        logger.warning('Dropped {} duplicate local sessions that share a trace id', duplicates)
    if not by_id:
        raise SessionLoadError(f'none of the {len(refs)} selected sessions could be loaded')
    metadata = _METADATA_JSON.dump_json({'source': 'local-sessions', 'sessions': len(by_id)})
    data = b''.join((_SNAPSHOT_HEAD, b','.join(item[1] for item in by_id.values()), _SNAPSHOT_MIDDLE, metadata, b'}'))
    if max_bytes is not None and len(data) > max_bytes:
        raise SnapshotTooLarge
    return FrozenSnapshot(data=data, n_sessions=len(by_id), failed=failed)
