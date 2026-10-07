"""Bounded search over local sessions, safe selection, and snapshot export. Stdlib + pydantic + loguru only."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from functools import lru_cache
from operator import itemgetter
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

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
from evaluatorq.trace_finder.models import Snapshot

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from evaluatorq.common.trace_document import TraceDocument
    from evaluatorq.local_sessions.models import ParsedSession, SessionFamily
    from evaluatorq.local_sessions.readers import SessionReader

_CACHE_MAX_ENTRIES = 50_000
_CHUNK_CHARS = 1024 * 1024

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
    # <repo>/.git/worktrees/<name>
    if gitdir.parent.name != 'worktrees' or gitdir.parent.parent.name != '.git':
        return None
    return str(gitdir.parent.parent.parent)


def project_matches(project_dir: str, base: str) -> bool:
    """True when `project_dir` equals or lies under `base`, or is a linked git worktree of a repo that does."""
    if not project_dir or not base:
        return False
    base_path = _real(base)
    directory = _real(project_dir)
    if directory.is_relative_to(base_path):
        return True
    repo = _worktree_main_repo(str(directory))
    return repo is not None and _real(repo).is_relative_to(base_path)


def _needles(text: str) -> tuple[str, ...]:
    variants = {
        text.lower(),
        json.dumps(text, ensure_ascii=False)[1:-1].lower(),
        json.dumps(text)[1:-1].lower(),
    }
    return tuple(variants)


def _file_may_contain(path: Path, needles: tuple[str, ...]) -> bool:
    overlap = max(len(needle) for needle in needles) - 1
    carry = ''
    with path.open(encoding='utf-8', errors='replace') as handle:
        while chunk := handle.read(_CHUNK_CHARS):
            window = carry + chunk.lower()
            if any(needle in window for needle in needles):
                return True
            carry = window[-overlap:] if overlap > 0 else ''
    return False


def _message_texts(parsed: ParsedSession) -> list[str]:
    texts: list[str] = []
    for item in parsed.items:
        kind = item.get('type')
        if kind == 'message':
            texts.append(blocks_text(item.get('content')))
        elif kind == 'reasoning':
            texts.append(blocks_text(item.get('summary')))
    return texts


def _confirm_text(reader: SessionReader, path: Path, needle: str) -> bool:
    parsed = reader.parse(path)
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


def search_sessions(
    query: SessionQuery,
    *,
    deadline_seconds: float = SEARCH_DEADLINE_SECONDS,
    cancelled: Callable[[], bool] | None = None,
) -> SessionSearchResult:
    """Scan session files newest first, stopping at `query.limit` matches, the deadline, or a cancel."""
    deadline = time.monotonic() + deadline_seconds
    families: list[SessionFamily] = [
        family for family in SESSION_FAMILIES if any(FAMILY_OF[source] == family for source in query.sources)
    ]
    start_ts = query.start.astimezone(timezone.utc).timestamp() if query.start is not None else None
    candidates: list[tuple[float, _CacheKey, Path, SessionReader]] = []
    for family in families:
        reader = READERS[family]
        for root in session_roots(family):
            for path in reader.iter_files(root):
                try:
                    stat = path.stat()
                except OSError as exc:
                    logger.warning('Skipping unreadable session file {}: {}', path, type(exc).__name__)
                    continue
                if start_ts is not None and stat.st_mtime < start_ts:
                    continue
                candidates.append((stat.st_mtime, (str(path), stat.st_mtime_ns, stat.st_size), path, reader))
    candidates.sort(key=itemgetter(0), reverse=True)

    needles = _needles(query.text) if query.text else None
    lowered = query.text.lower() if query.text else ''
    matches: list[SessionSummary] = []
    scanned = 0
    complete = True
    for _mtime, key, path, reader in candidates:
        if len(matches) >= query.limit:
            break
        if time.monotonic() >= deadline or (cancelled is not None and cancelled()):
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
            if needles is not None and not (_file_may_contain(path, needles) and _confirm_text(reader, path, lowered)):
                continue
        except (SessionLoadError, OSError, ValueError) as exc:
            logger.warning('Skipping session file {}: {}', path, type(exc).__name__)
            continue
        matches.append(summary)
    matches.sort(key=lambda summary: summary.updated_at, reverse=True)
    return SessionSearchResult(
        sessions=tuple(matches), scanned_files=scanned, candidate_files=len(candidates), complete=complete
    )


def resolve_session_ref(ref: SessionRef) -> SessionSummary:
    """Accept a ref only when it is a main session file inside a session root of its own source family."""
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
    if summary is None:
        raise SessionLoadError(f'{ref.path}: not a main session')
    if summary.source != ref.source:
        raise SessionLoadError(f'{ref.path}: source mismatch, the file is a {summary.source} session')
    return summary


def load_session_document(ref: SessionRef) -> TraceDocument:
    resolve_session_ref(ref)
    path = Path(os.path.realpath(ref.path))
    try:
        return session_document(READERS[FAMILY_OF[ref.source]].parse(path))
    except SessionLoadError:
        raise
    except (OSError, ValueError) as exc:
        raise SessionLoadError(f'{ref.path}: {type(exc).__name__}') from exc


def _document_end(document: TraceDocument) -> datetime:
    return datetime.fromisoformat(str(document.metadata.capture_metadata['end']))


def build_session_snapshot(refs: Sequence[SessionRef]) -> tuple[Snapshot, list[tuple[SessionRef, str]]]:
    """Freeze the selected sessions into a snapshot; also return `(ref, exception type)` for each that failed."""
    if len(refs) > MAX_SELECTED_SESSIONS:
        raise SessionLoadError(f'at most {MAX_SELECTED_SESSIONS} sessions can be selected, got {len(refs)}')
    by_id: dict[str, TraceDocument] = {}
    failed: list[tuple[SessionRef, str]] = []
    duplicates = 0
    for ref in refs:
        try:
            document = load_session_document(ref)
        except SessionLoadError as exc:
            failed.append((ref, type(exc).__name__))
            logger.warning('Skipping local session {}: {}', ref.path, type(exc).__name__)
            continue
        trace_id = document.metadata.trace_id
        existing = by_id.get(trace_id)
        if existing is not None:
            duplicates += 1
            if _document_end(document) <= _document_end(existing):
                continue
        by_id[trace_id] = document
    if duplicates:
        logger.warning('Dropped {} duplicate local sessions that share a trace id', duplicates)
    if not by_id:
        raise SessionLoadError(f'none of the {len(refs)} selected sessions could be loaded')
    documents = tuple(by_id.values())
    snapshot = Snapshot(
        traces=(),
        documents=documents,
        capture_metadata={'source': 'local-sessions', 'sessions': len(documents)},
    )
    return snapshot, failed
