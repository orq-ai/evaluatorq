"""Read local Claude Code, Claude desktop, Codex and omp sessions as ATIF."""

from __future__ import annotations

from evaluatorq.local_sessions.documents import session_document
from evaluatorq.local_sessions.models import (
    FAMILY_OF,
    MAX_SELECTED_SESSIONS,
    MAX_SESSION_TEXT_CHARS,
    MAX_TOOL_TEXT_CHARS,
    SEARCH_DEADLINE_SECONDS,
    SESSION_FAMILIES,
    SESSION_SOURCES,
    ParsedSession,
    SessionFamily,
    SessionLoadError,
    SessionQuery,
    SessionRef,
    SessionSearchResult,
    SessionSource,
    SessionSummary,
    day_window,
)
from evaluatorq.local_sessions.readers import READERS, SessionReader
from evaluatorq.local_sessions.roots import session_roots
from evaluatorq.local_sessions.search import (
    SnapshotTooLarge,
    build_session_snapshot,
    clear_summary_cache,
    load_session_document,
    project_matches,
    resolve_session_ref,
    search_sessions,
)

__all__ = [
    'FAMILY_OF',
    'MAX_SELECTED_SESSIONS',
    'MAX_SESSION_TEXT_CHARS',
    'MAX_TOOL_TEXT_CHARS',
    'READERS',
    'SEARCH_DEADLINE_SECONDS',
    'SESSION_FAMILIES',
    'SESSION_SOURCES',
    'ParsedSession',
    'SessionFamily',
    'SessionLoadError',
    'SessionQuery',
    'SessionReader',
    'SessionRef',
    'SessionSearchResult',
    'SessionSource',
    'SessionSummary',
    'SnapshotTooLarge',
    'build_session_snapshot',
    'clear_summary_cache',
    'day_window',
    'load_session_document',
    'project_matches',
    'resolve_session_ref',
    'search_sessions',
    'session_document',
    'session_roots',
]
