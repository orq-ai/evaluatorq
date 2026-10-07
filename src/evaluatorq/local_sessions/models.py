"""Types shared by every local-session reader."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime  # noqa: TC003 - pydantic field type
from pathlib import Path  # noqa: TC003 - pydantic field type
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from collections.abc import Mapping

SessionSource = Literal['claude-code', 'claude-desktop', 'codex', 'omp']
SESSION_SOURCES: tuple[SessionSource, ...] = ('claude-code', 'claude-desktop', 'codex', 'omp')
SessionFamily = Literal['claude', 'codex', 'omp']
SESSION_FAMILIES: tuple[SessionFamily, ...] = ('claude', 'codex', 'omp')
FAMILY_OF: Mapping[SessionSource, SessionFamily] = MappingProxyType({
    'claude-code': 'claude',
    'claude-desktop': 'claude',
    'codex': 'codex',
    'omp': 'omp',
})
assert set(FAMILY_OF) == set(SESSION_SOURCES)  # noqa: S101 - import-time registry check

MAX_SELECTED_SESSIONS = 1000
MAX_TOOL_TEXT_CHARS = 20_000
SEARCH_DEADLINE_SECONDS = 20.0


class SessionLoadError(ValueError):
    """A session file could not be read or is not a selectable main session."""


class SessionRef(BaseModel):
    """One selectable session file; `path` is absolute."""

    model_config = ConfigDict(frozen=True)

    source: SessionSource
    path: Path


class SessionSummary(SessionRef):
    """Listing metadata, read from a session file's head and tail only."""

    session_id: str = Field(min_length=1)
    title: str = ''
    project_dir: str = ''
    started_at: datetime
    updated_at: datetime
    size_bytes: int = Field(ge=0)
    first_prompt: str = ''
    agent_version: str = ''

    @field_validator('started_at', 'updated_at')
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('session timestamps must be timezone-aware')
        return value

    def ref(self) -> SessionRef:
        return SessionRef(source=self.source, path=self.path)


class SessionQuery(BaseModel):
    """Search filters: sessions active in [start, end), run in `project_dir`, whose messages contain `text`."""

    model_config = ConfigDict(frozen=True)

    sources: tuple[SessionSource, ...] = SESSION_SOURCES
    start: datetime | None = None
    end: datetime | None = None
    project_dir: str | None = None
    text: str | None = None
    limit: int = Field(default=50, ge=1, le=MAX_SELECTED_SESSIONS)


class SessionSearchResult(BaseModel):
    """Matches newest first; `complete` is False when the deadline or a cancel stopped the scan."""

    model_config = ConfigDict(frozen=True)

    sessions: tuple[SessionSummary, ...]
    scanned_files: int
    candidate_files: int
    complete: bool


@dataclass(frozen=True)
class ParsedSession:
    summary: SessionSummary
    items: list[dict[str, Any]]
    model: str
    total_tokens: int | None
    skipped_lines: int
