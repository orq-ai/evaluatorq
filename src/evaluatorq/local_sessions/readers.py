"""The reader registry: one `SessionReader` per tool family."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from evaluatorq.local_sessions import claude, codex, omp
from evaluatorq.local_sessions.models import SESSION_FAMILIES

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from evaluatorq.local_sessions.models import ParsedSession, SessionFamily, SessionSummary


class SessionReader(Protocol):
    family: SessionFamily

    def iter_files(self, root: Path) -> Iterator[Path]: ...

    def is_session_path(self, path: Path, root: Path) -> bool:
        """Pure path-shape check, no I/O."""
        ...

    def summarize(self, path: Path) -> SessionSummary | None:
        """`None` means readable but not a main session; an unreadable file raises `SessionLoadError`."""
        ...

    def parse(self, path: Path, summary: SessionSummary) -> ParsedSession:
        """Parse a session already summarized as a main session; any failure is a `SessionLoadError`."""
        ...


READERS: Mapping[SessionFamily, SessionReader] = MappingProxyType({
    'claude': claude.READER,
    'codex': codex.READER,
    'omp': omp.READER,
})
assert set(READERS) == set(SESSION_FAMILIES)  # noqa: S101
