"""Where each tool keeps its session files."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evaluatorq.local_sessions.models import SessionFamily


def claude_desktop_dir() -> Path:
    """Claude desktop's application-support directory (only the macOS path is verified)."""
    if sys.platform == 'darwin':
        return Path.home() / 'Library' / 'Application Support' / 'Claude'
    if sys.platform == 'win32':
        return Path(os.environ.get('APPDATA') or Path.home() / 'AppData' / 'Roaming') / 'Claude'
    return Path.home() / '.config' / 'Claude'


def session_roots(family: SessionFamily) -> tuple[Path, ...]:
    """Existing directories holding main session files for `family`, each tool's own env override first."""
    if family == 'claude':
        config = Path(os.environ.get('CLAUDE_CONFIG_DIR') or Path.home() / '.claude')
        candidates = (config / 'projects', claude_desktop_dir() / 'local-agent-mode-sessions')
    elif family == 'codex':
        home = Path(os.environ.get('CODEX_HOME') or Path.home() / '.codex')
        candidates = (home / 'sessions', home / 'archived_sessions')
    else:
        agent = Path(os.environ.get('PI_CODING_AGENT_DIR') or Path.home() / '.omp' / 'agent')
        candidates = (agent / 'sessions',)
    return tuple(path for path in candidates if path.is_dir())
