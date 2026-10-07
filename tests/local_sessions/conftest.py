"""Fixtures for local-session tests: roots are the isolated temp directories from `tests/conftest.py`."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from evaluatorq.local_sessions import clear_summary_cache, roots

if TYPE_CHECKING:
    from collections.abc import Iterable


@pytest.fixture(autouse=True)
def _fresh_summary_cache() -> None:
    clear_summary_cache()


def write_jsonl(path: Path, records: Iterable[dict[str, Any] | str], *, trailing_newline: bool = True) -> Path:
    """Write records (dicts as JSON, strings verbatim) one per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(record if isinstance(record, str) else json.dumps(record) for record in records)
    path.write_text(text + ('\n' if trailing_newline else ''), encoding='utf-8')
    return path


def _made(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def claude_projects() -> Path:
    return _made(Path(os.environ['CLAUDE_CONFIG_DIR']) / 'projects')


@pytest.fixture
def codex_home() -> Path:
    return _made(Path(os.environ['CODEX_HOME']))


@pytest.fixture
def omp_sessions() -> Path:
    return _made(Path(os.environ['PI_CODING_AGENT_DIR']) / 'sessions')


@pytest.fixture
def cowork_root() -> Path:
    return _made(roots.claude_desktop_dir() / 'local-agent-mode-sessions')
