"""`eq sessions`."""

# ruff: noqa: S101, SLF001

from __future__ import annotations

import json
import stat
from typing import TYPE_CHECKING

import typer
from typer.testing import CliRunner

from evaluatorq import cli as cli_root
from evaluatorq.local_sessions import cli as sessions_cli
from evaluatorq.local_sessions.models import SessionSearchResult
from evaluatorq.trace_finder.models import Snapshot

from .test_search import _claude, _codex, _utc

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _app() -> typer.Typer:
    app = typer.Typer()
    cli_root._register_subapps(app)
    return app


def _run(*args: str):
    return CliRunner().invoke(_app(), ['sessions', *args], env={'COLUMNS': '200'})


def test_table_output(claude_projects: Path, codex_home: Path) -> None:
    _claude(claude_projects, 'c1', text='fix the parser')
    _codex(codex_home, 'x1', text='write docs')
    result = _run()
    assert result.exit_code == 0, result.output
    assert 'fix the parser' in result.stdout
    assert 'write docs' in result.stdout
    assert 'claude-code' in result.stdout
    assert 'codex' in result.stdout
    assert 'Updated' in result.stdout


def test_json_output_shape(claude_projects: Path) -> None:
    path = _claude(claude_projects, 'c1')
    result = _run('--json', '--source', 'claude-code')
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert len(payload) == 1
    assert payload[0]['session_id'] == 'c1'
    assert payload[0]['source'] == 'claude-code'
    assert payload[0]['path'] == str(path)


def test_export_writes_snapshot_with_documents(claude_projects: Path, tmp_path: Path) -> None:
    _claude(claude_projects, 'c1')
    target = tmp_path / 'out' / 'snap.json'
    result = _run('--export', str(target))
    assert result.exit_code == 0, result.output
    snapshot = Snapshot.model_validate_json(target.read_text(encoding='utf-8'))
    assert len(snapshot.documents) == 1
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert 'Wrote 1 session to' in result.stderr


def test_export_empty_result_exits_2_and_writes_nothing(tmp_path: Path) -> None:
    target = tmp_path / 'snap.json'
    result = _run('--export', str(target))
    assert result.exit_code == 2
    assert not target.exists()


def test_unknown_source_exits_2() -> None:
    result = _run('--source', 'bogus')
    assert result.exit_code == 2
    assert 'bogus' in result.stderr


def test_bad_date_and_reversed_range_exit_2() -> None:
    assert _run('--from', 'yesterday').exit_code == 2
    reversed_range = _run('--from', '2026-10-05', '--to', '2026-10-01')
    assert reversed_range.exit_code == 2
    assert '--from' in reversed_range.stderr


def test_date_range_filters_by_local_day(claude_projects: Path) -> None:
    _claude(claude_projects, 'in', start=_utc(10, 12), end=_utc(10, 12))
    _claude(claude_projects, 'out', start=_utc(20), end=_utc(20))
    result = _run('--json', '--from', '2026-10-09', '--to', '2026-10-11')
    assert [s['session_id'] for s in json.loads(result.stdout)] == ['in']


def test_empty_result_message() -> None:
    result = _run()
    assert result.exit_code == 0
    assert 'No local sessions match.' in result.stdout


def test_cut_short_note(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sessions_cli,
        'search_sessions',
        lambda _query: SessionSearchResult(sessions=(), scanned_files=3, candidate_files=9, complete=False),
    )
    result = _run()
    assert result.exit_code == 0
    assert 'Searched 3 of 9 session files before the 20s limit' in result.stderr


def test_limit_note_when_the_list_is_full(claude_projects: Path, tmp_path: Path) -> None:
    _claude(claude_projects, 'c1')
    _claude(claude_projects, 'c2')
    table = _run('--limit', '2')
    as_json = _run('--limit', '2', '--json')
    exported = _run('--limit', '2', '--export', str(tmp_path / 'snap.json'))
    roomy = _run('--limit', '3')
    note = 'Listed the newest 2 sessions; raise --limit (up to 1000) to see more.'
    assert note in table.stderr
    assert note in as_json.stderr
    assert len(json.loads(as_json.stdout)) == 2, 'the note goes to stderr, stdout stays JSON'
    assert note in exported.stderr
    assert 'Listed the newest' not in roomy.stderr


def test_cut_short_export_says_it_is_partial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        sessions_cli,
        'search_sessions',
        lambda _query: SessionSearchResult(sessions=(), scanned_files=3, candidate_files=9, complete=False),
    )
    result = _run('--export', str(tmp_path / 'snap.json'))
    assert 'The export is partial' in result.stderr
    assert 'The export is partial' not in _run().stderr
