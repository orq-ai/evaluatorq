"""`eq agent-sessions`."""

# ruff: noqa: S101, SLF001

from __future__ import annotations

import json
import stat
import sys
from typing import TYPE_CHECKING

import pytest
import typer
from typer.testing import CliRunner

from evaluatorq import cli as cli_root
from evaluatorq.local_sessions import cli as sessions_cli
from evaluatorq.local_sessions.models import MAX_SESSION_TEXT_CHARS, SessionSearchResult
from evaluatorq.trace_finder.models import Snapshot

from .test_search import _claude, _codex, _utc

if TYPE_CHECKING:
    from pathlib import Path


def _app() -> typer.Typer:
    app = typer.Typer()
    cli_root._register_subapps(app)
    return app


def _run(*args: str):
    return CliRunner().invoke(_app(), ['agent-sessions', *args], env={'COLUMNS': '200'})




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
    if sys.platform != 'win32':  # Windows has no POSIX permission bits
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert 'Wrote 1 session to' in result.stderr


@pytest.mark.skipif(sys.platform == 'win32', reason='Windows has no POSIX permission bits')
def test_export_over_an_existing_world_readable_file_leaves_it_private(claude_projects: Path, tmp_path: Path) -> None:
    _claude(claude_projects, 'c1')
    target = tmp_path / 'snap.json'
    target.parent.chmod(0o755)
    target.write_text('old', encoding='utf-8')
    target.chmod(0o644)

    result = _run('--export', str(target))

    assert result.exit_code == 0, result.output
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o755
    assert Snapshot.model_validate_json(target.read_text(encoding='utf-8')).documents
    assert [path.name for path in target.parent.iterdir() if path.name != 'session-roots'] == ['snap.json']


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
    assert 'first day must not be after the last day' in reversed_range.stderr


def test_text_longer_than_the_cap_exits_2() -> None:
    result = _run('--text', 'x' * (MAX_SESSION_TEXT_CHARS + 1))
    assert result.exit_code == 2
    assert f'limited to {MAX_SESSION_TEXT_CHARS} characters' in result.stderr
    assert _run('--text', 'x' * MAX_SESSION_TEXT_CHARS).exit_code == 0


def test_export_to_an_unwritable_place_exits_2_without_a_traceback(claude_projects: Path, tmp_path: Path) -> None:
    _claude(claude_projects, 'c1')
    blocker = tmp_path / 'file'
    blocker.write_text('x', encoding='utf-8')

    result = _run('--export', str(blocker / 'snap.json'))

    assert result.exit_code == 2
    assert 'could not write' in result.stderr
    assert 'Traceback' not in result.output


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


def _capture_limit(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    seen: list[int] = []

    def fake(query):  # noqa: ANN001, ANN202
        seen.append(query.limit)
        return SessionSearchResult(sessions=(), scanned_files=0, candidate_files=0, complete=True)

    monkeypatch.setattr(sessions_cli, 'search_sessions', fake)
    return seen


def test_limit_defaults_to_50_when_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture_limit(monkeypatch)
    _run()
    assert seen == [50]


def test_export_without_limit_uses_the_maximum(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _capture_limit(monkeypatch)
    _run('--export', str(tmp_path / 'snap.json'))
    assert seen == [1000]


def test_explicit_limit_wins_with_export(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = _capture_limit(monkeypatch)
    _run('--limit', '7', '--export', str(tmp_path / 'snap.json'))
    assert seen == [7]


def test_cut_short_export_says_it_is_partial(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        sessions_cli,
        'search_sessions',
        lambda _query: SessionSearchResult(sessions=(), scanned_files=3, candidate_files=9, complete=False),
    )
    result = _run('--export', str(tmp_path / 'snap.json'))
    assert 'The export is partial' in result.stderr
    assert 'The export is partial' not in _run().stderr
