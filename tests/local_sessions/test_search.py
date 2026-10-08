"""Bounded search, safe resolution and snapshot export."""

# ruff: noqa: S101

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

import pytest
from loguru import logger

from evaluatorq.local_sessions import (
    MAX_SELECTED_SESSIONS,
    day_window,
    SessionLoadError,
    SessionQuery,
    SessionRef,
    SnapshotTooLarge,
    build_session_snapshot,
    load_session_document,
    project_matches,
    resolve_session_ref,
    search_sessions,
)
from evaluatorq.local_sessions import claude as claude_module
from evaluatorq.local_sessions.claude import READER as CLAUDE
from evaluatorq.local_sessions.readers import READERS
from evaluatorq.trace_finder.models import Snapshot

from .conftest import write_jsonl

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture
def warnings() -> Iterator[list[str]]:
    messages: list[str] = []
    handler = logger.add(lambda message: messages.append(str(message)), level='WARNING')
    yield messages
    logger.remove(handler)


def _utc(day: int, hour: int = 10) -> datetime:
    return datetime(2026, 10, day, hour, tzinfo=timezone.utc)


def _stamp(path: Path, when: datetime) -> Path:
    os.utime(path, (when.timestamp(), when.timestamp()))
    return path


def _claude(
    root: Path,
    sid: str,
    *,
    text: str = 'hello there',
    cwd: str = '/work/proj',
    start: datetime | None = None,
    end: datetime | None = None,
    entrypoint: str = 'cli',
    extra: list[dict[str, Any]] | None = None,
    stamp: bool = True,
) -> Path:
    start = start or _utc(1)
    end = end or start + timedelta(minutes=5)
    base = {'sessionId': sid, 'cwd': cwd, 'version': '2.0', 'entrypoint': entrypoint}
    records = [
        {
            **base,
            'type': 'user',
            'uuid': 'u1',
            'parentUuid': None,
            'timestamp': start.isoformat(),
            'message': {'role': 'user', 'content': text},
        },
        {
            **base,
            'type': 'assistant',
            'uuid': 'a1',
            'parentUuid': 'u1',
            'timestamp': end.isoformat(),
            'message': {'id': 'm1', 'model': 'claude-x', 'content': [{'type': 'text', 'text': 'ok'}]},
        },
        *(extra or []),
    ]
    path = write_jsonl(root / '-work-proj' / f'{sid}.jsonl', records)
    return _stamp(path, end) if stamp else path


def _codex(home: Path, sid: str, *, text: str = 'codex prompt', start: datetime | None = None) -> Path:
    start = start or _utc(2)
    ts = start.isoformat()
    path = write_jsonl(
        home / 'sessions' / '2026' / '10' / '02' / f'rollout-2026-10-02T10-00-00-{sid}.jsonl',
        [
            {'timestamp': ts, 'type': 'session_meta', 'payload': {'id': sid, 'timestamp': ts, 'cwd': '/work/codex'}},
            {
                'timestamp': ts,
                'type': 'response_item',
                'payload': {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': text}]},
            },
        ],
    )
    return _stamp(path, start)


def _omp(root: Path, sid: str, *, text: str = 'omp prompt', start: datetime | None = None) -> Path:
    start = start or _utc(3)
    ts = start.isoformat()
    path = write_jsonl(
        root / '-proj' / f'2026-10-03T10-00-00-000Z_{sid}.jsonl',
        [
            {'type': 'session', 'id': sid, 'cwd': '/work/omp', 'timestamp': ts, 'version': '3'},
            {
                'id': 'e1',
                'parentId': None,
                'timestamp': ts,
                'type': 'message',
                'message': {'role': 'user', 'content': [{'type': 'text', 'text': text}]},
            },
        ],
    )
    return _stamp(path, start)


def _all_sources(claude_projects: Path, codex_home: Path, omp_sessions: Path) -> None:
    _claude(claude_projects, 'c1')
    _claude(claude_projects, 'd1', entrypoint='claude-desktop', start=_utc(4))
    _codex(codex_home, 'x1')
    _omp(omp_sessions, 'o1')


def _ids(result: Any) -> set[str]:
    return {s.session_id for s in result.sessions}


def test_source_filter(claude_projects: Path, codex_home: Path, omp_sessions: Path) -> None:
    _all_sources(claude_projects, codex_home, omp_sessions)
    assert _ids(search_sessions(SessionQuery())) == {'c1', 'd1', 'x1', 'o1'}
    assert _ids(search_sessions(SessionQuery(sources=('codex',)))) == {'x1'}
    assert _ids(search_sessions(SessionQuery(sources=('claude-desktop',)))) == {'d1'}
    assert _ids(search_sessions(SessionQuery(sources=('claude-code', 'omp')))) == {'c1', 'o1'}


def test_window_includes_session_started_before_start_but_updated_inside(claude_projects: Path) -> None:
    _claude(claude_projects, 'long', start=_utc(1), end=_utc(10))
    _claude(claude_projects, 'old', start=_utc(1), end=_utc(2))
    _claude(claude_projects, 'late', start=_utc(20), end=_utc(21))
    result = search_sessions(SessionQuery(start=_utc(5), end=_utc(15)))
    assert _ids(result) == {'long'}


def test_project_matches_prefix_symlink_and_worktree(tmp_path: Path) -> None:
    base = tmp_path / 'a' / 'proj'
    (tmp_path / 'a' / 'proj2').mkdir(parents=True)
    (base / 'sub').mkdir(parents=True)
    assert project_matches(str(base), str(base))
    assert project_matches(str(base / 'sub'), str(base))
    assert not project_matches(str(tmp_path / 'a' / 'proj2'), str(base))
    assert not project_matches('', str(base))
    link = tmp_path / 'link'
    link.symlink_to(base)
    assert project_matches(str(base / 'sub'), str(link))

    worktree = tmp_path / 'wt'
    worktree.mkdir()
    (worktree / '.git').write_text(f'gitdir: {base}/.git/worktrees/x\n', encoding='utf-8')
    assert project_matches(str(worktree), str(base))
    other = tmp_path / 'other'
    assert not project_matches(str(worktree), str(other))
    gone = tmp_path / 'gone'
    assert not project_matches(str(gone), str(base))


def test_project_matches_from_inside_a_linked_worktree_finds_siblings(tmp_path: Path) -> None:
    repo = tmp_path / 'repo'
    (repo / '.git').mkdir(parents=True)
    here, sibling = tmp_path / 'wt-here', tmp_path / 'wt-sibling'
    for name, worktree in (('here', here), ('sibling', sibling)):
        worktree.mkdir()
        (worktree / '.git').write_text(f'gitdir: {repo}/.git/worktrees/{name}\n', encoding='utf-8')

    assert project_matches(str(sibling), str(here))
    assert project_matches(str(repo / 'src'), str(here))
    unrelated = tmp_path / 'unrelated'
    unrelated.mkdir()
    assert not project_matches(str(unrelated), str(here))


def test_project_filter_in_search(claude_projects: Path) -> None:
    _claude(claude_projects, 'p1', cwd='/work/proj')
    _claude(claude_projects, 'p2', cwd='/work/proj2')
    result = search_sessions(SessionQuery(project_dir='/work/proj'))
    assert _ids(result) == {'p1'}


def test_text_filter_matches_messages_not_tool_output(claude_projects: Path) -> None:
    needle = 'say "hi"\nplease'
    _claude(claude_projects, 'msg', text=needle)
    _claude(
        claude_projects,
        'tool',
        text='run it',
        extra=[
            {
                'type': 'user',
                'uuid': 'u2',
                'parentUuid': 'a1',
                'sessionId': 'tool',
                'cwd': '/work/proj',
                'timestamp': _utc(1).isoformat(),
                'message': {
                    'role': 'user',
                    'content': [{'type': 'tool_result', 'tool_use_id': 't1', 'content': 'SECRETNEEDLE output'}],
                },
            },
        ],
    )
    assert _ids(search_sessions(SessionQuery(text=needle))) == {'msg'}
    assert _ids(search_sessions(SessionQuery(text='SAY "HI"\nPLEASE'))) == {'msg'}
    assert _ids(search_sessions(SessionQuery(text='secretneedle'))) == set()


def test_text_filter_non_ascii(claude_projects: Path) -> None:
    _claude(claude_projects, 'u', text='café ☕ done')
    assert _ids(search_sessions(SessionQuery(text='café ☕'))) == {'u'}


def test_newest_first_and_early_stop_at_limit(claude_projects: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = [_claude(claude_projects, f's{day}', start=_utc(day), end=_utc(day, 11)) for day in (1, 2, 3, 4)]
    called: list[Path] = []
    original = CLAUDE.summarize

    def counting(path: Path) -> Any:
        called.append(path)
        return original(path)

    monkeypatch.setattr(CLAUDE, 'summarize', counting)
    result = search_sessions(SessionQuery(limit=2))
    assert [s.session_id for s in result.sessions] == ['s4', 's3']
    assert result.complete
    assert result.candidate_files == 4
    assert paths[0] not in called
    assert paths[1] not in called


def test_deadline_stops_scan_with_warning(claude_projects: Path, warnings: list[str]) -> None:
    _claude(claude_projects, 'a')
    result = search_sessions(SessionQuery(), deadline_seconds=0)
    assert result.complete is False
    assert result.scanned_files == 0
    assert any('Session file walk stopped after finding 0 files' in message for message in warnings)


def test_cancelled_stops_scan(claude_projects: Path) -> None:
    _claude(claude_projects, 'a')
    result = search_sessions(SessionQuery(), cancelled=lambda: True)
    assert result.complete is False
    assert result.sessions == ()


def test_unparsable_timestamp_falls_back_to_the_file_time(claude_projects: Path) -> None:
    _claude(claude_projects, 'good')
    bad = _claude(claude_projects, 'bad')
    text = bad.read_text(encoding='utf-8').replace(_utc(1).isoformat(), 'not-a-time')
    bad.write_text(text, encoding='utf-8')
    result = search_sessions(SessionQuery())
    assert _ids(result) == {'good', 'bad'}


def test_summary_cache_avoids_second_summarize(claude_projects: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _claude(claude_projects, 'a')
    calls: list[Path] = []
    original = CLAUDE.summarize
    monkeypatch.setattr(CLAUDE, 'summarize', lambda path: (calls.append(path), original(path))[1])
    search_sessions(SessionQuery())
    search_sessions(SessionQuery())
    assert len(calls) == 1


def test_resolve_accepts_listed_session(claude_projects: Path) -> None:
    path = _claude(claude_projects, 'ok')
    summary = resolve_session_ref(SessionRef(source='claude-code', path=path))
    assert summary.session_id == 'ok'


def test_resolve_rejections(claude_projects: Path, codex_home: Path, tmp_path: Path) -> None:
    outside = tmp_path / 'elsewhere'
    real = _claude(outside, 'x', stamp=False)
    with pytest.raises(SessionLoadError):
        resolve_session_ref(SessionRef(source='claude-code', path=real))

    inside = claude_projects / '-work-proj'
    inside.mkdir(parents=True, exist_ok=True)
    link = inside / 'link.jsonl'
    link.symlink_to(real)
    with pytest.raises(SessionLoadError):
        resolve_session_ref(SessionRef(source='claude-code', path=link))

    sub = write_jsonl(claude_projects / '-p' / 'sid' / 'subagents' / 'agent-1.jsonl', [{'type': 'user'}])
    with pytest.raises(SessionLoadError):
        resolve_session_ref(SessionRef(source='claude-code', path=sub))

    claude_file = _claude(claude_projects, 'real')
    with pytest.raises(SessionLoadError):
        resolve_session_ref(SessionRef(source='codex', path=claude_file))

    notes = claude_projects / '-work-proj' / 'notes.txt'
    notes.write_text('hi', encoding='utf-8')
    with pytest.raises(SessionLoadError):
        resolve_session_ref(SessionRef(source='claude-code', path=notes))

    with pytest.raises(SessionLoadError):
        resolve_session_ref(SessionRef(source='claude-code', path=claude_projects / '-work-proj' / 'missing.jsonl'))


def test_resolve_rejects_source_mismatch_within_family(claude_projects: Path) -> None:
    path = _claude(claude_projects, 'cli')
    with pytest.raises(SessionLoadError, match='source mismatch'):
        resolve_session_ref(SessionRef(source='claude-desktop', path=path))


def test_load_session_document(claude_projects: Path) -> None:
    path = _claude(claude_projects, 'doc')
    document = load_session_document(SessionRef(source='claude-code', path=path))
    assert document.metadata.trace_id == 'claude-code:doc'


def test_snapshot_deduplicates_trace_ids(claude_projects: Path, warnings: list[str]) -> None:
    older = _claude(claude_projects, 'same', end=_utc(1, 11))
    other_dir = claude_projects / '-other'
    newer = write_jsonl(
        other_dir / 'same.jsonl',
        [
            {
                'sessionId': 'same',
                'cwd': '/other',
                'type': 'user',
                'uuid': 'u1',
                'parentUuid': None,
                'timestamp': _utc(1).isoformat(),
                'message': {'role': 'user', 'content': 'again'},
            },
            {
                'sessionId': 'same',
                'cwd': '/other',
                'type': 'assistant',
                'uuid': 'a1',
                'parentUuid': 'u1',
                'timestamp': _utc(2).isoformat(),
                'message': {'id': 'm', 'model': 'x', 'content': [{'type': 'text', 'text': 'ok'}]},
            },
        ],
    )
    refs = [SessionRef(source='claude-code', path=older), SessionRef(source='claude-code', path=newer)]
    frozen = build_session_snapshot(refs)
    snapshot = Snapshot.model_validate_json(frozen.data)
    assert frozen.failed == []
    assert len(snapshot.documents) == 1
    assert snapshot.documents[0].metadata.project == '/other'
    assert snapshot.capture_metadata == {'source': 'local-sessions', 'sessions': 1}
    assert any('duplicate' in message for message in warnings)


def test_snapshot_reports_broken_ref_and_loads_rest(claude_projects: Path, warnings: list[str]) -> None:
    good = _claude(claude_projects, 'good')
    broken = SessionRef(source='claude-code', path=claude_projects / '-work-proj' / 'gone.jsonl')
    frozen = build_session_snapshot([SessionRef(source='claude-code', path=good), broken])
    assert frozen.n_sessions == 1
    assert [ref for ref, _ in frozen.failed] == [broken]
    assert 'not a file' in frozen.failed[0][1]
    assert not Snapshot.model_validate_json(frozen.data).is_empty


def test_snapshot_all_broken_raises(claude_projects: Path) -> None:
    broken = SessionRef(source='claude-code', path=claude_projects / 'gone.jsonl')
    with pytest.raises(SessionLoadError, match='none of the 1 selected sessions'):
        build_session_snapshot([broken])


def test_snapshot_rejects_too_many(claude_projects: Path) -> None:
    ref = SessionRef(source='claude-code', path=claude_projects / 'gone.jsonl')
    with pytest.raises(SessionLoadError, match=str(MAX_SELECTED_SESSIONS)):
        build_session_snapshot([ref] * (MAX_SELECTED_SESSIONS + 1))


def test_registry_has_every_family() -> None:
    assert set(READERS) == {'claude', 'codex', 'omp'}


def test_resolve_nul_byte_path_raises_session_load_error(claude_projects: Path) -> None:
    ref = SessionRef.model_construct(source='claude-code', path=claude_projects / 'a\x00b.jsonl')
    with pytest.raises(SessionLoadError):
        resolve_session_ref(ref)


def test_snapshot_reports_summarize_oserror_and_loads_rest(
    claude_projects: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = _claude(claude_projects, 'good')
    flaky = _claude(claude_projects, 'flaky')
    original = CLAUDE.summarize

    def summarize(path: Path) -> Any:
        if path.name == 'flaky.jsonl':
            raise OSError('gone')
        return original(path)

    monkeypatch.setattr(CLAUDE, 'summarize', summarize)
    refs = [SessionRef(source='claude-code', path=flaky), SessionRef(source='claude-code', path=good)]
    frozen = build_session_snapshot(refs)
    assert frozen.n_sessions == 1
    assert [ref for ref, _ in frozen.failed] == [refs[0]]
    assert 'OSError' in frozen.failed[0][1]
    assert 'gone' not in frozen.failed[0][1]


def test_ascii_needle_mixed_case_matches_through_bytes_path(claude_projects: Path) -> None:
    _claude(claude_projects, 'm', text='Please Fix The ParSer')
    assert _ids(search_sessions(SessionQuery(text='fix the PARSER'))) == {'m'}


def test_non_ascii_uppercase_needle_matches(claude_projects: Path) -> None:
    _claude(claude_projects, 'u', text='CAFÉ time')
    assert _ids(search_sessions(SessionQuery(text='café'))) == {'u'}
    assert _ids(search_sessions(SessionQuery(text='CAFÉ'))) == {'u'}


@pytest.mark.parametrize('needle', ['boundaryneedle', 'boundaryneedle é'])
def test_needle_spanning_chunk_boundary_is_found(
    claude_projects: Path, monkeypatch: pytest.MonkeyPatch, needle: str
) -> None:
    monkeypatch.setattr('evaluatorq.local_sessions.search._CHUNK_SIZE', 16)
    _claude(claude_projects, 'b', text=f'padding padding {needle} tail')
    assert _ids(search_sessions(SessionQuery(text=needle))) == {'b'}


def test_cancel_during_multi_chunk_file_stops_search(
    claude_projects: Path, monkeypatch: pytest.MonkeyPatch, warnings: list[str]
) -> None:
    monkeypatch.setattr('evaluatorq.local_sessions.search._CHUNK_SIZE', 16)
    _claude(claude_projects, 'big', text='x' * 400)
    calls = {'n': 0}

    def cancelled() -> bool:
        calls['n'] += 1
        return calls['n'] > 3

    result = search_sessions(SessionQuery(text='neverpresent'), cancelled=cancelled)
    assert result.complete is False
    assert result.sessions == ()
    assert any('Session search stopped' in message for message in warnings)


def test_loading_a_session_document_summarizes_it_once(claude_projects: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _claude(claude_projects, 'once')
    calls: list[Path] = []
    original = CLAUDE.summarize
    monkeypatch.setattr(CLAUDE, 'summarize', lambda p: (calls.append(p), original(p))[1])

    load_session_document(SessionRef(source='claude-code', path=path))

    assert len(calls) == 1


def test_searching_with_text_summarizes_a_match_once(claude_projects: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _claude(claude_projects, 'once', text='needle here')
    calls: list[Path] = []
    original = CLAUDE.summarize
    monkeypatch.setattr(CLAUDE, 'summarize', lambda p: (calls.append(p), original(p))[1])

    assert _ids(search_sessions(SessionQuery(text='needle'))) == {'once'}
    assert len(calls) == 1


def test_a_list_leaf_uuid_does_not_abort_search_or_snapshot(claude_projects: Path) -> None:
    bad = _claude(claude_projects, 'bad', text='needle', extra=[{'type': 'last-prompt', 'leafUuid': ['a1']}])
    _claude(claude_projects, 'good', text='needle')

    result = search_sessions(SessionQuery(text='needle'))
    frozen = build_session_snapshot([SessionRef(source='claude-code', path=bad)])

    assert 'good' in _ids(result)
    assert frozen.failed == [] and frozen.n_sessions == 1


def test_a_non_value_error_from_a_mapper_becomes_a_session_load_error(
    claude_projects: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _claude(claude_projects, 'boom', text='needle')

    def explode(_records: Any) -> Any:
        raise TypeError('secret user words')

    monkeypatch.setattr(claude_module, '_map_records', explode)

    with pytest.raises(SessionLoadError, match=r'unreadable session \(TypeError\)') as caught:
        load_session_document(SessionRef(source='claude-code', path=path))
    assert 'secret' not in str(caught.value)
    assert _ids(search_sessions(SessionQuery(text='needle'))) == set()
    snapshot_error = pytest.raises(SessionLoadError, build_session_snapshot, [SessionRef(source='claude-code', path=path)])
    assert snapshot_error.match('none of the 1 selected sessions')


def test_snapshot_budget_raises_as_soon_as_it_is_passed(claude_projects: Path) -> None:
    refs = [SessionRef(source='claude-code', path=_claude(claude_projects, sid)) for sid in ('a', 'b')]

    with pytest.raises(SnapshotTooLarge):
        build_session_snapshot(refs, max_bytes=10)
    frozen = build_session_snapshot(refs, max_bytes=10_000_000)
    assert frozen.n_sessions == 2
    assert len(Snapshot.model_validate_json(frozen.data).documents) == 2


def test_the_budget_counts_the_whole_snapshot_not_only_its_documents(claude_projects: Path) -> None:
    refs = [SessionRef(source='claude-code', path=_claude(claude_projects, 'only'))]
    size = len(build_session_snapshot(refs).data)
    document_bytes = len(load_session_document(refs[0]).model_dump_json().encode())

    assert build_session_snapshot(refs, max_bytes=size).n_sessions == 1
    with pytest.raises(SnapshotTooLarge):
        build_session_snapshot(refs, max_bytes=size - 1)
    assert document_bytes < size - 1, 'the envelope is what the second case rejects'


def test_the_frozen_bytes_are_what_the_snapshot_model_would_write(claude_projects: Path) -> None:
    refs = [
        SessionRef(source='claude-code', path=_claude(claude_projects, sid, text=text))
        for sid, text in (('a', 'plain'), ('b', 'non-ascii é “quoted” \u2028 line'))
    ]

    frozen = build_session_snapshot(refs)

    documents = tuple(load_session_document(ref) for ref in refs)
    expected = Snapshot(
        traces=(), documents=documents, capture_metadata={'source': 'local-sessions', 'sessions': 2}
    ).model_dump_json()
    assert frozen.data == expected.encode()
    assert Snapshot.model_validate_json(frozen.data).documents == documents


def test_cancel_during_candidate_collection_returns_an_incomplete_search(claude_projects: Path) -> None:
    for sid in ('a', 'b', 'c'):
        _claude(claude_projects, sid)
    checks = iter(range(100))

    result = search_sessions(SessionQuery(), cancelled=lambda: next(checks) >= 1)

    assert result.complete is False
    assert result.candidate_files == 1, 'the walk stopped after the first file instead of listing all three'
    assert result.scanned_files == 0
    assert result.sessions == ()


def test_day_window_is_half_open_in_the_given_zone() -> None:
    west = timezone(timedelta(hours=-5))

    start, end = day_window(date(2026, 10, 2), date(2026, 10, 2), west)

    assert start == datetime(2026, 10, 2, 5, tzinfo=timezone.utc)
    assert end == datetime(2026, 10, 3, 5, tzinfo=timezone.utc)
    assert day_window(None, None, west) == (None, None)
    assert day_window(date(2026, 10, 2), None, west) == (start, None)


def test_day_window_without_a_zone_reads_days_in_the_machine_zone() -> None:
    start, end = day_window(date(2026, 10, 2), date(2026, 10, 2), None)

    assert start == datetime(2026, 10, 2).astimezone()
    assert end == datetime(2026, 10, 3).astimezone()


def test_day_window_rejects_a_reversed_range_and_an_out_of_range_day() -> None:
    with pytest.raises(ValueError, match='first day must not be after the last day'):
        day_window(date(2026, 10, 5), date(2026, 10, 1), timezone.utc)
    with pytest.raises(ValueError, match='out of range'):
        day_window(None, date.max, timezone.utc)
    assert day_window(date(2026, 10, 1), date(2026, 10, 1), timezone.utc)[0] is not None
