from __future__ import annotations

# ruff: noqa: S101
import errno
import os
from pathlib import Path

import pytest

from evaluatorq.insights.store import list_run_paths, list_runs, load_run, save_run


def test_round_trip_and_truncated_file(minimal_run, tmp_path: Path) -> None:
    path = save_run(minimal_run, tmp_path)
    assert path.name.startswith('insights_20260901-000000_minimal-run')
    assert load_run(path) == minimal_run

    broken = tmp_path / 'insights_20260902-broken.json'
    broken.write_text('{truncated', encoding='utf-8')
    listed = list_runs(tmp_path)
    assert [p for p, _ in listed] == [broken, path]
    assert isinstance(listed[0][1], str)
    assert isinstance(listed[1][1], type(minimal_run))


def test_same_name_and_second_never_overwrites(tmp_path: Path, minimal_run) -> None:
    first = save_run(minimal_run, tmp_path)
    second_run = minimal_run.model_copy(update={'run_id': 'run-2'})
    second = save_run(second_run, tmp_path)

    assert first != second
    assert first.name == 'insights_20260901-000000_minimal-run.json'
    assert second.name == 'insights_20260901-000000_minimal-run-2.json'
    assert load_run(first).run_id == 'run-1'
    assert load_run(second).run_id == 'run-2'


def test_save_run_falls_back_when_hard_links_are_unsupported(tmp_path: Path, minimal_run, monkeypatch) -> None:
    def unsupported_link(source: Path, destination: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, 'hard links are unsupported')

    monkeypatch.setattr(os, 'link', unsupported_link)

    def unexpected_replace(self: Path, destination: Path) -> Path:
        raise AssertionError('fallback must never replace a claimed filename')

    monkeypatch.setattr(Path, 'replace', unexpected_replace)

    path = save_run(minimal_run, tmp_path)

    assert load_run(path) == minimal_run
    assert not path.is_symlink()


def test_fallback_copy_is_readable_with_relative_runs_dir(tmp_path: Path, minimal_run, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    def unsupported_link(source: Path, destination: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, 'hard links are unsupported')

    monkeypatch.setattr(os, 'link', unsupported_link)

    def unexpected_replace(self: Path, destination: Path) -> Path:
        raise AssertionError('fallback must never replace a claimed filename')

    monkeypatch.setattr(Path, 'replace', unexpected_replace)

    path = save_run(minimal_run, Path('runs'))
    assert load_run(path) == minimal_run


def test_save_run_fallback_does_not_overwrite_name_claimed_during_publication(
    tmp_path: Path, minimal_run, monkeypatch
) -> None:
    def unsupported_link(source: Path, destination: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, 'hard links are unsupported')

    monkeypatch.setattr(os, 'link', unsupported_link)
    original_open = os.open
    raced_path = tmp_path / 'insights_20260901-000000_minimal-run.json'
    claimed = False

    def claim_before_exclusive_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal claimed
        if Path(path) == raced_path and not claimed:
            claimed = True
            fd = original_open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666, dir_fd=dir_fd)
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                handle.write('claimed by another process')
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, 'open', claim_before_exclusive_open)

    saved = save_run(minimal_run, tmp_path)

    assert claimed
    assert raced_path.read_text(encoding='utf-8') == 'claimed by another process'
    assert saved.name == 'insights_20260901-000000_minimal-run-2.json'
    assert load_run(saved) == minimal_run


def test_save_run_fallback_cleanup_preserves_replacement_file(tmp_path: Path, minimal_run, monkeypatch) -> None:
    def unsupported_link(source: Path, destination: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, 'hard links are unsupported')

    monkeypatch.setattr(os, 'link', unsupported_link)
    original_fsync = os.fsync
    final_path = tmp_path / 'insights_20260901-000000_minimal-run.json'
    calls = 0

    def replace_reserved_file_before_failure(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            final_path.unlink()
            final_path.write_text('replacement from another process', encoding='utf-8')
            raise OSError(errno.EIO, 'simulated copy failure')
        original_fsync(descriptor)

    monkeypatch.setattr(os, 'fsync', replace_reserved_file_before_failure)

    with pytest.raises(OSError, match='simulated copy failure'):
        save_run(minimal_run, tmp_path)

    assert final_path.read_text(encoding='utf-8') == 'replacement from another process'


def test_save_run_fallback_closes_descriptor_when_fdopen_fails(tmp_path: Path, minimal_run, monkeypatch) -> None:
    def unsupported_link(source: Path, destination: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, 'hard links are unsupported')

    monkeypatch.setattr(os, 'link', unsupported_link)
    reserved_descriptors: list[int] = []

    def fail_fdopen(descriptor: int, *args, **kwargs):
        reserved_descriptors.append(descriptor)
        raise OSError(errno.EMFILE, 'simulated fdopen failure')

    monkeypatch.setattr(os, 'fdopen', fail_fdopen)

    with pytest.raises(OSError, match='simulated fdopen failure'):
        save_run(minimal_run, tmp_path)

    assert len(reserved_descriptors) == 1
    with pytest.raises(OSError, match='Bad file descriptor'):
        os.fstat(reserved_descriptors[0])
    assert list_run_paths(tmp_path) == []


def test_save_run_fallback_does_not_overwrite_collision(tmp_path: Path, minimal_run, monkeypatch) -> None:
    original = tmp_path / 'insights_20260901-000000_minimal-run.json'
    original.write_text('keep this file', encoding='utf-8')

    def unsupported_link(source: Path, destination: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, 'hard links are unsupported')

    monkeypatch.setattr(os, 'link', unsupported_link)

    saved = save_run(minimal_run, tmp_path)

    assert original.read_text(encoding='utf-8') == 'keep this file'
    assert saved.name == 'insights_20260901-000000_minimal-run-2.json'
    assert load_run(saved) == minimal_run


def test_save_run_uses_exclusive_copy_when_links_are_unavailable(tmp_path: Path, minimal_run, monkeypatch) -> None:
    def unsupported_link(*args) -> None:
        raise OSError(errno.EOPNOTSUPP, 'links are unsupported')

    monkeypatch.setattr(os, 'link', unsupported_link)
    monkeypatch.setattr(os, 'symlink', unsupported_link)

    path = save_run(minimal_run, tmp_path)

    assert load_run(path) == minimal_run


def test_save_run_does_not_hide_unrelated_link_errors(tmp_path: Path, minimal_run, monkeypatch) -> None:
    def permission_denied(*args) -> None:
        raise OSError(errno.ENOSPC, 'disk is full')

    monkeypatch.setattr(os, 'link', permission_denied)

    with pytest.raises(OSError, match='disk is full'):
        save_run(minimal_run, tmp_path)
    assert list_run_paths(tmp_path) == []


def test_list_run_paths_is_newest_first_and_missing_directory_is_empty(tmp_path: Path) -> None:
    older = tmp_path / 'insights_older.json'
    newer = tmp_path / 'insights_newer.json'
    older.write_text('{}', encoding='utf-8')
    newer.write_text('{}', encoding='utf-8')
    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))

    assert list_run_paths(tmp_path) == [newer, older]
    assert list_run_paths(tmp_path / 'missing') == []


def test_list_run_paths_keeps_listing_failure_non_fatal(tmp_path: Path, monkeypatch) -> None:
    def fail_glob(self, pattern):
        raise OSError('directory unavailable')

    monkeypatch.setattr(Path, 'glob', fail_glob)

    assert list_run_paths(tmp_path) == []
