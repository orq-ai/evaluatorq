from __future__ import annotations

# ruff: noqa: S101
import errno
import os
from pathlib import Path

import pytest

from evaluatorq.insights import store
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
    assert path.is_symlink()
    assert path.read_text(encoding='utf-8') == minimal_run.model_dump_json(indent=2) + '\n'


def test_fallback_symlink_is_readable_with_relative_runs_dir(tmp_path: Path, minimal_run, monkeypatch) -> None:
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
    original_symlink = Path.symlink_to
    raced_path = tmp_path / 'insights_20260901-000000_minimal-run.json'
    claimed = False

    def claim_before_symlink(path: Path, target: str, target_is_directory: bool = False) -> None:
        nonlocal claimed
        if Path(path) == raced_path and not claimed:
            claimed = True
            raced_path.write_text('claimed by another process', encoding='utf-8')
        original_symlink(path, target, target_is_directory)

    monkeypatch.setattr(Path, 'symlink_to', claim_before_symlink)

    saved = save_run(minimal_run, tmp_path)

    assert claimed
    assert raced_path.read_text(encoding='utf-8') == 'claimed by another process'
    assert saved.name == 'insights_20260901-000000_minimal-run-2.json'
    assert load_run(saved) == minimal_run


def test_fallback_never_exposes_json_until_atomic_publication(tmp_path: Path, minimal_run, monkeypatch) -> None:
    def unsupported_link(source: Path, destination: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, 'hard links are unsupported')

    monkeypatch.setattr(os, 'link', unsupported_link)
    original_symlink = Path.symlink_to
    observations: list[list[Path]] = []

    def inspect_before_publish(path: Path, target: str, target_is_directory: bool = False) -> None:
        observations.append(list_run_paths(tmp_path))
        original_symlink(path, target, target_is_directory)

    monkeypatch.setattr(Path, 'symlink_to', inspect_before_publish)
    path = save_run(minimal_run, tmp_path)

    assert observations == [[]]
    assert load_run(path) == minimal_run
    assert path.resolve().exists()


def test_save_run_fallback_fails_cleanly_when_symlinks_are_unavailable(
    tmp_path: Path, minimal_run, monkeypatch
) -> None:
    def unsupported_link(source: Path, destination: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, 'hard links are unsupported')

    monkeypatch.setattr(os, 'link', unsupported_link)
    def unsupported_symlink(target: str, target_is_directory: bool = False) -> None:
        raise OSError(errno.EOPNOTSUPP, 'symlinks are unsupported')

    monkeypatch.setattr(Path, 'symlink_to', unsupported_symlink)

    with pytest.raises(OSError, match='symlinks are unsupported'):
        save_run(minimal_run, tmp_path)

    assert list_run_paths(tmp_path) == []


def test_windows_exclusive_rename_fallback_handles_collision_without_symlinks(
    tmp_path: Path, minimal_run, monkeypatch
) -> None:
    def unsupported_link(*_args) -> None:
        raise OSError(errno.EOPNOTSUPP, 'no hard links')

    monkeypatch.setattr(os, 'link', unsupported_link)
    monkeypatch.setattr(store, '_rename_is_exclusive', lambda: True)
    monkeypatch.setattr(Path, 'symlink_to', lambda *_args: pytest.fail('Windows rename needs no symlink'))
    original_rename = Path.rename

    def exclusive_rename(source: Path, target: Path) -> Path:
        if target.exists():
            raise FileExistsError(target)
        return original_rename(source, target)

    monkeypatch.setattr(Path, 'rename', exclusive_rename)
    first = save_run(minimal_run, tmp_path)
    second = save_run(minimal_run.model_copy(update={'run_id': 'run-2'}), tmp_path)

    assert first.name == 'insights_20260901-000000_minimal-run.json'
    assert second.name == 'insights_20260901-000000_minimal-run-2.json'
    assert not first.is_symlink() and not second.is_symlink()
    assert load_run(first).run_id == 'run-1'
    assert load_run(second).run_id == 'run-2'


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
