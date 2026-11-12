# ruff: noqa: S101

from __future__ import annotations

import os
import stat
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from evaluatorq.common import private_files

if TYPE_CHECKING:
    from pathlib import Path


def test_write_private_atomic_sets_owner_only_permissions_on_posix(tmp_path: Path) -> None:
    if os.name == 'nt':
        pytest.skip('Windows access control is provided by inherited directory ACLs.')

    path = tmp_path / 'private.json'
    private_files.write_private_atomic(path, '{}')

    assert path.read_text() == '{}'
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_write_private_atomic_does_not_require_fchmod_on_windows(tmp_path: Path, monkeypatch) -> None:
    os_proxy = {
        'fdopen': os.fdopen,
        'close': os.close,
        'fsync': os.fsync,
    }
    if hasattr(os, 'getuid'):
        os_proxy['getuid'] = os.getuid
    monkeypatch.setattr(private_files, 'os', SimpleNamespace(name='nt', **os_proxy))

    path = tmp_path / 'private.json'
    private_files.write_private_atomic(path, '{}')

    assert path.read_text() == '{}'


@pytest.mark.skipif(os.name == 'nt', reason='POSIX permission bits do not control access on Windows')
def test_ensure_private_directory_tightens_an_existing_group_writable_folder(tmp_path: Path) -> None:
    folder = tmp_path / 'store'
    folder.mkdir()
    folder.chmod(0o775)

    private_files.ensure_private_directory(folder)

    assert stat.S_IMODE(folder.stat().st_mode) == 0o700


def test_ensure_private_directory_rejects_a_symlink(tmp_path: Path) -> None:
    target = tmp_path / 'elsewhere'
    target.mkdir()
    link = tmp_path / 'store'
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip('Creating symlinks is not permitted here.')

    with pytest.raises(OSError, match='is a symlink'):
        private_files.ensure_private_directory(link)


def test_ensure_private_directory_rejects_a_file(tmp_path: Path) -> None:
    path = tmp_path / 'store'
    path.write_text('not a folder')

    with pytest.raises(OSError, match='is not a directory'):
        private_files.ensure_private_directory(path)


@pytest.mark.skipif(os.name == 'nt', reason='POSIX permission bits do not control access on Windows')
def test_write_private_atomic_leaves_the_folder_mode_alone_for_a_user_export(tmp_path: Path) -> None:
    folder = tmp_path / 'exports'
    folder.mkdir()
    folder.chmod(0o755)

    private_files.write_private_atomic(folder / 'sessions.json', '{}', private_directory=False)

    assert stat.S_IMODE(folder.stat().st_mode) == 0o755
    assert stat.S_IMODE((folder / 'sessions.json').stat().st_mode) == 0o600
