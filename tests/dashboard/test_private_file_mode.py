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
