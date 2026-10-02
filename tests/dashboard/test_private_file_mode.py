# ruff: noqa: S101

from __future__ import annotations

import os
import stat
from typing import TYPE_CHECKING

import pytest

from evaluatorq.dashboard import insights_launch

if TYPE_CHECKING:
    from pathlib import Path


def test_private_file_mode_uses_fchmod_on_posix(tmp_path: Path) -> None:
    if os.name == 'nt':
        pytest.skip('Windows access control is provided by inherited directory ACLs.')

    path = tmp_path / 'private.json'
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
    try:
        insights_launch._set_private_file_mode(descriptor, 0o600)  # noqa: SLF001
        assert stat.S_IMODE(os.fstat(descriptor).st_mode) == 0o600
    finally:
        os.close(descriptor)


def test_private_file_mode_does_not_require_fchmod_on_windows(monkeypatch) -> None:
    monkeypatch.setattr(insights_launch.os, 'name', 'nt')
    monkeypatch.delattr(insights_launch.os, 'fchmod', raising=False)

    insights_launch._set_private_file_mode(1, 0o600)  # noqa: SLF001
