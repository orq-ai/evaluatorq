from __future__ import annotations

import builtins
import os
from types import SimpleNamespace
from pathlib import Path

import pytest

from evaluatorq.common import file_lock


def _mock_windows_locking(monkeypatch: pytest.MonkeyPatch, locking):
    real_import = builtins.__import__
    monkeypatch.setattr(file_lock.os, 'write', lambda *args, **kwargs: pytest.fail('lock must not write a sentinel'))

    def import_windows_lock(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'fcntl':
            raise ImportError('fcntl is unavailable')
        if name == 'msvcrt':
            return SimpleNamespace(LK_LOCK=1, LK_UNLCK=2, locking=locking)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, '__import__', import_windows_lock)


def test_windows_lock_uses_empty_byte_range_and_unlocks_when_body_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[int, int, int]] = []

    def locking(descriptor: int, mode: int, length: int) -> None:
        calls.append((descriptor, mode, length))

    _mock_windows_locking(monkeypatch, locking)
    lock_path = tmp_path / '.settings.lock'

    with pytest.raises(RuntimeError, match='body failed'):
        with file_lock.strict_file_lock(lock_path):
            raise RuntimeError('body failed')

    assert len(calls) == 2
    descriptor = calls[0][0]
    assert calls == [(descriptor, 1, 1), (descriptor, 2, 1)]
    with pytest.raises(OSError):
        os.fstat(descriptor)


def test_windows_lock_acquisition_failure_closes_descriptor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    descriptors: list[int] = []

    def fail_lock(descriptor: int, mode: int, length: int) -> None:
        descriptors.append(descriptor)
        raise OSError('simulated Windows lock failure')

    _mock_windows_locking(monkeypatch, fail_lock)

    with pytest.raises(OSError, match='simulated Windows lock failure'):
        with file_lock.strict_file_lock(tmp_path / '.settings.lock'):
            pytest.fail('transaction must not run without the inter-process lock')

    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
