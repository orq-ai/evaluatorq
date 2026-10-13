"""Strict advisory file locks shared by process and thread callers."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator  # noqa: TC003
from contextlib import contextmanager
from pathlib import Path  # noqa: TC003

_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.Lock] = {}


def _thread_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.Lock())


@contextmanager
def strict_file_lock(path: Path, *, thread_lock: threading.Lock | None = None) -> Iterator[None]:
    """Hold a permanent lock file across threads and processes; fail if unavailable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with thread_lock or _thread_lock(path):
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, 'O_NOFOLLOW'):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            if hasattr(os, 'fchmod'):
                os.fchmod(descriptor, 0o600)
            try:
                import fcntl
            except ImportError:
                try:
                    import msvcrt
                except ImportError as exc:
                    raise OSError('Inter-process file locking is unavailable') from exc
                # msvcrt can lock byte ranges beyond EOF; do not write a sentinel that races with peers.
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
