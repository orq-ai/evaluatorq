"""Atomic owner-only files in a validated destination directory."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


def _require_owned_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISDIR(info.st_mode) or (hasattr(os, 'getuid') and info.st_uid != os.getuid()):
        raise OSError(f'Private file storage is not an owned directory: {path}')


def ensure_private_directory(path: Path) -> None:
    """Create `path`, or take one this user already owns, and set it to mode 0o700.

    The mode is set explicitly rather than left to `mkdir`, because the process umask
    decides what `mkdir` grants: under umask 002 a new folder is group-writable, and the
    readers that check storage privacy reject it. Raises `OSError` for a symlink, a
    non-directory, or a directory owned by another user.
    """
    _require_owned_directory(path)
    path.chmod(0o700)


def write_private_atomic(path: Path, contents: str | bytes, *, private_directory: bool = True) -> None:
    """Replace a file atomically with mode 0o600, in an owned, non-symlink directory.

    Application storage defaults to mode 0o700. Explicit user exports may set
    `private_directory=False` to preserve an existing destination directory's permissions.

    Windows access control comes from the containing directory's inherited ACL;
    POSIX permission bits do not enforce access control there.
    """
    if private_directory:
        ensure_private_directory(path.parent)
    else:
        _require_owned_directory(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.stem}.', suffix='.tmp', dir=path.parent)
    try:
        try:
            handle = os.fdopen(descriptor, 'wb')
        except BaseException:
            os.close(descriptor)
            raise
        with handle:
            if os.name != 'nt':
                os.fchmod(handle.fileno(), 0o600)
            handle.write(contents.encode('utf-8') if isinstance(contents, str) else contents)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    finally:
        Path(temporary).unlink(missing_ok=True)
