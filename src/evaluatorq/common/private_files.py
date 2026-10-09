"""Atomic owner-only files in a validated destination directory."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


def write_private_atomic(path: Path, contents: str | bytes, *, private_directory: bool = True) -> None:
    """Replace a file atomically with mode 0o600, in an owned, non-symlink directory.

    Application storage defaults to mode 0o700. Explicit user exports may set
    `private_directory=False` to preserve an existing destination directory's permissions.

    Windows access control comes from the containing directory's inherited ACL;
    POSIX permission bits do not enforce access control there.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory_info = path.parent.lstat()
    if (
        path.parent.is_symlink()
        or not stat.S_ISDIR(directory_info.st_mode)
        or (hasattr(os, 'getuid') and directory_info.st_uid != os.getuid())
    ):
        raise OSError(f'Private file storage is not an owned directory: {path.parent}')
    if private_directory:
        path.parent.chmod(0o700)
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
