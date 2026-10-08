"""Atomic writes of files only their owner can read."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_private_atomic(path: Path, data: bytes | str) -> None:
    """Replace `path` with `data` atomically; the file is owner-only (0600) from creation and fsynced before the swap.

    The temp file lives in the target directory so `os.replace` stays on one filesystem, and is removed if the write
    fails. Text is written as UTF-8. The directory must already exist.
    """
    payload = data.encode('utf-8') if isinstance(data, str) else data
    descriptor, temp_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
