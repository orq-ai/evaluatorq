"""Private, validated source uploads for dashboard Insights runs."""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import time
import uuid
from typing import TYPE_CHECKING, Literal, cast

from pydantic import ValidationError

from evaluatorq.trace_finder.export import RunExport
from evaluatorq.trace_finder.models import Snapshot

if TYPE_CHECKING:
    from pathlib import Path

    from starlette.datastructures import UploadFile
    from starlette.requests import Request

MAX_INSIGHTS_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_FINDER_EXPORT_BYTES = 10 * 1024 * 1024
MAX_MULTIPART_OVERHEAD_BYTES = 1024 * 1024
ABANDONED_UPLOAD_TTL_SECONDS = 24 * 60 * 60
_UPLOAD_DIRECTORY = '.uploads'
UploadKind = Literal['finder', 'snapshot']
_UPLOAD_NAME = re.compile(r'^(finder|snapshot)-[0-9a-f]{32}\.json$')


def uploads_dir(runs_dir: Path) -> Path:
    """Return the private directory used only for validated Insights uploads."""
    return runs_dir / _UPLOAD_DIRECTORY


def cleanup_expired_uploads(runs_dir: Path, *, now: float | None = None) -> int:
    """Remove stale route-owned uploads that were never consumed by a worker."""
    directory = uploads_dir(runs_dir)
    if directory.is_symlink() or not directory.exists():
        return 0
    _ensure_private_directory(directory)
    cutoff = (time.time() if now is None else now) - ABANDONED_UPLOAD_TTL_SECONDS
    removed = 0
    for path in directory.iterdir():
        if _UPLOAD_NAME.fullmatch(path.name) is None or path.is_symlink():
            continue
        try:
            info = path.lstat()
            if stat.S_ISREG(info.st_mode) and info.st_mtime < cutoff:
                path.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed


class UploadTooLargeError(ValueError):
    """An upload body over its kind's size limit; the route answers 413 instead of 422."""


class UploadRequestTooLargeError(Exception):
    """Raised as multipart request bytes exceed the bounded parser allowance."""


def limit_request_body(request: Request) -> None:
    """Guard ASGI body consumption before Starlette's multipart parser spools it."""
    receive = request._receive  # noqa: SLF001 — Starlette exposes no receive wrapper hook.
    limit = MAX_INSIGHTS_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD_BYTES
    consumed = 0

    async def limited_receive() -> dict[str, object]:
        nonlocal consumed
        message = await receive()
        if message.get('type') == 'http.request':
            body = message.get('body', b'')
            consumed += len(body)
            if consumed > limit:
                raise UploadRequestTooLargeError('Upload exceeds the 100 MB size limit.')
        return message

    request._receive = limited_receive  # noqa: SLF001 — install the bounded ASGI receive wrapper.


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISDIR(info.st_mode):
        raise OSError('Insights upload storage must be a real directory.')
    if hasattr(os, 'getuid') and info.st_uid != os.getuid():
        raise OSError('Insights upload storage has unexpected ownership.')
    path.chmod(0o700)


def upload_size_error(kind: UploadKind, size: int) -> str | None:
    """The size-limit message for an upload of `kind`, or `None` when it fits."""
    if kind == 'finder' and size > MAX_FINDER_EXPORT_BYTES:
        return f'Finder export exceeds the {MAX_FINDER_EXPORT_BYTES // (1024 * 1024)} MiB size limit.'
    if size > MAX_INSIGHTS_UPLOAD_BYTES:
        return 'Upload exceeds the 100 MB size limit.'
    return None


def validate_upload(contents: bytes, kind: UploadKind | None = None) -> UploadKind:
    """Parse an upload once, classify it by its top-level keys unless `kind` is given, and validate it.

    Raises `UploadTooLargeError` above the kind's size limit and `ValueError` for any other invalid upload.
    """
    try:
        document = json.loads(contents)
    except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError are both ValueError
        raise ValueError('This file is not valid JSON.') from exc
    if kind is None:
        keys = set(document) if isinstance(document, dict) else set()
        if 'matched_trace_ids' in keys:  # a Finder export also carries `traces`, so this key decides
            kind = 'finder'
        elif 'traces' in keys:
            kind = 'snapshot'
        else:
            raise ValueError('This file is neither a Finder export nor a trace snapshot.')
    too_large = upload_size_error(kind, len(contents))
    if too_large:
        raise UploadTooLargeError(too_large)
    del document  # validate in JSON mode, the way the worker and launch path read the stored file
    try:
        if kind == 'finder':
            RunExport.model_validate_json(contents)
        elif not Snapshot.model_validate_json(contents).traces:
            raise ValueError('The trace snapshot contains no traces.')
    except ValidationError as exc:
        raise ValueError(f'Upload is not a valid {kind} JSON file.') from exc
    return kind


def store_upload(runs_dir: Path, contents: bytes, kind: UploadKind) -> Path:
    """Atomically store an upload that `validate_upload` accepted, under a route-owned random name."""
    directory = uploads_dir(runs_dir)
    _ensure_private_directory(directory)
    path = directory / f'{kind}-{uuid.uuid4().hex}.json'
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def read_uploaded_source(runs_dir: Path, path: Path, kind: UploadKind) -> bytes:
    """Read only a route-named regular upload from the private Insights directory."""
    directory = uploads_dir(runs_dir)
    if path.is_symlink() or directory.is_symlink():
        raise ValueError('Uploaded source must be inside private Insights upload storage.')
    try:
        directory = directory.resolve()
        path = path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError('Uploaded source must be inside private Insights upload storage.') from exc
    if path.parent != directory:
        raise ValueError('Uploaded source must be inside private Insights upload storage.')
    match = _UPLOAD_NAME.fullmatch(path.name)
    if match is None or match.group(1) != kind:
        raise ValueError('Uploaded source has the wrong file type.')
    directory_info = directory.lstat()
    if (
        not stat.S_ISDIR(directory_info.st_mode)
        or (os.name != 'nt' and stat.S_IMODE(directory_info.st_mode) != 0o700)
        or (hasattr(os, 'getuid') and directory_info.st_uid != os.getuid())
    ):
        raise ValueError('Insights upload storage has unsafe permissions.')
    directory_fd: int | None = None
    descriptor: int | None = None
    try:
        if os.name == 'nt':
            from evaluatorq.dashboard.insights_launch import _open_windows_approved_regular_file

            descriptor, _identity = _open_windows_approved_regular_file(directory, path, description='Insights upload')
        else:
            if not hasattr(os, 'O_DIRECTORY') or not hasattr(os, 'O_NOFOLLOW'):
                raise OSError('Safe Insights upload opening is unavailable on this platform.')
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            directory_opened = os.fstat(directory_fd)
            if (
                not stat.S_ISDIR(directory_opened.st_mode)
                or directory_opened.st_uid != os.getuid()
                or stat.S_IMODE(directory_opened.st_mode) != 0o700
            ):
                raise ValueError('Insights upload storage changed while opening the file.')
            descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or (
            os.name != 'nt' and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600)
        ):
            raise ValueError('Uploaded source must be a private regular file.')
        with os.fdopen(descriptor, 'rb') as handle:
            descriptor = None
            contents = handle.read(MAX_INSIGHTS_UPLOAD_BYTES + 1)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_fd is not None:
            os.close(directory_fd)
    validate_upload(contents, kind)
    return contents


def upload_kind(path: Path) -> UploadKind | None:
    """The kind a stored upload's name records, or `None` for a file this module did not name."""
    match = _UPLOAD_NAME.fullmatch(path.name)
    return cast('UploadKind', match.group(1)) if match else None


def is_uploaded_source(runs_dir: Path, path: Path) -> bool:
    """Return whether a path has the unambiguous route-owned upload shape."""
    directory = uploads_dir(runs_dir)
    if path.is_symlink() or directory.is_symlink() or _UPLOAD_NAME.fullmatch(path.name) is None:
        return False
    try:
        return path.resolve().parent == directory.resolve()
    except (OSError, RuntimeError):
        return False


def cleanup_uploaded_source(runs_dir: Path, path: Path) -> None:
    """Delete a valid upload file after worker consumption, leaving other files alone."""
    if not is_uploaded_source(runs_dir, path):
        return
    try:
        read_uploaded_source(runs_dir, path, upload_kind(path) or 'snapshot')
        path.unlink(missing_ok=True)
    except (FileNotFoundError, OSError, ValueError):
        return


async def receive_upload(upload: UploadFile) -> tuple[UploadKind, bytes]:
    """Read an UploadFile in bounded chunks, then classify and validate it off the event loop."""
    read = upload.read
    chunks = bytearray()
    while True:
        chunk = await read(min(1024 * 1024, MAX_INSIGHTS_UPLOAD_BYTES + 1 - len(chunks)))
        if not chunk:
            break
        chunks.extend(chunk)
        if len(chunks) > MAX_INSIGHTS_UPLOAD_BYTES:
            raise UploadTooLargeError('Upload exceeds the 100 MB size limit.')
    contents = bytes(chunks)
    return await asyncio.to_thread(validate_upload, contents), contents
