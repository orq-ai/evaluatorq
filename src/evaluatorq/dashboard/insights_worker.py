"""Detached worker for Insights runs started from the dashboard."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from evaluatorq.common.orq_client import close_orq_client
from evaluatorq.common.run_manifest import fail_if_running
from evaluatorq.contracts import RunManifest
from evaluatorq.dashboard.auth import auth_identity, build_auth_clients, resolve_dashboard_auth
from evaluatorq.dashboard.insights_launch import (
    _MANIFEST_ENV,
    _SNAPSHOT_ENV,
    InsightsLaunchPayload,
    _open_windows_approved_regular_file,
    _windows_directory_identity,
    finder_export_reference_path,
    read_launch_payload,
    start_worker_heartbeat,
    validate_private_finder_reference,
    worker_state_path,
)
from evaluatorq.dashboard.insights_uploads import (
    MAX_INSIGHTS_UPLOAD_BYTES,
    cleanup_uploaded_source,
    is_uploaded_source,
)
from evaluatorq.insights.models import InsightsPopulation
from evaluatorq.insights.pipeline import insights
from evaluatorq.trace_finder.export import RunExport
from evaluatorq.trace_finder.settings import effective_settings


def _fail_running(path: Path, error: str) -> None:
    fail_if_running(path, error, stage='setup')


def _read_private_snapshot(path: Path) -> bytes:
    """Read a private bounded snapshot through handles that cannot follow swaps."""
    expected_root = Path(tempfile.gettempdir()).resolve()
    if (
        path.name != 'finder-export.json'
        or not path.parent.name.startswith('evaluatorq-finder-snapshot-')
        or path.parent.parent.resolve() != expected_root
    ):
        raise ValueError('Finder export snapshot is outside its private temporary directory')

    directory_fd: int | None = None
    descriptor: int | None = None
    windows_directory_identity: tuple[int, int, int] | None = None
    try:
        if os.name == 'nt':
            descriptor, windows_directory_identity = _open_windows_approved_regular_file(
                path.parent, path, description='Finder export snapshot'
            )
        else:
            if not hasattr(os, 'O_DIRECTORY') or not hasattr(os, 'O_NOFOLLOW'):
                raise OSError('Safe Finder snapshot opening is unavailable on this platform.')
            root_fd = os.open(expected_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                directory_fd = os.open(
                    path.parent.name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=root_fd,
                )
            finally:
                os.close(root_fd)
            directory = os.fstat(directory_fd)
            if (
                not stat.S_ISDIR(directory.st_mode)
                or directory.st_uid != os.getuid()
                or stat.S_IMODE(directory.st_mode) != 0o700
            ):
                raise ValueError('Finder export snapshot directory has unsafe ownership or permissions')
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_NOFOLLOW,
                dir_fd=directory_fd,
            )

        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            os.name != 'nt' and (opened.st_uid != os.getuid() or stat.S_IMODE(opened.st_mode) != 0o600)
        ):
            raise ValueError('Finder export snapshot must be a private regular file')
        with os.fdopen(descriptor, 'rb') as snapshot_file:
            descriptor = None
            contents = snapshot_file.read(MAX_INSIGHTS_UPLOAD_BYTES + 1)
        if os.name == 'nt' and (
            windows_directory_identity is None or _windows_directory_identity(path.parent) != windows_directory_identity
        ):
            raise ValueError('Finder export snapshot directory changed while reading the file.')
        if len(contents) > MAX_INSIGHTS_UPLOAD_BYTES:
            raise ValueError('Missing or oversized validated Finder export snapshot')
        return contents
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_fd is not None:
            os.close(directory_fd)


def _cleanup_snapshot(path: Path) -> None:
    directory = path.parent
    expected_root = Path(tempfile.gettempdir()).resolve()
    if os.name != 'nt':
        _cleanup_snapshot_posix(path, directory, expected_root)
        return
    _cleanup_snapshot_windows(path, directory, expected_root)


def _cleanup_snapshot_windows(path: Path, directory: Path, expected_root: Path) -> None:  # noqa: C901 — each branch rejects an unsafe handle before deletion
    """Delete a validated Windows snapshot through pinned file handles."""
    try:
        info = directory.lstat()
        snapshot_info = path.lstat()
        if (
            path.name != 'finder-export.json'
            or directory.is_symlink()
            or not stat.S_ISREG(snapshot_info.st_mode)
            or not directory.name.startswith('evaluatorq-finder-snapshot-')
            or directory.resolve().parent != expected_root
            or (hasattr(os, 'getuid') and info.st_uid != os.getuid())
            or (hasattr(os, 'getuid') and snapshot_info.st_uid != os.getuid())
            or {entry.name for entry in directory.iterdir()} != {path.name}
        ):
            logger.warning('Leaving untrusted Finder snapshot path in place: {}', path)
            return

        import ctypes

        from evaluatorq.dashboard.insights_launch import _normalized_windows_path

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]
        create_file.restype = ctypes.c_void_p
        get_info = kernel32.GetFileInformationByHandle
        get_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        get_info.restype = ctypes.c_int
        get_final_path = kernel32.GetFinalPathNameByHandleW
        get_final_path.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32]
        get_final_path.restype = ctypes.c_uint32
        set_info = kernel32.SetFileInformationByHandle
        set_info.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        set_info.restype = ctypes.c_int
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int

        class FileTime(ctypes.Structure):
            _fields_ = [('low', ctypes.c_uint32), ('high', ctypes.c_uint32)]

        class ByHandleFileInfo(ctypes.Structure):
            _fields_ = [
                ('attributes', ctypes.c_uint32),
                ('creation_time', FileTime),
                ('access_time', FileTime),
                ('write_time', FileTime),
                ('volume_serial', ctypes.c_uint32),
                ('size_high', ctypes.c_uint32),
                ('size_low', ctypes.c_uint32),
                ('links', ctypes.c_uint32),
                ('index_high', ctypes.c_uint32),
                ('index_low', ctypes.c_uint32),
            ]

        class FileDispositionInfo(ctypes.Structure):
            _fields_ = [('delete_file', ctypes.c_ubyte)]

        invalid_handle = ctypes.c_void_p(-1).value
        share_read_write = 0x1 | 0x2  # Keep the path from being renamed while its handle is open.
        open_existing = 3
        open_reparse_point = 0x00200000
        backup_semantics = 0x02000000
        delete_and_read_attributes = 0x00010000 | 0x80
        file_attribute_directory = 0x10
        file_attribute_reparse_point = 0x400
        directory_identity = _windows_directory_identity(directory)

        def open_checked_handle(target: Path, *, is_directory: bool) -> tuple[int, ByHandleFileInfo]:
            flags = open_reparse_point | (backup_semantics if is_directory else 0)
            handle = create_file(
                str(target),
                delete_and_read_attributes,
                share_read_write,
                None,
                open_existing,
                flags,
                None,
            )
            if handle == invalid_handle or handle is None:
                raise ctypes.WinError(ctypes.get_last_error())
            file_info = ByHandleFileInfo()
            if not get_info(handle, ctypes.byref(file_info)):
                error = ctypes.WinError(ctypes.get_last_error())
                close_handle(handle)
                raise error
            final_path = ctypes.create_unicode_buffer(32768)
            length = get_final_path(handle, final_path, len(final_path), 0)
            if not length or length >= len(final_path):
                close_handle(handle)
                raise OSError('Could not verify the final path of the Finder snapshot handle.')
            if _normalized_windows_path(final_path.value) != _normalized_windows_path(str(target)):
                close_handle(handle)
                raise OSError('Finder snapshot handle resolved outside its approved path.')
            if is_directory:
                if (
                    not file_info.attributes & file_attribute_directory
                    or file_info.attributes & file_attribute_reparse_point
                ):
                    close_handle(handle)
                    raise OSError('Finder snapshot directory must not be a reparse point.')
                identity = (file_info.volume_serial, file_info.index_high, file_info.index_low)
                if identity != directory_identity:
                    close_handle(handle)
                    raise OSError('Finder snapshot directory changed while opening its handle.')
            elif file_info.attributes & (file_attribute_directory | file_attribute_reparse_point):
                close_handle(handle)
                raise OSError('Finder snapshot must be a regular file, not a reparse point.')
            return handle, file_info

        directory_handle: int | None = None
        snapshot_handle: int | None = None
        try:
            directory_handle, _directory_info = open_checked_handle(directory, is_directory=True)
            snapshot_handle, _snapshot_handle_info = open_checked_handle(path, is_directory=False)
            if not set_info(
                snapshot_handle,
                4,  # FileDispositionInfo
                ctypes.byref(FileDispositionInfo(1)),
                ctypes.sizeof(FileDispositionInfo),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
            close_handle(snapshot_handle)
            snapshot_handle = None

            if any(directory.iterdir()):
                logger.warning('Leaving non-empty Finder snapshot directory in place: {}', directory)
                return
            if not set_info(
                directory_handle,
                4,  # FileDispositionInfo
                ctypes.byref(FileDispositionInfo(1)),
                ctypes.sizeof(FileDispositionInfo),
            ):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            if snapshot_handle is not None:
                close_handle(snapshot_handle)
            if directory_handle is not None:
                close_handle(directory_handle)
    except OSError as exc:
        logger.warning('Could not remove Finder snapshot directory {}: {}', directory, exc)


def _cleanup_snapshot_posix(path: Path, directory: Path, expected_root: Path) -> None:
    """Remove only the snapshot file from the opened private directory."""
    if (
        path.name != 'finder-export.json'
        or not directory.name.startswith('evaluatorq-finder-snapshot-')
        or directory.parent.resolve() != expected_root
        or not hasattr(os, 'O_DIRECTORY')
        or not hasattr(os, 'O_NOFOLLOW')
    ):
        logger.warning('Leaving untrusted Finder snapshot path in place: {}', path)
        return

    root_fd: int | None = None
    directory_fd: int | None = None
    try:
        root_fd = os.open(expected_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        directory_fd = os.open(
            directory.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=root_fd,
        )
        info = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
            or os.listdir(directory_fd) != [path.name]  # noqa: PTH208 — enumerate the opened directory handle
        ):
            logger.warning('Leaving untrusted Finder snapshot path in place: {}', path)
            return

        snapshot_info = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(snapshot_info.st_mode)
            or snapshot_info.st_uid != os.getuid()
            or stat.S_IMODE(snapshot_info.st_mode) != 0o600
        ):
            logger.warning('Leaving untrusted Finder snapshot path in place: {}', path)
            return

        os.unlink(path.name, dir_fd=directory_fd)
        if os.listdir(directory_fd):  # noqa: PTH208 — enumerate the opened directory handle
            logger.warning('Leaving non-empty Finder snapshot directory in place: {}', directory)
            return
        current = os.stat(directory.name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
            logger.warning('Leaving changed Finder snapshot directory in place: {}', directory)
            return
        os.rmdir(directory.name, dir_fd=root_fd)
    except OSError as exc:
        logger.warning('Could not remove Finder snapshot directory {}: {}', directory, exc)
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        if root_fd is not None:
            os.close(root_fd)


def _snapshot_candidate_from_request() -> Path | None:
    """Recover a private Finder snapshot path when payload decoding fails.

    The separate environment value survives malformed request JSON. Treat both
    as untrusted: `_cleanup_snapshot` verifies the private temp directory,
    owner, permissions, and expected filename before deleting anything.
    """
    raw = os.environ.get('EVALUATORQ_INSIGHTS_LAUNCH_REQUEST')
    try:
        request = json.loads(raw) if raw is not None else None
    except (TypeError, ValueError):
        request = None
    if isinstance(request, dict):
        spec = request.get('spec')
        snapshot = request.get('finder_export_snapshot')
        if isinstance(spec, dict) and spec.get('source') == 'finder' and isinstance(snapshot, str):
            return Path(snapshot)
    snapshot = os.environ.get(_SNAPSHOT_ENV)
    return Path(snapshot) if snapshot else None


def _cleanup_finder_reference(runs_dir: Path, run_id: str) -> None:
    """Release a Finder export after its Insights worker reaches a terminal state."""
    try:
        reference = finder_export_reference_path(runs_dir, run_id)
        if reference.exists():
            validate_private_finder_reference(reference)
            reference.unlink(missing_ok=True)
    except (OSError, ValueError) as exc:
        logger.warning('Could not remove Finder export reference for run {}: {}', run_id, exc)


def _cleanup_finder_reference_for_manifest(manifest_path: Path) -> None:
    """Release a Finder lease when a launch payload could not be decoded."""
    try:
        if manifest_path.parent.name != '.manifests':
            return
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
        if manifest.surface.value != 'insights' or manifest_path.name != f'{manifest.run_id}.json':
            return
        _cleanup_finder_reference(manifest_path.parent.parent, manifest.run_id)
    except (OSError, ValueError) as exc:
        logger.warning('Could not recover Finder export reference from manifest {}: {}', manifest_path, exc)


async def _run_with_selected_auth(payload: InsightsLaunchPayload, population: InsightsPopulation, **extra: Any) -> bool:
    """Run Insights with the clients of the auth the dashboard had selected at launch."""
    spec = payload.spec
    settings = effective_settings()
    if payload.auth_method is not None and settings.orq_auth_method != payload.auth_method:
        raise ValueError('Dashboard authentication changed after this run started. Start the run again.')
    auth = resolve_dashboard_auth(settings)
    if (
        payload.auth_identity is not None
        and await asyncio.to_thread(auth_identity, auth, settings) != payload.auth_identity
    ):
        raise ValueError('Dashboard authentication or scope changed after this run started. Start the run again.')
    orq, llm = build_auth_clients(auth, workspace=settings.orq_workspace, project=settings.orq_project_id)
    try:
        run = await insights(
            population,
            labels=spec.label_specs(),
            dimensions=spec.dimension_names(),
            parallelism=spec.parallelism,
            coding_analysis=spec.coding_analysis,
            coding_labels=None if spec.coding_analysis else spec.coding_labels,
            run_name=payload.run_name,
            runs_dir=payload.runs_dir,
            _run_id=payload.run_id,
            llm_client=llm,
            orq_client=orq,
            **extra,
        )
        return run.status == 'completed'
    finally:
        await close_orq_client(orq)
        close = getattr(llm, 'close', None)
        if close is not None:
            await close()


def _cleanup_consumed_upload(runs_dir: Path, source: str, finder_export: str, snapshot_path: str) -> None:
    """Delete only a route-owned source artifact after its pipeline attempt."""
    source_value = finder_export if source == 'finder' else snapshot_path
    if source_value:
        source_path = Path(source_value)
        if is_uploaded_source(runs_dir, source_path):
            cleanup_uploaded_source(runs_dir, source_path)


def main() -> int:
    manifest_env = os.environ.get(_MANIFEST_ENV)
    heartbeat = start_worker_heartbeat(Path(manifest_env)) if manifest_env else None
    payload = None
    unvalidated_snapshot = _snapshot_candidate_from_request()
    try:
        payload = read_launch_payload()
        spec = payload.spec
        if spec.source == 'finder':
            snapshot = payload.finder_export_snapshot
            if snapshot is None:
                raise ValueError('Missing or oversized validated Finder export snapshot')
            raw_snapshot = _read_private_snapshot(snapshot)
            if (
                payload.finder_export_snapshot_sha256 is None
                or hashlib.sha256(raw_snapshot).hexdigest() != payload.finder_export_snapshot_sha256
            ):
                raise ValueError('Finder export snapshot changed after launch')
            validated_export = RunExport.model_validate_json(raw_snapshot)
            completed = asyncio.run(
                _run_with_selected_auth(
                    payload,
                    InsightsPopulation.from_finder_export(snapshot, export=validated_export),
                    _finder_export_source=Path(spec.finder_export),
                    _finder_export_sha256=hashlib.sha256(raw_snapshot).hexdigest(),
                    _source_name=spec.source_name or None,
                    coding_analysis=spec.coding_analysis,
                )
            )
        else:
            completed = asyncio.run(
                _run_with_selected_auth(payload, spec.population(), _source_name=spec.source_name or None)
            )
    except Exception as exc:  # noqa: BLE001 — record any failure before the pipeline owns its manifest
        logger.exception('Dashboard Insights worker failed')
        if payload is not None:
            _fail_running(payload.runs_dir / '.manifests' / f'{payload.run_id}.json', str(exc))
        else:
            manifest_path = os.environ.get(_MANIFEST_ENV)
            if manifest_path:
                _fail_running(Path(manifest_path), str(exc))
        return 1
    finally:
        if heartbeat is not None:
            heartbeat_stop, heartbeat_thread = heartbeat
            heartbeat_stop.set()
            heartbeat_thread.join()
        if payload is not None and payload.finder_export_snapshot is not None:
            _cleanup_snapshot(payload.finder_export_snapshot)
        elif unvalidated_snapshot is not None:
            _cleanup_snapshot(unvalidated_snapshot)
        if payload is not None and payload.spec.source == 'finder':
            _cleanup_finder_reference(payload.runs_dir, payload.run_id)
        if payload is not None:
            _cleanup_consumed_upload(
                payload.runs_dir, payload.spec.source, payload.spec.finder_export, payload.spec.snapshot_path
            )
        elif payload is None and manifest_env:
            _cleanup_finder_reference_for_manifest(Path(manifest_env))
        state_run_id = payload.run_id if payload is not None else Path(manifest_env).stem if manifest_env else None
        if state_run_id is not None:
            try:
                state_runs_dir = payload.runs_dir if payload is not None else Path(manifest_env).parent.parent
                worker_state_path(state_runs_dir, state_run_id).unlink(missing_ok=True)
            except (OSError, ValueError) as exc:
                logger.warning('Could not remove Insights worker state for run {}: {}', state_run_id, exc)
    return 0 if completed else 1


if __name__ == '__main__':
    raise SystemExit(main())
