"""Validate dashboard Insights requests and launch an independent run process."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Literal, NamedTuple

from loguru import logger
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from typing_extensions import Self

from evaluatorq.common.orq_client import DEFAULT_ORQ_BASE_URL, OrqProfile
from evaluatorq.common.run_manifest import fail_if_running, start_manifest
from evaluatorq.common.run_store_dir import get_store_dir
from evaluatorq.contracts import ManifestStatus, RunManifest
from evaluatorq.dashboard.insights_uploads import (
    MAX_FINDER_EXPORT_BYTES,
    is_uploaded_source,
    read_uploaded_source,
    upload_size_error,
)
from evaluatorq.insights.models import DimensionName, InsightsPopulation, LabelSpec
from evaluatorq.insights.presets import CODING_LABELS, LABEL_PRESETS
from evaluatorq.insights.progress import stage_plan
from evaluatorq.insights.store import get_insights_runs_dir
from evaluatorq.trace_finder.export import RunExport
from evaluatorq.trace_finder.models import FacetSelection, Snapshot

Source = Literal['recent', 'query', 'finder', 'snapshot']


class SourceInfo(NamedTuple):
    """How one launch `Source` shows up: its run form tab, its stored `population.mode`, and its label."""

    tab: Literal['orq', 'file']
    mode: str
    label: str


SOURCES: MappingProxyType[Source, SourceInfo] = MappingProxyType({
    'recent': SourceInfo('orq', 'filter', 'Orq traces'),
    'query': SourceInfo('orq', 'query', 'Orq traces matching a question'),
    'finder': SourceInfo('file', 'export', 'Trace file (Finder export)'),
    'snapshot': SourceInfo('file', 'snapshot', 'Trace file'),
})
SOURCE_BY_MODE: MappingProxyType[str, Source] = MappingProxyType({
    info.mode: source for source, info in SOURCES.items()
})
Preset = str
_CODING_PRESETS = {spec.name: spec for spec in CODING_LABELS[1:]}
_REQUEST_ENV = 'EVALUATORQ_INSIGHTS_LAUNCH_REQUEST'
_MANIFEST_ENV = 'EVALUATORQ_INSIGHTS_MANIFEST'
_SNAPSHOT_ENV = 'EVALUATORQ_INSIGHTS_FINDER_SNAPSHOT'
FINDER_EXPORT_REFERENCE_DIR = '.finder-export-leases'
INSIGHTS_WORKER_STATE_DIR = '.insights-workers'
INSIGHTS_WORKER_STALE_SECONDS = 90
INSIGHTS_WORKER_HEARTBEAT_SECONDS = 10
DEFAULT_WINDOW_DAYS = 7
DEFAULT_TRACE_LIMIT = 200


def _default_dimensions() -> list[DimensionName]:
    return ['intent']


def _set_private_file_mode(descriptor: int, mode: int) -> None:
    """Apply POSIX file permissions; Windows uses the containing directory ACL.

    Windows does not expose ``os.fchmod`` and does not implement POSIX mode
    bits as access control. Windows access therefore depends on the ACL
    inherited from the containing application or temporary directory; this
    helper does not set or inspect that ACL.
    """
    if os.name != 'nt':
        os.fchmod(descriptor, mode)


def worker_state_path(runs_dir: Path, run_id: str) -> Path:
    if re.fullmatch(r'[A-Za-z0-9_-]+', run_id) is None:
        raise ValueError('Invalid Insights run ID for worker state')
    return runs_dir / INSIGHTS_WORKER_STATE_DIR / f'{run_id}.json'


def _write_worker_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory_info = path.parent.lstat()
    if (
        path.parent.is_symlink()
        or not stat.S_ISDIR(directory_info.st_mode)
        or (hasattr(os, 'getuid') and directory_info.st_uid != os.getuid())
    ):
        raise OSError(f'Insights worker state directory is not a private directory: {path.parent}')
    Path(path.parent).chmod(0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.stem}.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            _set_private_file_mode(handle.fileno(), 0o600)
            json.dump(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def start_worker_heartbeat(manifest_path: Path) -> tuple[threading.Event, threading.Thread]:
    """Refresh the detached worker lease independently of its async workload."""
    run_id = manifest_path.stem
    path = worker_state_path(manifest_path.parent.parent, run_id)
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        state = {}
    if not isinstance(state, dict):
        state = {}
    state.update(pid=os.getpid(), process_identity=_worker_process_identity(os.getpid()), heartbeat_at=time.time())
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.is_set():
            try:
                _write_worker_state(path, state)
            except OSError as exc:
                logger.warning('Could not refresh Insights worker heartbeat {}: {}', path, exc)
            if stop.wait(INSIGHTS_WORKER_HEARTBEAT_SECONDS):
                return
            state['heartbeat_at'] = time.time()

    _write_worker_state(path, state)
    thread = threading.Thread(target=heartbeat, name=f'insights-heartbeat-{run_id}', daemon=True)
    thread.start()
    return stop, thread


def reconcile_stale_worker(runs_dir: Path, run_id: str) -> bool:
    """Fail a dashboard worker whose startup or heartbeat lease expired."""
    state_path = worker_state_path(runs_dir, run_id)
    manifest_path = runs_dir / '.manifests' / f'{run_id}.json'
    try:
        directory_info = state_path.parent.lstat()
        state_info = state_path.lstat()
        if (
            state_path.parent.is_symlink()
            or not stat.S_ISDIR(directory_info.st_mode)
            or (os.name != 'nt' and stat.S_IMODE(directory_info.st_mode) != 0o700)
            or (hasattr(os, 'getuid') and directory_info.st_uid != os.getuid())
            or not stat.S_ISREG(state_info.st_mode)
            or (os.name != 'nt' and stat.S_IMODE(state_info.st_mode) != 0o600)
            or (hasattr(os, 'getuid') and state_info.st_uid != os.getuid())
        ):
            return False
        manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if not isinstance(state, dict):
            return False
        heartbeat_at = state.get('heartbeat_at')
        stale = isinstance(heartbeat_at, (int, float)) and time.time() - heartbeat_at > INSIGHTS_WORKER_STALE_SECONDS
        worker_pid = state.get('pid')
        worker_identity = state.get('process_identity')
        launcher_pid = state.get('launcher_pid')
        launcher_identity = state.get('launcher_process_identity')
        # Until the worker PID is published, the launcher is the only process
        # that can prove startup is still in progress. The child waits for a
        # handshake after that PID has been written.
        checking_launcher = worker_pid is None
        pid = launcher_pid if checking_launcher else worker_pid
        process_identity = launcher_identity if checking_launcher else worker_identity
    except (OSError, ValueError, TypeError):
        return False
    if manifest.status != ManifestStatus.RUNNING or not stale:
        return False
    # A worker can be paused by the OS or debugger long enough for its lease
    # to age. A live PID is stronger evidence than an old heartbeat, so leave
    # its manifest and private Finder artifacts alone until it exits.
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or _worker_process_is_alive(pid, process_identity):
        return False
    # Re-read the lease immediately before failing. The worker may have
    # resumed and refreshed it while the manifest was being parsed.
    try:
        latest_state = json.loads(state_path.read_text(encoding='utf-8'))
        latest_manifest = RunManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError):
        return False
    if (
        latest_manifest.status != ManifestStatus.RUNNING
        or not isinstance(latest_state, dict)
        or latest_state.get('pid') != worker_pid
        or latest_state.get('process_identity') != worker_identity
        or latest_state.get('launcher_pid') != launcher_pid
        or latest_state.get('launcher_process_identity') != launcher_identity
        or not isinstance(latest_state.get('heartbeat_at'), (int, float))
        or time.time() - latest_state['heartbeat_at'] <= INSIGHTS_WORKER_STALE_SECONDS
        or _worker_process_is_alive(pid, process_identity)
    ):
        return False
    state = latest_state
    fail_if_running(
        manifest_path, 'Insights worker stopped before completing the run; start a new run to retry.', stage='worker'
    )
    try:
        persisted = RunManifest.model_validate_json(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    if persisted.status == ManifestStatus.RUNNING:
        return False
    snapshot = state.get('snapshot_path')
    if isinstance(snapshot, str):
        # Reuse the worker's strict private-temp ownership and mode checks.
        from evaluatorq.dashboard.insights_worker import _cleanup_snapshot

        _cleanup_snapshot(Path(snapshot))
    try:
        reference_path = finder_export_reference_path(runs_dir, run_id)
        if not reference_path.exists():
            pass
        elif reference_path.parent.is_symlink():
            logger.warning('Leaving untrusted Finder export reference directory in place: {}', reference_path.parent)
        else:
            validate_private_finder_reference(reference_path)
            reference_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning('Could not release Finder export reference for stale run {}: {}', run_id, exc)
    try:
        state_path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning('Could not remove stale Insights worker state {}: {}', state_path, exc)
    return True


def _worker_process_is_alive(pid: object, expected_identity: object = None) -> bool:
    """Return whether the recorded worker still exists with its original identity."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or not hasattr(os, 'kill'):
        return True
    if sys.platform == 'win32':
        status = _windows_worker_status(pid, expected_identity)
        if status is None:
            logger.warning('Could not safely verify Insights worker PID {} on Windows; leaving its run active', pid)
            return True
        return status
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        # Only ESRCH proves death; permission or platform errors are uncertain.
        return exc.errno != errno.ESRCH
    except OverflowError:
        return True
    if isinstance(expected_identity, str):
        observed = _read_worker_process_identity(pid)
        if observed is None:
            logger.warning('Could not verify Insights worker process identity for PID {}; leaving its run active', pid)
            return True
        identity, zombie = observed
        return identity == expected_identity and not zombie
    logger.warning('Insights worker PID {} has no recorded process identity; leaving its run active', pid)
    return True


def _read_worker_process_identity(pid: int) -> tuple[str, bool] | None:
    """Read a process start marker and zombie flag where the OS exposes them."""
    if sys.platform == 'win32':
        status = _windows_process_details(pid)
        if status is None or not status[1]:
            return None
        return status[0], False
    if sys.platform.startswith('linux'):
        try:
            contents = Path(f'/proc/{pid}/stat').read_text(encoding='ascii')
        except OSError:
            return None
        # The command name is parenthesized and can itself contain spaces or ')'.
        closing = contents.rfind(')')
        if closing < 0:
            return None
        fields = contents[closing + 1 :].split()
        if len(fields) <= 19:
            return None
        try:
            start_ticks = int(fields[19])
        except ValueError:
            return None
        return f'linux:{start_ticks}', fields[0] == 'Z'
    if sys.platform == 'darwin':
        try:
            result = subprocess.run(
                ['/bin/ps', '-o', 'lstart=', '-o', 'stat=', '-p', str(pid)],
                capture_output=True,
                check=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError, TypeError, ValueError):
            return None
        try:
            fields = result.stdout.split()
            if len(fields) < 6:
                return None
        except (AttributeError, TypeError):
            return None
        return f'darwin:{" ".join(fields[:5])}', fields[5].startswith('Z')
    return None


def _windows_worker_status(pid: int, expected_identity: object) -> bool | None:
    """Check a Windows process without sending it a signal; None means uncertain."""
    details = _windows_process_details(pid)
    if details is None:
        return None
    identity, alive = details
    if not alive:
        return False
    if not isinstance(expected_identity, str):
        return None
    return identity == expected_identity


def _windows_process_details(pid: int) -> tuple[str, bool] | None:
    """Return Windows process creation time and liveness using read-only APIs."""
    if not hasattr(ctypes, 'WinDLL') or pid > 0xFFFFFFFF:
        return None
    from ctypes import wintypes

    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        open_process = kernel32.OpenProcess
        wait_for_single_object = kernel32.WaitForSingleObject
        get_process_times = kernel32.GetProcessTimes
        close_handle = kernel32.CloseHandle
        # Querying and waiting are read-only rights, without process-control access.
        if hasattr(open_process, 'argtypes'):
            open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            open_process.restype = wintypes.HANDLE
            wait_for_single_object.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            get_process_times.argtypes = [
                wintypes.HANDLE,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ]
            close_handle.argtypes = [wintypes.HANDLE]
        handle = open_process(0x101000, 0, pid)
        if not handle:
            # ERROR_INVALID_PARAMETER means that the PID does not exist. Access
            # denied and all other failures remain uncertain.
            return ('windows:missing', False) if ctypes.get_last_error() == 87 else None
        try:
            wait_status = wait_for_single_object(handle, 0)
            if wait_status == 0:
                return 'windows:exited', False
            if wait_status != 258:  # WAIT_TIMEOUT means the process is still running.
                return None

            class FILETIME(ctypes.Structure):
                _fields_ = [('low', wintypes.DWORD), ('high', wintypes.DWORD)]

            created, exited, kernel, user = FILETIME(), FILETIME(), FILETIME(), FILETIME()
            if not get_process_times(
                handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)
            ):
                return None
            started = (created.high << 32) | created.low
            return f'windows:{started}', True
        finally:
            close_handle(handle)
    except Exception as exc:  # noqa: BLE001 - an unavailable Windows API must not crash dashboard recovery
        logger.warning('Windows worker process query failed for PID {}: {}', pid, exc)
        return None


def _worker_process_identity(pid: int) -> str | None:
    """Capture the worker's start marker, if this OS exposes one."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return None
    details = _read_worker_process_identity(pid)
    return details[0] if details is not None else None


# The child must record import-time failures too. This tiny stdlib-only wrapper
# runs before importing the package worker, then edits the already-created
# manifest if Python cannot import the worker or its dependencies.
_WORKER_BOOTSTRAP = """
import datetime, json, os, runpy, stat, sys, tempfile
manifest = os.environ.get("EVALUATORQ_INSIGHTS_MANIFEST")
request = os.environ.get("EVALUATORQ_INSIGHTS_LAUNCH_REQUEST")
def cleanup_snapshot():
    try:
        snapshot = os.environ.get("EVALUATORQ_INSIGHTS_FINDER_SNAPSHOT")
        if not snapshot:
            snapshot = json.loads(request or "{}").get("finder_export_snapshot")
        if not isinstance(snapshot, str) or os.path.basename(snapshot) != "finder-export.json":
            return
        directory = os.path.dirname(snapshot)
        if (not os.path.basename(directory).startswith("evaluatorq-finder-snapshot-")
                or os.path.dirname(os.path.realpath(directory)) != os.path.realpath(tempfile.gettempdir())):
            return
        if os.name == "nt":
            print("Leaving Finder snapshot after bootstrap failure; safe Windows cleanup is unavailable.",
                  file=__import__("sys").stderr)
            return
        if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
            return
        root = os.path.realpath(tempfile.gettempdir())
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            directory_fd = os.open(os.path.basename(directory), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                   dir_fd=root_fd)
            try:
                info = os.fstat(directory_fd)
                if (not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o700
                        or (hasattr(os, "getuid") and info.st_uid != os.getuid())
                        or os.listdir(directory_fd) != [os.path.basename(snapshot)]):
                    return
                snapshot_info = os.stat(os.path.basename(snapshot), dir_fd=directory_fd, follow_symlinks=False)
                if (not stat.S_ISREG(snapshot_info.st_mode) or stat.S_IMODE(snapshot_info.st_mode) != 0o600
                        or (hasattr(os, "getuid") and snapshot_info.st_uid != os.getuid())):
                    return
                os.unlink(os.path.basename(snapshot), dir_fd=directory_fd)
                if not os.listdir(directory_fd):
                    current = os.stat(os.path.basename(directory), dir_fd=root_fd, follow_symlinks=False)
                    if stat.S_ISDIR(current.st_mode) and (current.st_dev, current.st_ino) == (info.st_dev, info.st_ino):
                        os.rmdir(os.path.basename(directory), dir_fd=root_fd)
            finally:
                os.close(directory_fd)
        finally:
            os.close(root_fd)
    except (OSError, ValueError, TypeError) as cleanup_error:
        print(f"Could not remove validated Finder snapshot: {cleanup_error}", file=__import__("sys").stderr)
def cleanup_reference():
    try:
        directory = os.path.dirname(manifest or "")
        name = os.path.basename(manifest or "")
        run_id = name[:-5] if name.endswith(".json") else ""
        if os.path.basename(directory) != ".manifests" or not run_id or not all(ch.isalnum() or ch in "-_" for ch in run_id):
            return
        lease_dir = os.path.join(os.path.dirname(directory), ".finder-export-leases")
        lease_parent = os.path.dirname(lease_dir)
        parent_info = os.stat(lease_parent, follow_symlinks=False)
        directory_info = os.stat(lease_dir, follow_symlinks=False)
        reference = os.path.join(lease_dir, run_id + ".json")
        reference_info = os.stat(reference, follow_symlinks=False)
        if (os.path.islink(lease_parent) or not stat.S_ISDIR(parent_info.st_mode)
                or (hasattr(os, "getuid") and parent_info.st_uid != os.getuid())
                or (os.name != "nt" and stat.S_IMODE(parent_info.st_mode) & 0o022)
                or os.path.islink(lease_dir) or not stat.S_ISDIR(directory_info.st_mode)
                or (os.name != "nt" and stat.S_IMODE(directory_info.st_mode) != 0o700)
                or (hasattr(os, "getuid") and directory_info.st_uid != os.getuid())
                or not stat.S_ISREG(reference_info.st_mode)
                or (os.name != "nt" and stat.S_IMODE(reference_info.st_mode) != 0o600)
                or (hasattr(os, "getuid") and reference_info.st_uid != os.getuid())):
            return
        os.unlink(reference)
    except FileNotFoundError:
        pass
    except OSError as cleanup_error:
        print(f"Could not remove Finder export reference: {cleanup_error}", file=__import__("sys").stderr)
def cleanup_worker_state():
    try:
        directory = os.path.dirname(manifest or "")
        name = os.path.basename(manifest or "")
        run_id = name[:-5] if name.endswith(".json") else ""
        if os.path.basename(directory) != ".manifests" or not run_id or not all(ch.isalnum() or ch in "-_" for ch in run_id):
            return
        state_dir = os.path.join(os.path.dirname(directory), ".insights-workers")
        if os.path.islink(state_dir):
            return
        os.unlink(os.path.join(state_dir, run_id + ".json"))
    except FileNotFoundError:
        pass
    except OSError as cleanup_error:
        print(f"Could not remove Insights worker state: {cleanup_error}", file=__import__("sys").stderr)
try:
    # Wait until the parent records the PID. If the parent exits before
    # releasing the child, EOF stops the worker before it can begin the run.
    if sys.stdin.buffer.read(1) != b"1":
        raise SystemExit(0)
    runpy.run_module("evaluatorq.dashboard.insights_worker", run_name="__main__")
except BaseException as exc:
    if isinstance(exc, SystemExit) and exc.code in (None, 0):
        raise
    cleanup_snapshot()
    cleanup_reference()
    cleanup_worker_state()
    if manifest:
        lock = None
        try:
            # Same sibling lock as evaluatorq.common.run_manifest: other writers share this file.
            try:
                lock = open(manifest + ".lock", "a+b")
                if os.name == "nt":
                    import msvcrt
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                pass
            with open(manifest, encoding="utf-8") as source:
                data = json.load(source)
            if data.get("status") == "running":
                now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                data.update(status="error", stage="setup", error=f"Could not start Insights worker: {exc}", ended_at=now, updated_at=now)
                directory = os.path.dirname(manifest)
                fd, temporary = tempfile.mkstemp(prefix=".insights-failure-", dir=directory)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as target:
                        json.dump(data, target)
                    os.replace(temporary, manifest)
                finally:
                    if os.path.exists(temporary):
                        os.unlink(temporary)
        except BaseException as recovery_error:
            print(f"Could not mark Insights manifest failed: {recovery_error}", file=__import__("sys").stderr)
        finally:
            if lock is not None:
                lock.close()
    raise
"""


def get_finder_exports_dir() -> Path:
    """Return the only directory from which dashboard runs may read Finder exports."""
    return get_store_dir('finder-exports')


def _normalized_windows_path(value: str) -> str:
    def without_extended_prefix(path: str) -> str:
        if path.startswith('\\\\?\\UNC\\'):
            return '\\\\' + path[8:]
        if path.startswith('\\\\?\\'):
            return path[4:]
        return path

    value = without_extended_prefix(value)
    # GetFinalPathNameByHandleW returns the long path spelling, while paths
    # supplied by Windows APIs such as tempfile.gettempdir() can use 8.3
    # aliases (for example RUNNER~1). realpath resolves both spellings before
    # comparing them, while the opened handle's file ID still guards against
    # the directory being replaced between checks.
    canonical = without_extended_prefix(os.path.realpath(value)) if os.name == 'nt' else value
    return str(PureWindowsPath(canonical)).casefold()


def _windows_directory_identity(root: Path) -> tuple[int, int, int]:
    """Return the file ID of the approved Windows directory, rejecting reparse paths."""
    try:
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
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
    except (AttributeError, OSError) as exc:
        raise OSError('Safe Finder export directory checks are unavailable on Windows.') from exc

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

    invalid_handle = ctypes.c_void_p(-1).value
    handle = create_file(
        str(root), 0x80, 0x7, None, 3, 0x02000000 | 0x00200000, None
    )  # FILE_READ_ATTRIBUTES, shared read/write/delete, backup semantics, no reparse following.
    if handle == invalid_handle or handle is None:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        info = ByHandleFileInfo()
        if not get_info(handle, ctypes.byref(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        if not info.attributes & 0x10 or info.attributes & 0x400:
            raise OSError('Finder export directory must be a directory without reparse points.')

        path_buffer = ctypes.create_unicode_buffer(32768)
        length = get_final_path(handle, path_buffer, len(path_buffer), 0)
        if not length or length >= len(path_buffer):
            raise OSError('Could not verify the final path of the Finder export directory.')
        if _normalized_windows_path(path_buffer.value) != _normalized_windows_path(str(root)):
            raise OSError('Finder export directory handle resolved outside the approved path.')
        return info.volume_serial, info.index_high, info.index_low
    finally:
        close_handle(handle)


def _open_windows_approved_regular_file(
    root: Path, path: Path, *, description: str = 'Finder export'
) -> tuple[int, tuple[int, int, int]]:
    """Open a Windows file without following reparse points and verify its final path."""
    try:
        import msvcrt

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
        get_info = kernel32.GetFileInformationByHandleEx
        get_info.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
        get_info.restype = ctypes.c_int
        get_final_path = kernel32.GetFinalPathNameByHandleW
        get_final_path.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32]
        get_final_path.restype = ctypes.c_uint32
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [ctypes.c_void_p]
        close_handle.restype = ctypes.c_int
    except (AttributeError, ImportError, OSError) as exc:
        raise OSError(f'Safe {description.lower()} opening is unavailable on Windows.') from exc

    invalid_handle = ctypes.c_void_p(-1).value
    open_existing = 3
    share_read = 0x1
    share_read_write = 0x1 | 0x2
    file_read_attributes = 0x80
    flag_open_reparse_point = 0x00200000
    flag_backup_semantics = 0x02000000
    file_attribute_tag_info = 9
    file_attribute_directory = 0x10
    file_attribute_reparse_point = 0x400
    directory_identity = _windows_directory_identity(root)

    class FileAttributeTagInfo(ctypes.Structure):
        _fields_ = [('attributes', ctypes.c_uint32), ('reparse_tag', ctypes.c_uint32)]

    directory_handle = create_file(
        str(root),
        file_read_attributes,
        share_read_write,
        None,
        open_existing,
        flag_open_reparse_point | flag_backup_semantics,
        None,
    )
    if directory_handle == invalid_handle or directory_handle is None:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        # Deny FILE_SHARE_DELETE to pin this directory name until its child has been opened.
        # Recheck after pinning to reject a replacement that won the race before CreateFileW.
        if _windows_directory_identity(root) != directory_identity:
            raise OSError(f'{description} directory changed while pinning it.')

        handle = create_file(str(path), 0x80000000, share_read, None, open_existing, flag_open_reparse_point, None)
        if handle == invalid_handle or handle is None:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            info = FileAttributeTagInfo()
            if not get_info(handle, file_attribute_tag_info, ctypes.byref(info), ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            if info.attributes & (file_attribute_directory | file_attribute_reparse_point):
                raise OSError(f'{description} must be a regular file, not a directory or reparse point.')

            path_buffer = ctypes.create_unicode_buffer(32768)
            length = get_final_path(handle, path_buffer, len(path_buffer), 0)
            if not length or length >= len(path_buffer):
                raise OSError(f'Could not verify the final path of the {description.lower()}.')

            final_path = _normalized_windows_path(path_buffer.value)
            approved_root = _normalized_windows_path(str(root))
            if str(PureWindowsPath(final_path).parent) != approved_root:
                raise OSError(f'{description} handle resolved outside the approved directory.')

            if _windows_directory_identity(root) != directory_identity:
                raise OSError(f'{description} directory changed while opening the file.')

            descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | getattr(os, 'O_BINARY', 0))
            handle = None
            return descriptor, directory_identity
        finally:
            if handle is not None:
                close_handle(handle)
    finally:
        close_handle(directory_handle)


def _read_approved_finder_export(root: Path, path: Path) -> bytes:
    """Read a bounded export from the opened approved directory and file descriptors."""
    directory_fd: int | None = None
    directory_before: os.stat_result | None = None
    windows_directory_identity: tuple[int, int, int] | None = None
    try:
        if os.name != 'nt':
            if not hasattr(os, 'O_DIRECTORY') or not hasattr(os, 'O_NOFOLLOW'):
                raise ValueError('Safe Finder export file opening is unavailable on this platform.')
            directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            directory = os.fstat(directory_fd)
            if (
                not stat.S_ISDIR(directory.st_mode)
                or directory.st_uid != os.getuid()
                or stat.S_IMODE(directory.st_mode) & 0o022
            ):
                raise ValueError('Finder export directory has unsafe ownership or permissions.')
            # O_NONBLOCK keeps a FIFO named *.json from blocking this open until a writer appears;
            # the S_ISREG check below then rejects it.
            descriptor = os.open(
                path.name, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, 'O_NONBLOCK', 0), dir_fd=directory_fd
            )
        else:
            descriptor, windows_directory_identity = _open_windows_approved_regular_file(root, path)
        try:
            export_file = os.fdopen(descriptor, 'rb')
        except OSError:
            os.close(descriptor)
            raise
        with export_file:
            opened = os.fstat(export_file.fileno())
            current = path.lstat() if os.name != 'nt' else opened
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(current.st_mode)
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
                or (os.name != 'nt' and (opened.st_uid != os.getuid() or stat.S_IMODE(opened.st_mode) & 0o022))
            ):
                raise ValueError('Finder export must be a safe regular file in the approved directory.')
            contents = export_file.read(MAX_FINDER_EXPORT_BYTES + 1)
            if os.name == 'nt' and (
                windows_directory_identity is None or _windows_directory_identity(root) != windows_directory_identity
            ):
                raise ValueError('Finder export directory changed while reading the file.')
            return contents
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def _write_private_finder_snapshot(path: Path, contents: str) -> None:
    """Write a new private snapshot and close the raw descriptor if wrapping fails."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        snapshot_file = os.fdopen(descriptor, 'wb')
    except OSError:
        os.close(descriptor)
        raise
    with snapshot_file:
        snapshot_file.write(contents.encode('utf-8'))


def finder_export_reference_path(runs_dir: Path, run_id: str) -> Path:
    """Return the private lease path for a running Finder-sourced Insights run."""
    if re.fullmatch(r'[A-Za-z0-9_-]+', run_id) is None:
        raise ValueError('Invalid Insights run ID for Finder export reference')
    return runs_dir / FINDER_EXPORT_REFERENCE_DIR / f'{run_id}.json'


def ensure_private_finder_reference_dir(path: Path) -> None:
    """Create or validate the private Finder lease directory."""
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    validate_private_finder_reference_dir(path)


def validate_private_finder_reference_dir(path: Path) -> None:
    """Check an existing Finder lease directory without creating it."""
    parent_info = path.parent.lstat()
    if (
        path.parent.is_symlink()
        or not stat.S_ISDIR(parent_info.st_mode)
        or (hasattr(os, 'getuid') and parent_info.st_uid != os.getuid())
        or (os.name != 'nt' and stat.S_IMODE(parent_info.st_mode) & 0o022)
    ):
        raise OSError(f'Finder export lease parent is not private from other users: {path.parent}')
    info = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISDIR(info.st_mode)
        or (os.name != 'nt' and stat.S_IMODE(info.st_mode) != 0o700)
        or (hasattr(os, 'getuid') and info.st_uid != os.getuid())
    ):
        raise OSError(f'Finder export lease directory is not private: {path}')


def validate_private_finder_reference(path: Path) -> None:
    """Reject a Finder lease unless its directory and file are private and owned here."""
    validate_private_finder_reference_dir(path.parent)
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or (os.name != 'nt' and stat.S_IMODE(info.st_mode) != 0o600)
        or (hasattr(os, 'getuid') and info.st_uid != os.getuid())
    ):
        raise OSError(f'Finder export lease file is not private: {path}')


def read_private_finder_reference(path: Path) -> dict[str, object]:
    """Read a lease only after checking owner, mode, and regular-file type."""
    validate_private_finder_reference(path)
    if os.name == 'nt':
        descriptor, _directory_identity = _open_windows_approved_regular_file(
            path.parent, path, description='Finder export lease'
        )
    else:
        if not hasattr(os, 'O_DIRECTORY') or not hasattr(os, 'O_NOFOLLOW'):
            raise OSError('Safe Finder export lease opening is unavailable on this platform.')
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        directory_fd = os.open(path.parent, directory_flags)
        try:
            directory_info = os.fstat(directory_fd)
            path_info = path.parent.lstat()
            if (
                not stat.S_ISDIR(directory_info.st_mode)
                or (hasattr(os, 'getuid') and directory_info.st_uid != os.getuid())
                or stat.S_IMODE(directory_info.st_mode) & 0o022
                or (directory_info.st_dev, directory_info.st_ino) != (path_info.st_dev, path_info.st_ino)
            ):
                raise OSError(f'Finder export lease directory changed while opening: {path.parent}')
            flags = os.O_RDONLY | os.O_NOFOLLOW
            descriptor = os.open(path.name, flags, dir_fd=directory_fd)
        finally:
            os.close(directory_fd)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or (os.name != 'nt' and stat.S_IMODE(info.st_mode) != 0o600)
            or (hasattr(os, 'getuid') and info.st_uid != os.getuid())
        ):
            raise OSError(f'Finder export lease file changed while opening: {path}')
        with os.fdopen(descriptor, 'r', encoding='utf-8') as reference_file:
            descriptor = -1
            data = json.load(reference_file)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(data, dict):
        raise TypeError(f'Finder export lease is not an object: {path}')
    return data


def _initial_worker_state(path: Path, snapshot: Path | None) -> None:
    _write_worker_state(
        path,
        {
            'pid': None,
            'launcher_pid': os.getpid(),
            'launcher_process_identity': _worker_process_identity(os.getpid()),
            'heartbeat_at': time.time(),
            'snapshot_path': str(snapshot) if snapshot is not None else None,
        },
    )


class InsightsLaunchSpec(BaseModel):
    """Small, allow-listed subset of the Insights CLI for the dashboard wizard."""

    name: str = Field(default='', max_length=80)
    source: Source = 'recent'
    query: str = Field(default='', max_length=500)
    finder_export: str = Field(default='', max_length=4096)
    snapshot_path: str = Field(default='', max_length=4096)
    # Original name of an uploaded source file; the stored upload has a random name.
    source_name: str = Field(default='', max_length=255)
    window_days: int = Field(default=DEFAULT_WINDOW_DAYS, ge=1, le=90)
    limit: int = Field(default=DEFAULT_TRACE_LIMIT, ge=1, le=5000)
    facets: FacetSelection = FacetSelection()
    parallelism: int = Field(default=20, ge=1, le=200)
    labels: list[Preset] = Field(default_factory=list, max_length=len(LABEL_PRESETS))
    custom_labels: list[LabelSpec] = Field(default_factory=list, max_length=10)
    coding_labels: list[str] = Field(default_factory=list, max_length=len(_CODING_PRESETS))
    dimensions: list[DimensionName] = Field(default_factory=_default_dimensions)
    summary_model: str = Field(default='', max_length=200)
    classifier_model: str = Field(default='', max_length=200)
    embedding_model: str = Field(default='', max_length=200)
    compiler_model: str = Field(default='', max_length=200)
    _finder_export_snapshot: str | None = PrivateAttr(default=None)

    def validated_finder_export_snapshot(self) -> str | None:
        """Return the bounded Finder JSON captured during source validation."""
        return self._finder_export_snapshot

    def _validate_snapshot_source(self) -> None:
        if self.source != 'snapshot':
            return
        if self.facets != FacetSelection():
            raise ValueError('A local trace file already fixes the trace population; remove the facet filters.')
        if not self.snapshot_path.strip():
            raise ValueError('Browse to choose a trace file first.')
        path = Path(self.snapshot_path).expanduser()
        try:
            if is_uploaded_source(get_insights_runs_dir(), path):
                raw_snapshot = read_uploaded_source(get_insights_runs_dir(), path, 'snapshot')
                snapshot = Snapshot.model_validate_json(raw_snapshot)
            else:
                snapshot = Snapshot.model_validate_json(path.read_text(encoding='utf-8'))
        except (OSError, ValueError) as exc:
            raise ValueError(f'Could not read a valid local trace snapshot: {exc}') from exc
        if not snapshot.traces:
            raise ValueError('The local trace snapshot contains no traces.')

    @model_validator(mode='after')
    def validate_source(self) -> Self:  # noqa: C901
        if self.source == 'query' and not self.query.strip():
            raise ValueError('Enter a question to find matching traces.')
        if self.source == 'finder':
            if self.facets != FacetSelection():
                raise ValueError('A Finder export already fixes the trace population; remove the facet filters.')
            if not self.finder_export.strip():
                raise ValueError('Enter the path to a Finder JSON export.')
            try:
                root = get_finder_exports_dir().resolve()
                requested = Path(self.finder_export).expanduser()
                path = (requested if requested.is_absolute() else root / requested).resolve()
            except (OSError, RuntimeError) as exc:
                raise ValueError(f'Could not resolve Finder export path: {exc}') from exc
            if path.parent != root and not is_uploaded_source(get_insights_runs_dir(), path):
                raise ValueError(f'Finder exports must be in {root} or use a validated Insights upload.')
            try:
                if is_uploaded_source(get_insights_runs_dir(), path):
                    raw = read_uploaded_source(get_insights_runs_dir(), path, 'finder')
                else:
                    raw = _read_approved_finder_export(root, path)
                too_large = upload_size_error('finder', len(raw))
                if too_large:
                    raise ValueError(too_large)
                RunExport.model_validate_json(raw)
                self._finder_export_snapshot = raw.decode('utf-8')
            except (OSError, ValueError) as exc:
                raise ValueError(f'Could not read a valid Finder export: {exc}') from exc
            self.finder_export = str(path)
        self._validate_snapshot_source()
        if any(name not in LABEL_PRESETS for name in self.labels):
            raise ValueError('Select only available Insights label presets.')
        if any(name not in _CODING_PRESETS for name in self.coding_labels):
            raise ValueError('Select only available coding labels.')
        if len(set(self.labels)) != len(self.labels) or len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError('Select each label and dimension once.')
        if len(set(self.coding_labels)) != len(self.coding_labels):
            raise ValueError('Select each coding label once.')
        custom_names = [spec.name for spec in self.custom_labels]
        for spec in self.custom_labels:
            if not 1 <= len(spec.instructions) <= 2000:
                raise ValueError('Custom label instructions must contain between 1 and 2000 characters.')
            spec.to_question('')
        if len(set(custom_names)) != len(custom_names):
            raise ValueError('Custom label names must be unique.')
        all_names = [*self.labels, *custom_names]
        if len(set(all_names)) != len(all_names):
            raise ValueError('Preset and custom label names must be unique.')
        reserved = {spec.name for spec in CODING_LABELS}
        if reserved.intersection(all_names):
            raise ValueError('Custom labels cannot use names reserved for coding analysis.')
        if not self.labels and not self.custom_labels and not self.dimensions and not self.coding_labels:
            raise ValueError('Select at least one label or dimension.')
        return self

    def population(self) -> InsightsPopulation:
        if self.source == 'finder':
            return InsightsPopulation.from_finder_export(Path(self.finder_export).expanduser())
        if self.source == 'snapshot':
            return InsightsPopulation.from_snapshot(Path(self.snapshot_path).expanduser())
        return InsightsPopulation(
            query=self.query.strip() if self.source == 'query' else None,
            facets=self.facets,
            window_days=self.window_days,
            limit=self.limit,
        )

    @field_validator('summary_model', 'classifier_model', 'embedding_model', 'compiler_model')
    @classmethod
    def _strip_model(cls, value: str) -> str:
        return value.strip()

    def model_overrides(self) -> dict[str, str]:
        """The chosen models as `insights()` keywords; a blank field keeps that function's own default."""
        chosen = {
            'summary_model': self.summary_model,
            'classifier_model': self.classifier_model,
            'embedding_model': self.embedding_model,
            'compiler_model': self.compiler_model,
        }
        return {name: model for name, model in chosen.items() if model}

    def label_specs(self) -> list[LabelSpec]:
        return [*(LABEL_PRESETS[name] for name in self.labels), *self.custom_labels]

    @property
    def coding_enabled(self) -> bool:
        return bool(self.coding_labels)

    def coding_label_specs(self) -> list[LabelSpec]:
        return [_CODING_PRESETS[name] for name in self.coding_labels]

    def dimension_names(self) -> list[DimensionName]:
        return list(self.dimensions)

    def stages(self, population: InsightsPopulation | None = None) -> list[tuple[str, str]]:
        """The `(stage name, title)` plan a run with this spec reports, as `launch_insights` records it."""
        if population is None:
            population, _ = _population_for_launch_plan(self)
        return stage_plan(
            population,
            [*self.label_specs(), *(CODING_LABELS[:1] if self.coding_enabled else ()), *self.coding_label_specs()],
            self.dimension_names(),
        )


class InsightsLaunchPayload(BaseModel):
    run_id: str
    run_name: str
    runs_dir: Path
    spec: InsightsLaunchSpec
    finder_export_snapshot: Path | None = None
    finder_export_snapshot_sha256: str | None = None
    auth_method: str | None = None
    auth_identity: str | None = None


def _population_for_launch_plan(spec: InsightsLaunchSpec) -> tuple[InsightsPopulation, str | None]:
    """Plan against the validated Finder bytes, even if their source path changes."""
    finder_snapshot = spec.validated_finder_export_snapshot()
    if finder_snapshot is None:
        return spec.population(), None
    population = InsightsPopulation.from_finder_export(
        Path(spec.finder_export), export=RunExport.model_validate_json(finder_snapshot)
    )
    return population, finder_snapshot


def launch_insights(
    spec: InsightsLaunchSpec,
    runs_dir: Path,
    *,
    profile: OrqProfile | None = None,
    auth_method: str | None = None,
    auth_identity: str | None = None,
) -> str:
    """Create a visible manifest, then spawn a worker that survives dashboard reloads."""
    run_id = str(uuid.uuid4())
    run_name = spec.name.strip() or f'Insights {datetime.now().astimezone():%Y-%m-%d %H:%M}'
    population, finder_snapshot = _population_for_launch_plan(spec)
    plan = spec.stages(population)
    writer = start_manifest(
        run_id=run_id,
        surface='insights',
        run_name=run_name,
        runs_dir=runs_dir,
        planned_stages=[name for name, _ in plan],
        stage_labels=dict(plan),
        parallelism=spec.parallelism,
    )
    snapshot_path: Path | None = None
    reference_path: Path | None = None
    reference_temporary: Path | None = None
    state_path = worker_state_path(runs_dir, run_id)
    try:
        if finder_snapshot is not None:
            reference_path = finder_export_reference_path(runs_dir, run_id)
            ensure_private_finder_reference_dir(reference_path.parent)
            with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=reference_path.parent, prefix=f'.{run_id}.', suffix='.tmp', delete=False
            ) as reference_file:
                reference_temporary = Path(reference_file.name)
                _set_private_file_mode(reference_file.fileno(), 0o600)
                reference_file.write(json.dumps({'finder_export': spec.finder_export}))
                reference_file.flush()
                os.fsync(reference_file.fileno())
            reference_temporary.replace(reference_path)
            reference_temporary = None
            validate_private_finder_reference(reference_path)
            snapshot_directory = Path(tempfile.mkdtemp(prefix='evaluatorq-finder-snapshot-'))
            snapshot_path = snapshot_directory / 'finder-export.json'
            _write_private_finder_snapshot(snapshot_path, finder_snapshot)
        # Serialize the already validated spec without asking Pydantic to validate
        # the mutable Finder path a second time.
        worker_request = json.dumps({
            'run_id': run_id,
            'run_name': run_name,
            'runs_dir': str(runs_dir),
            'spec': spec.model_dump(mode='json'),
            'auth_method': auth_method,
            'auth_identity': auth_identity,
            'finder_export_snapshot': str(snapshot_path) if snapshot_path is not None else None,
            'finder_export_snapshot_sha256': hashlib.sha256(finder_snapshot.encode('utf-8')).hexdigest()
            if finder_snapshot is not None
            else None,
        })
        worker_env = {
            **os.environ,
            _REQUEST_ENV: worker_request,
            _MANIFEST_ENV: str(writer.path),
        }
        if snapshot_path is not None:
            worker_env[_SNAPSHOT_ENV] = str(snapshot_path)
        else:
            worker_env.pop(_SNAPSHOT_ENV, None)
        _initial_worker_state(state_path, snapshot_path)
        if profile is not None:
            worker_env.update(ORQ_API_KEY=profile.api_key, ORQ_BASE_URL=profile.server or DEFAULT_ORQ_BASE_URL)
        log_dir = runs_dir / '.logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / f'{run_id}.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(
                [sys.executable, '-c', _WORKER_BOOTSTRAP],
                stdin=subprocess.PIPE,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=worker_env,
                start_new_session=True,
            )
        try:
            state = json.loads(state_path.read_text(encoding='utf-8'))
            state['pid'] = process.pid
            state['process_identity'] = _worker_process_identity(process.pid)
            _write_worker_state(state_path, state)
            if process.stdin is None:
                raise OSError('Insights worker handshake is unavailable.')
            process.stdin.write(b'1')
        except (OSError, ValueError, TypeError) as exc:
            logger.warning('Could not record or release Insights worker process {}: {}', run_id, exc)
            raise
        finally:
            # Closing without a release byte stops the child when PID
            # persistence failed. Otherwise the child can start and clean up.
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError as exc:
                    logger.warning('Could not close Insights worker handshake {}: {}', run_id, exc)
    except Exception as exc:  # noqa: BLE001 — every ordinary setup failure must terminate this run.
        try:
            writer.fail(f'Could not start Insights worker: {exc}', stage='start')
        finally:
            for path in (reference_path, reference_temporary):
                if path is not None:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError as cleanup_error:
                        logger.warning('Could not remove Finder export reference {}: {}', path, cleanup_error)
            try:
                state_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                logger.warning('Could not remove Insights worker state {}: {}', state_path, cleanup_error)
            if snapshot_path is not None:
                from evaluatorq.dashboard.insights_worker import _cleanup_snapshot

                _cleanup_snapshot(snapshot_path)
    return run_id


def read_launch_payload() -> InsightsLaunchPayload:
    raw = os.environ.pop(_REQUEST_ENV, None)
    if raw is None:
        raise ValueError('Missing Insights launch request')
    data = json.loads(raw)
    spec_data = dict(data['spec'])
    finder_source = spec_data.get('source') == 'finder'
    finder_path = spec_data.get('finder_export', '')
    if finder_source:
        # Validate all user-selected options while neutralizing the original
        # pathname, which may have changed since launch validation.
        spec_data['source'] = 'recent'
        spec_data['finder_export'] = ''
    data['spec'] = spec_data
    payload = InsightsLaunchPayload.model_validate(data)
    if finder_source:
        payload.spec = payload.spec.model_copy(update={'source': 'finder', 'finder_export': finder_path})
    return payload
