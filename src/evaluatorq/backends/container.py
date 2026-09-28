"""Container options, Docker argv, lease heartbeats, and cleanup for coding agent targets."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import ctypes
import importlib.metadata
import os
import posixpath
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import FrameType


def image_tag_version() -> str:
    """Return evaluatorq's version as a Docker-safe tag."""
    try:
        version = importlib.metadata.version('evaluatorq')
    except importlib.metadata.PackageNotFoundError:
        version = '0.0.0.dev0'
    return re.sub(r'[^A-Za-z0-9_.-]', '-', version)


DEFAULT_CODING_AGENT_IMAGE = f'evaluatorq-coding-agent:{image_tag_version()}'
CONTAINER_LABEL = 'evaluatorq.coding-agent'
HOST_PID_LABEL = 'evaluatorq.host-pid'
HOST_LABEL = 'evaluatorq.host'
RESERVED_ENV = ('HOME', 'PATH')
HEARTBEAT_S = 30
LEASE_CHECK_S = 150
ORQ_ENV = ('ORQ_API_KEY', 'ORQ_BASE_URL')
ISOLATION_FLAGS = frozenset({
    '--privileged',
    '--pid=host',
    '--network=host',
    '--net=host',
    '--ipc=host',
    '--userns=host',
    '--uts=host',
})
ISOLATION_PAIRS = frozenset({'--pid', '--network', '--net', '--ipc', '--userns', '--uts'})
RESERVED_LABELS = frozenset({CONTAINER_LABEL, HOST_PID_LABEL, HOST_LABEL})
_MANAGED_VALUE_FLAGS = frozenset({'--entrypoint', '--user', '-u'})
# The packaged image's `agent` user. A Windows host has no uid to map, so the container runs as this one.
IMAGE_AGENT_ID = 1001
# Win32 constants for windows_pid_alive.
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
ERROR_ACCESS_DENIED = 5
STILL_ACTIVE = 259


def host_ids() -> tuple[int, int]:
    """Return the uid and gid the container runs as.

    The host user's on POSIX, so files written to the mounts stay theirs; the image's `agent` user (1001) on
    Windows, which has no uid to map.
    """
    if hasattr(os, 'getuid'):
        return os.getuid(), os.getgid()
    return IMAGE_AGENT_ID, IMAGE_AGENT_ID


class DockerOptions(BaseModel):
    """Options for running a coding agent in a container."""

    model_config = ConfigDict(frozen=True)

    image: str = DEFAULT_CODING_AGENT_IMAGE
    binary: str = 'docker'
    context: str | None = None
    workdir: str = '/work'
    name_prefix: str = 'evq'
    run_args: tuple[str, ...] = ()
    pass_env: tuple[str, ...] | None = None
    allow_privilege_escalation: bool = False

    @field_validator('name_prefix')
    @classmethod
    def name_prefix_must_be_safe_component(cls, value: str) -> str:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
            raise ValueError('name_prefix must be a Docker-safe single path component')
        return value

    @model_validator(mode='after')
    def podman_does_not_accept_docker_context(self) -> DockerOptions:
        if self.binary == 'podman' and self.context is not None:
            raise ValueError("Podman does not use Docker's --context; select a connection with CONTAINER_CONNECTION")
        validate_managed_run_args(self.run_args, allow_privilege_escalation=self.allow_privilege_escalation)
        if raw_environment_flags(self.run_args):
            logger.warning('DockerOptions.run_args contains raw environment flags; these bypass pass_env filtering')
        return self

    @field_validator('workdir')
    @classmethod
    def workdir_must_be_absolute(cls, value: str) -> str:
        if not value.startswith('/'):
            raise ValueError(f'workdir must be an absolute container path, got {value!r}')
        normalized = posixpath.normpath('/' + value.lstrip('/'))
        managed_mounts = ('/evq-lease', '/evq-home')
        if any(
            normalized == mount or normalized.startswith(f'{mount}/') or mount.startswith(f'{normalized.rstrip("/")}/')
            for mount in managed_mounts
        ):
            raise ValueError(f'workdir {value!r} overlaps evaluatorq-managed container mounts /evq-lease and /evq-home')
        return value

    def cli(self) -> list[str]:
        return cli_prefix(self.binary, self.context)


def cli_prefix(binary: str, context: str | None) -> list[str]:
    return [binary, '--context', context] if context else [binary]


def watchdog_script(check_s: int = LEASE_CHECK_S) -> str:
    """Return a BusyBox-compatible process that exits after the host lease stops changing."""
    return (
        f'p=; while sleep {check_s}; do c=$(cat /evq-lease/beat) || break; '
        '[ "$c" != "$p" ] || break; p=$c; done; '
        'echo "lease_expired $(date)" > /evq-home/.evq-exit'
    )


def build_run_argv(opts: DockerOptions, *, name: str, root: Path, lease_dir: Path, uid: int, gid: int) -> list[str]:
    """Build a container command, mounting its private lease beside shared workdir state."""
    argv = [
        *opts.cli(), 'run', '-d', '--rm', '--init', '--entrypoint', 'sh',
        '--name', name,
        '--label', f'{CONTAINER_LABEL}=1',
        '--label', f'{HOST_PID_LABEL}={os.getpid()}',
        '--label', f'{HOST_LABEL}={socket.gethostname()}',
        '--user', f'{uid}:{gid}',
        '-e', 'HOME=/evq-home',
        '-v', f'{root / "home"}:/evq-home',
        '-v', f'{lease_dir}:/evq-lease:ro',
        '-v', f'{root / "work"}:{opts.workdir}', '-w', opts.workdir,
    ]  # fmt: skip
    if not opts.allow_privilege_escalation:
        argv += ['--security-opt', 'no-new-privileges']
    return [*argv, *opts.run_args, opts.image, '-c', watchdog_script()]


def build_exec_argv(
    opts: DockerOptions, *, name: str, env_names: Sequence[str], agent_argv: Sequence[str]
) -> list[str]:
    env_flags = [flag for key in env_names for flag in ('-e', key)]
    return [*opts.cli(), 'exec', '-i', '-w', opts.workdir, *env_flags, name, 'evq-entrypoint', *agent_argv]


def forwarded_env_names(
    opts: DockerOptions, *, launcher: str, provider_env: Sequence[str], caller_env: Mapping[str, str]
) -> list[str]:
    """Return set environment names, with caller env keys taking precedence in order."""
    selected = opts.pass_env if opts.pass_env is not None else ORQ_ENV if launcher == 'orq' else tuple(provider_env)
    names = dict.fromkeys([*caller_env, *(key for key in selected if key in os.environ)])
    return [key for key in names if key not in RESERVED_ENV]


def isolation_breaking_flags(run_args: Sequence[str]) -> list[str]:
    found: list[str] = []
    for i, arg in enumerate(run_args):
        flag, eq, value = arg.partition('=')
        privileged_enabled = flag == '--privileged' and eq and value.lower() in ('1', 't', 'true')
        if arg in ISOLATION_FLAGS or privileged_enabled or arg.startswith('--cap-add'):
            found.append(arg)
        elif flag == '--volumes-from':
            found.append(arg if eq else f'{arg} {run_args[i + 1]}' if i + 1 < len(run_args) else arg)
        elif arg in ISOLATION_PAIRS and i + 1 < len(run_args) and run_args[i + 1] == 'host':
            found.append(f'{arg} host')
    return found


def unsafe_mounts(run_args: Sequence[str], root: Path) -> list[str]:
    """Return bind-mount host sources outside ``root`` and relative host sources."""
    args = list(run_args)
    sources: list[tuple[str, bool]] = []
    for i, arg in enumerate(args):
        flag, eq, value = arg.partition('=')
        if arg in ('-v', '--volume', '--mount'):
            if i + 1 >= len(args):
                continue
            flag, value = arg, args[i + 1]
        elif flag in ('--volume', '--mount') and eq:
            pass
        elif arg.startswith('-v') and len(arg) > 2:
            flag, value = '-v', arg[2:]
        else:
            continue
        if flag == '--mount':
            fields: dict[str, str] = {}
            for part in value.split(','):
                key, _, field_value = part.partition('=')
                fields[key] = field_value
            src = fields.get('source') or fields.get('src')
            if fields.get('type') == 'volume':
                src = None
        else:
            src = value.split(':', 1)[0] if ':' in value else None
        if src:
            sources.append((src, flag == '--mount'))
    resolved_root = root.resolve()
    unsafe: list[str] = []
    for src, mount_source in sources:
        if src.startswith('/'):
            if not Path(src).resolve().is_relative_to(resolved_root):
                unsafe.append(src)
        elif mount_source or src.startswith(('.', '~')) or '/' in src:
            unsafe.append(src)
    return unsafe


def raw_environment_flags(run_args: Sequence[str]) -> list[str]:
    """Return raw Docker environment flags that bypass ``pass_env`` selection."""
    return [
        arg
        for arg in run_args
        if arg in ('-e', '--env', '--env-file') or (arg.startswith(('--env=', '--env-file=', '-e')) and arg != '-e')
    ]


def validate_managed_run_args(run_args: Sequence[str], *, allow_privilege_escalation: bool = False) -> None:
    """Reject run flags that replace lifecycle controls supplied by evaluatorq."""
    args = list(run_args)
    violations: list[str] = []
    for i, arg in enumerate(args):
        flag, eq, value = arg.partition('=')
        mount_flag = flag if flag in ('--volume', '-v', '--mount') else '-v' if arg.startswith('-v') else None
        if not mount_flag:
            mount_value = ''
        elif eq:
            mount_value = value
        elif mount_flag == '-v' and arg != '-v':
            mount_value = arg[2:]
        else:
            mount_value = args[i + 1] if i + 1 < len(args) else ''
        if flag in _MANAGED_VALUE_FLAGS or (arg.startswith('-u') and arg != '-u'):
            violations.append(arg)
            continue
        if flag == '--name' or (
            flag == '--label'
            and _label_key(value if eq else args[i + 1] if i + 1 < len(args) else '') in RESERVED_LABELS
        ):
            violations.append(arg if eq else f'{arg} {args[i + 1] if i + 1 < len(args) else ""}'.rstrip())
            continue
        if mount_flag:
            target = _mount_target(mount_flag, mount_value)
            if _is_reserved_mount_target(target):
                violations.append(arg if eq or (arg.startswith('-v') and arg != '-v') else f'{arg} {mount_value}')
        if flag == '--rm' and eq and value.lower() in ('f', 'false', '0'):
            violations.append(arg)
        if not allow_privilege_escalation and flag == '--security-opt':
            security_opt = value if eq else args[i + 1] if i + 1 < len(args) else ''
            if security_opt.lower() == 'no-new-privileges=false':
                violations.append(arg if eq else f'{arg} {security_opt}')
    if violations:
        raise ValueError(
            'run_args cannot override evaluatorq-managed entrypoint, user, auto-remove, lease/home mounts, '
            'or no-new-privileges unless privilege escalation is enabled: '
            + ', '.join(repr(value) for value in violations)
        )


def _mount_target(flag: str, value: str) -> str | None:
    if flag == '--mount':
        fields: dict[str, str] = {}
        for part in value.split(','):
            key, _, field_value = part.partition('=')
            fields[key] = field_value
        return fields.get('target') or fields.get('dst') or fields.get('destination')
    parts = value.split(':')
    if len(parts) > 1:
        return parts[1]
    return value if value.startswith('/') else None


def _is_reserved_mount_target(target: str | None) -> bool:
    if target is None:
        return False
    normalized = posixpath.normpath(target)
    return normalized in ('/evq-lease', '/evq-home') or normalized.startswith(('/evq-lease/', '/evq-home/'))


def _label_key(label: str) -> str:
    return label.partition('=')[0]


class LiveContainer(BaseModel):
    """Identity and lease file for a container owned by this process."""

    model_config = ConfigDict(frozen=True)

    binary: str
    context: str | None
    beat: Path


# Name to container for every container this process started and has not removed.
# Mutations are synchronous; readers iterate snapshots because heartbeat runs in a thread.
LIVE_CONTAINERS: dict[str, LiveContainer] = {}
BEAT_WARNED: set[str] = set()
HOOKS_LOCK = threading.Lock()
CONTAINER_LOCK = threading.RLock()
atexit_installed = False
signal_hooks_installed = False
heartbeat_thread: threading.Thread | None = None
_installed_signal_handlers: dict[signal.Signals, Any] = {}


def _reset_after_fork() -> None:
    """Drop parent-owned container state and synchronization primitives in the child."""
    globals().update(
        LIVE_CONTAINERS={},
        BEAT_WARNED=set(),
        CONTAINER_LOCK=threading.RLock(),
        HOOKS_LOCK=threading.Lock(),
    )
    global heartbeat_thread, signal_hooks_installed
    heartbeat_thread = None

    # The atexit callback is inherited with the process and remains useful for child-owned containers.
    # Keep its flag set so register() does not stack a second copy in the child.
    for sig, previous in _installed_signal_handlers.items():
        with contextlib.suppress(OSError, ValueError):
            signal.signal(sig, previous)
    _installed_signal_handlers.clear()
    signal_hooks_installed = False


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_reset_after_fork)


def register(name: str, live: LiveContainer) -> None:
    """Track a container and ensure process cleanup and heartbeat are active."""
    with CONTAINER_LOCK:
        LIVE_CONTAINERS[name] = live
    install_exit_hooks()
    start_heartbeat()


def unregister(name: str) -> LiveContainer | None:
    """Forget a container, which also stops renewing its lease."""
    with CONTAINER_LOCK:
        BEAT_WARNED.discard(name)
        return LIVE_CONTAINERS.pop(name, None)


def remove_containers(binary: str, context: str | None, names: Sequence[str], *, timeout_s: float = 10) -> list[str]:
    """Remove containers in one batch, retry once, and return names that still fail."""
    pending = list(dict.fromkeys(names))
    reason = 'unknown removal error'
    for _attempt in range(2):
        if not pending:
            return []
        argv = [*cli_prefix(binary, context), 'rm', '-f', *pending]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            reason = str(exc)
        else:
            if proc.returncode == 0:
                return []
            lines = [line for line in proc.stderr.splitlines() if line.strip()]

            def mentions(line: str, name: str) -> bool:
                return re.search(rf'(?<![A-Za-z0-9_.-]){re.escape(name)}(?![A-Za-z0-9_.-])', line) is not None

            missing = {
                name for name in pending if any(mentions(line, name) and 'No such container' in line for line in lines)
            }
            failed = [name for name in pending if name not in missing and any(mentions(line, name) for line in lines)]
            if not failed and lines and all('No such container' in line for line in lines):
                return []
            pending = failed or [name for name in pending if name not in missing] or pending
            reason = proc.stderr.strip() or f'command exited with status {proc.returncode}'
    command = shlex.join([*cli_prefix(binary, context), 'rm', '-f', *pending])
    logger.error(
        f'could not remove container(s) {", ".join(pending)}: {reason}. Their lease ends them within '
        f'{LEASE_CHECK_S * 2}s; to remove now run: {command}'
    )
    return pending


async def remove_containers_async(binary: str, context: str | None, names: Sequence[str]) -> list[str]:
    """Run the synchronous removal primitive off the event loop."""
    return await asyncio.to_thread(remove_containers, binary, context, names)


def remove_all(reason: str, *, max_cleanup_s: float | None = None) -> None:
    """Remove registered containers; bound signal cleanup so termination can proceed."""
    deadline = time.monotonic() + max_cleanup_s if max_cleanup_s is not None else None
    with CONTAINER_LOCK:
        snapshot = list(LIVE_CONTAINERS.items())
    if not snapshot:
        return
    groups: dict[tuple[str, str | None], list[str]] = {}
    for name, live in snapshot:
        unregister(name)
        groups.setdefault((live.binary, live.context), []).append(name)
    for (binary, context), names in groups.items():
        remaining = deadline - time.monotonic() if deadline is not None else None
        if remaining is not None and remaining <= 0:
            logger.warning('host_exit cleanup deadline reached; remaining containers will expire with their leases')
            break
        logger.warning(f'host_exit ({reason}): removing container(s) {", ".join(names)}')
        if remaining is None:
            remove_containers(binary, context, names)
        else:
            remove_containers(binary, context, names, timeout_s=remaining / 2)


def release_containers(owned: list[str]) -> None:
    """Finalizer target that removes containers owned by an unclosed target."""
    for name in owned:
        live = unregister(name)
        if live is not None:
            logger.warning(f'CodingAgentTarget was garbage-collected without close(); removing container {name}')
            remove_containers(live.binary, live.context, [name])


def write_beat(path: Path, value: int) -> None:
    """Atomically replace a lease heartbeat counter."""
    temporary = path.with_name(f'{path.name}.tmp')
    temporary.write_text(str(value))
    temporary.replace(path)


def heartbeat_once(counter: int) -> None:
    """Renew all currently registered leases, warning only once for each failing name."""
    with CONTAINER_LOCK:
        for name, live in list(LIVE_CONTAINERS.items()):
            try:
                write_beat(live.beat, counter)
                BEAT_WARNED.discard(name)
            except OSError as exc:  # noqa: PERF203
                if name not in BEAT_WARNED:
                    BEAT_WARNED.add(name)
                    logger.warning(f'heartbeat for container {name} failed ({exc}); its lease ends it if this persists')


def heartbeat_loop() -> None:
    counter = 0
    while True:
        time.sleep(HEARTBEAT_S)
        counter += 1
        heartbeat_once(counter)


def start_heartbeat() -> None:
    """Start one daemon thread to renew leases independently of the event loop."""
    global heartbeat_thread
    with HOOKS_LOCK:
        if heartbeat_thread is None:
            heartbeat_thread = threading.Thread(
                target=heartbeat_loop, name='evaluatorq-container-heartbeat', daemon=True
            )
            heartbeat_thread.start()


def make_signal_handler(previous: Any) -> Callable[[int, FrameType | None], None]:
    """Build a signal handler that cleans up then chains or restores default behavior."""

    def handler(signum: int, frame: FrameType | None) -> None:
        remove_all(signal.Signals(signum).name, max_cleanup_s=5)
        if callable(previous):
            previous(signum, frame)
        elif previous != signal.SIG_IGN:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    return handler


def install_exit_hooks() -> None:
    """Install atexit cleanup once and signal hooks when called on the main thread."""
    global atexit_installed, signal_hooks_installed
    on_main_thread = threading.current_thread() is threading.main_thread()
    with HOOKS_LOCK:
        should_register_atexit = not atexit_installed
        if should_register_atexit:
            atexit_installed = True
        should_install_signals = on_main_thread and not signal_hooks_installed
        if should_install_signals:
            signal_hooks_installed = True
    if should_register_atexit:
        atexit.register(remove_all, 'atexit')
    if not on_main_thread:
        logger.warning(
            'container cleanup: not on the main thread, so no SIGTERM/SIGHUP handler; '
            'relying on atexit, labels and the lease'
        )
        return
    if should_install_signals:
        # Windows has no SIGHUP.
        for sig in (signal.SIGTERM, *((signal.SIGHUP,) if hasattr(signal, 'SIGHUP') else ())):
            previous = signal.getsignal(sig)
            _installed_signal_handlers[sig] = previous
            signal.signal(sig, make_signal_handler(previous))


def pid_alive(pid: int) -> bool:
    """Return whether a PID exists or is inaccessible to this process."""
    if sys.platform == 'win32':
        return windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def windows_pid_alive(pid: int) -> bool:
    """Windows liveness check. `os.kill(pid, 0)` is not a probe there: signal 0 is CTRL_C_EVENT.

    Limit: a process that exited with code 259 reads as alive, because 259 is also `STILL_ACTIVE`, so
    `sweep_orphans` leaves its container to the lease.
    """
    if sys.platform != 'win32':
        raise RuntimeError('windows_pid_alive is Windows only')
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong))
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
    if not handle:
        return ctypes.get_last_error() == ERROR_ACCESS_DENIED  # exists, owned by someone else
    try:
        code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            logger.warning(
                f'pid liveness: GetExitCodeProcess failed for pid {pid} (error {ctypes.get_last_error()}); '
                'treating it as alive'
            )
            return True
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def sweep_orphans(binary: str, context: str | None) -> None:
    """Remove labelled containers on this host whose owning process is no longer alive."""
    fmt = f'{{{{.Names}}}}\t{{{{.Label "{HOST_PID_LABEL}"}}}}\t{{{{.Label "{HOST_LABEL}"}}}}'
    argv = [*cli_prefix(binary, context), 'ps', '-a', '--filter', f'label={CONTAINER_LABEL}=1', '--format', fmt]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(f'orphan sweep skipped: {exc}')
        return
    if proc.returncode != 0:
        logger.warning(f'orphan sweep skipped: {proc.stderr.strip()}')
        return
    host, dead = socket.gethostname(), []
    for line in proc.stdout.splitlines():
        name, pid, owner = [*line.split('\t'), '', ''][:3]
        if owner != host:
            continue
        if not pid.isdigit():
            logger.warning(
                f'orphan sweep: container {name} has an unreadable host-pid label {pid!r}; leaving it to its lease'
            )
            continue
        if not pid_alive(int(pid)):
            logger.warning(f'orphan_sweep: removing container {name}, its host process {pid} is gone')
            dead.append(name)
    remove_containers(binary, context, dead)
