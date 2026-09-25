"""Container plumbing for ``CodingAgentTarget``: options and pure Docker argv helpers.

Nothing here knows about agents or turns. ``coding_agent.py`` builds on these pieces.
"""

from __future__ import annotations

import asyncio
import atexit
import importlib.metadata
import os
import re
import signal
import socket
import subprocess
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

    @model_validator(mode='after')
    def podman_does_not_accept_docker_context(self) -> DockerOptions:
        if self.binary == 'podman' and self.context is not None:
            raise ValueError("Podman does not use Docker's --context; select a connection with CONTAINER_CONNECTION")
        return self

    @field_validator('workdir')
    @classmethod
    def workdir_must_be_absolute(cls, value: str) -> str:
        if not value.startswith('/'):
            raise ValueError(f'workdir must be an absolute container path, got {value!r}')
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


def build_run_argv(opts: DockerOptions, *, name: str, root: Path, uid: int, gid: int) -> list[str]:
    argv = [
        *opts.cli(), 'run', '-d', '--rm', '--init', '--entrypoint', 'sh',
        '--name', name,
        '--label', f'{CONTAINER_LABEL}=1',
        '--label', f'{HOST_PID_LABEL}={os.getpid()}',
        '--label', f'{HOST_LABEL}={socket.gethostname()}',
        '--user', f'{uid}:{gid}',
        '-e', 'HOME=/evq-home',
        '-v', f'{root / "home"}:/evq-home',
        '-v', f'{root / "lease"}:/evq-lease:ro',
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
        if arg in ISOLATION_FLAGS or arg == '--privileged=true' or arg.startswith('--cap-add'):
            found.append(arg)
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
            fields = dict(part.partition('=')[::2] for part in value.split(','))
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
hooks_installed = False
heartbeat_thread: threading.Thread | None = None


def register(name: str, live: LiveContainer) -> None:
    """Track a container and ensure process cleanup and heartbeat are active."""
    LIVE_CONTAINERS[name] = live
    install_exit_hooks()
    start_heartbeat()


def unregister(name: str) -> LiveContainer | None:
    """Forget a container, which also stops renewing its lease."""
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
    command = ' '.join([*cli_prefix(binary, context), 'rm', '-f', *pending])
    logger.error(
        f'could not remove container(s) {", ".join(pending)}: {reason}. Their lease ends them within '
        f'{LEASE_CHECK_S * 2}s; to remove now run: {command}'
    )
    return pending


async def remove_containers_async(binary: str, context: str | None, names: Sequence[str]) -> list[str]:
    """Run the synchronous removal primitive off the event loop."""
    return await asyncio.to_thread(remove_containers, binary, context, names)


def remove_all(reason: str) -> None:
    """Remove all registered containers, grouped by engine and context."""
    snapshot = list(LIVE_CONTAINERS.items())
    if not snapshot:
        return
    groups: dict[tuple[str, str | None], list[str]] = {}
    for name, live in snapshot:
        unregister(name)
        groups.setdefault((live.binary, live.context), []).append(name)
    for (binary, context), names in groups.items():
        logger.warning(f'host_exit ({reason}): removing container(s) {", ".join(names)}')
        remove_containers(binary, context, names)


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
    for name, live in list(LIVE_CONTAINERS.items()):
        try:
            write_beat(live.beat, counter)
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
        remove_all(signal.Signals(signum).name)
        if callable(previous):
            previous(signum, frame)
        elif previous != signal.SIG_IGN:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    return handler


def install_exit_hooks() -> None:
    """Install atexit and signal cleanup hooks once, when first container is registered."""
    global hooks_installed
    with HOOKS_LOCK:
        if hooks_installed:
            return
        hooks_installed = True
    atexit.register(remove_all, 'atexit')
    if threading.current_thread() is not threading.main_thread():
        logger.warning(
            'container cleanup: not on the main thread, so no SIGTERM/SIGHUP handler; '
            'relying on atexit, labels and the lease'
        )
        return
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, make_signal_handler(signal.getsignal(sig)))


def pid_alive(pid: int) -> bool:
    """Return whether a PID exists or is inaccessible to this process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


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
