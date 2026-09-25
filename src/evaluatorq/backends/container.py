"""Container plumbing for ``CodingAgentTarget``: options and pure Docker argv helpers.

Nothing here knows about agents or turns. ``coding_agent.py`` builds on these pieces.
"""

from __future__ import annotations

import importlib.metadata
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, field_validator

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


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
    import socket

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
        if arg in ISOLATION_FLAGS or arg.startswith('--cap-add'):
            found.append(arg)
        elif arg in ISOLATION_PAIRS and i + 1 < len(run_args) and run_args[i + 1] == 'host':
            found.append(f'{arg} host')
    return found


def unsafe_mounts(run_args: Sequence[str], root: Path) -> list[str]:
    """Return bind-mount host sources outside ``root`` and relative host sources."""
    args = list(run_args)
    sources: list[str] = []
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
        else:
            src = value.split(':', 1)[0] if ':' in value else None
        if src:
            sources.append(src)
    resolved_root = root.resolve()
    unsafe: list[str] = []
    for src in sources:
        if src.startswith('/'):
            if not Path(src).resolve().is_relative_to(resolved_root):
                unsafe.append(src)
        elif src.startswith(('.', '~')) or '/' in src:
            unsafe.append(src)
    return unsafe
