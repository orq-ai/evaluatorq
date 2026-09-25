"""Pure pieces of container mode: options, argv, env selection, run_args checks."""

from __future__ import annotations

import os
import re
import shutil
import signal
import stat
import subprocess
import time
from pathlib import Path

import pydantic
import pytest
from loguru import logger

from evaluatorq.backends import DockerOptions, container as c
from evaluatorq.backends.coding_agent import AgentSpec, CodingAgentTarget
from evaluatorq.backends.container import (
    DEFAULT_CODING_AGENT_IMAGE,
    build_exec_argv,
    build_run_argv,
    forwarded_env_names,
    isolation_breaking_flags,
    unsafe_mounts,
    watchdog_script,
)


def test_default_image_tag_is_docker_safe() -> None:
    name, tag = DEFAULT_CODING_AGENT_IMAGE.split(':')
    assert name == 'evaluatorq-coding-agent'
    assert re.fullmatch(r'[A-Za-z0-9_.-]+', tag)


def test_options_frozen_and_workdir_absolute() -> None:
    with pytest.raises(pydantic.ValidationError):
        DockerOptions(workdir='app')
    opts = DockerOptions()
    with pytest.raises(pydantic.ValidationError):
        opts.image = 'x'  # pyright: ignore[reportAttributeAccessIssue]


def test_podman_context_is_rejected() -> None:
    with pytest.raises(pydantic.ValidationError, match='CONTAINER_CONNECTION'):
        DockerOptions(binary='podman', context='remote')


def test_run_argv(tmp_path: Path) -> None:
    opts = DockerOptions(image='img:1', context='orbstack', workdir='/app', run_args=('--memory', '4g'))
    argv = build_run_argv(opts, name='evq-claude-abcd1234', root=tmp_path, uid=501, gid=20)
    assert argv[:4] == ['docker', '--context', 'orbstack', 'run']
    expected_parts = (
        ['-d', '--rm', '--init', '--entrypoint', 'sh'],
        ['--name', 'evq-claude-abcd1234'],
        ['--label', 'evaluatorq.coding-agent=1'],
        ['--user', '501:20'],
        ['-e', 'HOME=/evq-home'],
        ['-v', f'{tmp_path / "home"}:/evq-home'],
        ['-v', f'{tmp_path / "lease"}:/evq-lease:ro'],
        ['-v', f'{tmp_path / "work"}:/app', '-w', '/app'],
        ['--security-opt', 'no-new-privileges'],
        ['--memory', '4g', 'img:1', '-c'],
    )
    for part in expected_parts:
        assert any(argv[i : i + len(part)] == part for i in range(len(argv) - len(part) + 1)), part
    assert argv[-1] == watchdog_script()


def test_privilege_escalation_opt_in(tmp_path: Path) -> None:
    argv = build_run_argv(DockerOptions(allow_privilege_escalation=True), name='n', root=tmp_path, uid=1, gid=1)
    assert 'no-new-privileges' not in argv


def test_exec_argv_has_no_values() -> None:
    argv = build_exec_argv(DockerOptions(), name='n', env_names=['ANTHROPIC_API_KEY'], agent_argv=['claude', '-p'])
    assert argv == ['docker', 'exec', '-i', '-w', '/work', '-e', 'ANTHROPIC_API_KEY', 'n', 'evq-entrypoint', 'claude', '-p']


def test_env_names_follow_launcher(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'ORQ_API_KEY', 'ORQ_BASE_URL'):
        monkeypatch.setenv(key, 'v')
    orq = forwarded_env_names(DockerOptions(), launcher='orq', provider_env=('ANTHROPIC_API_KEY',), caller_env={})
    assert orq == ['ORQ_API_KEY', 'ORQ_BASE_URL']
    direct = forwarded_env_names(DockerOptions(), launcher='direct', provider_env=('ANTHROPIC_API_KEY',), caller_env={})
    assert direct == ['ANTHROPIC_API_KEY']
    explicit = forwarded_env_names(
        DockerOptions(pass_env=('OPENAI_API_KEY', 'NOT_SET')), launcher='orq', provider_env=(), caller_env={'X': '1'}
    )
    assert explicit == ['X', 'OPENAI_API_KEY']


def test_path_and_home_never_forwarded() -> None:
    names = forwarded_env_names(DockerOptions(), launcher='orq', provider_env=(), caller_env={'PATH': '/x', 'HOME': '/h', 'K': 'v'})
    assert names == ['K']


def test_isolation_breaking_flags() -> None:
    assert isolation_breaking_flags(['--privileged', '--pid=host', '--pid', 'host', '--network=host', '--cpus', '2']) == [
        '--privileged', '--pid=host', '--pid host', '--network=host'
    ]


def test_unsafe_mounts(tmp_path: Path) -> None:
    inside = tmp_path / 'work' / 'data'
    args = [
        '-v', '/:/host',
        '--volume=/etc:/e',
        '-v/var:/v',
        '--mount', 'type=bind,source=/,target=/host',
        '--mount=type=bind,src=./rel,target=/r',
        '--mount=type=bind,source=secrets,target=/secrets',
        '-v', f'{inside}:/in',
        '-v', 'named-volume:/data',
    ]
    assert unsafe_mounts(args, tmp_path) == ['/', '/etc', '/var', '/', './rel', 'secrets']


def test_isolation_warning_includes_privileged_true() -> None:
    assert isolation_breaking_flags(['--privileged=true']) == ['--privileged=true']


def test_agent_spec_container_fields_are_required() -> None:
    base = AgentSpec.model_fields
    for field in ('container_permission', 'container_extra_args', 'provider_env'):
        assert base[field].is_required(), field


def test_bypass_defaults_and_caller_wins() -> None:
    claude = CodingAgentTarget('claude', container=DockerOptions())
    assert claude._permission_mode == 'bypassPermissions'
    codex = CodingAgentTarget('codex', container=DockerOptions(), permission_mode='read-only')
    assert codex._permission_mode == 'read-only'
    opencode = CodingAgentTarget('opencode', container=DockerOptions(), extra_args=['--auto'])
    assert opencode._extra_args.count('--auto') == 1
    assert CodingAgentTarget('opencode', container=DockerOptions())._extra_args == ['--auto']
    assert CodingAgentTarget('claude')._permission_mode is None


def test_new_carries_container_options() -> None:
    opts = DockerOptions(image='i:1', workdir='/app', name_prefix='task-7', run_args=('--cpus', '1'))
    clone = CodingAgentTarget('claude', container=opts).new()
    assert clone._container == opts
    assert clone._permission_mode == 'bypassPermissions'


def test_new_keeps_caller_values_before_container_defaults() -> None:
    implicit = CodingAgentTarget('claude', container=DockerOptions())
    assert implicit._kwargs['permission_mode'] is None
    assert implicit._kwargs['extra_args'] is None
    assert implicit.new()._permission_mode == 'bypassPermissions'

    explicit = CodingAgentTarget(
        'claude', container=DockerOptions(), permission_mode='read-only', extra_args=['--custom']
    )
    assert explicit._kwargs['permission_mode'] == 'read-only'
    assert explicit._kwargs['extra_args'] == ['--custom']
    clone = explicit.new()
    assert clone._permission_mode == 'read-only'
    assert clone._extra_args == ['--custom']


def test_run_args_and_reserved_env_warn_at_construction() -> None:
    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(str(m)), level='WARNING')
    try:
        CodingAgentTarget('claude', container=DockerOptions(run_args=('--privileged',)), env={'PATH': '/x'})
    finally:
        logger.remove(sink)
    assert any('--privileged' in s for s in seen)
    assert any('PATH' in s and 'not forwarded' in s for s in seen)


# Lifecycle cleanup tests exercise the real subprocess boundary with a tiny Docker stand-in.

FAKE_DOCKER = r"""#!/bin/sh
printf '%s\n' "$*" >> "$FAKE_LOG"
case "$1" in
  rm) shift 2; for n in "$@"; do
        case " $FAKE_RM_MISSING " in *" $n "*) echo "Error response from daemon: No such container: $n" >&2; st=1;; esac
        case " $FAKE_RM_FAIL " in *" $n "*) echo "Error response from daemon: cannot remove $n: busy" >&2; st=1;; esac
      done; exit ${st:-0};;
  ps) printf '%b' "$FAKE_PS"; exit 0;;
esac
"""


@pytest.fixture
def fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Path]:
    script = tmp_path / 'docker'
    script.write_text(FAKE_DOCKER)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / 'log'
    monkeypatch.setenv('FAKE_LOG', str(log))
    monkeypatch.setattr(c, 'LIVE_CONTAINERS', {})
    return str(script), log


def test_remove_treats_missing_as_success(fake_docker, monkeypatch) -> None:
    binary, log = fake_docker
    monkeypatch.setenv('FAKE_RM_MISSING', 'a')
    assert c.remove_containers(binary, None, ['a', 'b']) == []
    assert log.read_text().splitlines() == ['rm -f a b']


def test_remove_retries_once_then_errors(fake_docker, monkeypatch) -> None:
    binary, log = fake_docker
    monkeypatch.setenv('FAKE_RM_FAIL', 'b')
    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(str(m)), level='ERROR')
    try:
        assert c.remove_containers(binary, None, ['a', 'b']) == ['b']
    finally:
        logger.remove(sink)
    assert log.read_text().splitlines() == ['rm -f a b', 'rm -f b']
    assert any(f'{binary} rm -f b' in s for s in seen)


def test_heartbeat_writes_registered_and_warns_once(fake_docker, tmp_path) -> None:
    good, bad = tmp_path / 'g', tmp_path / 'missing' / 'x'
    good.mkdir()
    c.LIVE_CONTAINERS['g'] = c.LiveContainer(binary='docker', context=None, beat=good / 'beat')
    c.LIVE_CONTAINERS['b'] = c.LiveContainer(binary='docker', context=None, beat=bad / 'beat')
    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(str(m)), level='WARNING')
    try:
        c.heartbeat_once(1)
        c.heartbeat_once(2)
    finally:
        logger.remove(sink)
    assert (good / 'beat').read_text() == '2'
    assert sum('heartbeat' in s for s in seen) == 1
    c.unregister('g')
    c.heartbeat_once(3)
    assert (good / 'beat').read_text() == '2'


@pytest.mark.parametrize('shell', ['sh', 'busybox'])
def test_watchdog_script_lifecycle(tmp_path: Path, shell: str) -> None:
    if shutil.which(shell) is None:
        pytest.skip(f'{shell} not installed')
    lease, home = tmp_path / 'evq-lease', tmp_path / 'evq-home'
    lease.mkdir()
    home.mkdir()
    script = c.watchdog_script(check_s=1).replace('/evq-lease', str(lease)).replace('/evq-home', str(home))
    argv = [shell, 'sh', '-c', script] if shell == 'busybox' else ['sh', '-c', script]
    (lease / 'beat').write_text('0')
    proc = subprocess.Popen(argv)
    for n in range(1, 4):
        time.sleep(0.9)
        (lease / 'beat').write_text(str(n))
    assert proc.poll() is None
    assert proc.wait(timeout=5) == 0
    assert (home / '.evq-exit').read_text().startswith('lease_expired')
    (home / '.evq-exit').unlink()
    proc = subprocess.Popen(argv)
    (lease / 'beat').unlink()
    assert proc.wait(timeout=5) == 0


def test_sweep_removes_dead_owner_on_this_host_only(fake_docker, monkeypatch) -> None:
    import socket

    binary, log = fake_docker
    host = socket.gethostname()
    monkeypatch.setenv(
        'FAKE_PS',
        f'dead\\t999999999\\t{host}\\nalive\\t{os.getpid()}\\t{host}\\nelsewhere\\t999999999\\tother-box\\nodd\\tabc\\t{host}\\n',
    )
    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(str(m)), level='WARNING')
    try:
        c.sweep_orphans(binary, None)
    finally:
        logger.remove(sink)
    assert log.read_text().splitlines()[-1] == 'rm -f dead'
    assert any('orphan_sweep' in s and 'dead' in s for s in seen)
    assert any('odd' in s and 'abc' in s for s in seen)


def test_remove_all_groups_and_logs_reason(fake_docker, tmp_path) -> None:
    binary, log = fake_docker
    for name in ('a', 'b'):
        c.LIVE_CONTAINERS[name] = c.LiveContainer(binary=binary, context=None, beat=tmp_path / name)
    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(str(m)), level='WARNING')
    try:
        c.remove_all('SIGTERM')
    finally:
        logger.remove(sink)
    assert log.read_text().splitlines() == ['rm -f a b']
    assert c.LIVE_CONTAINERS == {}
    assert any('host_exit' in s and 'SIGTERM' in s for s in seen)


def test_signal_handler_cleans_up_and_chains(fake_docker, tmp_path, monkeypatch) -> None:
    binary, log = fake_docker
    c.LIVE_CONTAINERS['a'] = c.LiveContainer(binary=binary, context=None, beat=tmp_path / 'a')
    called: list[int] = []
    handler = c.make_signal_handler(lambda signum, frame: called.append(signum))
    handler(signal.SIGTERM, None)
    assert called == [signal.SIGTERM]
    assert log.read_text().splitlines() == ['rm -f a']


def test_release_containers_for_finalizer(fake_docker, tmp_path) -> None:
    binary, log = fake_docker
    c.LIVE_CONTAINERS['a'] = c.LiveContainer(binary=binary, context=None, beat=tmp_path / 'a')
    c.release_containers(['a', 'gone'])
    assert log.read_text().splitlines() == ['rm -f a']
    assert 'a' not in c.LIVE_CONTAINERS
