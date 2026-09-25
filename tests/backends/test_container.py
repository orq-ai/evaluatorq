"""Pure pieces of container mode: options, argv, env selection, run_args checks."""

from __future__ import annotations

import re
from pathlib import Path

import pydantic
import pytest
from loguru import logger

from evaluatorq.backends import DockerOptions
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


def test_run_argv(tmp_path: Path) -> None:
    opts = DockerOptions(image='img:1', context='orbstack', workdir='/app', run_args=('--memory', '4g'))
    argv = build_run_argv(opts, name='evq-claude-abcd1234', root=tmp_path, uid=501, gid=20)
    assert argv[:4] == ['docker', '--context', 'orbstack', 'run']
    joined = ' '.join(argv)
    for part in (
        '-d --rm --init --entrypoint sh',
        '--name evq-claude-abcd1234',
        '--label evaluatorq.coding-agent=1',
        '--user 501:20',
        '-e HOME=/evq-home',
        f'-v {tmp_path / "home"}:/evq-home',
        f'-v {tmp_path / "lease"}:/evq-lease:ro',
        f'-v {tmp_path / "work"}:/app -w /app',
        '--security-opt no-new-privileges',
        '--memory 4g img:1 -c',
    ):
        assert part in joined, part
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
        '-v', f'{inside}:/in',
        '-v', 'named-volume:/data',
    ]
    assert unsafe_mounts(args, tmp_path) == ['/', '/etc', '/var', '/', './rel']


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


def test_run_args_and_reserved_env_warn_at_construction() -> None:
    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(str(m)), level='WARNING')
    try:
        CodingAgentTarget('claude', container=DockerOptions(run_args=('--privileged',)), env={'PATH': '/x'})
    finally:
        logger.remove(sink)
    assert any('--privileged' in s for s in seen)
    assert any('PATH' in s and 'not forwarded' in s for s in seen)
