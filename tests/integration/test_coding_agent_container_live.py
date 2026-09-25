"""Container mode against a real daemon and the built image. Skipped without either. Not run in CI."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time

import pytest

from evaluatorq.backends import DockerOptions
from evaluatorq.backends.coding_agent import AgentName, CodingAgentTarget, CodingAgentUnavailableError, Launcher
from evaluatorq.backends.container import CONTAINER_LABEL, DEFAULT_CODING_AGENT_IMAGE, HOST_PID_LABEL
from evaluatorq.common.target_call import close_target
from evaluatorq.contracts import Message

pytestmark = pytest.mark.integration

KEYS = {'claude': 'ANTHROPIC_API_KEY', 'codex': 'OPENAI_API_KEY', 'opencode': 'ANTHROPIC_API_KEY'}


@pytest.fixture(autouse=True)
def needs_docker():
    if shutil.which('docker') is None or subprocess.run(['docker', 'info'], capture_output=True).returncode != 0:
        pytest.skip('no docker daemon')
    if subprocess.run(['docker', 'image', 'inspect', DEFAULT_CODING_AGENT_IMAGE], capture_output=True).returncode != 0:
        pytest.skip(f'{DEFAULT_CODING_AGENT_IMAGE} not built; run eq coding-agent build-image')
    yield
    left = subprocess.run(
        [
            'docker', 'ps', '-aq', '--filter', f'label={CONTAINER_LABEL}=1',
            '--filter', f'label={HOST_PID_LABEL}={os.getpid()}',
        ],
        capture_output=True,
        text=True,
    )
    assert left.stdout.strip() == '', 'a coding-agent container was left behind'


def _needs_key(name: str) -> None:
    if not os.environ.get(name):
        pytest.skip(f'{name} not set')


@pytest.mark.asyncio
@pytest.mark.parametrize('launcher', ['direct', 'orq'])
async def test_claude_turn(launcher: Launcher) -> None:
    _needs_key('ORQ_API_KEY' if launcher == 'orq' else 'ANTHROPIC_API_KEY')
    target = CodingAgentTarget('claude', launcher=launcher, container=DockerOptions())
    try:
        response = await target.respond([Message(role='user', content='Reply with the single word: pong')])
        assert 'pong' in response.text.lower()
    finally:
        await close_target(target)


@pytest.mark.asyncio
@pytest.mark.parametrize('agent', ['claude', 'codex', 'opencode'])
@pytest.mark.timeout(180)
async def test_idle_limit_is_real(agent: AgentName) -> None:
    _needs_key(KEYS[agent])
    slow = CodingAgentTarget(agent, container=DockerOptions(), timeout_ms=20_000)
    try:
        with pytest.raises(CodingAgentUnavailableError) as info:
            await slow.respond([Message(role='user', content='Run the shell command `sleep 60`, then say done.')])
        assert info.value.kill_reason == 'idle_timeout'
    finally:
        await close_target(slow)
    steady = CodingAgentTarget(agent, container=DockerOptions(), timeout_ms=20_000)
    try:
        started = time.monotonic()
        response = await steady.respond([
            Message(
                role='user',
                content='Run `sleep 10` six times, as six separate shell tool calls, one after another. '
                'Do not combine them into one command or a loop. Then say done.',
            )
        ])
        elapsed = time.monotonic() - started
        sleep_calls = []
        for call in response.tool_calls:
            if call.name not in {'Bash', 'bash', 'shell'}:
                continue
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError:
                arguments = call.arguments
            command_text = json.dumps(arguments) if isinstance(arguments, dict) else str(arguments)
            commands = re.findall(r'(?<![\w])sleep\s+10(?![\w])', command_text)
            if len(commands) == 1 and not re.search(r'\b(for|while)\b|&&|;|\|\|', command_text):
                sleep_calls.append(call)
        assert len(sleep_calls) == 6, f'expected six separate sleep 10 tool calls, got {len(sleep_calls)}'
        assert all(call.status.value == 'completed' for call in sleep_calls), 'one or more sleep calls did not complete'
        assert elapsed > steady.timeout_ms / 1000, f'turn took {elapsed:.1f}s, not longer than the idle limit'
        assert 'done' in response.text.lower()
    finally:
        await close_target(steady)


@pytest.mark.asyncio
async def test_isolation(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _needs_key('ANTHROPIC_API_KEY')
    secret = tmp_path / 'host-only.txt'
    secret.write_text('HOST-SECRET-FILE')
    workdir = tmp_path / 'work'
    workdir.mkdir()
    (workdir / 'inside.txt').write_text('WORKDIR-READABLE-CONTROL')
    monkeypatch.setenv('EVQ_HOST_ONLY', 'HOST-SECRET-ENV')
    target = CodingAgentTarget('claude', container=DockerOptions(), workdir=workdir)
    try:
        response = await target.respond([
            Message(
                role='user',
                content=(
                    f'Use separate shell tool calls to run `cat {secret}`, '
                    '`printf "HOST_ENV_VALUE="; printenv EVQ_HOST_ONLY`, '
                    'and `cat inside.txt`. Show each command output verbatim and then summarize.'
                ),
            )
        ])
        calls = response.tool_calls
        args_and_results = []
        for call in calls:
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError:
                arguments = call.arguments
            args_and_results.append((json.dumps(arguments), call.result or '', call.status.value))
        outside_read = [row for row in args_and_results if str(secret) in row[0]]
        env_read = [row for row in args_and_results if 'EVQ_HOST_ONLY' in row[0]]
        inside_read = [row for row in args_and_results if 'inside.txt' in row[0]]
        assert outside_read and outside_read[0][2] == 'completed', 'agent did not run the host-only path command'
        assert env_read and env_read[0][2] == 'completed', 'agent did not run the host-only environment command'
        assert inside_read and inside_read[0][2] == 'completed', 'agent did not run the in-workdir control command'
        assert 'No such file or directory' in outside_read[0][1], 'host-only path was unexpectedly readable'
        assert 'HOST_ENV_VALUE=' in env_read[0][1], 'host-only environment command did not produce its marker'
        assert 'WORKDIR-READABLE-CONTROL' in inside_read[0][1], 'in-workdir control was not readable'
        results = [row[1] for row in args_and_results]
        observed = response.text + '\n' + '\n'.join(results)
        assert 'HOST-SECRET-FILE' not in observed
        assert 'HOST-SECRET-ENV' not in observed
    finally:
        await close_target(target)


@pytest.mark.asyncio
@pytest.mark.skipif(os.uname().sysname != 'Linux', reason='macOS engines map ownership to the host user themselves')
async def test_written_file_owned_by_host_uid() -> None:
    _needs_key('ANTHROPIC_API_KEY')
    target = CodingAgentTarget('claude', container=DockerOptions(), keep_workdir=True)
    try:
        await target.respond([Message(role='user', content='Create a file named probe.txt containing x.')])
        assert target.workdir is not None
        assert (target.workdir / 'probe.txt').stat().st_uid == os.getuid()
    finally:
        await close_target(target)
