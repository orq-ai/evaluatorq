"""Container mode against a real daemon and the built image. Skipped without either. Not run in CI."""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from evaluatorq.backends import DockerOptions
from evaluatorq.backends.coding_agent import CodingAgentTarget, CodingAgentUnavailableError
from evaluatorq.backends.container import CONTAINER_LABEL, DEFAULT_CODING_AGENT_IMAGE
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
    left = subprocess.run(['docker', 'ps', '-aq', '--filter', f'label={CONTAINER_LABEL}=1'], capture_output=True, text=True)
    assert left.stdout.strip() == '', 'a coding-agent container was left behind'


def _needs_key(name: str) -> None:
    if not os.environ.get(name):
        pytest.skip(f'{name} not set')


@pytest.mark.asyncio
@pytest.mark.parametrize('launcher', ['direct', 'orq'])
async def test_claude_turn(launcher: str) -> None:
    _needs_key('ORQ_API_KEY' if launcher == 'orq' else 'ANTHROPIC_API_KEY')
    target = CodingAgentTarget('claude', launcher=launcher, container=DockerOptions())
    try:
        response = await target.respond([Message(role='user', content='Reply with the single word: pong')])
        assert 'pong' in response.text.lower()
    finally:
        await close_target(target)


@pytest.mark.asyncio
@pytest.mark.parametrize('agent', ['claude', 'codex', 'opencode'])
async def test_idle_limit_is_real(agent: str) -> None:
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
        response = await steady.respond([
            Message(
                role='user',
                content='Run `sleep 10` six times, as six separate shell tool calls, one after another. '
                'Do not combine them into one command or a loop. Then say done.',
            )
        ])
        if len(response.tool_calls) < 6:
            pytest.skip(f'model combined the sleeps into {len(response.tool_calls)} call(s); nothing to measure')
        assert 'done' in response.text.lower()
    finally:
        await close_target(steady)


@pytest.mark.asyncio
async def test_isolation(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _needs_key('ANTHROPIC_API_KEY')
    secret = tmp_path / 'host-only.txt'
    secret.write_text('HOST-SECRET-FILE')
    monkeypatch.setenv('EVQ_HOST_ONLY', 'HOST-SECRET-ENV')
    target = CodingAgentTarget('claude', container=DockerOptions())
    try:
        response = await target.respond([
            Message(role='user', content=f'Run `cat {secret}` and `echo $EVQ_HOST_ONLY` and paste both outputs verbatim.')
        ])
        assert 'HOST-SECRET-FILE' not in response.text and 'HOST-SECRET-ENV' not in response.text
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
