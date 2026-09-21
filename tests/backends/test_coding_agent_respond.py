"""respond() against a fake binary on PATH. The fake echoes a fixture and exits with a chosen code."""

from __future__ import annotations

import asyncio
import os
import stat
import time
from pathlib import Path
from typing import cast

import pytest

from evaluatorq.backends.coding_agent import (
    AgentName,
    CodingAgentError,
    CodingAgentTarget,
    CodingAgentUnavailableError,
)
from evaluatorq.contracts import AgentResponse, Message, ToolCallOutputItem
from evaluatorq.redteam.contracts import Turn, turns_to_messages

FIXTURES = Path(__file__).parent / 'fixtures'

_ECHO = """#!/bin/sh
cat - >/dev/null
[ -n "$FAKE_STDOUT" ] && cat "$FAKE_STDOUT"
exit ${FAKE_EXIT:-0}
"""

_SLEEPER = """#!/bin/sh
sleep 300 &
echo $! > "$FAKE_PIDFILE"
wait
"""


def _install(tmp_path: Path, name: str, body: str) -> str:
    bindir = tmp_path / 'bin'
    bindir.mkdir(exist_ok=True)
    script = bindir / name
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return f'{bindir}{os.pathsep}{os.environ.get("PATH", "")}'


def _target(tmp_path: Path, agent: str, *, stdout: str | None = None, exit_code: int = 0, body: str = _ECHO, **kw):
    path = _install(tmp_path, agent, body)
    env = {'PATH': path, 'FAKE_EXIT': str(exit_code)}
    if stdout is not None:
        env['FAKE_STDOUT'] = str(FIXTURES / f'{stdout}.jsonl')
    return CodingAgentTarget(cast(AgentName, agent), env=env, **kw)


@pytest.mark.asyncio
async def test_success_claude(tmp_path: Path) -> None:
    target = _target(tmp_path, 'claude', stdout='claude_tool')
    response = await target.respond([Message(role='user', content='run echo')])
    assert response.text == 'done'
    assert response.response_id == '9c0bf077-5b29-4b47-9953-3dd93ce096c9'
    assert response.usage is not None and response.usage.output_tokens == 83
    [call] = response.tool_calls
    assert call.name == 'Bash'
    assert isinstance(response.output[0], ToolCallOutputItem)  # tool calls precede the final text
    await target.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('agent,fixture', [('codex', 'codex_tool'), ('opencode', 'opencode_tool')])
async def test_success_other_agents(tmp_path: Path, agent: str, fixture: str) -> None:
    target = _target(tmp_path, agent, stdout=fixture)
    response = await target.respond([Message(role='user', content='run echo')])
    assert response.text == 'done'
    assert len(response.tool_calls) == 1
    await target.close()


@pytest.mark.asyncio
async def test_nonzero_exit_wins_over_parsed_result(tmp_path: Path) -> None:
    target = _target(tmp_path, 'claude', stdout='claude_tool', exit_code=3)
    with pytest.raises(CodingAgentError) as info:
        await target.respond([Message(role='user', content='x')])
    assert info.value.code == 'cli.exit.3'
    assert target.map_error(info.value) == ('cli.exit.3', info.value.message)


@pytest.mark.asyncio
async def test_exit_zero_empty_stdout_is_no_result(tmp_path: Path) -> None:
    target = _target(tmp_path, 'claude')
    with pytest.raises(CodingAgentError) as info:
        await target.respond([Message(role='user', content='x')])
    assert info.value.code == 'cli.no_result'


@pytest.mark.asyncio
async def test_garbage_stdout_is_parse_error(tmp_path: Path) -> None:
    garbage = tmp_path / 'garbage.txt'
    garbage.write_text('not json at all\n')
    path = _install(tmp_path, 'codex', _ECHO)
    target = CodingAgentTarget('codex', env={'PATH': path, 'FAKE_STDOUT': str(garbage)})
    with pytest.raises(CodingAgentError) as info:
        await target.respond([Message(role='user', content='x')])
    assert info.value.code == 'cli.parse_error'


@pytest.mark.asyncio
async def test_is_error_result_is_agent_error(tmp_path: Path) -> None:
    fx = tmp_path / 'err.jsonl'
    fx.write_text('{"type":"result","subtype":"error_during_execution","is_error":true,"result":"boom","session_id":"s"}\n')
    path = _install(tmp_path, 'claude', _ECHO)
    target = CodingAgentTarget('claude', env={'PATH': path, 'FAKE_STDOUT': str(fx)})
    with pytest.raises(CodingAgentError) as info:
        await target.respond([Message(role='user', content='x')])
    assert info.value.code == 'cli.agent_error'
    assert 'boom' in info.value.message


@pytest.mark.asyncio
async def test_missing_binary_is_not_found_and_non_retryable(tmp_path: Path) -> None:
    target = CodingAgentTarget('claude', env={'PATH': str(tmp_path)})
    with pytest.raises(CodingAgentUnavailableError) as info:
        await target.respond([Message(role='user', content='x')])
    assert info.value.code == 'cli.not_found'


@pytest.mark.asyncio
async def test_timeout_kills_and_is_non_retryable(tmp_path: Path) -> None:
    pidfile = tmp_path / 'pid'
    path = _install(tmp_path, 'claude', _SLEEPER)
    target = CodingAgentTarget('claude', env={'PATH': path, 'FAKE_PIDFILE': str(pidfile)}, timeout_ms=500)
    with pytest.raises(CodingAgentUnavailableError) as info:
        await target.respond([Message(role='user', content='x')])
    assert info.value.code == 'cli.timeout'
    _assert_gone(int(pidfile.read_text()))


@pytest.mark.asyncio
async def test_cancellation_kills_process_group(tmp_path: Path) -> None:
    pidfile = tmp_path / 'pid'
    path = _install(tmp_path, 'claude', _SLEEPER)
    target = CodingAgentTarget('claude', env={'PATH': path, 'FAKE_PIDFILE': str(pidfile)})
    task = asyncio.create_task(target.respond([Message(role='user', content='x')]))
    deadline = time.time() + 10
    while not pidfile.exists() and time.time() < deadline:
        await asyncio.sleep(0.05)
    assert pidfile.exists()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    _assert_gone(int(pidfile.read_text()))


def _assert_gone(pid: int) -> None:
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        # A zombie answers kill(pid, 0); waitpid tells us whether it is one.
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return
    raise AssertionError(f'process {pid} still alive')


@pytest.mark.asyncio
async def test_denied_tool_call_survives_turns_to_messages(tmp_path: Path) -> None:
    fx = tmp_path / 'denied.jsonl'
    fx.write_text(
        '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Bash","input":{"command":"rm -rf /"}}]}}\n'
        '{"type":"result","subtype":"success","is_error":false,"result":"not allowed","session_id":"s","usage":{"input_tokens":1,"output_tokens":1},"permission_denials":[{"tool_name":"Bash","tool_use_id":"t1","tool_input":{"command":"rm -rf /"}}]}\n'
    )
    path = _install(tmp_path, 'claude', _ECHO)
    target = CodingAgentTarget('claude', env={'PATH': path, 'FAKE_STDOUT': str(fx)})
    response = await target.respond([Message(role='user', content='wipe it')])
    turn = Turn(attacker=AgentResponse(text='wipe it'), target=response)
    rendered = turns_to_messages([turn])
    tool_rows = [m for m in rendered if m.role == 'tool']
    assert len(tool_rows) == 1
    assert tool_rows[0].content == '[denied by claude]'


@pytest.mark.asyncio
async def test_env_overlay_caller_wins(tmp_path: Path) -> None:
    body = '#!/bin/sh\ncat - >/dev/null\nprintf \'{"type":"item.completed","item":{"id":"a","type":"agent_message","text":"%s"}}\\n\' "$MARKER"\n'
    path = _install(tmp_path, 'codex', body)
    target = CodingAgentTarget('codex', env={'PATH': path, 'MARKER': 'from-env'})
    response = await target.respond([Message(role='user', content='x')])
    assert response.text == 'from-env'
