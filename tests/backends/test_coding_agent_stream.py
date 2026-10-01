"""_run's idle limit, hard cap, stream handling and kill-reason logging, against fake agent binaries."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from loguru import logger

from evaluatorq.backends import coding_agent
from evaluatorq.backends.coding_agent import CodingAgentTarget, CodingAgentUnavailableError, describe_last_event
from evaluatorq.contracts import Message
from tests.backends.fakes import install_fake

RESULT = '{"type":"result","result":"done","session_id":"s","usage":{"input_tokens":1,"output_tokens":1}}'
SYSTEM = '{"type":"system"}'

# Every fake drains the prompt, then writes unbuffered, LF-only lines so timing matches a real CLI.
PRELUDE = '''\
import os, subprocess, sys, time
sys.stdin.read()
sys.stdout.reconfigure(newline='\\n')
sys.stderr.reconfigure(newline='\\n')


def out(text, end='\\n'):
    sys.stdout.write(text + end)
    sys.stdout.flush()


def err(text):
    sys.stderr.write(text + '\\n')
    sys.stderr.flush()


'''


def _agent(tmp_path: Path, body: str) -> dict[str, str]:
    bindir = tmp_path / 'bin'
    install_fake(bindir, 'claude', PRELUDE + body)
    return {'PATH': f'{bindir}{os.pathsep}{os.environ["PATH"]}'}


def _warnings() -> tuple[list[str], int]:
    seen: list[str] = []
    return seen, logger.add(lambda m: seen.append(str(m)), level='WARNING')


@pytest.mark.asyncio
async def test_steady_output_outlives_idle_limit(tmp_path: Path) -> None:
    body = f'for _ in range(6):\n    out({SYSTEM!r})\n    time.sleep(0.5)\nout({RESULT!r})\n'
    target = CodingAgentTarget(agent='claude', env=_agent(tmp_path, body), timeout_ms=3000)
    response = await target.respond([Message(role='user', content='x')])
    assert response.text == 'done'
    await target.close()


@pytest.mark.asyncio
async def test_silence_past_idle_limit_is_idle_timeout(tmp_path: Path) -> None:
    tool_use = '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Bash","id":"t"}]}}'
    body = f'out({tool_use!r})\ntime.sleep(30)\n'
    target = CodingAgentTarget(agent='claude', env=_agent(tmp_path, body), timeout_ms=2000)
    seen, sink = _warnings()
    try:
        with pytest.raises(CodingAgentUnavailableError) as info:
            await target.respond([Message(role='user', content='x')])
    finally:
        logger.remove(sink)
    assert info.value.code == 'cli.timeout'
    assert info.value.kill_reason == 'idle_timeout'
    assert 'last event: assistant Bash' in info.value.message
    assert any('idle_timeout' in s for s in seen)
    await target.close()


@pytest.mark.asyncio
async def test_open_child_after_stdout_eof_still_hits_idle_limit(tmp_path: Path) -> None:
    body = 'os.close(1)\ntime.sleep(30)\n'
    target = CodingAgentTarget(agent='claude', env=_agent(tmp_path, body), timeout_ms=2000)
    seen, sink = _warnings()
    try:
        with pytest.raises(CodingAgentUnavailableError) as info:
            await asyncio.wait_for(target.respond([Message(role='user', content='x')]), 3)
    finally:
        logger.remove(sink)
        await target.close()
    assert info.value.kill_reason == 'idle_timeout'
    assert 'last event: none' in info.value.message
    assert any('idle_timeout' in s for s in seen)


@pytest.mark.asyncio
async def test_descendant_holding_stderr_open_still_hits_idle_limit(tmp_path: Path) -> None:
    sleeper = "[sys.executable, '-c', 'import time; time.sleep(30)']"
    body = f'out({RESULT!r})\nsubprocess.Popen({sleeper}, stdout=subprocess.DEVNULL)\n'
    target = CodingAgentTarget(agent='claude', env=_agent(tmp_path, body), timeout_ms=2000)
    seen, sink = _warnings()
    try:
        with pytest.raises(CodingAgentUnavailableError) as info:
            await asyncio.wait_for(target.respond([Message(role='user', content='x')]), 6)
    finally:
        logger.remove(sink)
        await target.close()
    assert info.value.kill_reason == 'idle_timeout'
    assert 'last event: result' in info.value.message
    assert any('idle_timeout' in s for s in seen)


@pytest.mark.asyncio
async def test_hard_cap_fires_on_steady_output(tmp_path: Path) -> None:
    body = f'while True:\n    out({SYSTEM!r})\n    time.sleep(0.1)\n'
    target = CodingAgentTarget(agent='claude', env=_agent(tmp_path, body), timeout_ms=10_000, max_turn_ms=3000)
    with pytest.raises(CodingAgentUnavailableError) as info:
        await target.respond([Message(role='user', content='x')])
    assert info.value.kill_reason == 'hard_cap'
    assert 'events' in info.value.message
    await target.close()


@pytest.mark.asyncio
async def test_sustained_stderr_resets_idle_limit(tmp_path: Path) -> None:
    body = f"for _ in range(5):\n    err('progress')\n    time.sleep(1)\nout({RESULT!r})\n"
    target = CodingAgentTarget(agent='claude', env=_agent(tmp_path, body), timeout_ms=3000)
    response = await target.respond([Message(role='user', content='x')])
    assert response.text == 'done'
    await target.close()


@pytest.mark.asyncio
async def test_idle_timeout_after_stderr_stops(tmp_path: Path) -> None:
    body = "err('progress')\ntime.sleep(30)\n"
    target = CodingAgentTarget(agent='claude', env=_agent(tmp_path, body), timeout_ms=2000)
    with pytest.raises(CodingAgentUnavailableError) as info:
        await asyncio.wait_for(target.respond([Message(role='user', content='x')]), 5)
    assert info.value.kill_reason == 'idle_timeout'
    await target.close()


@pytest.mark.asyncio
async def test_stderr_flood_does_not_stall(tmp_path: Path) -> None:
    body = f"err('x' * 1_048_576)\nout({RESULT!r})\n"
    target = CodingAgentTarget(agent='claude', env=_agent(tmp_path, body), timeout_ms=5000)
    response = await asyncio.wait_for(target.respond([Message(role='user', content='x')]), 20)
    assert response.text == 'done'
    await target.close()


@pytest.mark.asyncio
async def test_one_megabyte_line_parses_without_copying_for_each_deadline_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    big = 'y' * 1_048_576
    result = RESULT.replace('"done"', f'"{big}"')
    (tmp_path / 'out.jsonl').write_text(result + '\n')
    target = CodingAgentTarget(agent='claude', env=_agent(tmp_path, f"out(open({str(tmp_path / 'out.jsonl')!r}).read(), end='')\n"), timeout_ms=5000)
    original_deadline_error = coding_agent.deadline_error
    buffers: list[int] = []

    def inspect_buffer(
        agent: str,
        stdout: bytes | bytearray,
        started: float,
        last_output: float,
        idle_s: float,
        hard_s: float,
        now: float,
    ):
        if isinstance(stdout, bytearray):
            buffers.append(id(stdout))
        return original_deadline_error(agent, stdout, started, last_output, idle_s, hard_s, now)

    monkeypatch.setattr(coding_agent, 'deadline_error', inspect_buffer)
    response = await target.respond([Message(role='user', content='x')])
    assert len(response.text) == 1_048_576
    assert len(buffers) > 5
    assert len(set(buffers)) == 1
    await target.close()


@pytest.mark.asyncio
async def test_partial_line_counts_as_output(tmp_path: Path) -> None:
    half = RESULT[: len(RESULT) // 2]
    rest = RESULT[len(RESULT) // 2 :]
    body = f"time.sleep(2)\nout({half!r}, end='')\ntime.sleep(6)\nout({rest!r})\n"
    target = CodingAgentTarget(agent='claude', env=_agent(tmp_path, body), timeout_ms=8000)
    response = await target.respond([Message(role='user', content='x')])
    assert response.text == 'done'
    await target.close()


@pytest.mark.asyncio
async def test_cancellation_logs_cancelled(tmp_path: Path) -> None:
    target = CodingAgentTarget(agent='claude', env=_agent(tmp_path, 'time.sleep(30)\n'))
    seen, sink = _warnings()
    task = asyncio.create_task(target.respond([Message(role='user', content='x')]))
    await asyncio.sleep(1)
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        logger.remove(sink)
    assert any('cancelled' in s for s in seen)
    await target.close()


def test_describe_last_event() -> None:
    assert describe_last_event(b'{"type":"item.started","item":{"type":"command_execution"}}\n{"trunc') == (
        'item.started command_execution'
    )
    assert describe_last_event(b'') == 'none'


def test_coding_agent_manages_own_timeout() -> None:
    assert CodingAgentTarget.manages_own_timeout is True


def test_limits_must_be_positive() -> None:
    with pytest.raises(ValueError):
        CodingAgentTarget(agent='claude', max_turn_ms=0)
    with pytest.raises(ValueError):
        CodingAgentTarget(agent='claude', timeout_ms=0)
