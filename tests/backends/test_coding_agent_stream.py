"""_run's idle limit, hard cap, stream handling and kill-reason logging, against fake agent binaries."""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

import pytest
from loguru import logger

from evaluatorq.backends.coding_agent import CodingAgentTarget, CodingAgentUnavailableError, describe_last_event
from evaluatorq.contracts import Message

RESULT = '{"type":"result","result":"done","session_id":"s","usage":{"input_tokens":1,"output_tokens":1}}'


def _agent(tmp_path: Path, body: str) -> dict[str, str]:
    bindir = tmp_path / 'bin'
    bindir.mkdir(exist_ok=True)
    script = bindir / 'claude'
    script.write_text('#!/bin/sh\ncat - >/dev/null\n' + body)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return {'PATH': f'{bindir}{os.pathsep}{os.environ["PATH"]}'}


def _warnings() -> tuple[list[str], int]:
    seen: list[str] = []
    return seen, logger.add(lambda m: seen.append(str(m)), level='WARNING')


@pytest.mark.asyncio
async def test_steady_output_outlives_idle_limit(tmp_path: Path) -> None:
    body = 'for i in 1 2 3 4 5 6; do echo \'{"type":"system"}\'; sleep 0.3; done\n' + f"echo '{RESULT}'\n"
    target = CodingAgentTarget('claude', env=_agent(tmp_path, body), timeout_ms=1000)
    response = await target.respond([Message(role='user', content='x')])
    assert response.text == 'done'
    await target.close()


@pytest.mark.asyncio
async def test_silence_past_idle_limit_is_idle_timeout(tmp_path: Path) -> None:
    body = 'echo \'{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Bash","id":"t"}]}}\'\nsleep 30\n'
    target = CodingAgentTarget('claude', env=_agent(tmp_path, body), timeout_ms=800)
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
async def test_hard_cap_fires_on_steady_output(tmp_path: Path) -> None:
    body = 'while true; do echo \'{"type":"system"}\'; sleep 0.1; done\n'
    target = CodingAgentTarget('claude', env=_agent(tmp_path, body), timeout_ms=5000, max_turn_ms=800)
    with pytest.raises(CodingAgentUnavailableError) as info:
        await target.respond([Message(role='user', content='x')])
    assert info.value.kill_reason == 'hard_cap'
    assert 'events' in info.value.message
    await target.close()


@pytest.mark.asyncio
async def test_stderr_flood_does_not_stall(tmp_path: Path) -> None:
    body = 'head -c 1048576 /dev/zero | tr "\\0" x >&2\n' + f"echo '{RESULT}'\n"
    target = CodingAgentTarget('claude', env=_agent(tmp_path, body), timeout_ms=5000)
    response = await asyncio.wait_for(target.respond([Message(role='user', content='x')]), 20)
    assert response.text == 'done'
    await target.close()


@pytest.mark.asyncio
async def test_one_megabyte_line_parses(tmp_path: Path) -> None:
    big = 'y' * 1_048_576
    result = RESULT.replace('"done"', f'"{big}"')
    (tmp_path / 'out.jsonl').write_text(result + '\n')
    target = CodingAgentTarget('claude', env=_agent(tmp_path, f'cat {tmp_path / "out.jsonl"}\n'), timeout_ms=5000)
    response = await target.respond([Message(role='user', content='x')])
    assert len(response.text) == 1_048_576
    await target.close()


@pytest.mark.asyncio
async def test_partial_line_counts_as_output(tmp_path: Path) -> None:
    half = RESULT[: len(RESULT) // 2]
    rest = RESULT[len(RESULT) // 2 :]
    body = f"printf '%s' '{half}'\nsleep 0.5\nprintf '%s\\n' '{rest}'\nsleep 0.5\n"
    target = CodingAgentTarget('claude', env=_agent(tmp_path, body), timeout_ms=800)
    response = await target.respond([Message(role='user', content='x')])
    assert response.text == 'done'
    await target.close()


@pytest.mark.asyncio
async def test_cancellation_logs_cancelled(tmp_path: Path) -> None:
    target = CodingAgentTarget('claude', env=_agent(tmp_path, 'sleep 30\n'))
    seen, sink = _warnings()
    task = asyncio.create_task(target.respond([Message(role='user', content='x')]))
    await asyncio.sleep(0.5)
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
        CodingAgentTarget('claude', max_turn_ms=0)
    with pytest.raises(ValueError):
        CodingAgentTarget('claude', timeout_ms=0)
