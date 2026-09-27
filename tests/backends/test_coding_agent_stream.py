"""_run's idle limit, hard cap, stream handling and kill-reason logging, against fake agent binaries."""

from __future__ import annotations

import asyncio
import os
import stat
from pathlib import Path

import pytest
from loguru import logger

from evaluatorq.backends import coding_agent
from evaluatorq.backends.coding_agent import CodingAgentTarget, CodingAgentUnavailableError, describe_last_event
from evaluatorq.contracts import Message

RESULT = '{"type":"result","result":"done","session_id":"s","usage":{"input_tokens":1,"output_tokens":1}}'

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='requires Unix shell tooling')


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
async def test_open_child_after_stdout_eof_still_hits_idle_limit(tmp_path: Path) -> None:
    body = "exec python3 -c 'import os,time; os.close(1); time.sleep(30)'\n"
    target = CodingAgentTarget('claude', env=_agent(tmp_path, body), timeout_ms=300)
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
    body = f"echo '{RESULT}'\npython3 -c 'import subprocess; subprocess.Popen([\"sleep\",\"30\"], stdout=subprocess.DEVNULL)'\n"
    target = CodingAgentTarget('claude', env=_agent(tmp_path, body), timeout_ms=2000)
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
    body = 'while true; do echo \'{"type":"system"}\'; sleep 0.1; done\n'
    target = CodingAgentTarget('claude', env=_agent(tmp_path, body), timeout_ms=5000, max_turn_ms=800)
    with pytest.raises(CodingAgentUnavailableError) as info:
        await target.respond([Message(role='user', content='x')])
    assert info.value.kill_reason == 'hard_cap'
    assert 'events' in info.value.message
    await target.close()


@pytest.mark.asyncio
async def test_sustained_stderr_resets_idle_limit(tmp_path: Path) -> None:
    body = 'for i in 1 2 3 4; do echo progress >&2; sleep 1; done\n' + f"echo '{RESULT}'\n"
    target = CodingAgentTarget('claude', env=_agent(tmp_path, body), timeout_ms=2500)
    response = await target.respond([Message(role='user', content='x')])
    assert response.text == 'done'
    await target.close()


@pytest.mark.asyncio
async def test_idle_timeout_after_stderr_stops(tmp_path: Path) -> None:
    body = 'echo progress >&2; sleep 30\n'
    target = CodingAgentTarget('claude', env=_agent(tmp_path, body), timeout_ms=300)
    with pytest.raises(CodingAgentUnavailableError) as info:
        await asyncio.wait_for(target.respond([Message(role='user', content='x')]), 3)
    assert info.value.kill_reason == 'idle_timeout'
    await target.close()


@pytest.mark.asyncio
async def test_stderr_flood_does_not_stall(tmp_path: Path) -> None:
    body = 'head -c 1048576 /dev/zero | tr "\\0" x >&2\n' + f"echo '{RESULT}'\n"
    target = CodingAgentTarget('claude', env=_agent(tmp_path, body), timeout_ms=5000)
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
    target = CodingAgentTarget('claude', env=_agent(tmp_path, f'cat {tmp_path / "out.jsonl"}\n'), timeout_ms=5000)
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
    body = f"sleep 2\nprintf '%s' '{half}'\nsleep 4\nprintf '%s\\n' '{rest}'\n"
    target = CodingAgentTarget('claude', env=_agent(tmp_path, body), timeout_ms=5000)
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
