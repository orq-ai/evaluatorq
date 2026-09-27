"""Container mode against a fake docker binary. No daemon needed."""

from __future__ import annotations

import asyncio
import stat
import time
from pathlib import Path

import pytest
from loguru import logger

from evaluatorq.backends import DockerOptions
from evaluatorq.backends import coding_agent as coding_agent_module
from evaluatorq.backends import container as c
from evaluatorq.backends.coding_agent import CodingAgentError, CodingAgentTarget, CodingAgentUnavailableError
from evaluatorq.contracts import Message

FIXTURES = Path(__file__).parent / 'fixtures'


async def _wait_until(predicate, description: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError(f'timed out waiting for {description}')
        await asyncio.sleep(0.01)

FAKE = r"""#!/bin/sh
printf 'ARGV %s\n' "$*" >> "$FAKE_LOG"
[ "$1" = "--context" ] && shift 2
case "$1" in
  image) [ -n "$FAKE_IMAGE_ERROR" ] && echo "$FAKE_IMAGE_ERROR" >&2
         exit ${FAKE_IMAGE_EXIT:-0};;
  ps) exit 0;;
  run) [ -n "$FAKE_RUN_SLEEP" ] && sleep "$FAKE_RUN_SLEEP"
       [ -n "$FAKE_RUN_EXIT" ] && { echo "daemon down" >&2; exit $FAKE_RUN_EXIT; }; echo CREATED >> "$FAKE_LOG"; [ -n "$FAKE_RUN_STOPPED" ] && touch "$FAKE_STOPPED"; echo cid; exit 0;;
  inspect) if [ -f "$FAKE_STOPPED" ]; then rm "$FAKE_STOPPED"; echo false; else echo true; fi; exit 0;;
  exec) env | sed 's/^/ENV /' >> "$FAKE_LOG"; cat - >/dev/null
        [ -n "$FAKE_EXEC_SLEEP" ] && sleep "$FAKE_EXEC_SLEEP"
        [ -n "$FAKE_STDOUT" ] && cat "$FAKE_STDOUT"; exit ${FAKE_EXEC_EXIT:-0};;
  rm) [ -n "$FAKE_RM_SLEEP" ] && sleep "$FAKE_RM_SLEEP"; [ -n "$FAKE_RM_EXIT" ] && exit "$FAKE_RM_EXIT"; echo REMOVED >> "$FAKE_LOG"; exit 0;;
esac
"""


@pytest.fixture
def docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    script = tmp_path / 'docker'
    script.write_text(FAKE)
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / 'log'
    monkeypatch.setenv('FAKE_LOG', str(log))
    monkeypatch.setenv('FAKE_STDOUT', str(FIXTURES / 'claude_tool.jsonl'))
    monkeypatch.setattr(c, 'LIVE_CONTAINERS', {})
    monkeypatch.setattr(c, 'install_exit_hooks', lambda: None)
    monkeypatch.setattr(c, 'start_heartbeat', lambda: None)

    def calls(sub: str) -> list[str]:
        if not log.exists():
            return []
        return [l for l in log.read_text().splitlines() if l.startswith('ARGV') and f' {sub} ' in f'{l} ']

    return str(script), log, calls


def _target(binary: str, **kw) -> CodingAgentTarget:
    opts = kw.pop('opts', {})
    return CodingAgentTarget(agent='claude', container=DockerOptions(binary=binary, image='img:1', **opts), **kw)


def _log_has(log: Path, text: str) -> bool:
    return log.exists() and text in log.read_text()


@pytest.mark.asyncio
async def test_turn_runs_in_one_container_and_close_removes_it(docker, monkeypatch) -> None:
    binary, log, calls = docker
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-secret')
    target = _target(
        binary, env={'EXTRA': 'x'}, opts={'workdir': '/app', 'name_prefix': 'task-7', 'context': 'orbstack'}
    )
    await target.respond([Message(role='user', content='hi')])
    await target.respond([Message(role='user', content='again')])
    assert len(calls('run')) == 1 and len(calls('exec')) == 2
    [run] = calls('run')
    assert '--context orbstack' in run and '--name task-7-claude-' in run and '/work:/app' in run
    [exec1, _] = calls('exec')
    assert '-e EXTRA -e ANTHROPIC_API_KEY' in exec1 and 'evq-entrypoint claude' in exec1
    text = log.read_text()
    assert 'sk-secret' not in '\n'.join(l for l in text.splitlines() if l.startswith('ARGV'))
    assert 'ENV ANTHROPIC_API_KEY=sk-secret' in text and 'ENV EXTRA=x' in text
    name = run.split('--name ')[1].split()[0]
    assert name in c.LIVE_CONTAINERS
    await target.close()
    assert calls('rm')[-1].endswith(f'rm -f {name}')
    assert c.LIVE_CONTAINERS == {}


@pytest.mark.asyncio
async def test_caller_env_overrides_host_env_for_container_exec(docker, monkeypatch) -> None:
    binary, log, _ = docker
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'host-key')
    target = _target(binary, env={'ANTHROPIC_API_KEY': 'caller-key'})
    await target.respond([Message(role='user', content='hi')])
    assert 'ENV ANTHROPIC_API_KEY=caller-key' in log.read_text()
    await target.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('error', ['Error: No such image: img:1', 'Error: inspecting object: img:1: image not known'])
async def test_image_missing(docker, monkeypatch, error: str) -> None:
    binary, _, _ = docker
    monkeypatch.setenv('FAKE_IMAGE_EXIT', '1')
    monkeypatch.setenv('FAKE_IMAGE_ERROR', error)
    with pytest.raises(CodingAgentUnavailableError) as info:
        await _target(binary).respond([Message(role='user', content='x')])
    assert info.value.code == 'cli.image_missing'
    assert 'eq coding-agent build-image' in info.value.message and 'docker build' in info.value.message


@pytest.mark.asyncio
async def test_image_inspect_docker_context_failure(docker, monkeypatch) -> None:
    binary, _, _ = docker
    monkeypatch.setenv('FAKE_IMAGE_EXIT', '1')
    monkeypatch.setenv('FAKE_IMAGE_ERROR', 'Cannot connect to the Docker daemon at unix:///var/run/docker.sock')
    with pytest.raises(CodingAgentUnavailableError) as info:
        await _target(binary).respond([Message(role='user', content='x')])
    assert info.value.code == 'cli.container_start'
    assert 'image inspect failed' in info.value.message
    assert 'Cannot connect to the Docker daemon' in info.value.message


@pytest.mark.asyncio
async def test_container_start_failure(docker, monkeypatch) -> None:
    binary, _, _ = docker
    monkeypatch.setenv('FAKE_RUN_EXIT', '125')
    with pytest.raises(CodingAgentUnavailableError) as info:
        await _target(binary).respond([Message(role='user', content='x')])
    assert info.value.code == 'cli.container_start' and 'daemon down' in info.value.message
    assert c.LIVE_CONTAINERS == {}


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['timeout', 'oserror'])
async def test_docker_run_control_error_removes_registered_name(docker, monkeypatch, failure: str) -> None:
    binary, _, calls = docker
    import subprocess

    real_run = subprocess.run

    def fail_run(argv, *args, **kwargs):
        if 'run' in argv:
            if failure == 'timeout':
                raise subprocess.TimeoutExpired(argv, 0.01)
            raise OSError('docker run failed before returning a status')
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, 'run', fail_run)
    with pytest.raises(CodingAgentUnavailableError) as info:
        await _target(binary).respond([Message(role='user', content='x')])
    assert info.value.code == 'cli.container_start'
    assert 'run control call failed' in info.value.message
    assert calls('rm')
    assert c.LIVE_CONTAINERS == {}


@pytest.mark.asyncio
async def test_container_that_exits_immediately_is_removed(docker, monkeypatch, tmp_path: Path) -> None:
    binary, _, calls = docker
    stopped = tmp_path / 'stopped'
    monkeypatch.setenv('FAKE_STOPPED', str(stopped))
    monkeypatch.setenv('FAKE_RUN_STOPPED', '1')
    with pytest.raises(CodingAgentUnavailableError) as info:
        await _target(binary).respond([Message(role='user', content='x')])
    assert info.value.code == 'cli.container_start'
    assert 'not running' in info.value.message
    assert calls('inspect') and calls('rm')
    assert c.LIVE_CONTAINERS == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'exit_code,code', [(127, 'cli.agent_not_found'), (126, 'cli.agent_not_found'), (137, 'cli.timeout')]
)
async def test_exec_exit_codes(docker, monkeypatch, exit_code: int, code: str) -> None:
    binary, _, calls = docker
    monkeypatch.setenv('FAKE_EXEC_EXIT', str(exit_code))
    target = _target(binary)
    span_attrs: dict[str, object] = {}
    original_set_span_attrs = coding_agent_module.set_span_attrs

    def record_span_attrs(span, attrs):
        span_attrs.update(attrs)
        original_set_span_attrs(span, attrs)

    monkeypatch.setattr(coding_agent_module, 'set_span_attrs', record_span_attrs)
    with pytest.raises(CodingAgentUnavailableError) as info:
        await target.respond([Message(role='user', content='x')])
    assert info.value.code == code
    assert calls('rm')
    assert c.LIVE_CONTAINERS == {}
    if code == 'cli.agent_not_found':
        assert 'img:1' in info.value.message
    if exit_code == 137:
        assert calls('rm')
        assert c.LIVE_CONTAINERS == {}
        assert _log_has(docker[1], 'REMOVED')
        assert info.value.kill_reason == 'container_lost'
        assert span_attrs['evaluatorq.coding_agent.kill_reason'] == 'container_lost'
    await target.close()


@pytest.mark.asyncio
async def test_empty_agent_output_removes_container(docker, monkeypatch, tmp_path: Path) -> None:
    binary, _, calls = docker
    empty_output = tmp_path / 'empty.jsonl'
    empty_output.write_text('')
    monkeypatch.setenv('FAKE_STDOUT', str(empty_output))
    target = _target(binary)
    with pytest.raises(CodingAgentError, match='cli.no_result'):
        await target.respond([Message(role='user', content='x')])
    assert calls('rm')
    assert c.LIVE_CONTAINERS == {}


@pytest.mark.asyncio
async def test_exec_process_creation_error_removes_container(docker, monkeypatch) -> None:
    binary, _, calls = docker
    target = _target(binary)

    async def fail_exec(*args, **kwargs):
        raise FileNotFoundError('docker exec client could not start')

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fail_exec)
    with pytest.raises(CodingAgentUnavailableError) as info:
        await target.respond([Message(role='user', content='x')])
    assert info.value.code == 'cli.not_found'
    assert calls('rm')
    assert c.LIVE_CONTAINERS == {}


@pytest.mark.asyncio
async def test_idle_timeout_removes_container(docker, monkeypatch) -> None:
    binary, _, calls = docker
    monkeypatch.setenv('FAKE_EXEC_SLEEP', '30')
    target = _target(binary, timeout_ms=800)
    with pytest.raises(CodingAgentUnavailableError) as info:
        await target.respond([Message(role='user', content='x')])
    assert info.value.kill_reason == 'idle_timeout'
    assert len(calls('rm')) == 1 and c.LIVE_CONTAINERS == {}
    monkeypatch.delenv('FAKE_EXEC_SLEEP')
    await target.respond([Message(role='user', content='x')])  # next turn starts a fresh container, no restart warning
    assert len(calls('run')) == 2
    await target.close()


@pytest.mark.asyncio
async def test_cancellation_removes_container_before_propagating(docker, monkeypatch) -> None:
    binary, log, calls = docker
    monkeypatch.setenv('FAKE_EXEC_SLEEP', '30')
    monkeypatch.setenv('FAKE_RM_SLEEP', '0.3')
    target = _target(binary)
    task = asyncio.create_task(target.respond([Message(role='user', content='x')]))
    await _wait_until(lambda: bool(calls('exec')), 'docker exec call')
    task.cancel()
    await _wait_until(lambda: bool(calls('rm')), 'docker rm call')
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done()
    await _wait_until(lambda: _log_has(log, 'REMOVED'), 'container removal')
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(calls('rm')) == 1 and c.LIVE_CONTAINERS == {}
    await target.close()


@pytest.mark.asyncio
async def test_cancel_during_run_still_cleans_up(docker, monkeypatch) -> None:
    binary, log, calls = docker
    target = _target(binary)
    monkeypatch.setenv('FAKE_RUN_SLEEP', '0.3')
    task = asyncio.create_task(target.respond([Message(role='user', content='x')]))
    await _wait_until(lambda: bool(c.LIVE_CONTAINERS), 'container registration')
    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done()
    await _wait_until(lambda: _log_has(log, 'CREATED'), 'container creation')
    await _wait_until(lambda: bool(calls('rm')), 'docker rm call')
    with pytest.raises(asyncio.CancelledError):
        await task
    assert c.LIVE_CONTAINERS == {} and calls('rm')
    assert log.read_text().index('CREATED') < log.read_text().index('REMOVED')
    await target.close()


@pytest.mark.asyncio
async def test_cancel_during_restart_removal_waits_until_old_container_is_removed(docker, monkeypatch, tmp_path) -> None:
    binary, log, calls = docker
    target = _target(binary)
    await target.respond([Message(role='user', content='first')])
    old_name = next(iter(c.LIVE_CONTAINERS))
    stopped = tmp_path / 'stopped'
    stopped.write_text('')
    monkeypatch.setenv('FAKE_STOPPED', str(stopped))
    monkeypatch.setenv('FAKE_RM_SLEEP', '0.3')

    task = asyncio.create_task(target.respond([Message(role='user', content='second')]))
    await _wait_until(lambda: bool(calls('rm')), 'docker rm call')
    task.cancel()
    await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done()
    await _wait_until(lambda: _log_has(log, 'REMOVED'), 'container removal')
    with pytest.raises(asyncio.CancelledError):
        await task
    assert old_name not in c.LIVE_CONTAINERS
    assert c.LIVE_CONTAINERS == {}
    assert log.read_text().index(f'rm -f {old_name}') < log.read_text().index('REMOVED')
    await target.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('marker,reason', [(True, 'lease_expired'), (False, 'container_lost')])
async def test_restart_between_turns(docker, monkeypatch, tmp_path, marker: bool, reason: str) -> None:
    binary, _, calls = docker
    target = _target(binary)
    await target.respond([Message(role='user', content='x')])
    first = next(iter(c.LIVE_CONTAINERS))
    work = target.workdir
    assert work is not None
    root = work.parent
    if marker:
        (root / 'home' / '.evq-exit').write_text('lease_expired Thu')
    stopped = tmp_path / 'stopped'
    stopped.write_text('')
    monkeypatch.setenv('FAKE_STOPPED', str(stopped))
    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(str(m)), level='WARNING')
    try:
        await target.respond([Message(role='user', content='y')])
    finally:
        logger.remove(sink)
    assert any(reason in s and 'processes' in s for s in seen)
    runs = calls('run')
    assert len(runs) == 2 and first not in runs[1]
    assert list(c.LIVE_CONTAINERS) != [first] and len(c.LIVE_CONTAINERS) == 1
    assert f'{root / "home"}:/evq-home' in runs[1] and f'{root / "work"}:/work' in runs[1]
    await target.close()


@pytest.mark.asyncio
async def test_failed_restart_removal_does_not_renew_old_container_lease(docker, monkeypatch, tmp_path) -> None:
    binary, _, calls = docker
    target = _target(binary)
    await target.respond([Message(role='user', content='first')])
    old_name, old_live = next(iter(c.LIVE_CONTAINERS.items()))
    old_beat = old_live.beat
    old_value = old_beat.read_text()

    stopped = tmp_path / 'stopped'
    stopped.write_text('')
    monkeypatch.setenv('FAKE_STOPPED', str(stopped))
    monkeypatch.setenv('FAKE_RM_EXIT', '1')
    await target.respond([Message(role='user', content='second')])

    new_name, new_live = next(iter(c.LIVE_CONTAINERS.items()))
    assert new_name != old_name
    assert new_live.beat != old_beat
    assert old_beat.exists()
    c.heartbeat_once(123)
    assert old_beat.read_text() == old_value
    assert new_live.beat.read_text() == '123'
    runs = calls('run')
    assert len(runs) == 2
    assert f'-v {old_beat.parent}:/evq-lease:ro' in runs[0]
    assert f'-v {new_live.beat.parent}:/evq-lease:ro' in runs[1]
    await target.close()


@pytest.mark.asyncio
async def test_ten_clones_concurrently(docker) -> None:
    binary, _, calls = docker
    base = _target(binary)
    clones = [base.new() for _ in range(10)]
    await asyncio.gather(*(t.respond([Message(role='user', content='x')]) for t in clones))
    assert len(c.LIVE_CONTAINERS) == 10
    await asyncio.gather(*(t.close() for t in clones))
    assert c.LIVE_CONTAINERS == {} and len(calls('rm')) == 10


@pytest.mark.asyncio
async def test_keep_workdir_outlives_container(docker) -> None:
    binary, _, calls = docker
    target = _target(binary, keep_workdir=True)
    await target.respond([Message(role='user', content='x')])
    work = target.workdir
    await target.close()
    assert work is not None
    assert work.exists()
    assert not (work.parent / 'lease').exists()
    assert calls('rm')


@pytest.mark.asyncio
async def test_skills_are_copied_not_linked(docker, tmp_path) -> None:
    binary, _, _ = docker
    skill = tmp_path / 'grill-me'
    skill.mkdir()
    (skill / 'SKILL.md').write_text('x')
    (skill / 'outside').symlink_to('/etc/hosts')
    target = _target(binary, skills=[skill])
    work = target._ensure_workdir()
    copied = work / '.claude/skills/grill-me'
    assert copied.is_dir() and not copied.is_symlink()
    assert (copied / 'outside').is_symlink()  # kept as a link, not dereferenced
    await target.close()


@pytest.mark.asyncio
async def test_unsafe_mount_warns_before_run(docker) -> None:
    binary, _, _ = docker
    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(str(m)), level='WARNING')
    try:
        target = _target(binary, opts={'run_args': ('-v', '/:/host')})
        await target.respond([Message(role='user', content='x')])
    finally:
        logger.remove(sink)
    assert any("'/'" in s and 'mount' in s for s in seen)
    await target.close()


@pytest.mark.asyncio
async def test_orphan_sweep_runs_before_first_container(docker) -> None:
    binary, log, _ = docker
    target = _target(binary)
    await target.respond([Message(role='user', content='x')])
    subcommands = [l.split()[1] for l in log.read_text().splitlines() if l.startswith('ARGV')]
    assert subcommands[:3] == ['image', 'ps', 'run']
    await target.close()
