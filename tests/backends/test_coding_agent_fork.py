"""Fork ownership rules for Docker-backed coding-agent targets."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from evaluatorq.backends import coding_agent as coding_agent_module
from evaluatorq.backends import container as container_module
from evaluatorq.backends import DockerOptions
from evaluatorq.backends.coding_agent import CodingAgentTarget
from evaluatorq.contracts import Message


pytestmark = pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires Unix fork')


def test_inherited_target_is_rejected_but_new_target_is_process_owned(tmp_path: Path) -> None:
    target = CodingAgentTarget(agent='claude', container=DockerOptions(binary='docker', image='img:1'))
    root = tmp_path / 'parent-work'
    root.mkdir()
    target._root = root
    target._workdir = root / 'work'
    target._workdir.mkdir()
    target._container_name = 'parent-container'
    target._owned.append('parent-container')

    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        try:
            outcomes: list[str] = []
            for operation in (
                lambda: asyncio.run(target.respond([Message(role='user', content='hello')])),
                lambda: asyncio.run(target.close()),
            ):
                try:
                    operation()
                except RuntimeError as exc:
                    outcomes.append(str(exc))
                else:
                    outcomes.append('unexpected success')
            fresh = target.new()
            outcomes.append('fresh owner' if fresh._creator_pid == os.getpid() else 'wrong owner')
            coding_agent_module.remove_tree_if_owner(target._creator_pid, root)
            os.write(write_fd, '\n'.join(outcomes).encode())
        finally:
            os.close(write_fd)
            os._exit(0)

    os.close(write_fd)
    result = os.read(read_fd, 8192).decode()
    os.close(read_fd)
    _, status = os.waitpid(child, 0)

    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    lines = result.splitlines()
    assert len(lines) == 3
    assert all('inherited across os.fork()' in line and 'target.new()' in line for line in lines[:2])
    assert lines[2] == 'fresh owner'
    assert root.exists()
    assert target._container_name == 'parent-container'
    assert target._owned == ['parent-container']


def test_child_resets_parent_container_registry_and_starts_its_own_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    live_containers: dict[str, container_module.LiveContainer] = {}
    monkeypatch.setattr(container_module, 'LIVE_CONTAINERS', live_containers)
    monkeypatch.setattr(container_module, 'BEAT_WARNED', set())
    monkeypatch.setattr(container_module, 'install_exit_hooks', lambda: None)
    monkeypatch.setattr(container_module, 'start_heartbeat', lambda: None)
    parent_beat = tmp_path / 'parent-beat'
    container_module.register(
        'parent-container', container_module.LiveContainer(binary='docker', context=None, beat=parent_beat)
    )

    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        try:
            reset = not container_module.LIVE_CONTAINERS and not container_module.BEAT_WARNED
            container_module.install_exit_hooks = original_install_exit_hooks
            container_module.start_heartbeat = original_start_heartbeat
            child_beat = tmp_path / 'child-beat'
            container_module.register(
                'child-container', container_module.LiveContainer(binary='docker', context=None, beat=child_beat)
            )
            container_module.heartbeat_once(23)
            heartbeat_started = (
                container_module.heartbeat_thread is not None and container_module.heartbeat_thread.is_alive()
            )
            removed: list[str] = []
            container_module.remove_containers = lambda _binary, _context, names: removed.extend(names) or []
            container_module.remove_all('fork-test')
            result = f'{reset}|{heartbeat_started}|{child_beat.read_text()}|{",".join(removed)}'
            os.write(write_fd, result.encode())
        finally:
            os.close(write_fd)
            os._exit(0)

    os.close(write_fd)
    result = os.read(read_fd, 8192).decode()
    os.close(read_fd)
    _, status = os.waitpid(child, 0)

    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0
    assert result == 'True|True|23|child-container'
    assert set(container_module.LIVE_CONTAINERS) == {'parent-container'}


original_install_exit_hooks = container_module.install_exit_hooks
original_start_heartbeat = container_module.start_heartbeat
