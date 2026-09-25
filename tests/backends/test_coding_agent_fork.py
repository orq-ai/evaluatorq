"""Fork ownership rules for Docker-backed coding-agent targets."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from evaluatorq.backends import coding_agent as coding_agent_module
from evaluatorq.backends import DockerOptions
from evaluatorq.backends.coding_agent import CodingAgentTarget
from evaluatorq.contracts import Message


pytestmark = pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires Unix fork')


def test_inherited_target_is_rejected_but_new_target_is_process_owned(tmp_path: Path) -> None:
    target = CodingAgentTarget('claude', container=DockerOptions(binary='docker', image='img:1'))
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
            coding_agent_module._remove_tree_if_owner(target._creator_pid, root)
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

