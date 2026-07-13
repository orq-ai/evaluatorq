"""Thread-context ContextVar isolation — the load-bearing property is that
concurrent conversations (asyncio tasks) don't leak thread ids into each other.
"""

from __future__ import annotations

import asyncio

from evaluatorq.common.thread_context import (
    build_thread_id,
    conversation_thread,
    current_thread_id,
    thread_body_param,
)


def test_build_thread_id_sim_and_redteam_shapes() -> None:
    # Sim: {run_id}:{index}; redteam: {run_id}:{agent_key}:{index}. Both share the
    # run_id prefix so `contains:{run_id}` matches every conversation in the run.
    assert build_thread_id('run1', 3) == 'run1:3'
    assert build_thread_id('run1', 'agent-a', 7) == 'run1:agent-a:7'
    sim_thread_id = build_thread_id('run1', 3)
    redteam_thread_id = build_thread_id('run1', 'agent-a', 7)
    assert sim_thread_id is not None and sim_thread_id.startswith('run1')
    assert redteam_thread_id is not None and redteam_thread_id.startswith('run1')


def test_build_thread_id_none_without_run() -> None:
    assert build_thread_id(None, 3) is None
    assert build_thread_id('', 3) is None


def test_unset_is_none_and_empty() -> None:
    assert current_thread_id() is None
    assert thread_body_param() == {}


def test_bind_and_restore() -> None:
    with conversation_thread('t1') as tid:
        assert tid == 't1'
        assert current_thread_id() == 't1'
        assert thread_body_param() == {'thread': {'id': 't1'}}
    assert current_thread_id() is None  # restored on exit


def test_generated_id_when_omitted() -> None:
    with conversation_thread() as tid:
        assert tid and current_thread_id() == tid


def test_concurrent_tasks_are_isolated() -> None:
    async def convo(name: str, seen: dict[str, str | None]) -> None:
        with conversation_thread(name):
            await asyncio.sleep(0.01)  # yield so tasks interleave
            seen[name] = current_thread_id()

    async def main() -> dict[str, str | None]:
        seen: dict[str, str | None] = {}
        await asyncio.gather(*(convo(f't{i}', seen) for i in range(5)))
        return seen

    seen = asyncio.run(main())
    assert seen == {f't{i}': f't{i}' for i in range(5)}


if __name__ == '__main__':
    test_unset_is_none_and_empty()
    test_bind_and_restore()
    test_generated_id_when_omitted()
    test_concurrent_tasks_are_isolated()
    print('ok')
