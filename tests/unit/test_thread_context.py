"""Thread-context ContextVar isolation — the load-bearing property is that
concurrent conversations (asyncio tasks) don't leak thread ids into each other.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from evaluatorq.common.thread_context import (
    _evaluatorq_run_scope,
    build_thread_id,
    conversation_thread,
    current_thread_id,
    evaluatorq_pipeline,
    evaluatorq_run_id,
    pipeline_metadata,
    pipeline_metadata_param,
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


def test_pipeline_metadata_bind_and_restore() -> None:
    assert pipeline_metadata_param() == {}  # unset
    with evaluatorq_pipeline('red_teaming') as label:
        assert label == 'red_teaming'
        assert pipeline_metadata_param() == {'metadata': {'evaluatorq_pipeline': 'red_teaming'}}
    assert pipeline_metadata_param() == {}  # restored on exit


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


def test_run_id_absent_when_unset() -> None:
    assert 'evaluatorq_run_id' not in pipeline_metadata()
    assert pipeline_metadata_param() == {} or 'evaluatorq_run_id' not in pipeline_metadata_param().get('metadata', {})


def test_run_id_present_in_both_metadata_forms() -> None:
    with evaluatorq_run_id('r1'):
        assert pipeline_metadata()['evaluatorq_run_id'] == 'r1'
        assert pipeline_metadata_param()['metadata']['evaluatorq_run_id'] == 'r1'
    # restored after scope
    assert 'evaluatorq_run_id' not in pipeline_metadata()


def test_run_id_and_pipeline_label_travel_together() -> None:
    with evaluatorq_pipeline('agent_simulation'), evaluatorq_run_id('r2'):
        md = pipeline_metadata()
        assert md['evaluatorq_pipeline'] == 'agent_simulation'
        assert md['evaluatorq_run_id'] == 'r2'
        assert pipeline_metadata_param()['metadata'] == md


def test_run_id_resets_on_exception() -> None:
    try:
        with evaluatorq_run_id('r3'):
            raise RuntimeError('boom')
    except RuntimeError:
        pass
    assert 'evaluatorq_run_id' not in pipeline_metadata()


def test_shared_run_scope_stamps_and_restores() -> None:
    attrs: dict[str, object] = {}
    span = SimpleNamespace(set_attribute=lambda key, value: attrs.__setitem__(key, value))

    with _evaluatorq_run_scope('outer', span):
        assert pipeline_metadata()['evaluatorq_run_id'] == 'outer'

    assert attrs == {'orq.evaluatorq_run_id': 'outer'}
    assert 'evaluatorq_run_id' not in pipeline_metadata()


def test_shared_run_scope_without_span_still_binds() -> None:
    with _evaluatorq_run_scope('run-without-span', None):
        assert pipeline_metadata()['evaluatorq_run_id'] == 'run-without-span'


def test_shared_run_scope_restores_after_exception() -> None:
    with pytest.raises(RuntimeError, match='boom'):
        with _evaluatorq_run_scope('temporary', None):
            raise RuntimeError('boom')

    assert 'evaluatorq_run_id' not in pipeline_metadata()


def test_run_id_concurrent_tasks_are_isolated() -> None:
    async def _run() -> set[str]:
        seen: set[str] = set()

        async def worker(rid: str) -> None:
            with evaluatorq_run_id(rid):
                await asyncio.sleep(0)
                seen.add(pipeline_metadata()['evaluatorq_run_id'])

        await asyncio.gather(worker('a'), worker('b'), worker('c'))
        return seen

    assert asyncio.run(_run()) == {'a', 'b', 'c'}
    assert 'evaluatorq_run_id' not in pipeline_metadata()


if __name__ == '__main__':
    test_unset_is_none_and_empty()
    test_bind_and_restore()
    test_generated_id_when_omitted()
    test_concurrent_tasks_are_isolated()
    print('ok')
