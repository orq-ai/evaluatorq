"""Per-conversation target isolation: parallel conversations must never share
mutable target state (the shared-instance ``_task_id`` race).

The fake target below mimics ``ORQAgentTarget``'s statefulness: ``respond``
reads its ``_task_id``, yields to the event loop (as a real HTTP call would),
then writes a conversation-specific value back. On a shared instance,
concurrent conversations interleave those read/write pairs and stitch turns
onto the wrong chain — which is exactly what the runner's per-conversation
``new()`` clone prevents.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from evaluatorq.contracts import AgentResponse, AgentTarget, Message
from evaluatorq.simulation.runner.simulation import SimulationRunner
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Persona,
    Scenario,
    SimulationDatapoint,
)


def _datapoint(idx: int) -> SimulationDatapoint:
    persona = Persona(
        name=f'P{idx}',
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background='A test user.',
    )
    scenario = Scenario(name=f'S{idx}', goal='Get help')
    return SimulationDatapoint(
        id=f'dp-{idx}',
        persona=persona,
        scenario=scenario,
        user_system_prompt='',
        first_message='hello',
    )


class StatefulFakeTarget(AgentTarget):
    """Mimics ORQAgentTarget's per-conversation state and clone semantics."""

    def __init__(
        self, *, memory_entity_id: str | None = None, registry: list[StatefulFakeTarget] | None = None
    ) -> None:
        super().__init__(memory_entity_id=memory_entity_id)
        self._seeded = memory_entity_id is not None
        self._task_id: str | None = None
        self.task_writes: list[tuple[str | None, str]] = []
        self.closed = False
        # Shared across clones so tests can inspect every instance that served.
        self.registry: list[StatefulFakeTarget] = registry if registry is not None else []
        self.registry.append(self)

    async def respond(self, messages: list[Message]) -> AgentResponse:
        conversation = next(m.content for m in messages if m.role == 'user')
        observed = self._task_id
        # Yield twice so concurrent conversations interleave here, exactly like
        # a real HTTP round-trip between the _task_id read and write.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self._task_id = f'task-{conversation}'
        self.task_writes.append((observed, self._task_id))
        return AgentResponse(text=f'reply to {conversation}')

    def new(self) -> StatefulFakeTarget:
        return StatefulFakeTarget(
            memory_entity_id=self.memory_entity_id if self._seeded else None,
            registry=self.registry,
        )

    async def close(self) -> None:
        self.closed = True


def _runner(target: AgentTarget) -> SimulationRunner:
    # The runner validates injected agents by duck typing, so the plain fakes
    # below suffice; cast for the type checker only.
    return SimulationRunner(
        target_agent=target,
        max_turns=1,
        user_simulator=cast('Any', _FakeUserSimulator()),
        judge=cast('Any', _FakeJudge()),
    )


class _FakeUserSimulator:
    def update_context(self, *, persona_context=None, scenario_context=None) -> None:
        return None

    async def generate_first_message(self) -> str:
        return 'hello'

    async def respond_async(self, messages, *, llm_purpose=None) -> str:
        return 'user follow-up'

    def get_usage(self):
        from evaluatorq.contracts import TokenUsage

        return TokenUsage()

    def reset_usage(self) -> None:
        return None


class _FakeJudge:
    async def evaluate(self, messages):
        from evaluatorq.simulation.types import Judgment

        return Judgment(
            should_terminate=True,
            reason='ok',
            goal_achieved=True,
            rules_broken=[],
            goal_completion_score=1.0,
        )

    def get_usage(self):
        from evaluatorq.contracts import TokenUsage

        return TokenUsage()

    def reset_usage(self) -> None:
        return None


@pytest.mark.asyncio
async def test_parallel_conversations_never_share_a_target_instance() -> None:
    """Each conversation runs on its own clone with a self-consistent task chain.

    On a shared instance (main before this fix) the interleaved read/write
    pairs cross conversations: some clone would observe a prior task id
    belonging to a different conversation.
    """
    registry: list[StatefulFakeTarget] = []
    root = StatefulFakeTarget(registry=registry)
    runner = _runner(root)

    datapoints = [_datapoint(i) for i in range(4)]
    results = await runner.run_batch(datapoints, max_concurrency=4)

    assert len(results) == 4
    assert all(r.terminated_by.value != 'error' for r in results), [r.reason for r in results]
    # The root never served a conversation; each conversation used a fresh clone.
    assert root.task_writes == []
    serving = [t for t in registry if t.task_writes]
    assert len(serving) == 4
    for clone in serving:
        # A fresh clone starts with no task id, and every write on it belongs
        # to one conversation only (no cross-conversation observations).
        observed_ids = {obs for obs, _new in clone.task_writes}
        written_ids = {new for _obs, new in clone.task_writes}
        assert observed_ids <= {None} | written_ids
        assert len(written_ids) == 1


@pytest.mark.asyncio
async def test_seeded_memory_entity_reaches_every_conversation() -> None:
    registry: list[StatefulFakeTarget] = []
    root = StatefulFakeTarget(memory_entity_id='seeded-entity', registry=registry)
    runner = _runner(root)

    await runner.run_batch([_datapoint(i) for i in range(3)], max_concurrency=3)

    serving = [t for t in registry if t.task_writes]
    assert len(serving) == 3
    assert all(t.memory_entity_id == 'seeded-entity' for t in serving)


@pytest.mark.asyncio
async def test_unseeded_clones_get_isolated_memory_scopes() -> None:
    registry: list[StatefulFakeTarget] = []
    root = StatefulFakeTarget(registry=registry)
    runner = _runner(root)

    await runner.run_batch([_datapoint(i) for i in range(3)], max_concurrency=3)

    serving = [t for t in registry if t.task_writes]
    assert all(t.memory_entity_id is None for t in serving)


@pytest.mark.asyncio
async def test_close_releases_spawned_clones() -> None:
    registry: list[StatefulFakeTarget] = []
    root = StatefulFakeTarget(registry=registry)
    runner = _runner(root)

    await runner.run_batch([_datapoint(i) for i in range(2)], max_concurrency=2)
    await runner.close()

    clones = [t for t in registry if t is not root]
    assert clones
    assert all(t.closed for t in clones)
    # The shared root target is owned by the caller, not the runner.
    assert root.closed is False


@pytest.mark.asyncio
async def test_plain_callable_target_still_works() -> None:
    async def target(messages: list[Message]) -> str:
        return 'callable reply'

    runner = SimulationRunner(
        target=target,
        max_turns=1,
        user_simulator=cast('Any', _FakeUserSimulator()),
        judge=cast('Any', _FakeJudge()),
    )
    results = await runner.run_batch([_datapoint(0)], max_concurrency=1)

    assert len(results) == 1
    assert results[0].terminated_by.value != 'error'
    assert any(m.content == 'callable reply' for m in results[0].messages)
