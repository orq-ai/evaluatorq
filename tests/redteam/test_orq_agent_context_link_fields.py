import asyncio
from types import SimpleNamespace

from evaluatorq.redteam.backends.orq import ORQAgentTarget


class _FakeAgents:
    def retrieve(self, agent_key):
        return SimpleNamespace(
            id='agent-uuid-1',
            workspace_id='ws-42',
            display_name='Support Bot',
            description='desc',
            system_prompt=None,
            instructions=None,
            settings=SimpleNamespace(tools=[]),
            knowledge_bases=[],
            memory_stores=[],
            model='gpt-4o',
        )


class _FakeClient:
    def __init__(self):
        self.agents = _FakeAgents()


def test_context_carries_link_fields():
    target = ORQAgentTarget(agent_key='support-bot', orq_client=_FakeClient())
    ctx = asyncio.run(target.get_agent_context())
    assert ctx.id == 'agent-uuid-1'
    assert ctx.workspace_id == 'ws-42'
    assert ctx.target_kind == 'agent'
