"""Agent-context retrieval must survive provider metadata it cannot model (RES-1177).

An orq POC died in the ``context_retrieval`` stage before a single attack was
sent: the orq SDK leaves unset optional fields holding an ``Unset()`` placeholder
rather than ``None``, so ``getattr(agent, name, None)`` returns the sentinel and
it lands in a ``str``-typed ``AgentContext`` field. Context is enrichment, so
neither the sentinel nor a failure to model it may end a run — but an
unreachable agent still must.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from evaluatorq.contracts import AgentContext
from evaluatorq.redteam.backends.orq import ORQAgentTarget, ORQBackend


class _Unset:
    """Stand-in for ``orq_ai_sdk.types.basemodel.Unset``.

    Reproduced locally so the test pins our handling of *any* placeholder
    object, and does not silently start passing if the SDK renames its own.
    """

    def __repr__(self) -> str:
        return 'Unset()'


def _agent_payload(**overrides):
    payload = {
        'id': 'agent-uuid-1',
        'workspace_id': 'ws-42',
        'display_name': 'Thales Engineering Companion',
        'description': 'desc',
        'system_prompt': _Unset(),
        'instructions': 'be helpful',
        'settings': SimpleNamespace(tools=[]),
        'knowledge_bases': [],
        'memory_stores': [],
        'model': SimpleNamespace(id='openai/gpt-4o'),
    }
    payload.update(overrides)
    return SimpleNamespace(**payload)


class _FakeAgents:
    def __init__(self, agent):
        self._agent = agent

    def retrieve(self, agent_key):  # noqa: ARG002 - signature mirrors the SDK
        return self._agent


class _FakeClient:
    def __init__(self, agent=None):
        self.agents = _FakeAgents(agent if agent is not None else _agent_payload())


def test_unset_placeholder_does_not_abort_context_retrieval():
    """The exact RES-1177 crash: ``system_prompt`` present but holding a sentinel."""
    target = ORQAgentTarget(agent_key='thales-engineering-companion-2', orq_client=_FakeClient())
    ctx = asyncio.run(target.get_agent_context())

    assert ctx.system_prompt == ''
    # Enrichment around the bad field survives — the point of the fix is to keep
    # the context, not merely to stop the exception.
    assert ctx.display_name == 'Thales Engineering Companion'
    assert ctx.model == 'openai/gpt-4o'


def test_unset_collection_placeholders_are_treated_as_empty():
    agent = _agent_payload(
        settings=SimpleNamespace(tools=_Unset()),
        knowledge_bases=_Unset(),
        memory_stores=_Unset(),
    )

    ctx = asyncio.run(ORQAgentTarget(agent_key='k', orq_client=_FakeClient(agent)).get_agent_context())

    assert ctx.tools == []
    assert ctx.knowledge_bases == []
    assert ctx.memory_stores == []


def test_int_and_bool_metadata_are_stringified_not_dropped():
    agent = _agent_payload(description=42, display_name=True)
    ctx = asyncio.run(ORQAgentTarget(agent_key='k', orq_client=_FakeClient(agent)).get_agent_context())

    assert ctx.description == '42'
    assert ctx.display_name == 'True'


def test_required_text_fields_are_never_none():
    agent = _agent_payload(system_prompt=None, instructions=_Unset())
    ctx = asyncio.run(ORQAgentTarget(agent_key='k', orq_client=_FakeClient(agent)).get_agent_context())

    assert ctx.system_prompt == ''
    assert ctx.instructions == ''


@pytest.mark.parametrize(
    ('system_prompt', 'instructions'),
    [
        (_Unset(), 'agent behavioral instructions'),  # agent: instructions only
        ('deployment system prompt', _Unset()),  # deployment: system_prompt only
        (_Unset(), _Unset()),  # bring-your-own target: neither introspectable
    ],
)
def test_system_prompt_and_instructions_are_mutually_exclusive(system_prompt, instructions):
    """Supplying one and omitting the other is the normal case, not missing data.

    Agents carry ``instructions``, deployments carry ``system_prompt``. Neither
    field may be promoted to mandatory, and the empty one must read as ``''``
    so an ``instructions or system_prompt`` chain falls through cleanly.
    """
    agent = _agent_payload(system_prompt=system_prompt, instructions=instructions)
    ctx = asyncio.run(ORQAgentTarget(agent_key='k', orq_client=_FakeClient(agent)).get_agent_context())

    expected_prompt = system_prompt if isinstance(system_prompt, str) else ''
    expected_instructions = instructions if isinstance(instructions, str) else ''
    assert ctx.system_prompt == expected_prompt
    assert ctx.instructions == expected_instructions
    assert (ctx.instructions or ctx.system_prompt) == (expected_instructions or expected_prompt)


def test_both_directive_fields_populated_is_allowed():
    """Exclusivity is how targets behave, not a rule we enforce.

    Rejecting a payload that fills both would re-introduce a fatal error in the
    enrichment path — the exact failure mode RES-1177 is about.
    """
    agent = _agent_payload(system_prompt='sp', instructions='ins')
    ctx = asyncio.run(ORQAgentTarget(agent_key='k', orq_client=_FakeClient(agent)).get_agent_context())

    assert ctx.system_prompt == 'sp'
    assert ctx.instructions == 'ins'


def test_optional_text_fields_stay_none_when_absent():
    agent = _agent_payload(id=_Unset(), workspace_id=None, description=_Unset())
    ctx = asyncio.run(ORQAgentTarget(agent_key='k', orq_client=_FakeClient(agent)).get_agent_context())

    assert ctx.id is None
    assert ctx.workspace_id is None
    assert ctx.description is None


def _ctx(**overrides):
    agent = _agent_payload(**overrides)
    return asyncio.run(ORQAgentTarget(agent_key='k', orq_client=_FakeClient(agent)).get_agent_context())


def test_version_read_from_new_sdk_field():
    """orq-ai-sdk 4.12.x shape: ``version`` present, ``version_hash`` gone."""
    assert _ctx(version='v7').version == 'v7'


def test_version_falls_back_to_version_hash_on_old_sdk():
    """orq-ai-sdk 4.4.x shape: ``version`` absent, ``version_hash`` present."""
    assert _ctx(version_hash='ab12cd').version == 'ab12cd'


def test_unset_version_falls_through_to_version_hash():
    """The placeholder is *truthy*, so an ``or`` chain would stop at it and never
    reach the fallback. Selection has to be by type."""
    assert _ctx(version=_Unset(), version_hash='ab12cd').version == 'ab12cd'


def test_unset_version_with_no_fallback_is_none():
    assert _ctx(version=_Unset()).version is None


def test_new_capability_fields_are_populated():
    ctx = _ctx(skills=['refund', 'lookup'], type='a2a', engine='jinja')

    assert ctx.skills == ['refund', 'lookup']
    assert ctx.agent_type == 'a2a'  # SDK calls this field 'type'
    assert ctx.engine == 'jinja'


def test_old_sdk_payload_omitting_new_fields_is_fine():
    """The version CI installs (the pin floor) exposes none of these."""
    ctx = _ctx()

    assert ctx.version is None
    assert ctx.skills == []
    assert ctx.agent_type is None
    assert ctx.engine is None


def test_unknown_literal_members_are_dropped_not_raised():
    """Providers add enum values faster than we follow; an unrecognized one is not
    worth failing a scan over."""
    ctx = _ctx(type='swarm', engine=_Unset())

    assert ctx.agent_type is None
    assert ctx.engine is None


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [(_Unset(), []), (None, []), (['a', 3, None], ['a']), ('not-a-list', [])],
)
def test_skills_tolerates_bad_shapes(raw, expected):
    assert _ctx(skills=raw).skills == expected


def test_key_is_not_coerced():
    """Identity is strict: a bad key must fail, not resolve to something plausible."""
    with pytest.raises(ValidationError):
        AgentContext(key=_Unset())  # type: ignore[arg-type]


def test_resolve_context_degrades_on_validation_error(monkeypatch):
    async def _boom(self):
        raise ValidationError.from_exception_data('AgentContext', [])

    monkeypatch.setattr(ORQAgentTarget, 'get_agent_context', _boom)
    backend = ORQBackend(orq_client=_FakeClient())

    ctx = asyncio.run(backend.resolve_context('thales-engineering-companion-2'))

    assert ctx == AgentContext(key='thales-engineering-companion-2')


def test_resolve_context_reraises_api_errors():
    """Guard against widening the ``except``: red-teaming nothing beats no error."""

    class _AuthError(Exception):
        pass

    async def _boom(self):
        raise _AuthError('401 Unauthorized')

    class _BadAgents:
        def retrieve(self, agent_key):  # noqa: ARG002 - signature mirrors the SDK
            raise _AuthError('401 Unauthorized')

    backend = ORQBackend(orq_client=SimpleNamespace(agents=_BadAgents()))

    with pytest.raises(_AuthError):
        asyncio.run(backend.resolve_context('unreachable-agent'))
