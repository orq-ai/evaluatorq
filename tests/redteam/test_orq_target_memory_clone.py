"""``ORQAgentTarget.new()`` memory-entity semantics.

A seeded ``memory_entity_id`` (constructor arg or later assignment, the sim
layer's --memory-entity path) must survive cloning, otherwise the seeded id
silently reverts to a random one the day a runner clones targets for isolation.
An auto-minted id must NOT survive: unseeded parallel jobs get their own
memory scope per clone.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from evaluatorq.redteam.backends.orq import ORQAgentTarget


def _target(**kwargs) -> ORQAgentTarget:
    return ORQAgentTarget(agent_key='k', orq_client=MagicMock(), **kwargs)


def test_constructor_seeded_id_survives_new() -> None:
    target = _target(memory_entity_id='seeded-entity')
    assert target.new().memory_entity_id == 'seeded-entity'


def test_assignment_seeded_id_survives_new() -> None:
    # The sim layer seeds via attribute assignment after create_target().
    target = _target()
    target.memory_entity_id = 'seeded-entity'
    assert target.new().memory_entity_id == 'seeded-entity'


def test_auto_minted_id_is_not_carried_into_clones() -> None:
    target = _target()
    assert target.memory_entity_id
    assert target.memory_entity_id.startswith('red-team-')
    clone = target.new()
    assert clone.memory_entity_id
    assert clone.memory_entity_id != target.memory_entity_id  # isolation preserved


def test_bare_backend_clone_path_keeps_seeded_id() -> None:
    # BareTargetBackend.create_target() calls target.new(); a bring-your-own
    # seeded target must keep its entity through that path.
    from evaluatorq.redteam.backends.base import BareTargetBackend

    seeded = _target(memory_entity_id='seeded-entity')
    cloned = BareTargetBackend(seeded).create_target('ignored')
    assert cloned.memory_entity_id == 'seeded-entity'
