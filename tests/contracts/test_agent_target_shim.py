"""Tests for AgentTarget contract after RES-877 Task 9.

The ``send_prompt`` back-compat shim has been removed. ``respond(messages)`` is
the sole response method; callers own the conversation transcript.
"""

from __future__ import annotations

import pytest

from evaluatorq.contracts import AgentTarget


def test_agent_target_has_no_send_prompt():
    assert not hasattr(AgentTarget, "send_prompt")


def test_respond_is_abstract_subclass_without_it_cannot_instantiate():
    """respond is abstract: a subclass that implements only ``new`` is incomplete."""

    class _Bare(AgentTarget):  # pyright: ignore[reportImplicitAbstractClass]
        def new(self) -> _Bare:
            return _Bare()  # pyright: ignore[reportAbstractUsage]

    with pytest.raises(TypeError, match="abstract"):
        _Bare()  # type: ignore[abstract]  # pyright: ignore[reportAbstractUsage]
