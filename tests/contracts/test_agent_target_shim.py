"""Tests for the AgentTarget.send_prompt back-compat shim (RES-808 PR3).

``respond(messages)`` is the canonical interface; ``send_prompt(prompt)`` is a
concrete shim on the ABC that wraps the prompt in a single user message and
delegates to ``respond``. A target that overrides only ``respond`` must get a
working ``send_prompt`` for free.
"""

from __future__ import annotations

import pytest

from evaluatorq.contracts import AgentResponse, AgentTarget, Message


class _RespondOnlyTarget(AgentTarget):
    def __init__(self) -> None:
        super().__init__()
        self.received: list[list[Message]] = []

    async def respond(self, messages: list[Message]) -> AgentResponse:
        self.received.append(messages)
        return AgentResponse(text=f"echo: {messages[-1].content}")

    def new(self) -> _RespondOnlyTarget:
        return _RespondOnlyTarget()


@pytest.mark.asyncio
async def test_send_prompt_delegates_to_respond_with_single_user_message():
    target = _RespondOnlyTarget()
    result = await target.send_prompt("hello")

    assert result.text == "echo: hello"
    assert len(target.received) == 1
    assert len(target.received[0]) == 1
    assert target.received[0][0].role == "user"
    assert target.received[0][0].content == "hello"


def test_respond_is_abstract_subclass_without_it_cannot_instantiate():
    """respond is abstract: a subclass that implements only ``new`` is incomplete."""

    class _Bare(AgentTarget):  # pyright: ignore[reportImplicitAbstractClass]
        def new(self) -> _Bare:
            return _Bare()  # pyright: ignore[reportAbstractUsage]

    with pytest.raises(TypeError, match="abstract"):
        _Bare()  # type: ignore[abstract]  # pyright: ignore[reportAbstractUsage]
