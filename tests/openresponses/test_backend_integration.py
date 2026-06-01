"""End-to-end integration test for OpenResponsesAgentTarget.

Hits the real ``/responses`` endpoint when ``ORQ_API_KEY`` (or
``OPENAI_API_KEY``) and ``EVALUATORQ_OPENRESPONSES_TEST_AGENT`` are set,
otherwise skips. Verifies:

- The on-the-wire payload built by the target is consumed without error.
- The response is parsed into an :class:`AgentResponse` with text.
- Multi-turn threading carries ``previous_response_id`` correctly.

Run with::

    EVALUATORQ_OPENRESPONSES_TEST_AGENT=<agent-id> \\
        uv run pytest -m integration tests/openresponses/test_backend_integration.py
"""

from __future__ import annotations

import os

import pytest

from evaluatorq.redteam.backends.openresponses import OpenResponsesAgentTarget


def _agent_id() -> str | None:
    return os.environ.get("EVALUATORQ_OPENRESPONSES_TEST_AGENT")


def _has_credentials() -> bool:
    return bool(os.environ.get("ORQ_API_KEY") or os.environ.get("OPENAI_API_KEY"))


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _has_credentials(),
        reason="ORQ_API_KEY or OPENAI_API_KEY required for integration test",
    ),
    pytest.mark.skipif(
        not _agent_id(),
        reason="Set EVALUATORQ_OPENRESPONSES_TEST_AGENT to a real agent id",
    ),
]


@pytest.mark.asyncio
async def test_single_turn_round_trip():
    """The simplest possible attack: send a benign probe, get back text."""
    agent_id = _agent_id()
    assert agent_id is not None
    target = OpenResponsesAgentTarget(agent_id=agent_id)
    response = await target.send_prompt("Reply with the single word: pong")
    assert isinstance(response.text, str)
    assert response.text, "expected non-empty response text from /responses endpoint"
    assert response.model is not None


@pytest.mark.asyncio
async def test_multi_turn_threads_via_previous_response_id():
    """Two-turn exchange should reuse the prior response id for threading."""
    agent_id = _agent_id()
    assert agent_id is not None
    target = OpenResponsesAgentTarget(agent_id=agent_id)

    first = await target.send_prompt("My name is Alice. Acknowledge.")
    assert first.response_id, "first response should carry an id for threading"
    captured_id = target._previous_response_id  # noqa: SLF001
    assert captured_id == first.response_id

    second = await target.send_prompt("What did I just tell you my name was?")
    assert isinstance(second.text, str)
    assert second.text
