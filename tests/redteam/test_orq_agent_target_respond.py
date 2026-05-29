"""Tests for ORQAgentTarget.respond(messages) — RES-808 PR3.

The ORQ agents endpoint holds conversation state server-side via ``task_id``,
so ``respond`` forwards only the last user message and rejects a transcript
whose final message is not a user turn.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai")

from evaluatorq.contracts import AgentResponse, Message
from evaluatorq.redteam.backends.orq import ORQAgentTarget


def _make_orq_response(text: str = "ok", task_id: str = "task-1") -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = 5
    usage.completion_tokens = 3
    usage.total_tokens = 8
    part = MagicMock()
    part.kind = "text"
    part.text = text
    item = MagicMock()
    item.parts = [part]
    response = MagicMock()
    response.task_id = task_id
    response.usage = usage
    response.output = [item]
    response.pending_tool_calls = []
    response.model = None
    return response


@pytest.mark.asyncio
async def test_respond_rejects_non_user_last_message():
    target = ORQAgentTarget(agent_key="a", orq_client=MagicMock())
    with pytest.raises(ValueError, match=r"messages\[-1\].role"):
        await target.respond(
            [
                Message(role="user", content="x"),
                Message(role="assistant", content="y"),
            ]
        )


@pytest.mark.asyncio
async def test_respond_rejects_empty_messages():
    target = ORQAgentTarget(agent_key="a", orq_client=MagicMock())
    with pytest.raises(ValueError, match=r"messages\[-1\].role"):
        await target.respond([])


@pytest.mark.asyncio
async def test_respond_forwards_only_last_user_message():
    """respond([..., assistant, user]) sends only the last user turn to the endpoint.

    Prior turns are not in the SDK payload — the server holds them via task_id.
    """
    target = ORQAgentTarget(agent_key="a", orq_client=MagicMock())

    captured: dict[str, Any] = {}

    async def fake_to_thread(fn: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _make_orq_response()

    with (
        patch("asyncio.to_thread", side_effect=fake_to_thread),
        patch("evaluatorq.redteam.tracing.get_tracer", return_value=None),
    ):
        result = await target.respond(
            [
                Message(role="user", content="prior turn"),
                Message(role="assistant", content="prior reply"),
                Message(role="user", content="LATEST USER"),
            ]
        )

    assert isinstance(result, AgentResponse)
    assert captured["message"] == {
        "role": "user",
        "parts": [{"kind": "text", "text": "LATEST USER"}],
    }


def test_send_prompt_shim_removed():
    """send_prompt back-compat shim was removed in RES-877 Task 9; respond is the sole method."""
    assert not hasattr(ORQAgentTarget, "send_prompt")
