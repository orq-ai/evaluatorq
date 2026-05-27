"""Integration test: OpenAIAgentTarget with mocked Runner.

The OpenAI Agents SDK doesn't provide a fake model, so we mock Runner.run()
to return realistic results. This tests the full wrapper wiring: stateless
respond(), clone(), and new().

After RES-877 Task 8/9: OpenAIAgentTarget is fully stateless — no _history,
no reset_conversation(). The orchestrator owns the conversation transcript
and passes the full message list to respond().
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("agents")

from evaluatorq.contracts import Message  # noqa: E402
from evaluatorq.integrations.openai_agents_integration import OpenAIAgentTarget  # noqa: E402
from evaluatorq.integrations.openai_agents_integration.target import (  # noqa: E402
    _message_to_responses_input_items,
)


def _make_result(output: str, history: list[dict[str, Any]]) -> MagicMock:
    """Create a realistic RunResult mock."""
    result = MagicMock()
    result.final_output = output
    result.to_input_list.return_value = history
    return result


class TestOpenAIAgentIntegration:
    @pytest.mark.asyncio
    async def test_stateless_respond_two_turns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """respond() is stateless: each call runs independently with the provided messages."""
        call_count = 0

        async def fake_run(agent, input_data, **kwargs):  # noqa: ANN001, ANN003
            nonlocal call_count
            call_count += 1
            reply = f"Reply {call_count}"
            return _make_result(reply, [*input_data, {"role": "assistant", "content": reply}])

        mock_runner = MagicMock()
        mock_runner.run = fake_run
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", mock_runner)

        agent = MagicMock()
        target = OpenAIAgentTarget(agent)

        r1 = await target.respond([Message(role="user", content="Hello")])
        assert r1.text == "Reply 1"

        r2 = await target.respond([Message(role="user", content="Follow up")])
        assert r2.text == "Reply 2"

        # Target is stateless; no _history attribute should be present
        assert not hasattr(target, "_history")

    @pytest.mark.asyncio
    async def test_clone_produces_independent_target(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cloned target should work independently from the original."""
        async def fake_run(agent, input_data, **kwargs):  # noqa: ANN001, ANN003
            return _make_result("response", [{"role": "assistant", "content": "response"}])

        mock_runner = MagicMock()
        mock_runner.run = fake_run
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", mock_runner)

        target = OpenAIAgentTarget(MagicMock())
        await target.respond([Message(role="user", content="Build up history")])

        cloned = target.clone()
        assert cloned is not target
        # Stateless — clone is just a fresh instance sharing the same agent
        r = await cloned.respond([Message(role="user", content="Hello from clone")])
        assert r.text == "response"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_to_input_list_echoes_input_prefix() -> None:
    """Real Agents SDK: to_input_list() echoes our input 1:1 at the front.

    Guards the core assumption in ``OpenAIAgentTarget.respond``: the new-items
    slice ``full_list[prev_len:]`` only yields this turn's output because the SDK
    returns ``input + appended`` from ``to_input_list()``. If a future SDK version
    normalizes or reorders the input, this test fails loud (and the runtime guard
    in ``respond`` warns). Requires ``OPENAI_API_KEY``.
    """
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("requires OPENAI_API_KEY")

    from agents import Agent, Runner

    agent = Agent(name="echo-probe", instructions="Reply with a single short word.")
    messages = [Message(role="user", content="Say hello in one word.")]
    input_data = [item for m in messages for item in _message_to_responses_input_items(m)]
    prev_len = len(input_data)

    result = await Runner.run(agent, input_data)
    full_list = result.to_input_list()

    assert full_list[:prev_len] == input_data, "SDK no longer echoes input 1:1 at the front"
    assert len(full_list) > prev_len, "run should append at least the assistant reply"
