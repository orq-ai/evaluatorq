"""Integration test: OpenAIAgentTarget with mocked Runner.

The OpenAI Agents SDK doesn't provide a fake model, so we mock Runner.run()
to return realistic results. This tests the full wrapper wiring: history
management, multi-turn conversations, reset, and clone.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("agents")

from evaluatorq.integrations.openai_agents_integration import OpenAIAgentTarget  # noqa: E402


def _make_result(output: str, history: list[dict[str, Any]]) -> MagicMock:
    """Create a realistic RunResult mock."""
    result = MagicMock()
    result.final_output = output
    result.to_input_list.return_value = history
    return result


class TestOpenAIAgentIntegration:
    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Simulate a multi-turn conversation and verify history builds up."""
        call_count = 0

        async def fake_run(agent, input_data, **kwargs):  # noqa: ANN001, ANN003
            nonlocal call_count
            call_count += 1
            if isinstance(input_data, str):
                history = [
                    {"role": "user", "content": input_data},
                    {"role": "assistant", "content": f"Reply {call_count}"},
                ]
            else:
                history = [
                    *input_data,
                    {"role": "assistant", "content": f"Reply {call_count}"},
                ]
            return _make_result(f"Reply {call_count}", history)

        mock_runner = MagicMock()
        mock_runner.run = fake_run
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", mock_runner)

        agent = MagicMock()
        target = OpenAIAgentTarget(agent)

        r1 = await target.send_prompt("Hello")
        assert r1.text == "Reply 1"

        r2 = await target.send_prompt("Follow up")
        assert r2.text == "Reply 2"

        # History should have grown
        assert len(target._history) > 2

    @pytest.mark.asyncio
    async def test_reset_clears_conversation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """After reset, the next prompt should be sent as a fresh string, not history."""
        inputs_received: list[Any] = []

        async def fake_run(agent, input_data, **kwargs):  # noqa: ANN001, ANN003
            inputs_received.append(input_data)
            return _make_result("response", [
                {"role": "user", "content": "msg"},
                {"role": "assistant", "content": "response"},
            ])

        mock_runner = MagicMock()
        mock_runner.run = fake_run
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", mock_runner)

        target = OpenAIAgentTarget(MagicMock())
        await target.send_prompt("First")
        target.reset_conversation()
        await target.send_prompt("After reset")

        # First call: string. Second call (after reset): also string, not list
        assert isinstance(inputs_received[0], str)
        assert isinstance(inputs_received[1], str)

    @pytest.mark.asyncio
    async def test_clone_has_empty_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cloned target should start with fresh history."""
        async def fake_run(agent, input_data, **kwargs):  # noqa: ANN001, ANN003
            return _make_result("response", [{"role": "assistant", "content": "response"}])

        mock_runner = MagicMock()
        mock_runner.run = fake_run
        monkeypatch.setattr("evaluatorq.integrations.openai_agents_integration.target.Runner", mock_runner)

        target = OpenAIAgentTarget(MagicMock())
        await target.send_prompt("Build up history")
        assert len(target._history) > 0

        cloned = target.clone()
        assert len(cloned._history) == 0
