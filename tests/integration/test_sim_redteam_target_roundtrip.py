"""RES-808 acceptance: one OrqResponsesTarget works as both a sim target_agent
and a redteam AgentTarget without state corruption.

Mock-based (no network) — runs in the default unit suite. Verifies the
stateless contract: the sim path (respond) and the redteam path (send_prompt
shim) do not leak each other's payloads, and neither threads
previous_response_id.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.contracts import AgentResponse, LLMCallConfig, Message
from evaluatorq.simulation.target import OrqResponsesTarget


def _make_response(text: str) -> MagicMock:
    part = MagicMock()
    part.type = "output_text"
    part.text = text
    msg_item = MagicMock()
    msg_item.type = "message"
    msg_item.content = [part]
    usage = MagicMock()
    usage.input_tokens = 5
    usage.output_tokens = 3
    response = MagicMock()
    response.id = "resp"
    response.usage = usage
    response.output = [msg_item]
    return response


@pytest.mark.asyncio
async def test_one_instance_used_for_sim_then_redteam_path():
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock(
        side_effect=[_make_response("sim-r1"), _make_response("redteam-r1")]
    )
    target = OrqResponsesTarget(LLMCallConfig(model="gpt-4o"), client=client)

    # Sim path: caller hands the runner an AgentTarget; runner calls respond().
    sim_result = await target.respond([Message(role="user", content="sim-q")])
    # Redteam path: orchestrator calls the single-prompt shim.
    redteam_result = await target.send_prompt("redteam-q")

    assert isinstance(sim_result, AgentResponse)
    assert isinstance(redteam_result, AgentResponse)
    assert sim_result.text == "sim-r1"
    assert redteam_result.text == "redteam-r1"

    call_args = client.responses.create.await_args_list
    # No state leakage: each call's input is exactly what was passed.
    assert call_args[0].kwargs["input"] == [{"role": "user", "content": "sim-q"}]
    # send_prompt routes through the respond shim → [Message(user, "redteam-q")].
    assert call_args[1].kwargs["input"] == [{"role": "user", "content": "redteam-q"}]
    # Stateless contract: no previous_response_id threading anywhere.
    assert "previous_response_id" not in call_args[0].kwargs
    assert "previous_response_id" not in call_args[1].kwargs
