from __future__ import annotations

# ruff: noqa: S101, SLF001
import asyncio
import importlib.util
from pathlib import Path
from typing import ClassVar

from evaluatorq.contracts import AgentResponse
from evaluatorq.simulation.types import Message


def _load_script_module():
    root = Path(__file__).parents[2]
    path = root / "examples" / "agent_simulation" / "orq_agent_tailscale_openai.py"
    spec = importlib.util.spec_from_file_location("orq_agent_tailscale_openai", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_refund_agent_target_uses_local_tool_wrapper(monkeypatch) -> None:
    module = _load_script_module()

    class FakeRefundTarget:
        def __init__(self, agent_key: str) -> None:
            self.agent_key = agent_key

        def new(self):
            return type(self)(self.agent_key)

        async def send_prompt(self, prompt: str) -> AgentResponse:
            return AgentResponse(text=f"handled locally: {prompt}", model="refund-model")

    monkeypatch.setattr(
        module,
        "_load_refund_agent_target_class",
        lambda: FakeRefundTarget,
    )

    target = module._build_target("refund-agent-fixed")

    assert isinstance(target, module.LocalRefundToolTarget)
    assert target.name == "refund-agent-fixed"

    response = asyncio.run(
        target.respond(
            [
                Message(role="user", content="first"),
                Message(role="assistant", content="previous"),
                Message(role="user", content="please refund ord_a1"),
            ]
        )
    )

    assert response.text == "handled locally: please refund ord_a1"
    assert response.model == "refund-model"


def test_refund_agent_target_isolates_inner_target_per_message_list(monkeypatch) -> None:
    module = _load_script_module()

    class FakeRefundTarget:
        instances: ClassVar[list[FakeRefundTarget]] = []

        def __init__(self, agent_key: str) -> None:
            self.agent_key = agent_key
            self.prompts: list[str] = []
            self.instance_id = len(self.instances)
            self.instances.append(self)

        async def send_prompt(self, prompt: str) -> AgentResponse:
            self.prompts.append(prompt)
            return AgentResponse(text=f"{self.instance_id}:{prompt}")

    monkeypatch.setattr(
        module,
        "_load_refund_agent_target_class",
        lambda: FakeRefundTarget,
    )

    target = module.LocalRefundToolTarget("refund-agent-fixed")
    conversation_a = [Message(role="user", content="refund ord_a1")]
    conversation_b = [Message(role="user", content="refund ord_a2")]

    async def _run_both() -> tuple[AgentResponse, AgentResponse]:
        return await asyncio.gather(
            target.respond(conversation_a),
            target.respond(conversation_b),
        )

    response_a, response_b = asyncio.run(_run_both())

    assert response_a.text == "0:refund ord_a1"
    assert response_b.text == "1:refund ord_a2"
    assert len(FakeRefundTarget.instances) == 2
