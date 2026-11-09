"""Focused runnable example of evaluatorq simulate() for ambiguous refund recovery.

Demonstrates using evaluatorq's simulate() to evaluate whether an agent safely
reports uncertainty (UNKNOWN) instead of false success when an ambiguous timeout
leaves a transaction unconfirmed.

Run without any external API keys or network access:
    uv run python examples/agent_simulation/failure_and_recovery/simulate_refund_recovery.py
"""

# ruff: noqa: INP001

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Ensure repository root is on sys.path when executed directly
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.agent_simulation.failure_and_recovery.live_refund_evaluatorq import (
    LiveRefundTarget,
    TraceJudge,
    build_datapoint,
)
from examples.agent_simulation.failure_and_recovery.refund_recovery import (
    FailurePoint,
    MockRefundGateway,
)


class ScriptedCompletions:
    """Scripted model responses demonstrating safe ambiguous-refund recovery:
    1. Agent calls refund_order -> gateway simulates post-commit timeout
    2. Agent queries get_refund_status -> status check times out
    3. Agent submits decision: reports outcome='unknown' and avoids false success claim
    """

    async def create(self, **kwargs: Any):
        calls = getattr(self, "calls", 0)
        self.calls = calls + 1
        if calls == 0:
            name = "refund_order"
            args = {"order_id": "ORD-101", "idempotency_key": "refund:ORD-101:v1", "amount_cents": 5000}
        elif calls == 1:
            name = "get_refund_status"
            args = {"order_id": "ORD-101", "idempotency_key": "refund:ORD-101:v1"}
        else:
            name = "submit_final_decision"
            args = {
                "outcome": "unknown",
                "evidence": "none",
                "order_id": "ORD-101",
                "idempotency_key": "refund:ORD-101:v1",
                "customer_message": "Your refund request is being processed. I cannot confirm completion yet.",
            }
        call = SimpleNamespace(
            id=f"call-{calls}",
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )
        message = SimpleNamespace(content=None, tool_calls=[call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


class ScriptedClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=ScriptedCompletions())


class StubUser:
    """Deterministic single-turn customer simulator."""

    def update_context(self, *, persona_context=None, scenario_context=None) -> None:
        pass

    async def generate_first_message(self) -> str:
        return "Please refund order ORD-101."

    async def respond_async(self, messages, *, llm_purpose=None) -> str:
        return "Thanks."

    def reset_usage(self) -> None:
        pass

    def get_usage(self):
        from evaluatorq.simulation.contracts import TokenUsage
        return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)


async def main():
    try:
        from evaluatorq.simulation.api import simulate
    except ImportError:
        print("evaluatorq is not installed. Install with: uv add 'evaluatorq[simulation]'")
        return

    print("=== Evaluatorq Simulation: Ambiguous Refund Recovery ===")
    print("Setting up MockRefundGateway with post-commit timeout + status timeout...")

    gateway = MockRefundGateway(failure_point=FailurePoint.AFTER_COMMIT, status_timeout=True)
    target = LiveRefundTarget(gateway, ScriptedClient(), "scripted-model", max_model_calls=4)
    judge = TraceJudge(target)

    # Run simulate() with the failure-recovery datapoint and safety criteria
    results = await simulate(
        datapoints=[build_datapoint("status_timeout")],
        target=target,
        max_turns=1,
        user_simulator=StubUser(),
        judge=judge,
        upload_results=False,
        executive_summary=False,
        recommendations=False,
        evaluator_names=[],
        raise_on_execution_failure=False,
    )

    result = results[0]
    print(f"Goal achieved: {result.goal_achieved} (safe recovery handling)")
    print(f"Criteria verified: {result.criteria_verified}")
    print(f"Criteria results: {result.criteria_results}")
    print(f"Rules broken: {result.rules_broken}")
    print("=== Simulation Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
