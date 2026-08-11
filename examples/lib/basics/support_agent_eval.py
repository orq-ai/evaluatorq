#!/usr/bin/env python3
"""Example: compare two versions of a support agent on the same questions.

Runs entirely offline — no API key, no account. The two "agents" are stand-ins
for whatever you actually ship: swap the bodies for a model call, a LangChain
agent, or an HTTP request to your service, and the evaluation loop is unchanged.

The script exits non-zero: `agent-v1` misses two of the three questions, and a
failing evaluator fails the run. That is the CI gate working, not a bug.

Usage:
    # from the evaluatorq repository root
    uv run python examples/lib/basics/support_agent_eval.py
"""

from __future__ import annotations

import asyncio

from evaluatorq import DataPoint, evaluatorq, job, string_contains_evaluator

POLICY = {
    "refund": "Refunds are available within 30 days of delivery.",
    "ship": "Orders ship within 2 business days.",
    "warranty": "Every device carries a 12 months warranty.",
}


@job("agent-v1")
async def agent_v1(data: DataPoint, _row: int) -> str:  # noqa: RUF029
    """First attempt: answers from memory, so it only knows about refunds."""
    question = str(data.inputs["question"]).lower()
    if "refund" in question:
        return "Sure — you can request a refund within 30 days of delivery."
    return "Our support team is happy to help with that."


@job("agent-v2")
async def agent_v2(data: DataPoint, _row: int) -> str:  # noqa: RUF029
    """Second attempt: looks the answer up in the support policy first."""
    question = str(data.inputs["question"]).lower()
    for topic, answer in POLICY.items():
        if topic in question:
            return answer
    return "Our support team is happy to help with that."


def data_points() -> list[DataPoint]:
    return [
        DataPoint(inputs={"question": "How do I get a refund?"}, expected_output="30 days"),
        DataPoint(inputs={"question": "When will my order ship?"}, expected_output="2 business days"),
        DataPoint(inputs={"question": "How long is the warranty?"}, expected_output="12 months"),
    ]


async def main() -> None:
    await evaluatorq(
        "support-agent-eval",
        data=data_points(),
        jobs=[agent_v1, agent_v2],
        evaluators=[string_contains_evaluator()],
        parallelism=3,
    )


if __name__ == "__main__":
    asyncio.run(main())
