#!/usr/bin/env python3
"""Example: compare two versions of a support agent on the same questions.

This is the quick start from the README and the Getting Started guide, so the
three stay in sync — edit this file first.

Needs no API key and no account. (If `ORQ_API_KEY` happens to be set, the run
also uploads its results to Orq — unset it for a purely local run.) The two
"agents" are stand-ins for whatever you actually ship: swap the bodies for a
model call, a LangChain agent, or an HTTP request to your service, and the
evaluation loop is unchanged.

The script exits 0 even though `agent-v1` misses two of the three questions:
evaluator failures are returned in the results for callers to inspect. To make
this a CI gate, check the results and raise an exit error in your own script.

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
    """Answers from memory — so it only really knows about refunds."""
    question = str(data.inputs["question"]).lower()
    if "refund" in question:
        return "Sure — you can request a refund within 30 days of delivery."
    return "Our support team is happy to help with that."


@job("agent-v2")
async def agent_v2(data: DataPoint, _row: int) -> str:  # noqa: RUF029
    """Looks the answer up in the support policy first."""
    question = str(data.inputs["question"]).lower()
    for topic, answer in POLICY.items():
        if topic in question:
            return answer
    return "Our support team is happy to help with that."


async def main() -> None:
    data = [
        DataPoint(inputs={"question": "How do I get a refund?"}, expected_output="30 days"),
        DataPoint(inputs={"question": "When will my order ship?"}, expected_output="2 business days"),
        DataPoint(inputs={"question": "How long is the warranty?"}, expected_output="12 months"),
    ]
    await evaluatorq(
        "support-agent-eval",
        data=data,
        jobs=[agent_v1, agent_v2],
        evaluators=[string_contains_evaluator()],
        parallelism=3,
    )


if __name__ == "__main__":
    asyncio.run(main())
