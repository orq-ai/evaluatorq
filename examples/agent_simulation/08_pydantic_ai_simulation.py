#!/usr/bin/env python3
"""Example: Simulating a Pydantic AI agent through the unified AgentTarget.

Demonstrates that a Pydantic AI ``Agent`` plugs into the three-part simulation
loop by wrapping it in ``PydanticAITarget`` and passing it as ``target=``.

Framework-specific quirks handled by PydanticAITarget:
- Message format: Pydantic AI threads context via typed ``message_history``,
  not a role/content list. The target owns history internally and forwards only
  the latest user turn, re-feeding ``result.all_messages()`` each turn.
- Async: ``agent.run`` is awaitable; no thread offload needed.
- Tokens: ``RunUsage`` has input/output tokens but no total (derived).

Prerequisites:
    uv sync --extra pydantic-ai --extra simulation
    .env with ORQ_API_KEY (+ OPENAI_API_KEY / OPENAI_BASE_URL).

Usage:
    cd packages/evaluatorq-py
    uv run python examples/agent_simulation/08_pydantic_ai_simulation.py
    uv run python examples/agent_simulation/08_pydantic_ai_simulation.py --upload
"""

from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from evaluatorq.integrations.pydantic_ai_integration import PydanticAITarget
from evaluatorq.simulation import simulate
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Criterion,
    Persona,
    Scenario,
    StartingEmotion,
)

AGENT_MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4o-mini")


def build_agent() -> object:
    """Build a Pydantic AI support agent with one tool, driven via the Orq router."""
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    model = OpenAIChatModel(
        AGENT_MODEL,
        provider=OpenAIProvider(
            base_url=os.environ.get("OPENAI_BASE_URL"),
            api_key=os.environ["OPENAI_API_KEY"],
        ),
    )
    agent = Agent(
        model,
        system_prompt=(
            "You are a customer support agent for an online store. "
            "Use the get_order_status tool to look up orders. "
            "Never invent tracking details you did not retrieve from a tool."
        ),
    )

    @agent.tool_plain
    def get_order_status(order_id: str) -> str:
        """Look up the current status of an order by its order ID."""
        return (
            f"Order {order_id}: status=shipped, carrier=FedEx, "
            "estimated_delivery=in 2 days."
        )

    return agent


async def main() -> None:
    parser = argparse.ArgumentParser(description="Pydantic AI simulation example")
    parser.add_argument("--upload", action="store_true", help="Upload results to Orq as an experiment")
    parser.add_argument("--max-turns", type=int, default=6)
    args = parser.parse_args()

    if not os.getenv("ORQ_API_KEY"):
        raise SystemExit("ORQ_API_KEY is not set - needed for the UserSimulator and Judge LLMs")

    target = PydanticAITarget(build_agent())

    persona = Persona(
        name="Curious Shopper",
        patience=0.7,
        assertiveness=0.5,
        politeness=0.8,
        technical_level=0.4,
        communication_style=CommunicationStyle.casual,
        background="Waiting on a package and wants an update",
    )

    scenario = Scenario(
        name="Order Status Check",
        goal="Find out where order ORD-12345 is",
        context="Customer placed an order a week ago and hasn't received it yet",
        starting_emotion=StartingEmotion.neutral,
        criteria=[
            Criterion(description="Agent looks up the order status with the tool", type="must_happen"),
            Criterion(description="Agent provides a specific delivery estimate", type="must_happen"),
            Criterion(description="Agent invents tracking info without checking", type="must_not_happen"),
        ],
    )

    results = await simulate(
        evaluation_name="pydantic-ai-simulation-example",
        target=target,
        personas=[persona],
        scenarios=[scenario],
        max_turns=args.max_turns,
        evaluator_names=["goal_achieved", "criteria_met"],
        upload_results=args.upload,
        exit_on_failure=False,
    )

    if not results:
        logger.error("Simulation produced no results - the run failed; check OTel spans under orq.simulation.pipeline")
        raise SystemExit(1)

    result = results[0]
    logger.info(f"Goal achieved: {result.goal_achieved}")
    logger.info(f"Goal completion score: {result.goal_completion_score:.2f}")
    logger.info(f"Turns: {result.turn_count}  terminated_by={result.terminated_by}")
    logger.info(f"Criteria results: {result.criteria_results}")
    logger.info(f"Token usage: {result.token_usage}")

    logger.info("--- Conversation ---")
    for msg in result.messages:
        role = "User" if msg.role == "user" else "Agent"
        logger.info(f"{role}: {msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
