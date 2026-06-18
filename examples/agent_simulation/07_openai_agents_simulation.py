#!/usr/bin/env python3
"""Example: Simulating an OpenAI Agents SDK agent through the unified AgentTarget.

Demonstrates that an ``agents.Agent`` (OpenAI Agents SDK) plugs into the
three-part simulation loop by wrapping it in ``OpenAIAgentTarget`` and passing
it as ``target=``.

Framework-specific quirks handled by OpenAIAgentTarget:
- Message format: the SDK is stateless per run, so the target renders the full
  transcript into Responses-API input items each turn (the orchestrator owns
  continuity), preserving tool calls and tool results across turns.
- Tool calls: function_call / function_call_output items round-trip into
  AgentResponse, so tool ordering survives.
- Tokens: read from the run's usage (context_wrapper.usage).

Note on routing: the Orq router is a Chat Completions endpoint, so this example
drives the agent with ``OpenAIChatCompletionsModel`` over a custom AsyncOpenAI
client pointed at OPENAI_BASE_URL, and disables the SDK's OpenAI-platform
tracing (which would need a real OpenAI key).

Prerequisites:
    uv sync --extra openai-agents --extra simulation
    .env with ORQ_API_KEY (+ OPENAI_API_KEY / OPENAI_BASE_URL).

Usage:
    cd packages/evaluatorq-py
    uv run python examples/agent_simulation/07_openai_agents_simulation.py
    uv run python examples/agent_simulation/07_openai_agents_simulation.py --upload
"""

from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from evaluatorq.integrations.openai_agents_integration import OpenAIAgentTarget
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
    """Build a small support Agent with one tool, driven via the Orq router."""
    from agents import (
        Agent,
        OpenAIChatCompletionsModel,
        function_tool,
        set_tracing_disabled,
    )
    from openai import AsyncOpenAI

    # The SDK ships traces to the OpenAI platform by default; disable it since we
    # authenticate against the Orq router, not OpenAI directly.
    set_tracing_disabled(True)

    @function_tool
    def get_order_status(order_id: str) -> str:
        """Look up the current status of an order by its order ID."""
        return (
            f"Order {order_id}: status=shipped, carrier=FedEx, "
            "estimated_delivery=in 2 days."
        )

    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )
    model = OpenAIChatCompletionsModel(model=AGENT_MODEL, openai_client=client)
    return Agent(
        name="support-agent",
        instructions=(
            "You are a customer support agent for an online store. "
            "Use the get_order_status tool to look up orders. "
            "Never invent tracking details you did not retrieve from a tool."
        ),
        model=model,
        tools=[get_order_status],
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI Agents SDK simulation example")
    parser.add_argument("--upload", action="store_true", help="Upload results to Orq as an experiment")
    parser.add_argument("--max-turns", type=int, default=6)
    args = parser.parse_args()

    if not os.getenv("ORQ_API_KEY"):
        raise SystemExit("ORQ_API_KEY is not set - needed for the UserSimulator and Judge LLMs")

    # _resolve_target() routes AgentTarget instances to the respond(messages) path.
    target = OpenAIAgentTarget(build_agent())

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
        evaluation_name="openai-agents-simulation-example",
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
