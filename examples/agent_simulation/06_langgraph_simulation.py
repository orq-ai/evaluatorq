#!/usr/bin/env python3
"""Example: Simulating a LangGraph agent through the unified AgentTarget.

Demonstrates that Agent Simulation is framework-agnostic: a compiled LangGraph
``StateGraph`` plugs into the three-part loop (user simulator -> agent under
test -> judge) by wrapping it in ``LangGraphTarget`` and passing it as
``target=``. No per-framework code lives in the simulation engine.

Framework-specific quirks handled by LangGraphTarget:
- Message format: LangGraph owns thread state (keyed by thread_id), so the
  target forwards only the latest user turn rather than the full transcript.
- State: the graph needs a checkpointer so thread_id continuity works across
  turns; LangGraphTarget generates a fresh thread per instance.
- Tool calls: interleaved text/tool ordering is preserved in AgentResponse.
- Tokens: collected via a LangChain callback handler.

Prerequisites:
    uv sync --extra langgraph --extra simulation
    .env with ORQ_API_KEY (+ OPENAI_API_KEY / OPENAI_BASE_URL for the agent's
    own model). Both sim-side and agent-side calls route through the Orq router.

Usage:
    cd packages/evaluatorq-py
    uv run python examples/agent_simulation/06_langgraph_simulation.py
    uv run python examples/agent_simulation/06_langgraph_simulation.py --upload
"""

from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from evaluatorq.integrations.langgraph_integration import LangGraphTarget
from evaluatorq.simulation import simulate
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Criterion,
    Persona,
    Scenario,
    StartingEmotion,
)

# The agent under test calls its own model. Route it through the Orq router via
# OPENAI_BASE_URL so a single Orq key powers both the agent and the simulator.
AGENT_MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4o-mini")


def build_graph() -> object:
    """Build a small ReAct support agent with one tool, backed by a checkpointer."""
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.prebuilt import create_react_agent

    @tool
    def get_order_status(order_id: str) -> str:
        """Look up the current status of an order by its order ID."""
        return (
            f"Order {order_id}: status=shipped, carrier=FedEx, "
            "estimated_delivery=in 2 days."
        )

    model = ChatOpenAI(
        model=AGENT_MODEL,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),
        temperature=0,
    )
    return create_react_agent(
        model,
        tools=[get_order_status],
        prompt=(
            "You are a customer support agent for an online store. "
            "Use the get_order_status tool to look up orders. "
            "Never invent tracking details you did not retrieve from a tool."
        ),
        checkpointer=InMemorySaver(),
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph agent simulation example")
    parser.add_argument("--upload", action="store_true", help="Upload results to Orq as an experiment")
    parser.add_argument("--max-turns", type=int, default=6)
    args = parser.parse_args()

    if not os.getenv("ORQ_API_KEY"):
        raise SystemExit("ORQ_API_KEY is not set - needed for the UserSimulator and Judge LLMs")

    # Wrap the compiled LangGraph app as a unified AgentTarget. _resolve_target()
    # routes AgentTarget instances to the runner's respond(messages) path.
    target = LangGraphTarget(build_graph())

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
        evaluation_name="langgraph-simulation-example",
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
