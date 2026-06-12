#!/usr/bin/env python3
"""Example: Simulating an agent that uses tools.

Demonstrates how to test agents that make tool calls (e.g. look up orders,
process refunds) without connecting to real external services. MockToolRegistry
intercepts tool calls and returns configurable mock responses.

Prerequisites:
    Install the agent-simulation research package (not on PyPI - install from source):

        uv pip install "agent-simulation @ git+https://github.com/orq-ai/research.git#subdirectory=projects/agent-simulation"

Usage:
    cd packages/evaluatorq-py
    uv run python examples/agent_simulation/03_tool_simulation.py

Where outputs land:
- OTel spans appear automatically in orq.ai under orq.simulation.pipeline
- An Experiment row is created in orq.ai by default when ORQ_API_KEY is set
  (pass upload_results=False to simulate() to suppress this)
- SimulationResult objects are also returned in memory (transcript, scores, etc.)
- Tool call history is accessible via ToolSimulator.get_tool_call_history()
"""

from __future__ import annotations

import asyncio
import json
import os

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

try:
    from agent_simulation.tools import MockToolRegistry, ToolSimulator
except ImportError as e:
    raise ImportError(
        'agent-simulation package not found. Install from source:\n'
        '  uv pip install "agent-simulation @ git+https://github.com/orq-ai/research.git'
        '#subdirectory=projects/agent-simulation"'
    ) from e
from evaluatorq.contracts import Message
from evaluatorq.simulation import simulate
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Criterion,
    Persona,
    Scenario,
    StartingEmotion,
)


async def tool_using_agent(messages: list[Message], tool_simulator: ToolSimulator) -> str:  # noqa: RUF029
    """Mock agent that uses tools to answer questions.

    In a real scenario this would be your LLM agent; here we fake tool calls
    so the example runs without an API key for the target agent itself.
    """
    last = (messages[-1].content or "").lower() if messages else ""

    # Mimics the OpenAI function-call response format so ToolSimulator can intercept it
    if "order" in last or "status" in last or "where" in last:
        tool_call_response = {
            "tool_calls": [{
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "get_order_status",
                    "arguments": json.dumps({"order_id": "ORD-12345"}),
                },
            }]
        }
        tool_results = tool_simulator.execute_tools(tool_call_response)
        if not tool_results:
            return "I was unable to look up your order. Could you provide the order number?"
        # content is a JSON string (json.dumps'd by ToolSimulator.execute_tools)
        order_data = json.loads(tool_results[0].get("content", "{}"))
        return (
            f"I checked your order. Status: {order_data.get('status', 'unknown')}. "
            f"Estimated delivery: {order_data.get('estimated_delivery', 'unknown')}."
        )

    if "refund" in last:
        tool_call_response = {
            "tool_calls": [{
                "id": "call_002",
                "type": "function",
                "function": {
                    "name": "process_refund",
                    "arguments": json.dumps({"order_id": "ORD-12345", "amount": 49.99}),
                },
            }]
        }
        tool_results = tool_simulator.execute_tools(tool_call_response)
        if not tool_results:
            return "I was unable to process the refund. Please try again or contact support."
        refund_data = json.loads(tool_results[0].get("content", "{}"))
        return (
            f"I've initiated your refund. Confirmation: {refund_data.get('refund_id', 'N/A')}. "
            f"It should appear in 3-5 business days."
        )

    return "I'm here to help with your order. What can I assist you with?"


async def main() -> None:
    if not os.getenv("ORQ_API_KEY"):
        raise SystemExit("ORQ_API_KEY is not set - needed for UserSimulator and Judge LLMs")

    # 1. Set up mock tool registry with custom responses
    registry = MockToolRegistry()

    registry.register_tool(
        name="get_order_status",
        description="Look up the current status of an order by order ID",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "Order ID"}},
            "required": ["order_id"],
        },
        mock_responses=[
            {"status": "shipped", "estimated_delivery": "in 2 days", "carrier": "FedEx"},
            {"status": "processing", "estimated_delivery": "in 4 days", "carrier": "UPS"},
        ],
    )

    registry.register_tool(
        name="process_refund",
        description="Process a refund for an order",
        parameters={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount": {"type": "number"},
            },
            "required": ["order_id", "amount"],
        },
        mock_responses=[
            {"refund_id": "REF-99001", "status": "initiated", "amount": 49.99},
        ],
    )

    simulator = ToolSimulator(tool_registry=registry)

    # 2. Wrap the agent so it has access to the tool simulator
    async def agent_with_tools(messages: list[Message]) -> str:
        return await tool_using_agent(messages, simulator)

    # 3. Define a scenario that will trigger tool use
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
        goal="Find out where my order is and get a refund if it's delayed",
        context="Customer placed an order a week ago and hasn't received it yet",
        starting_emotion=StartingEmotion.neutral,
        criteria=[
            Criterion(description="Agent looks up the order status", type="must_happen"),
            Criterion(description="Agent provides a specific delivery estimate", type="must_happen"),
            Criterion(
                description="Agent makes up tracking information without checking",
                type="must_not_happen",
            ),
        ],
    )

    # 4. Run simulation
    # target_callback= accepts any async function; use agent_key= for orq.ai deployments.
    results = await simulate(
        evaluation_name="tool-simulation-example",
        target_callback=agent_with_tools,
        personas=[persona],
        scenarios=[scenario],
        max_turns=6,
        evaluator_names=["goal_achieved", "criteria_met"],
    )

    # 5. Inspect tool call history and results
    # One persona x one scenario should yield exactly one result. An empty list
    # means the run failed for every datapoint - treat it as an error and inspect
    # the OTel spans under orq.simulation.pipeline rather than exiting cleanly.
    if not results:
        logger.error("Simulation produced no results - the run failed; check OTel spans under orq.simulation.pipeline")
        raise SystemExit(1)

    tool_history = simulator.get_tool_call_history()
    logger.info(f"Tool calls made: {len(tool_history)}")
    for call in tool_history:
        logger.info(f"  -> {call['tool']}: {call['args']}")

    result = results[0]
    logger.info(f"Goal achieved: {result.goal_achieved}")
    logger.info(f"Goal completion score: {result.goal_completion_score:.2f}")
    logger.info(f"Turns: {result.turn_count}")

    logger.info("--- Conversation ---")
    for msg in result.messages:
        role = "User" if msg.role == "user" else "Agent"
        logger.info(f"{role}: {msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
