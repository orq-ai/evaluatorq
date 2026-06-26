#!/usr/bin/env python3
"""Example: Basic agent simulation with a mock agent.

Demonstrates the core simulation loop with a local mock agent:
- Define a persona and scenario manually
- Run a simulation against a local callback function
- Inspect the conversation and result

Usage:
    cd packages/evaluatorq-py
    uv run python examples/agent_simulation/01_basic_simulation.py

Where outputs land:
- OTel spans appear automatically in orq.ai under orq.simulation.pipeline
  (requires ORQ_API_KEY to be set)
- An Experiment row is created in orq.ai by default when ORQ_API_KEY is set
  (pass upload_results=False to simulate() to suppress this)
- Results are also returned in memory as SimulationResult objects
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

# load_dotenv() runs before local imports so env vars are set before any
# library init code that reads them (e.g. evaluatorq tracing setup).
load_dotenv()

from evaluatorq.contracts import Message
from evaluatorq.simulation import simulate
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Criterion,
    EmotionalArc,
    Persona,
    Scenario,
    StartingEmotion,
)


async def support_agent(messages: list[Message]) -> str:  # noqa: RUF029
    """Simple mock customer support agent - replace with your own logic.

    Declared `async` because target_callback must be awaitable per the simulation
    runner protocol; a real agent here would `await` an LLM/HTTP call.
    """
    last = (messages[-1].content or "").lower() if messages else ""
    if "refund" in last:
        return "I can help with that. Could you share your order number?"
    if "order" in last or "status" in last:
        return "Let me look that up. What email is on the account?"
    if "thank" in last:
        return "Happy to help! Anything else I can do for you?"
    return "Thanks for reaching out. How can I assist you today?"


async def main() -> None:
    if not os.getenv("ORQ_API_KEY"):
        raise SystemExit("ORQ_API_KEY is not set - needed for UserSimulator and Judge LLMs")

    # 1. Define a persona - who the simulated user is
    persona = Persona(
        name="Impatient Customer",
        patience=0.2,
        assertiveness=0.8,
        politeness=0.4,
        technical_level=0.3,
        communication_style=CommunicationStyle.terse,
        background="Received the wrong item and wants a refund urgently",
        emotional_arc=EmotionalArc.escalating,  # optional: tone escalates each turn
    )

    # 2. Define a scenario - what the user wants to achieve
    scenario = Scenario(
        name="Wrong Item Refund",
        goal="Get a full refund for the wrong item received",
        context="Customer ordered headphones but received a phone case instead",
        starting_emotion=StartingEmotion.frustrated,
        criteria=[
            Criterion(description="Agent asks for order details", type="must_happen"),
            Criterion(description="Agent acknowledges the mistake", type="must_happen"),
            Criterion(description="Agent blames the customer", type="must_not_happen"),
        ],
        is_edge_case=False,  # set True to flag adversarial/edge-case scenarios for separate analysis
    )

    # 3. Run simulation
    # target_callback=: pass any async function; use target="agent:<key>" for orq.ai agents.
    # sim_model=: the LLM used for the UserSimulator and Judge (defaults to openai/gpt-5.4-mini).
    # evaluator_names=: scorers applied to each result (default: goal_achieved, criteria_met).
    logger.info("Running simulation...")
    results = await simulate(
        evaluation_name="basic-simulation-example",
        target_callback=support_agent,
        personas=[persona],
        scenarios=[scenario],
        max_turns=6,
        evaluator_names=["goal_achieved", "criteria_met"],
    )

    # 4. Inspect results
    # One persona x one scenario should yield exactly one result. An empty list
    # means the simulation runner failed for every datapoint - treat it as an
    # error, not a benign "nothing happened". Inspect the OTel spans under
    # orq.simulation.pipeline to see where the run broke.
    if not results:
        logger.error("Simulation produced no results - the run failed; check OTel spans under orq.simulation.pipeline")
        raise SystemExit(1)
    result = results[0]
    logger.info(f"Goal achieved: {result.goal_achieved}")
    logger.info(f"Goal completion score: {result.goal_completion_score:.2f}")
    logger.info(f"Turns: {result.turn_count}")
    logger.info(f"Terminated by: {result.terminated_by}")
    if result.rules_broken:
        logger.warning(f"Rules broken: {result.rules_broken}")
    if result.criteria_results:
        logger.info(f"Criteria results: {result.criteria_results}")

    logger.info("--- Conversation ---")
    for msg in result.messages:
        role = "User" if msg.role == "user" else "Agent"
        logger.info(f"{role}: {msg.content}")


if __name__ == "__main__":
    asyncio.run(main())
