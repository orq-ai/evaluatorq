#!/usr/bin/env python3
"""Example: Iterative agent instruction improvement with the hardening loop.

NOTE: This example uses two packages:
- evaluatorq.simulation.types - production SDK types (Persona, Scenario, Criterion, enums)
- agent_simulation - research package for HardeningLoop (not in production evaluatorq yet)

String literals (e.g. communication_style="casual") are used instead of enum values
because agent_simulation.models.persona.Persona uses Pydantic Literal types, not enums.
Pydantic coerces them correctly at runtime.

Demonstrates how to automatically improve an agent's system prompt by:
1. Running simulations to find failures
2. Diagnosing why the agent failed
3. Generating targeted instruction fixes
4. Re-running to verify improvement

This is especially useful when you have a set of test scenarios and want the
agent's instructions to pass them reliably.

Prerequisites:
    Install the agent-simulation research package (not on PyPI - install from source):

        uv pip install "agent-simulation @ git+https://github.com/orq-ai/research.git#subdirectory=projects/agent-simulation"

Usage:
    cd packages/evaluatorq-py
    uv run python examples/agent_simulation/04_hardening_loop.py
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

try:
    from agent_simulation import SimulationRunner
    from agent_simulation.hardening import HardeningLoop
    from agent_simulation.models.datapoint import Datapoint
    from agent_simulation.models.persona import Persona
    from agent_simulation.models.scenario import Criterion, Scenario
except ImportError as e:
    raise ImportError(
        'agent-simulation package not found. Install from source:\n'
        '  uv pip install "agent-simulation @ git+https://github.com/orq-ai/research.git'
        '#subdirectory=projects/agent-simulation"'
    ) from e

from evaluatorq.contracts import Message

# A deliberately weak set of instructions - the loop will improve these
INITIAL_INSTRUCTIONS = """
You are a customer support agent. Help customers with their questions.
Be helpful and polite.
"""


def make_agent(instructions: str) -> Callable[[list[Message]], Awaitable[str]]:
    """Return an agent callback whose behaviour reflects the given instructions.

    The mock checks whether key phrases have been added to the instructions by
    the hardening loop, and unlocks better responses accordingly. This lets the
    example show a real pass-rate improvement across iterations without needing
    a live LLM for the target agent.

    In production, pass `instructions` as the system prompt to your LLM agent.
    """
    can_cancel = "cancellation" in instructions.lower() or "cancel" in instructions.lower()

    async def agent(messages: list[Message]) -> str:  # noqa: RUF029
        last = (messages[-1].content or "").lower() if messages else ""
        if "refund" in last:
            return "I'll process that refund. What's your order number?"
        if "cancel" in last:
            if can_cancel:
                return "Of course - I can cancel that order for you. What's the order number?"
            # Deliberately bad until the hardening loop improves the instructions
            return "I'm sorry, I cannot help with cancellations."
        if "speak" in last and "manager" in last:
            return "Of course, let me connect you with a manager right away."
        return "How can I help you today?"

    return agent


async def main() -> None:
    if not os.getenv("ORQ_API_KEY"):
        raise SystemExit("ORQ_API_KEY is not set - needed for DiagnosisAgent and FixGeneratorAgent")

    datapoints = [
        Datapoint.generate(
            persona=Persona(
                name="Cancellation Customer",
                patience=0.4,
                assertiveness=0.7,
                politeness=0.6,
                technical_level=0.3,
                communication_style="casual",
                background="Wants to cancel an order placed by mistake",
                emotional_arc="escalating",
            ),
            scenario=Scenario(
                name="Order Cancellation",
                goal="Cancel my order",
                context="Customer placed an order 10 minutes ago and wants to cancel",
                starting_emotion="urgent",
                criteria=[
                    Criterion(description="Agent helps with the cancellation", type="must_happen"),
                    Criterion(description="Agent refuses or deflects the request", type="must_not_happen"),
                ],
            ),
        ),
        Datapoint.generate(
            persona=Persona(
                name="Refund Seeker",
                patience=0.3,
                assertiveness=0.8,
                politeness=0.5,
                technical_level=0.2,
                communication_style="terse",
                background="Received a damaged product",
                emotional_arc="escalating",
            ),
            scenario=Scenario(
                name="Damaged Product Refund",
                goal="Get a full refund for the damaged item",
                context="Customer received a broken laptop charger",
                starting_emotion="frustrated",
                criteria=[
                    Criterion(description="Agent initiates the refund process", type="must_happen"),
                    Criterion(description="Agent asks for proof of damage", type="must_happen"),
                ],
            ),
        ),
    ]

    # Run the hardening loop.
    # Use a mutable container so on_instructions_updated can swap the active agent
    # without a nonlocal rebind across the closure boundary. In production, use this
    # hook to push new_instructions as a system prompt to your LLM instead.
    logger.info("Starting hardening loop...")
    logger.info(f"Initial instructions:\n{INITIAL_INSTRUCTIONS.strip()}")

    current_agent: list[Callable[[list[Message]], Awaitable[str]]] = [make_agent(INITIAL_INSTRUCTIONS)]

    async def agent_proxy(messages: list[Message]) -> str:
        return await current_agent[0](messages)

    runner = SimulationRunner(target_callback=agent_proxy, max_turns=6)

    def on_instructions_updated(new_instructions: str) -> None:
        current_agent[0] = make_agent(new_instructions)

    async with HardeningLoop(
        runner,
        INITIAL_INSTRUCTIONS,
        on_instructions_updated=on_instructions_updated,
    ) as loop:
        report = await loop.harden(datapoints, max_iterations=3)

    logger.info(f"Pass rate: {report.original_pass_rate:.0%} -> {report.final_pass_rate:.0%}")
    logger.info(f"Iterations: {report.total_iterations}")

    logger.info("--- Improved Instructions ---")
    logger.info(report.improved_instructions)

    logger.info("--- Iteration Summary ---")
    for i, iteration in enumerate(report.iterations, 1):
        logger.info(f"Iteration {i}: {iteration.pass_rate_before:.0%} -> {iteration.pass_rate_after:.0%}")


if __name__ == "__main__":
    asyncio.run(main())
