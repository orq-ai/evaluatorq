#!/usr/bin/env python3
"""Example: Simulating a CrewAI crew through the unified AgentTarget.

CrewAI is the biggest format divergence among the supported frameworks, so it is
the generality stress test for the simulation protocol. A multi-agent ``Crew``
plugs into the three-part loop by wrapping it in ``CrewAITarget`` and passing it
as ``target=``.

Framework-specific quirks handled by CrewAITarget:
- Sync API: ``Crew.kickoff`` is blocking, so it is run via ``asyncio.to_thread``.
- No message-list interface: the transcript is flattened into a single string
  injected under ``{conversation}`` in the task description.
- Multi-agent: "the response" is the crew's final output (CrewOutput.raw);
  intermediate agent/tool steps are not surfaced.
- Tokens: mapped from CrewOutput.token_usage (successful_requests -> calls).

Prerequisites:
    uv sync --extra crewai --extra simulation
    .env with ORQ_API_KEY (+ OPENAI_API_KEY / OPENAI_BASE_URL).

Usage:
    cd packages/evaluatorq-py
    uv run python examples/agent_simulation/09_crewai_simulation.py
    uv run python examples/agent_simulation/09_crewai_simulation.py --upload
"""

from __future__ import annotations

import argparse
import asyncio
import os

# Quiet CrewAI's first-run tracing/telemetry prompts before importing it.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from evaluatorq.integrations.crewai_integration import CrewAITarget
from evaluatorq.simulation import simulate
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Criterion,
    Persona,
    Scenario,
    StartingEmotion,
)

AGENT_MODEL = os.getenv("AGENT_MODEL", "openai/gpt-4o-mini")


def make_crew() -> object:
    """Build a single-agent support crew driven via the Orq router.

    The task description interpolates the flattened conversation under
    ``{conversation}``, which CrewAITarget fills each turn.
    """
    from crewai import LLM, Agent, Crew, Task

    llm = LLM(
        model=AGENT_MODEL,
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ["OPENAI_API_KEY"],
    )
    agent = Agent(
        role="Customer Support Agent",
        goal="Resolve the customer's order questions accurately and concisely",
        backstory=(
            "You are a support agent for an online store. You answer order "
            "questions. You never invent tracking details you cannot confirm."
        ),
        llm=llm,
        verbose=False,
    )
    task = Task(
        description=(
            "You are continuing a live customer support chat. Here is the "
            "conversation so far:\n\n{conversation}\n\n"
            "Write only the support agent's next reply to the customer. "
            "Order ORD-12345 is shipped via FedEx with estimated delivery in 2 days."
        ),
        expected_output="The support agent's next reply to the customer.",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task], verbose=False)


async def main() -> None:
    parser = argparse.ArgumentParser(description="CrewAI simulation example")
    parser.add_argument("--upload", action="store_true", help="Upload results to Orq as an experiment")
    parser.add_argument("--max-turns", type=int, default=6)
    args = parser.parse_args()

    if not os.getenv("ORQ_API_KEY"):
        raise SystemExit("ORQ_API_KEY is not set - needed for the UserSimulator and Judge LLMs")

    # Pass crew_factory so parallel datapoints each get an independent crew.
    target = CrewAITarget(make_crew(), crew_factory=make_crew)

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
            Criterion(description="Agent provides a specific delivery estimate", type="must_happen"),
            Criterion(description="Agent is polite and helpful", type="must_happen"),
            Criterion(description="Agent invents tracking info it cannot confirm", type="must_not_happen"),
        ],
    )

    results = await simulate(
        evaluation_name="crewai-simulation-example",
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
