#!/usr/bin/env python3
"""Example: Production simulation with wrap_simulation_agent() + evaluatorq().

This is the recommended pattern for production use. It differs from the bare
simulate() call in two important ways:

1. wrap_simulation_agent() creates a job function that evaluatorq() calls once per
   DataPoint, reusing the resolved agent callback across the batch.
2. evaluatorq() handles CI gating, result display, and auto-upload to orq.ai
   Experiments - so results land in the Experiments table and are linked to the
   deployment, not just returned in memory.

Use this pattern when:
- You want Experiment rows in orq.ai with a URL you can share with stakeholders
- You are running simulations as part of a CI/CD pipeline
- You need to compose simulation scoring with other evaluatorq evaluators

Use simulate() or generate_and_simulate() (see examples 01-02) when you want
a simpler call without composing alongside other evaluatorq evaluators, or
when you want to pass upload_results=False to suppress the Experiment upload.

Usage:
    cd packages/evaluatorq-py
    uv run python examples/agent_simulation/05_wrap_and_experiment.py \
        --deployment my-support-agent

Where outputs land:
- Experiment row created in orq.ai - URL printed to stdout on completion
- OTel spans under orq.job / orq.simulation.run / orq.simulation.turn
- SimulationResult objects converted to OpenResponses format and returned
  as DataPointResult.output by evaluatorq()
"""

from __future__ import annotations

import argparse
import asyncio
import os

from dotenv import load_dotenv
from loguru import logger

# load_dotenv() runs before local imports so env vars are set before any
# library init code that reads them (e.g. evaluatorq tracing setup).
load_dotenv()

from evaluatorq import DataPoint, evaluatorq
from evaluatorq.simulation import wrap_simulation_agent
from evaluatorq.simulation.types import (
    CommunicationStyle,
    Criterion,
    EmotionalArc,
    Persona,
    Scenario,
    StartingEmotion,
)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Simulation wired into evaluatorq() - the production pattern")
    parser.add_argument("--deployment", "-d", required=True, help="orq.ai deployment key (from AI Studio)")
    parser.add_argument("--max-turns", type=int, default=6)
    args = parser.parse_args()

    if not os.getenv("ORQ_API_KEY"):
        raise SystemExit("ORQ_API_KEY is not set")

    # 1. Define personas and scenarios.
    #    The DataPoint wraps each (persona, scenario) pair for the evaluatorq() framework.
    personas = [
        Persona(
            name="Impatient Customer",
            patience=0.2,
            assertiveness=0.8,
            politeness=0.4,
            technical_level=0.3,
            communication_style=CommunicationStyle.terse,
            background="Received the wrong item and wants a refund urgently",
            emotional_arc=EmotionalArc.escalating,
        ),
        Persona(
            name="Polite First-Timer",
            patience=0.8,
            assertiveness=0.3,
            politeness=0.9,
            technical_level=0.2,
            communication_style=CommunicationStyle.formal,
            background="First time contacting support, unfamiliar with the process",
        ),
    ]

    scenarios = [
        Scenario(
            name="Wrong Item Refund",
            goal="Get a full refund for the wrong item received",
            context="Customer ordered headphones but received a phone case instead",
            starting_emotion=StartingEmotion.frustrated,
            criteria=[
                Criterion(description="Agent asks for order details", type="must_happen"),
                Criterion(description="Agent acknowledges the mistake", type="must_happen"),
                Criterion(description="Agent blames the customer", type="must_not_happen"),
            ],
        ),
        Scenario(
            name="Delivery Status",
            goal="Find out when the order will arrive",
            context="Customer placed an order 5 days ago and hasn't received it",
            starting_emotion=StartingEmotion.neutral,
            criteria=[
                Criterion(description="Agent provides a specific timeline or tracking info", type="must_happen"),
                Criterion(description="Agent makes up a delivery date without checking", type="must_not_happen"),
            ],
            is_edge_case=False,  # explicitly False here to contrast with edge-case scenarios
        ),
    ]

    # 2. Build a DataPoint for each (persona, scenario) pair.
    #    wrap_simulation_agent() reads inputs["persona"] and inputs["scenario"].
    data = [
        DataPoint(inputs={
            "persona": persona.model_dump(),
            "scenario": scenario.model_dump(),
        })
        for persona in personas
        for scenario in scenarios
    ]
    logger.info(f"Running {len(data)} simulations ({len(personas)} personas x {len(scenarios)} scenarios)")

    # 3. Create the simulation job.
    #    agent_key= is the orq.ai deployment key - routes through from_orq_deployment() internally.
    #    Use target_callback= for local functions or third-party agents.
    job = wrap_simulation_agent(
        name="support-simulation",
        agent_key=args.deployment,
        max_turns=args.max_turns,
    )

    # 4. Run via evaluatorq() for Experiment upload, CI gating, and result display.
    #    The Experiment URL is printed to stdout when the run finishes.
    #    job.aclose() releases the wrapper's long-lived simulation runner and HTTP client.
    try:
        await evaluatorq(
            "support-agent-simulation",
            data=data,
            jobs=[job],
            evaluators=[],  # add evaluatorq scorers here if needed
        )
    finally:
        await job.aclose()

    logger.info("Simulation complete - check orq.ai Experiments for results")


if __name__ == "__main__":
    asyncio.run(main())
