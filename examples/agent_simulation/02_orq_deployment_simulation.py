#!/usr/bin/env python3
"""Example: Batch simulation against an orq.ai deployment or A2A agent.

Demonstrates how to:
- Auto-generate personas and scenarios from an agent description
- Run a batch of simulations against a live orq.ai deployment (--deployment)
  or a live A2A agent via the orq Responses API (--agent)
- Export results to JSONL

Usage:
    cd packages/evaluatorq-py

    # Against an orq.ai deployment (prompt + model config in AI Studio)
    uv run python examples/agent_simulation/02_orq_deployment_simulation.py \
        --deployment my-support-agent

    # Against an A2A agent via the orq Responses API
    uv run python examples/agent_simulation/02_orq_deployment_simulation.py \
        --agent my-a2a-agent

    # Faster test run
    uv run python examples/agent_simulation/02_orq_deployment_simulation.py \
        --deployment my-support-agent --num-personas 2 --num-scenarios 3

Where outputs land:
- OTel spans appear automatically in orq.ai under orq.simulation.pipeline
- An Experiment row is created in orq.ai by default (URL printed to stdout);
  pass upload_results=False to generate_and_simulate() to suppress this
- Results are exported to JSONL for offline analysis or dataset seeding
"""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# generate_and_simulate() synthesises Persona x Scenario pairs from a plain-text
# description, then runs the full simulation batch in one call.
from evaluatorq.contracts import Message
from evaluatorq.simulation import (
    export_results_to_jsonl,
    generate_and_simulate,
)


def make_a2a_callback(agent_key: str) -> Callable[[list[Message]], Coroutine[Any, Any, str]]:
    """Return a target_callback that calls an orq A2A agent via the Responses API.

    The Responses API (client.agents.responses.create) is the production path for
    full A2A agents in orq.ai - agents with memory, tool use, and multi-step
    reasoning, as opposed to stateless deployments (prompt + model config).

    Each call sends only the latest user message. Conversation context is
    preserved by threading the task_id returned from the first response into
    subsequent calls (task_id= continues the existing agent execution), so the
    agent's server-side state carries across turns.
    """
    from orq_ai_sdk import Orq
    from orq_ai_sdk.models import A2AMessage, TextPart

    client = Orq(api_key=os.getenv("ORQ_API_KEY", ""))

    # Persists across turns within this callback's lifetime: the first call
    # returns a task_id; later calls pass it back to continue the same execution.
    state: dict[str, str] = {}

    async def callback(messages: list[Message]) -> str:
        last = messages[-1]
        message = A2AMessage(
            role="user",
            parts=[TextPart(kind="text", text=last.content or "")],
        )
        response = await asyncio.to_thread(
            client.agents.responses.create,
            agent_key=agent_key,
            message=message,
            task_id=state.get("task_id"),
        )
        # Remember the task so the next turn continues this conversation.
        # If the response carries no task_id, threading is broken: every turn
        # sends task_id=None and the agent loses server-side state, silently
        # degrading the multi-turn simulation into disconnected single turns.
        # Surface that once so a degraded run is distinguishable from a healthy one.
        if getattr(response, "task_id", None):
            state["task_id"] = response.task_id
        elif not state.get("warned_no_task_id"):
            logger.warning(
                f"A2A agent '{agent_key}' returned no task_id - multi-turn context "
                "will not thread; the agent will see each turn in isolation"
            )
            state["warned_no_task_id"] = "1"
        # A2A response: output is List[AgentResponseMessage]; agent turns have
        # role == "agent" and TextPart entries in .parts (kind="text").
        texts = [
            part.text
            for msg in response.output
            if msg.role == "agent"
            for part in msg.parts
            if isinstance(part, TextPart)
        ]
        if not texts:
            raise RuntimeError(
                f"A2A agent '{agent_key}' returned no text output - "
                "check agent_key and API connectivity"
            )
        return " ".join(texts)

    return callback


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch simulation against an orq.ai deployment or A2A agent"
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--deployment", "-d", help="orq.ai deployment key (from AI Studio -> Deployments)")
    target_group.add_argument("--agent", "-a", help="orq.ai A2A agent key (from AI Studio -> Agents)")
    parser.add_argument(
        "--description",
        default="",
        help="Plain-text description of what the agent does (improves persona/scenario generation)",
    )
    parser.add_argument("--num-personas", type=int, default=3)
    parser.add_argument("--num-scenarios", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--output", default="data/results.jsonl", help="Output path for JSONL results (relative to packages/evaluatorq-py/)")
    args = parser.parse_args()

    if not os.getenv("ORQ_API_KEY"):
        raise SystemExit("ORQ_API_KEY is not set")

    if args.deployment:
        # Deployment path: target="deployment:<key>" routes through
        # from_orq_deployment() internally, which calls evaluatorq.deployment.invoke
        # - stateless prompt + model config.
        target_key = args.deployment
        agent_description = args.description or f"orq.ai deployment '{args.deployment}'"
        target_kwargs: dict[str, Any] = {"target": f"deployment:{target_key}"}
        logger.info(f"Target: deployment '{target_key}'")
    else:
        # A2A agent path: wrap client.agents.responses.create as a target_callback.
        # Use this for full agents with memory, tools, and multi-step reasoning.
        assert args.agent is not None, "argparse mutually-exclusive group guarantees one of --deployment/--agent"  # noqa: S101
        agent_description = args.description or f"orq.ai A2A agent '{args.agent}'"
        target_kwargs = {"target_callback": make_a2a_callback(args.agent)}
        logger.info(f"Target: A2A agent '{args.agent}' via Responses API")

    logger.info(f"Generating {args.num_personas} personas x {args.num_scenarios} scenarios...")
    results = await generate_and_simulate(
        evaluation_name="orq-deployment-simulation-example",
        agent_description=agent_description,
        num_personas=args.num_personas,
        num_scenarios=args.num_scenarios,
        max_turns=args.max_turns,
        evaluator_names=["goal_achieved", "criteria_met"],
        parallelism=5,
        **target_kwargs,
    )

    # Summary
    if not results:
        logger.warning("No results to summarise")
    else:
        passed = sum(r.goal_achieved for r in results)
        logger.info(f"Pass rate: {passed}/{len(results)} ({100 * passed / len(results):.0f}%)")

    for r in results:
        status = "PASS" if r.goal_achieved else "FAIL"
        logger.info(f"  [{status}] score={r.goal_completion_score:.2f} turns={r.turn_count}")
        logger.info(f"         terminated_by={r.terminated_by} rules_broken={r.rules_broken or []}")

    # Export to JSONL for offline analysis or seeding an orq.ai Dataset
    output_path = Path(__file__).parent.parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_results_to_jsonl(results, str(output_path))
    logger.info(f"Results written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
