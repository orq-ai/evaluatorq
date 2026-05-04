"""High-level simulation API functions."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from evaluatorq.simulation.types import DEFAULT_MODEL

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from evaluatorq.simulation.generators import FirstMessageGenerator
    from evaluatorq.simulation.types import (
        ChatMessage,
        Datapoint,
        Persona,
        Scenario,
        SimulationResult,
    )


async def simulate(
    *,
    evaluation_name: str = "",
    agent_key: str | None = None,
    target_callback: Callable[[list[ChatMessage]], str | Awaitable[str]] | None = None,
    personas: list[Persona] | None = None,
    scenarios: list[Scenario] | None = None,
    datapoints: list[Datapoint] | None = None,
    max_turns: int = 10,
    model: str = DEFAULT_MODEL,
    evaluator_names: list[str] | None = None,
    parallelism: int = 5,
) -> list[SimulationResult]:
    """High-level function to run agent simulations.

    Handles:
    - Creating persona x scenario combinations
    - Generating first messages for each combination
    - Running simulations in parallel
    - Applying evaluators to results
    """
    from openai import AsyncOpenAI

    from evaluatorq.simulation.adapters import from_orq_deployment
    from evaluatorq.simulation.evaluators import get_evaluator
    from evaluatorq.simulation.generators import FirstMessageGenerator
    from evaluatorq.simulation.runner.simulation import SimulationRunner

    # Validate evaluator names early
    resolved_evaluator_names = evaluator_names or ["goal_achieved", "criteria_met"]
    scorers = [(name, get_evaluator(name)) for name in resolved_evaluator_names]

    # Build datapoints from personas x scenarios if not provided
    if datapoints is None:
        if personas is None or scenarios is None:
            raise ValueError(
                "Either provide 'datapoints' or both 'personas' and 'scenarios'"
            )
        if not personas or not scenarios:
            raise ValueError("'personas' and 'scenarios' arrays must both be non-empty")

        api_key = os.environ.get("ORQ_API_KEY")
        if not api_key:
            raise ValueError(
                "ORQ_API_KEY environment variable is not set. Set it before calling simulate()."
            )

        shared_client = AsyncOpenAI(
            api_key=api_key,
            base_url=f"{os.environ.get('ORQ_BASE_URL', 'https://api.orq.ai')}/v2/router",
        )

        first_msg_gen = FirstMessageGenerator(model=model, client=shared_client)

        try:
            pairs = [
                (persona, scenario) for persona in personas for scenario in scenarios
            ]

            generated_datapoints: list[Datapoint] = []
            batch_size = 5
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i : i + batch_size]
                batch_results = await asyncio.gather(
                    *[
                        _generate_single_datapoint(first_msg_gen, persona, scenario)
                        for persona, scenario in batch
                    ]
                )
                generated_datapoints.extend(batch_results)
            datapoints = generated_datapoints
        finally:
            await shared_client.close()

    if not datapoints:
        raise ValueError(
            "No datapoints to simulate — persona or scenario generation may have failed"
        )

    # Bridge agentKey to invoke() if no callback is provided
    resolved_callback = target_callback
    if not resolved_callback and agent_key:
        resolved_callback = from_orq_deployment(agent_key)

    if not resolved_callback:
        raise ValueError("Either target_callback or agent_key is required")

    # Create simulation runner
    runner = SimulationRunner(
        target_callback=resolved_callback,
        model=model,
        max_turns=max_turns,
    )

    try:
        # Run simulations
        results = await runner.run_batch(
            datapoints,
            max_turns=max_turns,
            max_concurrency=parallelism,
        )

        # Apply evaluators to results
        for result in results:
            scores: dict[str, float] = {}
            for scorer_name, fn in scorers:
                scores[scorer_name] = fn(result)
            result.metadata["evaluator_scores"] = scores
            if evaluation_name:
                result.metadata["evaluation_name"] = evaluation_name

        return results
    finally:
        await runner.close()


async def _generate_single_datapoint(
    gen: FirstMessageGenerator, persona: Persona, scenario: Scenario
) -> Datapoint:
    from evaluatorq.simulation.utils.prompt_builders import generate_datapoint

    first_message = await gen.generate(persona, scenario)
    return generate_datapoint(persona, scenario, first_message)


async def generate_and_simulate(
    *,
    evaluation_name: str = "",
    agent_description: str,
    agent_key: str | None = None,
    target_callback: Callable[[list[ChatMessage]], str | Awaitable[str]] | None = None,
    num_personas: int = 5,
    num_scenarios: int = 5,
    max_turns: int = 10,
    model: str = DEFAULT_MODEL,
    evaluator_names: list[str] | None = None,
    parallelism: int = 5,
) -> list[SimulationResult]:
    """Generate personas/scenarios and run simulations.

    Convenience function that combines generation and simulation.
    """
    from openai import AsyncOpenAI

    from evaluatorq.simulation.adapters import from_orq_deployment
    from evaluatorq.simulation.generators import PersonaGenerator, ScenarioGenerator

    # Bridge agentKey to invoke() if no callback is provided
    resolved_callback = target_callback
    if not resolved_callback and agent_key:
        resolved_callback = from_orq_deployment(agent_key)

    if not resolved_callback:
        raise ValueError(
            "Either target_callback or agent_key is required for generate_and_simulate"
        )

    api_key = os.environ.get("ORQ_API_KEY")
    if not api_key:
        raise ValueError(
            "ORQ_API_KEY environment variable is not set. Set it before calling generate_and_simulate()."
        )

    shared_client = AsyncOpenAI(
        api_key=api_key,
        base_url=f"{os.environ.get('ORQ_BASE_URL', 'https://api.orq.ai')}/v2/router",
    )

    try:
        persona_gen = PersonaGenerator(model=model, client=shared_client)
        scenario_gen = ScenarioGenerator(model=model, client=shared_client)

        personas, scenarios = await asyncio.gather(
            persona_gen.generate(
                agent_description=agent_description, num_personas=num_personas
            ),
            scenario_gen.generate(
                agent_description=agent_description, num_scenarios=num_scenarios
            ),
        )
    finally:
        await shared_client.close()

    # Run simulations
    return await simulate(
        evaluation_name=evaluation_name,
        target_callback=resolved_callback,
        personas=personas,
        scenarios=scenarios,
        max_turns=max_turns,
        model=model,
        evaluator_names=evaluator_names,
        parallelism=parallelism,
    )
