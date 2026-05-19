"""High-level simulation API functions."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from evaluatorq.simulation.types import DEFAULT_MODEL

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from opentelemetry.trace import Span

    from evaluatorq.simulation.agents.base import BaseAgent
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
    target: Callable[[list[ChatMessage]], str | Awaitable[str]] | None = None,
    personas: list[Persona] | None = None,
    scenarios: list[Scenario] | None = None,
    datapoints: list[Datapoint] | None = None,
    max_turns: int = 10,
    model: str = DEFAULT_MODEL,
    evaluator_names: list[str] | None = None,
    parallelism: int = 5,
    user_simulator: BaseAgent | None = None,
    judge: BaseAgent | None = None,
    upload_results: bool = False,
    evaluation_description: str | None = None,
    path: str | None = None,
) -> list[SimulationResult]:
    """High-level function to run agent simulations.

    Handles:
    - Creating persona x scenario combinations
    - Generating first messages for each combination
    - Running simulations in parallel
    - Applying evaluators to results
    - Optional upload of results to the Orq platform — pass
      ``upload_results=True`` and set ``ORQ_API_KEY`` to enable.

    Args:
        target_callback: Callable that receives the conversation history and
            returns the agent's response.  Kept for backwards compatibility.
        target: Alias for ``target_callback``.  Takes precedence when both
            are supplied.
        user_simulator: Pre-constructed ``BaseAgent`` to drive the user side
            of the conversation.  When omitted a default ``UserSimulatorAgent``
            is built from ``model``.  If the provided agent exposes an
            ``update_context(persona_context, scenario_context)`` method it will
            be called before each individual simulation so the agent is grounded
            in the current datapoint's persona and scenario.
        judge: Pre-constructed ``BaseAgent`` used to evaluate each turn.
            When omitted a default ``JudgeAgent`` is built from ``model``.
        upload_results: When ``True``, results are uploaded to the Orq
            platform if ``ORQ_API_KEY`` is set in the environment. Upload
            errors are raised so callers can detect failed uploads. Defaults to
            ``False`` (opt-in) so simulate() does not perform network
            uploads silently — callers must opt in explicitly to surface
            results in the platform.
        evaluation_description: Optional human-readable description attached
            to the experiment uploaded to Orq. Mirrors ``evaluatorq()``.
        path: Optional Orq folder path (e.g. ``"MyProject/MyFolder"``) under
            which the experiment is created. Mirrors ``evaluatorq()``.
    """
    from datetime import datetime, timezone

    from evaluatorq.simulation.tracing import with_simulation_span
    from evaluatorq.simulation.upload import upload_simulation_results
    from evaluatorq.tracing.setup import flush_tracing, init_tracing_if_needed

    # Initialize OTel tracing (no-op if already initialized or not configured)
    await init_tracing_if_needed()

    # Capture start_time AFTER tracing init so the duration we report to the
    # platform reflects only simulation work, not first-time OTel setup.
    start_time = datetime.now(tz=timezone.utc)

    try:
        async with with_simulation_span(
            "orq.simulation.pipeline",
            {
                "orq.simulation.evaluation_name": evaluation_name,
                "orq.simulation.max_turns": max_turns,
                "orq.simulation.parallelism": parallelism,
            },
        ) as pipeline_span:
            results = await _simulate_core(
                evaluation_name=evaluation_name,
                agent_key=agent_key,
                target_callback=target_callback,
                target=target,
                personas=personas,
                scenarios=scenarios,
                datapoints=datapoints,
                max_turns=max_turns,
                model=model,
                evaluator_names=evaluator_names,
                parallelism=parallelism,
                user_simulator=user_simulator,
                judge=judge,
                pipeline_span=pipeline_span,
            )

        # Upload runs OUTSIDE the pipeline span — matches evaluatorq core's
        # pattern (evaluatorq.py) where upload happens after the eval span
        # closes. Keeps the trace timing focused on simulation work.
        api_key = os.environ.get("ORQ_API_KEY")
        if upload_results and not api_key:
            raise ValueError(
                "ORQ_API_KEY environment variable is not set. Set it before calling simulate(upload_results=True)."
            )
        if upload_results:
            await upload_simulation_results(
                api_key=api_key,
                evaluation_name=evaluation_name or "simulation",
                evaluation_description=evaluation_description,
                results=results,
                start_time=start_time,
                end_time=datetime.now(tz=timezone.utc),
                model=model,
                path=path,
            )

        return results
    finally:
        # Flush pending spans to ensure they're exported before the process exits
        await flush_tracing()


async def _simulate_core(
    *,
    evaluation_name: str,
    agent_key: str | None,
    target_callback: Callable[[list[ChatMessage]], str | Awaitable[str]] | None,
    target: Callable[[list[ChatMessage]], str | Awaitable[str]] | None,
    personas: list[Persona] | None,
    scenarios: list[Scenario] | None,
    datapoints: list[Datapoint] | None,
    max_turns: int,
    model: str,
    evaluator_names: list[str] | None,
    parallelism: int,
    user_simulator: BaseAgent | None,
    judge: BaseAgent | None,
    pipeline_span: Span | None,
) -> list[SimulationResult]:
    """Core simulation logic (runs inside the orq.simulation.pipeline span)."""
    from openai import AsyncOpenAI

    from evaluatorq.simulation.adapters import from_orq_deployment
    from evaluatorq.simulation.evaluators import get_evaluator
    from evaluatorq.simulation.generators import FirstMessageGenerator
    from evaluatorq.simulation.runner.simulation import SimulationRunner
    from evaluatorq.simulation.tracing import record_token_usage, set_span_attrs

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

    set_span_attrs(
        pipeline_span,
        {"orq.simulation.datapoints_count": len(datapoints)},
    )

    # Bridge agentKey to invoke() if no callback is provided.
    # ``target`` takes precedence over ``target_callback`` when both are given.
    resolved_callback = target or target_callback
    if not resolved_callback and agent_key:
        resolved_callback = from_orq_deployment(agent_key)

    if not resolved_callback:
        raise ValueError("Either target_callback (or target) or agent_key is required")

    # Create simulation runner
    runner = SimulationRunner(
        target_callback=resolved_callback,
        model=model,
        max_turns=max_turns,
        user_simulator=user_simulator,
        judge=judge,
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

        # Aggregate token usage onto the pipeline span
        total_prompt = sum((r.token_usage.prompt_tokens or 0) for r in results)
        total_completion = sum(
            (r.token_usage.completion_tokens or 0) for r in results
        )
        total_total = sum((r.token_usage.total_tokens or 0) for r in results)
        record_token_usage(
            pipeline_span,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_total,
        )
        set_span_attrs(
            pipeline_span,
            {
                "orq.simulation.results_count": len(results),
                "orq.simulation.goal_achieved_count": sum(
                    1 for r in results if r.goal_achieved
                ),
            },
        )

        return results
    finally:
        await runner.close()


async def _generate_single_datapoint(
    gen: FirstMessageGenerator, persona: Persona, scenario: Scenario
) -> Datapoint:
    from evaluatorq.simulation.tracing import with_simulation_span
    from evaluatorq.simulation.utils.prompt_builders import generate_datapoint

    async with with_simulation_span(
        "orq.simulation.first_message_generation",
        {
            "orq.simulation.persona": persona.name,
            "orq.simulation.scenario": scenario.name,
            "orq.simulation.model": getattr(gen, "_model", None),
        },
    ):
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
    upload_results: bool = False,
    evaluation_description: str | None = None,
    path: str | None = None,
) -> list[SimulationResult]:
    """Generate personas/scenarios and run simulations.

    Convenience function that combines generation and simulation. Pass
    ``upload_results=True`` to upload results to the Orq platform after
    the run; defaults to ``False`` so the call performs no network
    upload unless explicitly requested.

    ``ORQ_API_KEY`` is required for this function to work at all (persona
    and scenario generation calls the Orq router). The ``upload_results``
    flag only controls the final results upload, not the generation calls.
    """
    from datetime import datetime, timezone

    from openai import AsyncOpenAI

    from evaluatorq.simulation.adapters import from_orq_deployment
    from evaluatorq.simulation.generators import PersonaGenerator, ScenarioGenerator
    from evaluatorq.simulation.tracing import with_simulation_span
    from evaluatorq.simulation.upload import upload_simulation_results
    from evaluatorq.tracing.setup import flush_tracing, init_tracing_if_needed

    # Initialize OTel tracing early so generation spans are captured
    await init_tracing_if_needed()

    # Capture after init so duration excludes first-time OTel setup.
    start_time = datetime.now(tz=timezone.utc)

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

    try:
        async with with_simulation_span(
            "orq.simulation.pipeline",
            {
                "orq.simulation.evaluation_name": evaluation_name,
                "orq.simulation.mode": "generate_and_simulate",
                "orq.simulation.num_personas": num_personas,
                "orq.simulation.num_scenarios": num_scenarios,
                "orq.simulation.max_turns": max_turns,
                "orq.simulation.parallelism": parallelism,
            },
        ) as pipeline_span:
            shared_client = AsyncOpenAI(
                api_key=api_key,
                base_url=f"{os.environ.get('ORQ_BASE_URL', 'https://api.orq.ai')}/v2/router",
            )

            try:
                persona_gen = PersonaGenerator(model=model, client=shared_client)
                scenario_gen = ScenarioGenerator(model=model, client=shared_client)

                personas, scenarios = await asyncio.gather(
                    persona_gen.generate(
                        agent_description=agent_description,
                        num_personas=num_personas,
                    ),
                    scenario_gen.generate(
                        agent_description=agent_description,
                        num_scenarios=num_scenarios,
                    ),
                )
            finally:
                await shared_client.close()

            # Delegate to core (no duplicate pipeline span)
            results = await _simulate_core(
                evaluation_name=evaluation_name,
                agent_key=None,
                target_callback=resolved_callback,
                target=None,
                personas=personas,
                scenarios=scenarios,
                datapoints=None,
                max_turns=max_turns,
                model=model,
                evaluator_names=evaluator_names,
                parallelism=parallelism,
                user_simulator=None,
                judge=None,
                pipeline_span=pipeline_span,
            )

        # Upload runs outside the pipeline span (see simulate() for rationale).
        # api_key is guaranteed non-empty here (validated above), so the
        # symmetric `and api_key` guard is omitted.
        if upload_results:
            await upload_simulation_results(
                api_key=api_key,
                evaluation_name=evaluation_name or "simulation",
                evaluation_description=evaluation_description,
                results=results,
                start_time=start_time,
                end_time=datetime.now(tz=timezone.utc),
                model=model,
                path=path,
            )

        return results
    finally:
        await flush_tracing()
