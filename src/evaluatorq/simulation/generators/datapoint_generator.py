"""Datapoint generator for creating test datasets.

Combines personas and scenarios into complete datapoints with first messages.
"""

from __future__ import annotations

import asyncio
import logging
import os
from itertools import starmap
from typing import Any

from openai import AsyncOpenAI

from evaluatorq.simulation.generators.first_message_generator import (
    FirstMessageGenerator,
)
from evaluatorq.simulation.generators.persona_generator import PersonaGenerator
from evaluatorq.simulation.generators.scenario_generator import ScenarioGenerator
from evaluatorq.simulation.quality.message_perturbation import apply_random_perturbation
from evaluatorq.simulation.types import (
    DEFAULT_MODEL,
    Datapoint,
    Persona,
    Scenario,
)
from evaluatorq.simulation.utils.prompt_builders import generate_datapoint

logger = logging.getLogger(__name__)

_DEFAULT_RATE_LIMIT_DELAY = 0.1  # 100ms
_DEFAULT_MAX_CONCURRENT_CALLS = 5


class DatapointGenerator:
    """Generates complete datapoints for simulation.

    Orchestrates persona, scenario, and first message generation
    to produce ready-to-use test datapoints.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        rate_limit_delay: float = _DEFAULT_RATE_LIMIT_DELAY,
        max_concurrent_calls: int = _DEFAULT_MAX_CONCURRENT_CALLS,
    ) -> None:
        self._model = model
        self._rate_limit_delay = rate_limit_delay
        self._semaphore = asyncio.Semaphore(max_concurrent_calls)

        resolved_key = os.environ.get("ORQ_API_KEY")
        if not resolved_key:
            raise ValueError(
                "ORQ_API_KEY environment variable is not set. "
                "Set it before creating a DatapointGenerator."
            )
        self._shared_client = AsyncOpenAI(
            base_url=f"{os.environ.get('ORQ_BASE_URL', 'https://api.orq.ai')}/v2/router",
            api_key=resolved_key,
        )
        self._persona_generator = PersonaGenerator(model=model, client=self._shared_client)
        self._scenario_generator = ScenarioGenerator(model=model, client=self._shared_client)
        self._first_message_generator = FirstMessageGenerator(model=model, client=self._shared_client)

    async def close(self) -> None:
        """Close the shared HTTP client."""
        await self._shared_client.close()

    async def generate_from_description(
        self,
        *,
        agent_description: str,
        context: str = "",
        num_personas: int = 3,
        num_scenarios: int = 5,
        edge_case_percentage: float = 0.2,
        perturbation_rate: float = 0.0,
        include_boundary: bool = False,
        num_boundary: int = 5,
        include_security: bool = False,
        num_security: int = 5,
        security_seed_examples: list[dict[str, Any]] | None = None,
        security_categories: list[str] | None = None,
    ) -> list[Datapoint]:
        """Generate datapoints from agent description.

        Creates personas and scenarios, then combines them into datapoints.
        Total datapoints = numPersonas x (numScenarios + boundary + security)
        """
        logger.info(
            "Generating %d personas and %d scenarios...", num_personas, num_scenarios
        )

        # Build tasks for parallel generation
        tasks: dict[str, Any] = {
            "personas": self._persona_generator.generate(
                agent_description=agent_description,
                context=context,
                num_personas=num_personas,
                edge_case_percentage=edge_case_percentage,
            ),
            "scenarios": self._scenario_generator.generate(
                agent_description=agent_description,
                context=context,
                num_scenarios=num_scenarios,
                edge_case_percentage=edge_case_percentage,
            ),
        }

        if include_boundary:
            logger.info("Including %d boundary scenarios", num_boundary)
            tasks["boundary"] = self._scenario_generator.generate_boundary_scenarios(
                agent_description=agent_description,
                num_scenarios=num_boundary,
            )

        if include_security:
            logger.info("Including %d security scenarios", num_security)
            tasks["security"] = self._scenario_generator.generate_security_scenarios(
                agent_description=agent_description,
                seed_examples=security_seed_examples,
                categories=security_categories,
                num_scenarios=num_security,
            )

        task_keys = list(tasks.keys())
        raw_results = await asyncio.gather(*tasks.values())
        results = dict(zip(task_keys, raw_results, strict=False))

        personas: list[Persona] = results["personas"]
        scenarios: list[Scenario] = results["scenarios"]

        # Merge additional scenario types
        if "boundary" in results:
            boundary: list[Scenario] = results["boundary"]
            logger.info("Generated %d boundary scenarios", len(boundary))
            scenarios = [*scenarios, *boundary]
        if "security" in results:
            security: list[Scenario] = results["security"]
            logger.info("Generated %d security scenarios", len(security))
            scenarios = [*scenarios, *security]

        if not personas:
            raise RuntimeError(
                "Persona generation failed: no personas were produced. "
                "Check LLM connectivity and response parsing."
            )

        if not scenarios:
            raise RuntimeError(
                "Scenario generation failed: no scenarios were produced. "
                "Check LLM connectivity and response parsing."
            )

        # Generate datapoints from all combinations
        datapoints = await self.generate_from_combinations(personas, scenarios)

        # Apply message perturbations for robustness testing
        if perturbation_rate > 0.0:
            datapoints = self._apply_perturbations(datapoints, perturbation_rate)

        return datapoints

    async def generate_from_combinations(
        self,
        personas: list[Persona],
        scenarios: list[Scenario],
    ) -> list[Datapoint]:
        """Generate datapoints from persona-scenario combinations."""
        combinations = [(p, s) for p in personas for s in scenarios]

        logger.info(
            "Generating %d datapoints from %d personas x %d scenarios",
            len(combinations),
            len(personas),
            len(scenarios),
        )

        async def generate_single(persona: Persona, scenario: Scenario) -> Datapoint:
            async with self._semaphore:
                first_message = await self._first_message_generator.generate(
                    persona, scenario
                )
                await asyncio.sleep(self._rate_limit_delay)
                return generate_datapoint(persona, scenario, first_message)

        tasks = list(starmap(generate_single, combinations))
        datapoints = await asyncio.gather(*tasks)

        logger.info("Generated %d datapoints", len(datapoints))
        return list(datapoints)

    @staticmethod
    def _apply_perturbations(
        datapoints: list[Datapoint],
        perturbation_rate: float,
    ) -> list[Datapoint]:
        """Apply random input perturbations to first messages for robustness testing."""
        import random

        perturbed_count = 0
        result = []
        for dp in datapoints:
            if dp.first_message and random.random() < perturbation_rate:
                perturbed_msg, p_type = apply_random_perturbation(dp.first_message)
                perturbed_count += 1
                logger.debug("Applied %s perturbation to: %s", p_type, dp.scenario.name)
                result.append(dp.model_copy(update={"first_message": perturbed_msg}))
            else:
                result.append(dp)

        if perturbed_count > 0:
            logger.info(
                "Applied perturbations to %d/%d first messages",
                perturbed_count,
                len(datapoints),
            )

        return result
