"""Wraps the simulation framework as an evaluatorq Job.

Follows the same pattern as wrap_langchain_agent().
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from evaluatorq.simulation.adapters import from_orq_deployment
from evaluatorq.simulation.convert import to_open_responses
from evaluatorq.simulation.types import (
    DEFAULT_MODEL,
    ChatMessage,
    Datapoint,
    Persona,
    Scenario,
    SimulationResult,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from evaluatorq.types import DataPoint

logger = logging.getLogger(__name__)


def _validate_shape(value: Any, label: str, required_keys: list[str]) -> None:
    """Lightweight runtime check that an object has the expected keys."""
    if not isinstance(value, dict):
        raise ValueError(  # noqa: TRY004
            f"Expected '{label}' to be an object, got {type(value).__name__}"
        )
    for key in required_keys:
        if key not in value:
            raise ValueError(f"Invalid '{label}': missing required field '{key}'")


def wrap_simulation_agent(
    *,
    name: str = "simulation",
    target_callback: Callable[[list[ChatMessage]], str | Awaitable[str]] | None = None,
    agent_key: str | None = None,
    max_turns: int = 10,
    model: str | None = None,
    evaluators: list[str] | None = None,
) -> Callable[[DataPoint, int], Awaitable[dict[str, Any]]]:
    """Create an evaluatorq Job that runs agent simulations.

    Each DataPoint should have inputs containing simulation data:
    - ``persona`` and ``scenario``, or
    - ``datapoint`` (full Datapoint object), or
    - ``personas`` and ``scenarios`` for batch generation
    """

    async def job_fn(data: DataPoint, _row: int) -> dict[str, Any]:
        # Lazy import to avoid circular imports
        from evaluatorq.simulation import simulate

        # Resolve the target callback
        resolved_callback = target_callback
        if not resolved_callback and agent_key:
            resolved_callback = from_orq_deployment(agent_key)

        if not resolved_callback:
            raise ValueError(
                "wrap_simulation_agent requires either target_callback or agent_key"
            )

        # Extract simulation inputs from DataPoint
        inputs = data.inputs

        datapoints: list[Datapoint] | None = None
        personas: list[Persona] | None = None
        scenarios: list[Scenario] | None = None

        if "datapoint" in inputs:
            dp = inputs["datapoint"]
            _validate_shape(dp, "datapoint", ["persona", "scenario", "first_message"])
            datapoints = [Datapoint.model_validate(dp)]
        elif "datapoints" in inputs:
            dps = inputs["datapoints"]
            if not isinstance(dps, list):
                raise ValueError("Expected 'datapoints' to be an array")
            if len(dps) != 1:
                raise ValueError(
                    "wrap_simulation_agent DataPoint must encode exactly one datapoint. "
                    "For batch simulations use simulate() directly."
                )
            for dp in dps:
                _validate_shape(
                    dp, "datapoints[]", ["persona", "scenario", "first_message"]
                )
            datapoints = [Datapoint.model_validate(dp) for dp in dps]
        elif "persona" in inputs and "scenario" in inputs:
            _validate_shape(inputs["persona"], "persona", ["name"])
            _validate_shape(inputs["scenario"], "scenario", ["name", "goal"])
            personas = [Persona.model_validate(inputs["persona"])]
            scenarios = [Scenario.model_validate(inputs["scenario"])]
        elif "personas" in inputs and "scenarios" in inputs:
            if not isinstance(inputs["personas"], list) or not isinstance(
                inputs["scenarios"], list
            ):
                raise ValueError("Expected 'personas' and 'scenarios' to be arrays")
            if len(inputs["personas"]) != 1 or len(inputs["scenarios"]) != 1:
                raise ValueError(
                    "wrap_simulation_agent DataPoint must encode exactly one persona-scenario pair. "
                    "For batch simulations use simulate() directly."
                )
            for p in inputs["personas"]:
                _validate_shape(p, "personas[]", ["name"])
            for s in inputs["scenarios"]:
                _validate_shape(s, "scenarios[]", ["name", "goal"])
            personas = [Persona.model_validate(p) for p in inputs["personas"]]
            scenarios = [Scenario.model_validate(s) for s in inputs["scenarios"]]
        else:
            raise ValueError(
                "Expected data.inputs to contain 'persona' + 'scenario', 'datapoint', "
                "'datapoints', or 'personas' + 'scenarios'"
            )

        # Run simulation
        effective_model = model or DEFAULT_MODEL
        results: list[SimulationResult] = await simulate(
            evaluation_name=name,
            target_callback=resolved_callback,
            datapoints=datapoints,
            personas=personas,
            scenarios=scenarios,
            max_turns=max_turns,
            model=effective_model,
            evaluator_names=evaluators,
        )

        if not results:
            raise RuntimeError("Simulation produced no results")

        if len(results) > 1:
            logger.warning(
                "wrap_simulation_agent ran %s simulations but only returns the first result. "
                "Use simulate() directly to collect all outputs.",
                len(results),
            )

        # Convert the first result to OpenResponses format
        result = results[0]

        return {
            "name": name,
            "output": to_open_responses(result, effective_model),
        }

    return job_fn
