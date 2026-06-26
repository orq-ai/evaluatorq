"""Shared helpers for converting between evaluatorq ``DataPoint`` and the
simulation-side ``SimulationDatapoint``.

Both ``simulation/api.py`` (the ``simulate()`` path) and
``simulation/wrap_agent.py`` (the public ``wrap_simulation_agent()`` job)
need to normalise an evaluatorq ``DataPoint`` into a single simulation
``SimulationDatapoint``, so the logic lives here to avoid the cross-module import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from evaluatorq.simulation.types import Persona, Scenario, SimulationDatapoint

if TYPE_CHECKING:
    from evaluatorq.types import DataPoint


def _validate_shape(value: Any, label: str, required_keys: list[str]) -> None:
    """Lightweight runtime check that an object has the expected keys."""
    if not isinstance(value, dict):
        raise ValueError(  # noqa: TRY004
            f"Expected '{label}' to be an object, got {type(value).__name__}"
        )
    for key in required_keys:
        if key not in value:
            raise ValueError(f"Invalid '{label}': missing required field '{key}'")


def _extract_single_datapoint(data: DataPoint) -> SimulationDatapoint:
    """Extract exactly one simulation SimulationDatapoint from an evaluatorq DataPoint.

    Accepts the four input shapes (``datapoint``, ``datapoints`` of length one,
    ``persona`` + ``scenario``, ``personas`` + ``scenarios`` each of length
    one) and normalises to a single ``SimulationDatapoint``.
    """
    inputs = data.inputs
    if "datapoint" in inputs:
        dp = inputs["datapoint"]
        _validate_shape(dp, "datapoint", ["persona", "scenario", "first_message"])
        return SimulationDatapoint.model_validate(dp)
    if "datapoints" in inputs:
        dps = inputs["datapoints"]
        if not isinstance(dps, list):
            raise ValueError("Expected 'datapoints' to be an array")
        if len(dps) != 1:
            raise ValueError(
                "wrap_simulation_agent DataPoint must encode exactly one datapoint. "
                "For batch simulations use simulate() directly."
            )
        _validate_shape(
            dps[0], "datapoints[]", ["persona", "scenario", "first_message"]
        )
        return SimulationDatapoint.model_validate(dps[0])
    if "persona" in inputs and "scenario" in inputs:
        _validate_shape(inputs["persona"], "persona", ["name"])
        _validate_shape(inputs["scenario"], "scenario", ["name", "goal"])
        persona = Persona.model_validate(inputs["persona"])
        scenario = Scenario.model_validate(inputs["scenario"])
        return SimulationDatapoint(
            id=f"{persona.name}-{scenario.name}",
            persona=persona,
            scenario=scenario,
            user_system_prompt="",
            first_message=inputs.get("first_message", ""),
        )
    if "personas" in inputs and "scenarios" in inputs:
        if not isinstance(inputs["personas"], list) or not isinstance(
            inputs["scenarios"], list
        ):
            raise ValueError("Expected 'personas' and 'scenarios' to be arrays")
        if len(inputs["personas"]) != 1 or len(inputs["scenarios"]) != 1:
            raise ValueError(
                "wrap_simulation_agent DataPoint must encode exactly one "
                "persona-scenario pair. For batch simulations use simulate() directly."
            )
        _validate_shape(inputs["personas"][0], "personas[]", ["name"])
        _validate_shape(inputs["scenarios"][0], "scenarios[]", ["name", "goal"])
        persona = Persona.model_validate(inputs["personas"][0])
        scenario = Scenario.model_validate(inputs["scenarios"][0])
        return SimulationDatapoint(
            id=f"{persona.name}-{scenario.name}",
            persona=persona,
            scenario=scenario,
            user_system_prompt="",
            first_message=inputs.get("first_message", ""),
        )
    raise ValueError(
        "Expected data.inputs to contain 'persona' + 'scenario', 'datapoint', "
        "'datapoints', or 'personas' + 'scenarios'"
    )
