"""The data-shaped keyword arguments of `simulate` and `generate_and_simulate`, as models a JSON file can carry.

``eq sim simulate --config`` validates against `SimulateRunConfig` and ``eq sim run --config`` against
`GenerateAndSimulateRunConfig`; ``eq sim schema`` prints either. Fields use the public keyword names.
Python objects (a callable or ``AgentTarget`` target, ``user_simulator``, ``judge``, ``hooks``,
``generation_client``, ``emit_datapoints``), the deprecated ``parallelism`` alias and the older spelling of
each renamed keyword have no field here; ``tests/test_cli_config_parity.py`` keeps the models in step.

The internal ``SimulationConfig`` is not reused: it uses internal names (``run_output``, ``evaluation_name``)
and carries the Python objects this file has to exclude.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — pydantic resolves field annotations at runtime
from typing import Any

from pydantic import BaseModel, ConfigDict

from evaluatorq.contracts import LLMCallConfig  # noqa: TC001 — pydantic resolves field annotations at runtime
from evaluatorq.simulation._config import (
    DEFAULT_MAX_TARGET_RETRIES,
    DEFAULT_MAX_TOOL_RESULT_CHARS,
    DEFAULT_TARGET_AGENT_TIMEOUT_MS,
)
from evaluatorq.simulation.evaluators.scorers import SimulationScoringConfig  # noqa: TC001
from evaluatorq.simulation.reports.recommendations import SimulationRecommendationConfig  # noqa: TC001
from evaluatorq.simulation.types import Persona, Scenario, SimulationDatapoint  # noqa: TC001

# Public keyword -> the name the internal run functions take it under, for the fields the CLI forwards
# untouched. ``run_name`` and ``report_path`` are absent because the CLI owns them (``--name``, ``--report``).
_INTERNAL_NAMES = {
    'experiment_description': 'evaluation_description',
    'orq_folder_path': 'orq_results_path',
    'raise_on_execution_failure': 'exit_on_failure',
}


class _SharedRunConfig(BaseModel):
    """Keyword arguments `simulate` and `generate_and_simulate` both take. Defaults are the SDK's own."""

    model_config = ConfigDict(extra='forbid')

    run_name: str | None = None
    target: str | None = None
    memory_entity_id: str | None = None
    max_turns: int | None = None
    llm_config: LLMCallConfig | None = None
    evaluator_names: list[str] | None = None
    scoring: SimulationScoringConfig | None = None
    datapoint_parallelism: int | None = None
    llm_parallelism: int | None = None
    target_agent_timeout_ms: int = DEFAULT_TARGET_AGENT_TIMEOUT_MS
    max_target_retries: int = DEFAULT_MAX_TARGET_RETRIES
    target_reasoning_effort: str | None = None
    max_tool_result_chars: int = DEFAULT_MAX_TOOL_RESULT_CHARS
    per_simulation_timeout_s: float | None = None
    upload_results: bool = True
    experiment_description: str | None = None
    orq_folder_path: str | None = None
    raise_on_execution_failure: bool | None = None
    save: bool = False
    report_path: Path | None = None
    executive_summary: bool = True
    recommendations: bool | SimulationRecommendationConfig = False


class SimulateRunConfig(_SharedRunConfig):
    """Keyword arguments of `simulate` that JSON can express. ``target`` takes the string forms only."""

    personas: list[Persona] | None = None
    scenarios: list[Scenario] | None = None
    datapoints: list[SimulationDatapoint] | None = None
    dataset_id: str | None = None
    experiment_id: str | None = None
    experiment_run_id: str | None = None
    previous_run: str | None = None


class GenerateAndSimulateRunConfig(_SharedRunConfig):
    """Keyword arguments of `generate_and_simulate` that JSON can express. ``target`` takes the string forms only."""

    agent_description: str | None = None
    num_personas: int = 5
    num_scenarios: int = 5
    edge_case_percentage: float | None = None
    persona_seeds: list[str] | None = None
    scenario_seeds: list[str] | None = None
    generation_instructions: str = ''


def to_internal_kwargs(public: dict[str, Any]) -> dict[str, Any]:
    """Rename public keyword names to the internal run functions' names.

    ``raise_on_execution_failure=None`` is dropped: the public functions read it as "default", the internal
    ones take a plain ``bool`` and default it themselves.
    """
    renamed = {_INTERNAL_NAMES.get(name, name): value for name, value in public.items()}
    if renamed.get('exit_on_failure', False) is None:
        del renamed['exit_on_failure']
    return renamed
