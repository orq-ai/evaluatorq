"""The data-shaped keyword arguments of `simulate` and `generate_and_simulate`, as models a JSON file can carry.

``eq sim simulate --config`` validates against `SimulateRunConfig` and ``eq sim run --config`` against
`GenerateAndSimulateRunConfig`; ``eq sim schema`` prints either. Fields use the public keyword names.
Python objects (a callable or ``AgentTarget`` target, ``user_simulator``, ``judge``, ``hooks``,
``generation_client``, ``emit_datapoints``), the deprecated ``parallelism`` alias and the older spelling of
each renamed keyword have no field here; ``tests/test_cli_config_parity.py`` keeps the models in step.

The internal ``SimulationConfig`` is not reused: it carries the Python objects this file has to exclude.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — pydantic resolves field annotations at runtime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from evaluatorq.common.llm_limit import check_llm_parallelism
from evaluatorq.contracts import LLMCallConfig  # noqa: TC001 — pydantic resolves field annotations at runtime
from evaluatorq.simulation._config import (
    DEFAULT_MAX_TARGET_RETRIES,
    DEFAULT_MAX_TOOL_RESULT_CHARS,
    DEFAULT_TARGET_AGENT_TIMEOUT_MS,
)
from evaluatorq.simulation.evaluators.scorers import SimulationScoringConfig  # noqa: TC001
from evaluatorq.simulation.reports.recommendations import SimulationRecommendationConfig  # noqa: TC001
from evaluatorq.simulation.types import Persona, Scenario, SimulationDatapoint  # noqa: TC001


class _SharedRunConfig(BaseModel):
    """Keyword arguments `simulate` and `generate_and_simulate` both take. Defaults are the SDK's own."""

    model_config = ConfigDict(extra='forbid')

    run_name: str | None = None
    target: str | None = None
    memory_entity_id: str | None = None
    max_turns: int | None = Field(default=None, ge=1)
    llm_config: LLMCallConfig | None = None
    evaluator_names: list[str] | None = None
    scoring: SimulationScoringConfig | None = None
    datapoint_parallelism: int | None = Field(default=None, ge=1)
    llm_parallelism: Annotated[int | None, AfterValidator(check_llm_parallelism)] = None
    target_agent_timeout_ms: int = Field(default=DEFAULT_TARGET_AGENT_TIMEOUT_MS, gt=0)
    max_target_retries: int = Field(default=DEFAULT_MAX_TARGET_RETRIES, ge=0)
    target_reasoning_effort: str | None = None
    max_tool_result_chars: int = Field(default=DEFAULT_MAX_TOOL_RESULT_CHARS, gt=0)
    per_simulation_timeout_s: float | None = Field(default=None, gt=0)
    upload_results: bool = True
    experiment_description: str | None = None
    orq_folder_path: str | None = None
    raise_on_execution_failure: bool | None = None
    save: bool = False
    report_path: Path | None = None
    executive_summary: bool = True
    recommendations: bool | SimulationRecommendationConfig = True


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
    num_personas: int = Field(default=5, ge=1)
    num_scenarios: int = Field(default=5, ge=1)
    edge_case_percentage: float | None = None
    persona_seeds: list[str] | None = None
    scenario_seeds: list[str] | None = None
    generation_instructions: str = ''


class SimulateCliConfig(SimulateRunConfig):
    """What ``eq sim simulate --config`` accepts, with the CLI's defaults: it saves the run unless told not to."""

    save: bool = True


class GenerateAndSimulateCliConfig(GenerateAndSimulateRunConfig):
    """What ``eq sim run --config`` accepts, with the CLI's defaults: it saves the run unless told not to."""

    save: bool = True
