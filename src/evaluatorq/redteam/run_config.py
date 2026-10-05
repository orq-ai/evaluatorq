"""The data-shaped keyword arguments of `red_team`, as one model a JSON file can carry.

``eq redteam run --config`` validates its file against `RedTeamRunConfig` and ``eq redteam schema`` prints
its JSON schema. Python objects (an ``AgentTarget``, ``hooks``, ``llm_client``) and the deprecated
``parallelism`` alias have no field here; ``tests/test_cli_config_parity.py`` keeps the two in step.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 — pydantic resolves field annotations at runtime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict

from evaluatorq.common.llm_limit import check_llm_parallelism
from evaluatorq.redteam.contracts import (
    AttackTechnique,
    DeliveryMethod,
    LLMConfig,
    Pipeline,
    RedTeamRecommendationConfig,
    SaveMode,
    TargetConfig,
)
from evaluatorq.types import DataPoint  # noqa: TC001 — pydantic resolves field annotations at runtime


class RedTeamRunConfig(BaseModel):
    """Keyword arguments of `red_team` that JSON can express. Defaults are the SDK's own.

    ``target`` takes the string forms only (``"agent:<key>"``, ``"deployment:<key>"``) or a list of them.
    Unknown keys are rejected, so a typo fails instead of silently doing nothing.
    """

    model_config = ConfigDict(extra='forbid')

    target: str | list[str] | None = None
    llm_config: LLMConfig | None = None
    mode: Pipeline | None = None
    categories: list[str] | None = None
    vulnerabilities: list[str] | None = None
    strategies: list[str] | None = None
    datapoints: list[DataPoint] | None = None
    attack_techniques: list[AttackTechnique | str] | None = None
    delivery_methods: list[DeliveryMethod | str] | None = None
    max_turns: int | None = None
    max_per_category: int | None = None
    datapoint_parallelism: int | None = None
    llm_parallelism: Annotated[int | None, AfterValidator(check_llm_parallelism)] = None
    generate_strategies: bool = True
    generated_strategy_count: int = 2
    max_dynamic_datapoints: int | None = None
    max_static_datapoints: int | None = None
    cleanup_memory: bool = True
    name: str | None = None
    description: str | None = None
    dataset: str | None = None
    previous_run: str | None = None
    artifacts_dir: Path | None = None
    target_config: TargetConfig | None = None
    recommendations: bool | RedTeamRecommendationConfig = True
    generate_executive_summary: bool = True
    attacker_instructions: str | None = None
    verbosity: int = 0
    save: SaveMode = SaveMode.FINAL


class RedTeamCliConfig(RedTeamRunConfig):
    """What ``eq redteam run --config`` accepts, with the CLI's defaults.

    One differs from `red_team`: ``verbosity`` is 1, a summary progress bar, because a terminal user is
    watching. ``-v`` raises it, ``-q`` sets it to 0.
    """

    verbosity: int = 1
