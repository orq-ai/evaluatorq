"""Every data-shaped SDK keyword argument is reachable from the CLI's ``--config`` file.

``eq redteam run``, ``eq sim simulate`` and ``eq sim run`` validate ``--config`` against a model that mirrors
the SDK signature. A keyword added to the SDK without a field there is unreachable from the CLI, so this
test fails until the field is added or the keyword is named below with the reason it cannot be JSON.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import pytest

from evaluatorq.redteam.run_config import RedTeamRunConfig
from evaluatorq.redteam.runner import red_team
from evaluatorq.simulation.api import generate_and_simulate, simulate
from evaluatorq.simulation.run_config import GenerateAndSimulateRunConfig, SimulateRunConfig

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import BaseModel

# Keyword -> why a JSON document cannot carry it. Import-path references ("module:attr") are deliberately
# not offered as a workaround: loading code named by a config file is out of scope for security reasons.
_SIMULATION_NOT_JSON: dict[str, str] = {
    'user_simulator': 'a BaseAgent instance',
    'judge': 'a BaseAgent instance',
    'hooks': 'SimulationHooks objects',
    'generation_client': 'an AsyncOpenAI client',
    'parallelism': 'deprecated alias of datapoint_parallelism',
    'evaluation_name': 'older spelling of run_name',
    'evaluation_description': 'older spelling of experiment_description',
    'orq_results_path': 'older spelling of orq_folder_path',
    'exit_on_failure': 'older spelling of raise_on_execution_failure',
    'report': 'older spelling of report_path',
}

CASES: list[tuple[Callable[..., Any], type[BaseModel], dict[str, str]]] = [
    (
        red_team,
        RedTeamRunConfig,
        {
            'llm_client': 'an AsyncOpenAI client',
            'hooks': 'a PipelineHooks object',
            'parallelism': 'deprecated alias of datapoint_parallelism',
        },
    ),
    (simulate, SimulateRunConfig, _SIMULATION_NOT_JSON),
    (generate_and_simulate, GenerateAndSimulateRunConfig, {**_SIMULATION_NOT_JSON, 'emit_datapoints': 'a callable'}),
]
IDS = [fn.__name__ for fn, _, _ in CASES]


@pytest.mark.parametrize(('fn', 'model', 'not_json'), CASES, ids=IDS)
def test_every_sdk_keyword_is_a_config_field_or_named_as_not_json(
    fn: Callable[..., Any], model: type[BaseModel], not_json: dict[str, str]
) -> None:
    params = set(inspect.signature(fn).parameters)
    fields = set(model.model_fields)

    assert params - fields - set(not_json) == set(), 'add a config field, or name the keyword in the allowlist'
    assert fields - params == set(), 'a config field the SDK function does not take'
    assert set(not_json) - params == set(), 'a stale allowlist entry'
    assert fields & set(not_json) == set(), 'a keyword is both a field and allowlisted'


@pytest.mark.parametrize(('fn', 'model', 'not_json'), CASES, ids=IDS)
def test_config_defaults_are_the_sdk_defaults(
    fn: Callable[..., Any], model: type[BaseModel], not_json: dict[str, str]
) -> None:
    del not_json
    for name, param in inspect.signature(fn).parameters.items():
        if name not in model.model_fields or param.default is inspect.Parameter.empty:
            continue
        assert model.model_fields[name].get_default(call_default_factory=True) == param.default, name
