"""Every data-shaped SDK keyword argument is reachable from the CLI's ``--config`` file.

``eq redteam run``, ``eq sim simulate`` and ``eq sim run`` validate ``--config`` against a model that mirrors
the SDK signature. A keyword added to the SDK without a field there is unreachable from the CLI, so this
test fails until the field is added or the keyword is named below with the reason it cannot be JSON.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

import click
import pytest
import typer.main

from evaluatorq.common.cli_config import Flag
from evaluatorq.redteam import cli as redteam_cli
from evaluatorq.redteam.run_config import RedTeamCliConfig, RedTeamRunConfig
from evaluatorq.redteam.runner import red_team
from evaluatorq.simulation import cli as sim_cli
from evaluatorq.simulation.api import generate_and_simulate, simulate
from evaluatorq.simulation.run_config import (
    GenerateAndSimulateCliConfig,
    GenerateAndSimulateRunConfig,
    SimulateCliConfig,
    SimulateRunConfig,
)

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
IDS = [getattr(fn, '__name__', repr(fn)) for fn, _, _ in CASES]


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


_CLI_DIFFERS_FROM_SDK_ONLY_IN: list[tuple[type[BaseModel], type[BaseModel], set[str]]] = [
    (RedTeamCliConfig, RedTeamRunConfig, {'verbosity'}),
    (SimulateCliConfig, SimulateRunConfig, {'save'}),
    (GenerateAndSimulateCliConfig, GenerateAndSimulateRunConfig, {'save'}),
]


@pytest.mark.parametrize(
    ('cli_model', 'sdk_model', 'overridden'),
    _CLI_DIFFERS_FROM_SDK_ONLY_IN,
    ids=[cli.__name__ for cli, _, _ in _CLI_DIFFERS_FROM_SDK_ONLY_IN],
)
def test_cli_model_differs_from_sdk_model_only_where_named(
    cli_model: type[BaseModel], sdk_model: type[BaseModel], overridden: set[str]
) -> None:
    assert set(cli_model.model_fields) == set(sdk_model.model_fields)

    differing = {
        name
        for name, field in cli_model.model_fields.items()
        if field.get_default(call_default_factory=True)
        != sdk_model.model_fields[name].get_default(call_default_factory=True)
    }
    assert differing == overridden


# Config-backed flags that must keep their own default: presentation counters, not SDK values.
_FLAG_DEFAULT_EXEMPT: dict[str, str] = {
    'verbose': 'a -v counter; its default 0 means "not passed" and the CLI default verbosity lives on the model',
    'quiet': 'a -q switch; its default False means "not passed"',
}

_FLAG_TABLES: list[tuple[typer.Typer, str, tuple[Flag, ...]]] = [
    (redteam_cli.app, 'run', redteam_cli._RUN_FLAGS),
    (sim_cli.app, 'simulate', sim_cli._SIMULATE_FLAGS),
    (sim_cli.app, 'run', sim_cli._RUN_FLAGS),
]


@pytest.mark.parametrize(
    ('app', 'command', 'flags'),
    _FLAG_TABLES,
    ids=['redteam run', 'sim simulate', 'sim run'],
)
def test_config_backed_flags_declare_no_default(app: typer.Typer, command: str, flags: tuple[Flag, ...]) -> None:
    group = typer.main.get_command(app)
    assert isinstance(group, click.Group)
    params = {param.name: param for param in group.commands[command].params}

    with_defaults = {
        flag.param: params[flag.param].default
        for flag in flags
        if flag.path is not None and flag.param not in _FLAG_DEFAULT_EXEMPT and params[flag.param].default is not None
    }
    assert with_defaults == {}, 'a config-backed flag must default to None; the model owns the default'


def test_every_default_exemption_names_a_flag_that_exists() -> None:
    params = {flag.param for _, _, flags in _FLAG_TABLES for flag in flags}
    assert set(_FLAG_DEFAULT_EXEMPT) <= params, 'a renamed flag leaves a stale exemption'
