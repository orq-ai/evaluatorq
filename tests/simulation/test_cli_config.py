"""`eq sim simulate|run --config / --llm-config / --json` and `eq sim schema` (RES-1685)."""
# ruff: noqa: S101

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from evaluatorq.contracts import LLMCallConfig
from evaluatorq.simulation.cli import app
from evaluatorq.simulation.run_config import GenerateAndSimulateRunConfig, SimulateRunConfig
from evaluatorq.simulation.types import SimulationRun

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()

PERSONA = {
    'name': 'Impatient',
    'patience': 0.2,
    'assertiveness': 0.8,
    'politeness': 0.4,
    'technical_level': 0.3,
    'communication_style': 'terse',
    'background': 'Wants a refund',
}
SCENARIO = {'name': 'Refund', 'goal': 'Get a refund'}
# --no-executive-summary: the post-run summary is an LLM call the CLI makes outside the faked run function.
QUIET = ['--no-save', '--no-executive-summary', '--yes']


def _run(mode: str) -> SimulationRun:
    return SimulationRun(
        run_name='cfg',
        created_at=datetime.now(tz=timezone.utc),
        mode=mode,  # pyright: ignore[reportArgumentType]
        target_kind='orq_agent',
        evaluator_names=[],
        total_results=0,
        scorer_averages={},
        results=[],
    )


def _invoke(command: str, args: list[str], *, stdin: str | None = None) -> tuple[Any, AsyncMock]:
    internal = '_simulate_run' if command == 'simulate' else '_generate_and_simulate_run'
    fake = AsyncMock(return_value=_run('simulate' if command == 'simulate' else 'run'))
    with (
        patch(f'evaluatorq.simulation.api.{internal}', new=fake),
        patch('evaluatorq.simulation.cli._resolve_target', return_value=MagicMock()),
    ):
        result = runner.invoke(app, [command, *args, *QUIET], input=stdin)
    return result, fake


def _flat(output: str) -> str:
    """Unwrap the Rich error box, which breaks long messages across lines."""
    return ' '.join(output.replace('│', ' ').split())


def _write(tmp_path: Path, payload: dict[str, Any]) -> str:
    path = tmp_path / 'sim.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    return str(path)


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('EVALUATORQ_DIR', str(tmp_path / '.evaluatorq'))


def test_simulate_takes_inline_personas_and_scenarios_as_its_input_source(tmp_path: Path) -> None:
    config = _write(
        tmp_path,
        {
            'target': 'agent:x',
            'personas': [PERSONA],
            'scenarios': [SCENARIO],
            'max_turns': 4,
            'experiment_description': 'nightly',
            'raise_on_execution_failure': False,
            'per_simulation_timeout_s': 30,
            'scoring': {},
        },
    )

    result, fake = _invoke('simulate', ['--config', config])

    assert result.exit_code == 0, result.output
    kwargs = fake.call_args.kwargs
    assert kwargs['personas'][0].name == 'Impatient'
    assert kwargs['scenarios'][0].goal == 'Get a refund'
    assert kwargs['datapoints'] is None
    assert kwargs['max_turns'] == 4
    # Public keyword names in the file, internal names at the run function.
    assert kwargs['evaluation_description'] == 'nightly'
    assert kwargs['exit_on_failure'] is False
    assert kwargs['per_simulation_timeout_s'] == 30


def test_simulate_rejects_an_inline_source_beside_another_one(tmp_path: Path) -> None:
    config = _write(tmp_path, {'target': 'agent:x', 'personas': [PERSONA], 'scenarios': [SCENARIO]})

    result, fake = _invoke('simulate', ['--config', config, '--from-run', 'latest'])

    assert result.exit_code == 2
    flat = _flat(result.output)
    assert 'Provide exactly one of --input' in flat
    assert 'inline datapoints/personas/scenarios in --config' in flat
    fake.assert_not_called()


def test_simulate_reads_its_config_from_stdin_and_a_flag_wins() -> None:
    payload = {'target': 'agent:x', 'previous_run': 'latest', 'max_turns': 9, 'run_name': 'from-config'}

    result, fake = _invoke('simulate', ['--config', '-', '--max-turns', '3'], stdin=json.dumps(payload))

    assert result.exit_code == 0, result.output
    kwargs = fake.call_args.kwargs
    assert kwargs['previous_run'] == 'latest'
    assert kwargs['max_turns'] == 3
    assert kwargs['evaluation_name'] == 'from-config'


def test_llm_config_blob_with_sim_model_winning_the_model() -> None:
    blob = json.dumps({'model': 'blob/model', 'temperature': 0.2})

    result, fake = _invoke(
        'simulate', ['--from-run', 'latest', '--target', 'agent:x', '--llm-config', blob, '--sim-model', 'flag/model']
    )

    assert result.exit_code == 0, result.output
    llm_config: LLMCallConfig = fake.call_args.kwargs['llm_config']
    assert llm_config.model == 'flag/model'
    assert llm_config.temperature == 0.2


def test_llm_config_blob_model_is_kept_when_sim_model_is_not_passed() -> None:
    result, fake = _invoke(
        'simulate', ['--from-run', 'latest', '--target', 'agent:x', '--llm-config', '{"model": "blob/model"}']
    )

    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs['llm_config'].model == 'blob/model'


def test_llm_config_typo_is_rejected() -> None:
    result, fake = _invoke('simulate', ['--from-run', 'latest', '--target', 'agent:x', '--llm-config', '{"temprature": 1}'])

    assert result.exit_code == 2
    assert 'temprature' in _flat(result.output)
    fake.assert_not_called()


@pytest.mark.parametrize(
    ('payload', 'message'),
    [
        ({'target': 'agent:x', 'num_personas': 3}, 'Extra inputs are not permitted'),
        ({'target': 'agent:x', 'personas': [{**PERSONA, 'patiense': 0.1}], 'scenarios': [SCENARIO]}, 'personas[0]'),
    ],
    ids=['run-only key on simulate', 'nested typo'],
)
def test_simulate_rejects_unknown_keys(tmp_path: Path, payload: dict[str, Any], message: str) -> None:
    result, fake = _invoke('simulate', ['--config', _write(tmp_path, payload)])

    assert result.exit_code == 2
    assert message in _flat(result.output)
    fake.assert_not_called()


def test_run_reaches_generate_and_simulate_keywords_without_a_flag(tmp_path: Path) -> None:
    config = _write(
        tmp_path,
        {
            'target': 'agent:x',
            'agent_description': 'A refunds bot',
            'num_personas': 2,
            'persona_seeds': ['angry retiree'],
            'edge_case_percentage': 0.5,
            'generation_instructions': 'reply in German',
            'target_agent_timeout_ms': 1000,
            'llm_config': {'model': 'cfg/model', 'reasoning_effort': 'low'},
        },
    )

    result, fake = _invoke('run', ['--config', config, '--num-scenarios', '3'])

    assert result.exit_code == 0, result.output
    kwargs = fake.call_args.kwargs
    assert kwargs['agent_description'] == 'A refunds bot'
    assert kwargs['num_personas'] == 2
    assert kwargs['num_scenarios'] == 3
    assert kwargs['persona_seeds'] == ['angry retiree']
    assert kwargs['edge_case_percentage'] == 0.5
    assert kwargs['generation_instructions'] == 'reply in German'
    assert kwargs['target_agent_timeout_ms'] == 1000
    assert kwargs['llm_config'].model == 'cfg/model'
    assert kwargs['llm_config'].reasoning_effort == 'low'


def test_every_run_config_field_reaches_the_run_function(tmp_path: Path) -> None:
    """A field the command maps and also forwards would raise a duplicate-keyword error at call time."""
    payload: dict[str, Any] = {
        'run_name': 'r',
        'target': 'agent:x',
        'memory_entity_id': 'm',
        'max_turns': 2,
        'llm_config': {'model': 'm/m'},
        'evaluator_names': ['goal_achieved'],
        'scoring': {},
        'datapoint_parallelism': 2,
        'llm_parallelism': 3,
        'target_agent_timeout_ms': 1000,
        'max_target_retries': 1,
        'target_reasoning_effort': 'low',
        'max_tool_result_chars': 100,
        'per_simulation_timeout_s': 5.0,
        'upload_results': False,
        'experiment_description': 'd',
        'orq_folder_path': 'A/B',
        'raise_on_execution_failure': None,
        'save': False,
        'report_path': str(tmp_path / 'report.json'),
        'executive_summary': False,
        'recommendations': False,
        'agent_description': 'bot',
        'num_personas': 1,
        'num_scenarios': 1,
        'edge_case_percentage': 0.1,
        'persona_seeds': ['p'],
        'scenario_seeds': ['s'],
        'generation_instructions': 'g',
    }
    assert set(payload) == set(GenerateAndSimulateRunConfig.model_fields)

    result, fake = _invoke('run', ['--config', _write(tmp_path, payload)])

    assert result.exit_code == 0, result.output
    kwargs = fake.call_args.kwargs
    assert kwargs['upload_results'] is False
    assert kwargs['evaluation_description'] == 'd'
    assert kwargs['orq_results_path'] == 'A/B'
    assert 'exit_on_failure' not in kwargs
    assert (tmp_path / 'report.json').exists()


def test_json_prints_only_the_simulation_run_on_stdout() -> None:
    result, _ = _invoke('simulate', ['--from-run', 'latest', '--target', 'agent:x', '--json'])

    assert result.exit_code == 0, result.output
    assert SimulationRun.model_validate_json(result.stdout).mode == 'simulate'
    assert 'Using for generations' in result.stderr


def test_a_null_config_value_falls_back_to_the_flag_default() -> None:
    payload = {'target': 'agent:x', 'previous_run': 'latest', 'run_name': None, 'datapoint_parallelism': None}

    result, fake = _invoke('simulate', ['--config', '-'], stdin=json.dumps(payload))

    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs['evaluation_name'] == 'sim'
    assert fake.call_args.kwargs['datapoint_parallelism'] == 10


def test_json_on_run() -> None:
    result, _ = _invoke('run', ['--agent-description', 'bot', '--target', 'agent:x', '--json'])

    assert result.exit_code == 0, result.output
    assert SimulationRun.model_validate_json(result.stdout).mode == 'run'


@pytest.mark.parametrize(
    ('args', 'title'),
    [
        ([], SimulateRunConfig.__name__),
        (['--input'], SimulateRunConfig.__name__),
        (['--command', 'run'], GenerateAndSimulateRunConfig.__name__),
        (['--output'], SimulationRun.__name__),
    ],
)
def test_schema(args: list[str], title: str) -> None:
    result = runner.invoke(app, ['schema', *args])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)['title'] == title
