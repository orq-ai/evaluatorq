"""`eq sim simulate|run --config / --llm-config / --json` and `eq sim schema` (RES-1685)."""
# ruff: noqa: S101

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from evaluatorq.contracts import LLMCallConfig
from evaluatorq.simulation.cli import app
from evaluatorq.simulation.run_config import GenerateAndSimulateCliConfig, SimulateCliConfig
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


def _run(mode: Literal['run', 'simulate', 'generate']) -> SimulationRun:
    return SimulationRun(
        run_name='cfg',
        created_at=datetime.now(tz=timezone.utc),
        mode=mode,
        target_kind='orq_agent',
        evaluator_names=[],
        total_results=0,
        scorer_averages={},
        results=[],
    )


def _invoke_raw(
    command: str, args: list[str], *, stdin: str | None = None, env: dict[str, str] | None = None
) -> tuple[Any, AsyncMock]:
    """Invoke with the run function faked and exactly ``args`` on the command line."""
    internal = '_simulate_run' if command == 'simulate' else '_generate_and_simulate_run'
    fake = AsyncMock(return_value=_run('simulate' if command == 'simulate' else 'run'))
    with (
        patch(f'evaluatorq.simulation.api.{internal}', new=fake),
        patch('evaluatorq.simulation.cli._resolve_target', return_value=MagicMock()),
    ):
        result = runner.invoke(app, [command, *args], input=stdin, env=env)
    return result, fake


def _invoke(
    command: str, args: list[str], *, stdin: str | None = None, env: dict[str, str] | None = None
) -> tuple[Any, AsyncMock]:
    return _invoke_raw(command, [*args, *QUIET], stdin=stdin, env=env)


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def _flat(output: str) -> str:
    """Strip ANSI codes and unwrap the Rich error box, which breaks long messages across lines.

    Typer forces colour when ``GITHUB_ACTIONS`` is set, and the highlighting splits
    option tokens like ``--input`` into separately coloured runs, so the codes must
    go before the whitespace is collapsed (same reason as ``test_cli_simulate_validation``).
    """
    return ' '.join(_ANSI_RE.sub('', output).replace('│', ' ').split())


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
    assert 'datapoints' not in kwargs
    assert kwargs['max_turns'] == 4
    # Public keyword names in the file, internal names at the run function.
    assert kwargs['evaluation_description'] == 'nightly'
    assert kwargs['exit_on_failure'] is False
    assert kwargs['per_simulation_timeout_s'] == 30


def test_two_input_sources_inside_the_config_still_conflict(tmp_path: Path) -> None:
    payload = {'target': 'agent:x', 'personas': [PERSONA], 'scenarios': [SCENARIO], 'previous_run': 'latest'}

    result, fake = _invoke('simulate', ['--config', _write(tmp_path, payload)])

    assert result.exit_code == 2
    flat = _flat(result.output)
    assert 'Provide exactly one of --input' in flat
    assert 'inline personas/scenarios in --config' in flat
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
    result, fake = _invoke(
        'simulate', ['--from-run', 'latest', '--target', 'agent:x', '--llm-config', '{"temprature": 1}']
    )

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


def test_config_only_fields_reach_the_run_function_under_the_sdk_names(tmp_path: Path) -> None:
    payload: dict[str, Any] = {
        'target': 'agent:x',
        'agent_description': 'bot',
        'upload_results': False,
        'experiment_description': 'd',
        'orq_folder_path': 'A/B',
        'raise_on_execution_failure': None,
        'report_path': str(tmp_path / 'report.json'),
    }

    result, fake = _invoke('run', ['--config', _write(tmp_path, payload)])

    assert result.exit_code == 0, result.output
    kwargs = fake.call_args.kwargs
    assert kwargs['upload_results'] is False
    assert kwargs['evaluation_description'] == 'd'
    assert kwargs['orq_results_path'] == 'A/B'
    assert 'exit_on_failure' not in kwargs
    assert 'report' not in kwargs
    assert 'report_path' not in kwargs
    assert (tmp_path / 'report.json').exists()


def test_json_prints_only_the_simulation_run_on_stdout() -> None:
    result, _ = _invoke('simulate', ['--from-run', 'latest', '--target', 'agent:x', '--json'])

    assert result.exit_code == 0, result.output
    assert SimulationRun.model_validate_json(result.stdout).mode == 'simulate'
    assert 'Using for generations' in result.stderr


def test_a_null_config_value_is_unset_so_the_sdk_default_applies() -> None:
    payload = {'target': 'agent:x', 'previous_run': 'latest', 'run_name': None, 'datapoint_parallelism': None}

    result, fake = _invoke('simulate', ['--config', '-'], stdin=json.dumps(payload))

    assert result.exit_code == 0, result.output
    assert 'evaluation_name' not in fake.call_args.kwargs
    assert 'datapoint_parallelism' not in fake.call_args.kwargs


def test_a_null_inline_source_is_not_an_input_source() -> None:
    payload = {'target': 'agent:x', 'previous_run': 'latest', 'personas': None, 'scenarios': None, 'datapoints': None}

    result, fake = _invoke('simulate', ['--config', '-'], stdin=json.dumps(payload))

    assert result.exit_code == 0, _flat(result.output)
    fake.assert_called_once()


def test_save_is_on_by_default_and_reaches_the_sdk() -> None:
    result, fake = _invoke_raw('simulate', ['--from-run', 'latest', '--target', 'agent:x', '--yes'])

    assert result.exit_code == 0, _flat(result.output)
    assert fake.call_args.kwargs['save'] is True


@pytest.mark.parametrize(
    ('payload', 'flags', 'saved'),
    [
        ({}, ['--no-save'], False),
        ({'save': False}, [], False),
        ({'save': True}, ['--no-save'], False),
        ({'save': False}, ['--save'], True),
    ],
    ids=['--no-save', 'config save=false', '--no-save beats config', '--save beats config'],
)
def test_save_follows_the_flag_then_the_config_then_the_cli_default(
    payload: dict[str, Any], flags: list[str], saved: bool
) -> None:
    stdin = json.dumps({'target': 'agent:x', 'previous_run': 'latest', **payload})

    result, fake = _invoke_raw('simulate', ['--config', '-', '--yes', '--no-executive-summary', *flags], stdin=stdin)

    assert result.exit_code == 0, _flat(result.output)
    assert fake.call_args.kwargs['save'] is saved


@pytest.mark.parametrize('command', ['simulate', 'run'])
def test_without_a_config_the_sdk_receives_the_cli_defaults_and_nothing_else(command: str) -> None:
    args = ['--from-run', 'latest'] if command == 'simulate' else ['--agent-description', 'bot']

    result, fake = _invoke_raw(command, [*args, '--target', 'agent:x', '--yes'])

    assert result.exit_code == 0, _flat(result.output)
    kwargs = fake.call_args.kwargs
    assert kwargs['save'] is True
    assert kwargs['recommendations'] is True
    assert kwargs['executive_summary'] is True
    for unset in ('datapoint_parallelism', 'max_turns', 'evaluation_name', 'llm_config', 'llm_parallelism'):
        assert unset not in kwargs


@pytest.mark.parametrize('command', ['simulate', 'run'])
def test_explicit_flags_at_their_default_values_still_win_over_the_config(command: str) -> None:
    base = {'target': 'agent:x', 'max_turns': 3, 'datapoint_parallelism': 2, 'recommendations': False}
    base.update({'previous_run': 'latest'} if command == 'simulate' else {'agent_description': 'bot'})

    result, fake = _invoke(
        command,
        ['--config', '-', '--max-turns', '10', '--datapoint-parallelism', '10', '--recommendations'],
        stdin=json.dumps(base),
    )

    assert result.exit_code == 0, _flat(result.output)
    kwargs = fake.call_args.kwargs
    assert kwargs['max_turns'] == 10
    assert kwargs['datapoint_parallelism'] == 10
    assert kwargs['recommendations'] is True


def _datapoints_file(tmp_path: Path) -> str:
    from evaluatorq.simulation.types import SimulationDatapoint

    datapoint = SimulationDatapoint.model_validate({
        'id': 'dp-1',
        'persona': PERSONA,
        'scenario': SCENARIO,
        'user_system_prompt': 'You are a customer.',
        'first_message': 'Hi',
    })
    path = tmp_path / 'dp.jsonl'
    path.write_text(datapoint.model_dump_json(), encoding='utf-8')
    return str(path)


def _invoke_real_target(command: str, args: list[str], *, stdin: str) -> tuple[Any, AsyncMock, MagicMock]:
    """Like `_invoke`, but runs the real target check, wrapped so the resolved flags are visible."""
    from evaluatorq.simulation import cli

    internal = '_simulate_run' if command == 'simulate' else '_generate_and_simulate_run'
    fake = AsyncMock(return_value=_run('simulate' if command == 'simulate' else 'run'))
    resolver = MagicMock(wraps=cli._resolve_target)
    with (
        patch(f'evaluatorq.simulation.api.{internal}', new=fake),
        patch('evaluatorq.simulation.cli._resolve_target', new=resolver),
    ):
        result = runner.invoke(app, [command, *args, *QUIET], input=stdin, env={'OPENAI_API_KEY': 'test-key'})
    return result, fake, resolver


def test_a_target_flag_replaces_the_config_target(tmp_path: Path) -> None:
    stdin = json.dumps({'target': 'agent:prod'})

    result, fake, resolver = _invoke_real_target(
        'simulate', ['--config', '-', '--openai-model', 'gpt-x', '-i', _datapoints_file(tmp_path)], stdin=stdin
    )

    assert result.exit_code == 0, result.output
    assert resolver.call_args.kwargs['target'] is None
    assert resolver.call_args.kwargs['openai_model'] == 'gpt-x'
    assert fake.call_args.kwargs['datapoints'][0].id == 'dp-1'


_EVERY_INPUT = {
    'datapoints': [
        {
            'id': 'cfg-dp',
            'persona': PERSONA,
            'scenario': SCENARIO,
            'user_system_prompt': 'You are a customer.',
            'first_message': 'Hi',
        }
    ],
    'personas': [PERSONA],
    'scenarios': [SCENARIO],
    'dataset_id': 'ds_cfg',
    'experiment_id': 'exp_cfg',
    'experiment_run_id': 'run_cfg',
    'previous_run': 'prev_cfg',
}


@pytest.mark.parametrize(
    ('flag', 'field'),
    [
        ('--input', 'datapoints'),
        ('--dataset-id', 'dataset_id'),
        ('--experiment-id', 'experiment_id'),
        ('--from-run', 'previous_run'),
    ],
)
def test_each_input_flag_replaces_every_config_input_source(tmp_path: Path, flag: str, field: str) -> None:
    """Whichever trigger flag is passed, the file's other inputs (and its experiment_run_id) are dropped."""
    value = _datapoints_file(tmp_path) if flag == '--input' else 'from-flag'

    result, fake = _invoke(
        'simulate',
        ['--config', '-', flag, value],
        stdin=json.dumps({'target': 'agent:x', **_EVERY_INPUT}),
        env={'ORQ_API_KEY': 'k'},
    )

    assert result.exit_code == 0, result.output
    kwargs = fake.call_args.kwargs
    assert set(kwargs) & set(_EVERY_INPUT) == {field}


def test_the_input_flag_loads_the_jsonl_into_datapoints(tmp_path: Path) -> None:
    result, fake = _invoke('simulate', ['--input', _datapoints_file(tmp_path), '--target', 'agent:x'])

    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs['datapoints'][0].id == 'dp-1'


@pytest.mark.parametrize(
    ('flag', 'value', 'target_kwarg'),
    [
        ('--target', 'agent:new', 'agent:new'),
        ('--vercel-url', 'https://x.test/api', None),
        ('--openai-model', 'gpt-x', None),
    ],
)
@pytest.mark.parametrize('command', ['simulate', 'run'])
def test_each_target_flag_replaces_the_config_target(
    command: str, flag: str, value: str, target_kwarg: str | None
) -> None:
    config = (
        {'target': 'agent:prod', 'agent_description': 'bot'}
        if command == 'run'
        else {'target': 'agent:prod', 'previous_run': 'latest'}
    )
    stdin = json.dumps(config)
    resolver = MagicMock(return_value=MagicMock())
    internal = '_simulate_run' if command == 'simulate' else '_generate_and_simulate_run'
    fake = AsyncMock(return_value=_run('simulate' if command == 'simulate' else 'run'))
    with (
        patch(f'evaluatorq.simulation.api.{internal}', new=fake),
        patch('evaluatorq.simulation.cli._resolve_target', new=resolver),
    ):
        result = runner.invoke(app, [command, '--config', '-', flag, value, *QUIET], input=stdin)

    assert result.exit_code == 0, result.output
    assert resolver.call_args.kwargs['target'] == target_kwarg
    assert 'target' in fake.call_args.kwargs
    assert fake.call_args.kwargs['target'] is resolver.return_value


def test_experiment_run_id_flag_narrows_the_config_experiment() -> None:
    payload = {'target': 'agent:x', 'experiment_id': 'exp_1'}

    result, fake = _invoke(
        'simulate',
        ['--config', '-', '--experiment-run-id', 'run_2'],
        stdin=json.dumps(payload),
        env={'ORQ_API_KEY': 'k'},
    )

    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs['experiment_id'] == 'exp_1'
    assert fake.call_args.kwargs['experiment_run_id'] == 'run_2'


def test_two_target_flags_on_the_command_line_still_conflict() -> None:
    result, fake, _ = _invoke_real_target(
        'simulate',
        ['--config', '-', '--target', 'agent:x', '--openai-model', 'gpt-x'],
        stdin='{"previous_run": "latest"}',
    )

    assert result.exit_code == 2
    assert 'Only one target flag allowed' in _flat(result.output)
    fake.assert_not_called()


def test_two_input_flags_on_the_command_line_still_conflict(tmp_path: Path) -> None:
    result, fake = _invoke(
        'simulate',
        ['--config', '-', '--input', _datapoints_file(tmp_path), '--from-run', 'latest'],
        stdin='{"target": "agent:x"}',
    )

    assert result.exit_code == 2
    assert 'Provide exactly one of --input' in _flat(result.output)
    fake.assert_not_called()


def test_json_on_run() -> None:
    result, _ = _invoke('run', ['--agent-description', 'bot', '--target', 'agent:x', '--json'])

    assert result.exit_code == 0, result.output
    assert SimulationRun.model_validate_json(result.stdout).mode == 'run'


@pytest.mark.parametrize(
    ('args', 'title'),
    [
        ([], SimulateCliConfig.__name__),
        (['--input'], SimulateCliConfig.__name__),
        (['--command', 'run'], GenerateAndSimulateCliConfig.__name__),
        (['--output'], SimulationRun.__name__),
    ],
)
def test_schema(args: list[str], title: str) -> None:
    result = runner.invoke(app, ['schema', *args])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)['title'] == title


def test_a_missing_datapoints_file_is_a_bad_parameter() -> None:
    result, fake = _invoke('simulate', ['--input', 'absent.jsonl', '--target', 'agent:x'])

    assert result.exit_code == 2
    assert 'Datapoints file not found: absent.jsonl' in _flat(result.output)
    fake.assert_not_called()


@pytest.mark.parametrize(
    'content',
    ['', 'not json\n', '{"persona": {}, "scenario": {}}\n'],
    ids=['empty', 'unparseable', 'invalid datapoint'],
)
def test_an_empty_or_invalid_datapoints_file_is_a_one_line_error_with_exit_1(tmp_path: Path, content: str) -> None:
    bad = tmp_path / 'bad.jsonl'
    bad.write_text(content, encoding='utf-8')

    result, fake = _invoke('simulate', ['--input', str(bad), '--target', 'agent:x'])

    assert result.exit_code == 1
    assert 'Traceback' not in result.output
    fake.assert_not_called()


def test_an_empty_evaluator_list_in_the_config_is_forwarded_as_unset() -> None:
    payload = {'target': 'agent:x', 'previous_run': 'latest', 'evaluator_names': []}

    result, fake = _invoke('simulate', ['--config', '-'], stdin=json.dumps(payload))

    assert result.exit_code == 0, result.output
    assert 'evaluator_names' not in fake.call_args.kwargs


CLI_OWNED = {'target', 'memory_entity_id', 'report_path'}


@pytest.mark.parametrize(
    ('cfg_model', 'impl_name', 'fake_name', 'impl_kwargs'),
    [
        (SimulateCliConfig, '_simulate_impl', '_simulate_run', {}),
        (GenerateAndSimulateCliConfig, '_run_impl', '_generate_and_simulate_run', {'agent_description': 'bot'}),
    ],
)
def test_every_config_field_reaches_the_run_function_or_is_cli_owned(
    cfg_model: type[Any], impl_name: str, fake_name: str, impl_kwargs: dict[str, Any]
) -> None:
    """A field added to the config model cannot be dropped silently: it is forwarded or declared CLI-owned."""
    from evaluatorq.simulation import cli
    from evaluatorq.simulation.run_config import to_internal_kwargs

    sentinel = object()
    resolved_target = object()
    cfg = cfg_model.model_construct(**{name: sentinel for name in cfg_model.model_fields})
    fake = AsyncMock(return_value=_run('simulate'))
    with patch(f'evaluatorq.simulation.api.{fake_name}', new=fake):
        asyncio.run(getattr(cli, impl_name)(cfg, target=resolved_target, hooks=None, **impl_kwargs))

    received = fake.call_args.kwargs
    for field in cfg_model.model_fields:
        (internal,) = to_internal_kwargs({field: sentinel})
        if field == 'target':
            assert received[internal] is resolved_target
        elif field in CLI_OWNED:
            assert internal not in received, field
        elif field == 'agent_description':
            assert received[internal] == impl_kwargs['agent_description']  # the CLI's resolved value wins
        else:
            assert received[internal] is sentinel, field
