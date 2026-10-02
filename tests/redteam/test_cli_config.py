"""`eq redteam run --config / --llm-config / --json` and `eq redteam schema` (RES-1685)."""
# ruff: noqa: S101

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from evaluatorq.redteam.cli import app
from evaluatorq.redteam.contracts import (
    LLMConfig,
    Pipeline,
    RedTeamRecommendationConfig,
    RedTeamReport,
    ReportSummary,
    SaveMode,
    TargetConfig,
)
from evaluatorq.redteam.run_config import RedTeamRunConfig

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _report(*, evaluated: int = 5) -> RedTeamReport:
    return RedTeamReport(
        created_at=datetime.now(tz=timezone.utc),
        description=None,
        pipeline=Pipeline.DYNAMIC,
        framework=None,
        categories_tested=[],
        tested_agents=['agent:x'],
        total_results=0,
        results=[],
        summary=ReportSummary(total_attacks=5, evaluated_attacks=evaluated, resistance_rate=0.8),
    )


def _invoke(args: list[str], *, stdin: str | None = None, report: RedTeamReport | None = None) -> tuple[Any, AsyncMock]:
    fake = AsyncMock(return_value=report or _report())
    with patch('evaluatorq.redteam.red_team', new=fake):
        result = runner.invoke(app, ['run', *args, '--yes'], input=stdin)
    return result, fake


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def _flat(output: str) -> str:
    """Strip ANSI codes and unwrap the Rich error box, which breaks long messages across lines.

    Typer forces colour when ``GITHUB_ACTIONS`` is set, and the highlighting splits
    option tokens like ``--input`` into separately coloured runs, so the codes must
    go before the whitespace is collapsed (same reason as ``test_cli_simulate_validation``).
    """
    return ' '.join(_ANSI_RE.sub('', output).replace('│', ' ').split())


def _write(tmp_path: Path, payload: dict[str, Any]) -> str:
    path = tmp_path / 'run.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    return str(path)


def test_config_file_supplies_the_red_team_keywords(tmp_path: Path) -> None:
    config = _write(
        tmp_path,
        {
            'target': 'agent:x',
            'mode': 'hybrid',
            'max_turns': 8,
            'description': 'nightly',
            'attack_techniques': ['indirect-injection'],
            'datapoints': [{'inputs': {'category': 'ASI01'}}],
            'generate_strategies': False,
            'target_config': {'system_prompt': 'be careful'},
            'recommendations': {'max_areas': 2},
            'save': 'none',
        },
    )

    result, fake = _invoke(['--config', config])

    assert result.exit_code == 0, result.output
    kwargs = fake.call_args.kwargs
    assert kwargs['target'] == 'agent:x'
    assert kwargs['mode'] is Pipeline.HYBRID
    assert kwargs['max_turns'] == 8
    assert kwargs['description'] == 'nightly'
    assert kwargs['attack_techniques'] == ['indirect-injection']
    assert kwargs['datapoints'][0].inputs == {'category': 'ASI01'}
    assert kwargs['generate_strategies'] is False
    assert kwargs['target_config'] == TargetConfig(system_prompt='be careful')
    assert kwargs['recommendations'] == RedTeamRecommendationConfig(max_areas=2)
    assert kwargs['save'] is SaveMode.NONE


def test_config_is_read_from_stdin() -> None:
    result, fake = _invoke(['--config', '-'], stdin=json.dumps({'target': ['agent:a', 'agent:b'], 'max_turns': 2}))

    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs['target'] == ['agent:a', 'agent:b']
    assert fake.call_args.kwargs['max_turns'] == 2


def test_an_explicit_flag_beats_the_config_even_at_its_default_value(tmp_path: Path) -> None:
    # 2 is --generated-strategy-count's default: precedence must come from where the value came from,
    # not from comparing it with the default.
    config = _write(tmp_path, {'target': 'agent:x', 'generated_strategy_count': 4, 'name': 'from-config'})

    result, fake = _invoke(['--config', config, '--generated-strategy-count', '2', '-t', 'agent:flag'])

    assert result.exit_code == 0, result.output
    kwargs = fake.call_args.kwargs
    assert kwargs['generated_strategy_count'] == 2
    assert kwargs['target'] == 'agent:flag'
    assert kwargs['name'] == 'from-config'


def test_without_config_red_team_gets_the_cli_defaults() -> None:
    result, fake = _invoke(['-t', 'agent:x'])

    assert result.exit_code == 0, result.output
    kwargs = fake.call_args.kwargs
    # Unset values are left to red_team's own defaults.
    assert kwargs['datapoint_parallelism'] is None
    assert kwargs['max_turns'] is None
    assert kwargs['name'] is None
    assert kwargs['llm_config'] == LLMConfig()
    assert kwargs['verbosity'] == 1
    assert kwargs['generate_strategies'] is True
    assert kwargs['cleanup_memory'] is True
    assert kwargs['save'] is SaveMode.FINAL


@pytest.mark.parametrize(
    ('payload', 'message'),
    [
        ({'target': 'agent:x', 'max_turn': 3}, 'Extra inputs are not permitted'),
        ({'target': 'agent:x', 'llm_config': {'attacker': {'temprature': 0.2}}}, 'llm_config.attacker.temprature'),
        ({'target': 'agent:x', 'max_turns': 'many'}, 'max_turns'),
    ],
    ids=['top-level typo', 'nested typo', 'wrong type'],
)
def test_a_bad_config_is_rejected_before_anything_runs(tmp_path: Path, payload: dict[str, Any], message: str) -> None:
    result, fake = _invoke(['--config', _write(tmp_path, payload)])

    assert result.exit_code == 2
    assert message in _flat(result.output)
    fake.assert_not_called()


def test_a_missing_config_file_is_a_usage_error(tmp_path: Path) -> None:
    result, fake = _invoke(['--config', str(tmp_path / 'absent.json')])

    assert result.exit_code == 2
    assert 'cannot read' in _flat(result.output)
    fake.assert_not_called()


def test_target_is_required_from_the_flag_or_the_config(tmp_path: Path) -> None:
    result, fake = _invoke(['--config', _write(tmp_path, {'max_turns': 3})])

    assert result.exit_code == 2
    assert 'provide --target' in _flat(result.output)
    fake.assert_not_called()


def test_llm_config_blob_with_a_narrow_flag_winning_its_own_field() -> None:
    blob = json.dumps({'attacker': {'model': 'blob/attacker', 'temperature': 0.9}, 'retry_count': 5})

    result, fake = _invoke(['-t', 'agent:x', '--llm-config', blob, '--attack-model', 'flag/attacker'])

    assert result.exit_code == 0, result.output
    llm_config: LLMConfig = fake.call_args.kwargs['llm_config']
    assert llm_config.attacker.model == 'flag/attacker'
    assert llm_config.attacker.temperature == 0.9
    assert llm_config.retry_count == 5
    # A narrow flag left at its default does not overwrite the blob.
    assert 'max_target_retries' not in llm_config.model_fields_set


def test_llm_config_flag_merges_into_the_config_files_llm_config(tmp_path: Path) -> None:
    config = _write(tmp_path, {'target': 'agent:x', 'llm_config': {'retry_count': 7, 'max_probe_turns': 3}})

    result, fake = _invoke(['--config', config, '--llm-config', '{"retry_count": 1}', '--max-target-retries', '4'])

    assert result.exit_code == 0, result.output
    llm_config: LLMConfig = fake.call_args.kwargs['llm_config']
    assert llm_config.retry_count == 1
    assert llm_config.max_probe_turns == 3
    assert llm_config.max_target_retries == 4


def test_system_prompt_flag_overrides_the_config_files_target_config(tmp_path: Path) -> None:
    config = _write(tmp_path, {'target': 'agent:x', 'target_config': {'system_prompt': 'from file'}})

    result, fake = _invoke(['--config', config, '--system-prompt', 'from flag'])

    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs['target_config'] == TargetConfig(system_prompt='from flag')


def test_a_negative_bool_flag_beats_the_config(tmp_path: Path) -> None:
    config = _write(tmp_path, {'target': 'agent:x', 'generate_strategies': True, 'cleanup_memory': False})

    result, fake = _invoke(['--config', config, '--no-generate-strategies', '--cleanup-memory'])

    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs['generate_strategies'] is False
    assert fake.call_args.kwargs['cleanup_memory'] is True


def test_the_positive_bool_flag_beats_a_false_config_value(tmp_path: Path) -> None:
    config = _write(tmp_path, {'target': 'agent:x', 'generate_strategies': False})

    result, fake = _invoke(['--config', config, '--generate-strategies'])

    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs['generate_strategies'] is True


def test_a_null_config_value_means_unset(tmp_path: Path) -> None:
    config = _write(tmp_path, {'target': 'agent:x', 'max_turns': None, 'datapoint_parallelism': None})

    result, fake = _invoke(['--config', config])

    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs['max_turns'] is None
    assert fake.call_args.kwargs['datapoint_parallelism'] is None


@pytest.mark.parametrize(('flags', 'verbosity'), [([], 3), (['-v'], 2), (['-q'], 0), (['-v', '-q'], 0)])
def test_verbosity_flags_beat_the_config(tmp_path: Path, flags: list[str], verbosity: int) -> None:
    config = _write(tmp_path, {'target': 'agent:x', 'verbosity': 3})

    result, fake = _invoke(['--config', config, *flags])

    assert result.exit_code == 0, result.output
    assert fake.call_args.kwargs['verbosity'] == verbosity


def test_json_prints_only_the_report_on_stdout(tmp_path: Path) -> None:
    report_path = tmp_path / 'report.json'

    result, _ = _invoke(['-t', 'agent:x', '--json', '--report', str(report_path)])

    assert result.exit_code == 0, result.output
    parsed = RedTeamReport.model_validate_json(result.stdout)
    assert parsed.tested_agents == ['agent:x']
    assert 'Report saved' in result.stderr
    assert report_path.exists()


def test_json_keeps_the_coverage_gate_exit_code() -> None:
    result, _ = _invoke(['-t', 'agent:x', '--json'], report=_report(evaluated=0))

    assert result.exit_code == 1
    assert RedTeamReport.model_validate_json(result.stdout).summary.evaluated_attacks == 0
    assert 'could be evaluated' in result.stderr


def test_every_config_field_reaches_red_team(tmp_path: Path) -> None:
    """A field the command maps and also forwards would raise a duplicate-keyword error at call time."""
    payload: dict[str, Any] = {
        'target': 'agent:x',
        'llm_config': {'retry_count': 1},
        'mode': 'dynamic',
        'categories': ['ASI01'],
        'vulnerabilities': ['goal_hijacking'],
        'strategies': None,
        'datapoints': None,
        'attack_techniques': None,
        'delivery_methods': ['direct_request'],
        'max_turns': 2,
        'max_per_category': 1,
        'datapoint_parallelism': 3,
        'llm_parallelism': 4,
        'generate_strategies': False,
        'generated_strategy_count': 1,
        'max_dynamic_datapoints': 5,
        'max_static_datapoints': 6,
        'cleanup_memory': False,
        'name': 'n',
        'description': 'd',
        'dataset': 'hf:org/repo',
        'previous_run': None,
        'artifacts_dir': str(tmp_path),
        'target_config': {'system_prompt': 's'},
        'recommendations': False,
        'generate_executive_summary': False,
        'attacker_instructions': 'i',
        'verbosity': 2,
        'save': 'detail',
    }
    assert set(payload) == set(RedTeamRunConfig.model_fields)

    result, fake = _invoke(['--config', _write(tmp_path, payload)])

    assert result.exit_code == 0, result.output
    kwargs = fake.call_args.kwargs
    assert kwargs['verbosity'] == 2
    assert kwargs['cleanup_memory'] is False
    assert kwargs['generate_executive_summary'] is False
    assert kwargs['max_static_datapoints'] == 6


def test_schema_defaults_to_the_config_input_shape() -> None:
    result = runner.invoke(app, ['schema'])

    assert result.exit_code == 0, result.output
    schema = json.loads(result.stdout)
    assert schema['title'] == 'RedTeamCliConfig'
    assert schema['additionalProperties'] is False
    assert {'target', 'llm_config', 'datapoints'} <= set(schema['properties'])


def test_schema_output_describes_the_report() -> None:
    result = runner.invoke(app, ['schema', '--output'])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)['title'] == 'RedTeamReport'
