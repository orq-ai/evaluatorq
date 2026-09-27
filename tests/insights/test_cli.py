"""CLI tests for ``eq insights``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import typer
from click import unstyle
from typer.testing import CliRunner

from evaluatorq import cli as cli_root
from evaluatorq.common.orq_client import OrqProfile
from evaluatorq.insights import cli as cli_module
from evaluatorq.insights.models import LabelSpec
from evaluatorq.trace_finder.export import (
    ExportCounts,
    ExportFilters,
    ExportNumericFilters,
    ExportTask,
    ExportTimes,
    ExportValuesSelection,
    RunExport,
)
from evaluatorq.trace_finder import cli as finder_cli
from evaluatorq.trace_finder.settings import DashboardSettings, save_settings


def _app() -> typer.Typer:
    app = typer.Typer()
    cli_root._register_subapps(app)
    return app


def test_help_lists_population_and_clustering_options() -> None:
    result = CliRunner().invoke(_app(), ['insights', '--help'], env={'COLUMNS': '120'})

    assert result.exit_code == 0, result.output
    help_text = unstyle(result.output)
    for option in ('--query', '--profile', '--label', '--dimension', '--from-finder', '--max-clusters', '--classifier-model'):
        assert option in help_text


@pytest.mark.parametrize(
    ('override', 'expected_key', 'expected_host'),
    [(False, 'saved-key', 'https://saved.example'), (True, 'override-key', 'https://my.orq.ai')],
)
def test_cli_uses_selected_profile_for_traces_and_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    minimal_run: Any,
    override: bool,
    expected_key: str,
    expected_host: str,
) -> None:
    settings_path = tmp_path / 'settings.json'
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(settings_path))
    monkeypatch.setenv('ORQ_API_KEY', 'environment-key')
    save_settings(DashboardSettings.model_validate({'orq_profile': 'saved'}), settings_path)
    profiles = (
        OrqProfile('saved', 'saved-key', 'https://saved.example', False),
        OrqProfile('override', 'override-key', None, False),
    )
    monkeypatch.setattr(finder_cli, 'list_orq_profiles', lambda: profiles)
    orq = object()
    llm = SimpleNamespace(close=AsyncMock())
    captured: dict[str, Any] = {}
    closed_orq = AsyncMock()

    async def fake_insights(population: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return minimal_run

    def fake_orq(*args: Any, **kwargs: Any) -> Any:
        captured['orq_args'] = (args, kwargs)
        return orq

    def fake_llm(**kwargs: Any) -> Any:
        captured['llm_args'] = kwargs
        return SimpleNamespace(client=llm, owned=True)

    monkeypatch.setattr(cli_module, 'insights', fake_insights)
    monkeypatch.setattr(cli_module, 'resolve_orq_client', fake_orq)
    monkeypatch.setattr(cli_module, 'resolve_llm_client', fake_llm)
    monkeypatch.setattr(cli_module, 'close_orq_client', closed_orq)

    args = ['insights', '--profile', 'override'] if override else ['insights']
    result = CliRunner().invoke(_app(), args)

    assert result.exit_code == 0, result.output
    assert captured['orq_args'] == ((expected_key,), {'base_url': expected_host})
    assert captured['llm_args']['extra_api_key'] == expected_key
    assert captured['llm_args']['orq_host'] == expected_host
    assert captured['orq_client'] is orq
    assert captured['llm_client'] is llm
    closed_orq.assert_awaited_once_with(orq)
    llm.close.assert_awaited_once()


def test_missing_selected_profile_does_not_use_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings_path = tmp_path / 'settings.json'
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(settings_path))
    monkeypatch.setenv('ORQ_API_KEY', 'environment-key')
    save_settings(DashboardSettings.model_validate({'orq_profile': 'deleted'}), settings_path)
    monkeypatch.setattr(finder_cli, 'list_orq_profiles', lambda: ())
    client = AsyncMock()
    monkeypatch.setattr(cli_module, 'resolve_orq_client', client)

    result = CliRunner().invoke(_app(), ['insights'])

    assert result.exit_code == 2
    assert "Orq profile 'deleted' is unavailable" in result.output
    client.assert_not_called()


def test_repeated_labels_resolve_preset_and_json_spec(
    monkeypatch: Any,
    tmp_path: Path,
    minimal_run: Any,
) -> None:
    custom = LabelSpec(name='tier', kind='choice', instructions='Classify the support tier.', criteria={'one': None})
    label_path = tmp_path / 'tier.json'
    label_path.write_text(custom.model_dump_json(), encoding='utf-8')
    captured: dict[str, Any] = {}

    async def fake_insights(population: Any, **kwargs: Any) -> Any:
        captured['population'] = population
        captured.update(kwargs)
        return minimal_run

    monkeypatch.setattr(cli_module, 'insights', fake_insights)

    result = CliRunner().invoke(_app(), ['insights', '--label', 'sentiment', '--label', str(label_path)])

    assert result.exit_code == 0, result.output
    assert [spec.name for spec in captured['labels']] == ['sentiment', 'tier']


def test_repeated_dimensions_are_rejected_before_pipeline_runs(monkeypatch: Any) -> None:
    invoked: list[bool] = []

    async def fake_insights(population: Any, **kwargs: Any) -> Any:
        invoked.append(True)
        raise AssertionError('pipeline must not run for duplicate dimensions')

    monkeypatch.setattr(cli_module, 'insights', fake_insights)
    result = CliRunner().invoke(
        _app(), ['insights', '--dimension', 'intent', '--dimension', 'intent']
    )

    assert result.exit_code == 2, result.output
    assert '--dimension cannot be repeated: intent' in unstyle(result.output)
    assert invoked == []


def test_cli_prints_report_path_without_loading_historical_runs(
    monkeypatch: Any, minimal_run: Any, tmp_path: Path
) -> None:
    report_path = tmp_path / 'insights-new-run.json'

    async def fake_run(population: Any, profile: Any, **kwargs: Any) -> Any:
        kwargs['_on_saved'](report_path)
        return minimal_run

    def unexpected_history_scan() -> Any:
        raise AssertionError('CLI must use the path reported by the pipeline')

    monkeypatch.setattr(cli_module, '_run_insights_with_profile', fake_run)
    monkeypatch.setattr(cli_module, 'list_runs', unexpected_history_scan, raising=False)

    result = CliRunner().invoke(_app(), ['insights'])

    assert result.exit_code == 0, result.output
    assert str(report_path) in ''.join(unstyle(result.output).split())


def test_error_run_exits_one(monkeypatch: Any, minimal_run: Any) -> None:
    async def fake_insights(population: Any, **kwargs: Any) -> Any:
        return minimal_run.model_copy(update={'status': 'error'})

    monkeypatch.setattr(cli_module, 'insights', fake_insights)

    result = CliRunner().invoke(_app(), ['insights'])

    assert result.exit_code == 1, result.output


def test_operational_pipeline_failure_is_reported_without_traceback(monkeypatch: Any) -> None:
    async def fail_pipeline(population: Any, **kwargs: Any) -> Any:
        raise OSError('connection refused')

    monkeypatch.setattr(cli_module, 'insights', fail_pipeline)

    result = CliRunner().invoke(_app(), ['insights'])

    assert result.exit_code == 1
    assert 'Error: connection refused' in result.output
    assert 'Traceback' not in result.output


def test_programmer_error_from_pipeline_is_not_hidden(monkeypatch: Any) -> None:
    async def fail_pipeline(population: Any, **kwargs: Any) -> Any:
        raise AssertionError('unexpected invariant failure')

    monkeypatch.setattr(cli_module, 'insights', fail_pipeline)

    result = CliRunner().invoke(_app(), ['insights'])

    assert isinstance(result.exception, AssertionError)
    assert 'unexpected invariant failure' in str(result.exception)


def test_finder_export_and_query_are_usage_error() -> None:
    result = CliRunner().invoke(_app(), ['insights', '--from-finder', 'finder.json', '--query', 'refunds'])

    assert result.exit_code == 2, result.output
    assert '--from-finder cannot be combined with --query' in result.output


@pytest.mark.parametrize(
    ('option', 'value'),
    [
        ('--window-days', '3'),
        ('--limit', '10'),
        ('--project', 'project-a'),
        ('--model', 'model-a'),
        ('--provider', 'provider-a'),
        ('--status', 'completed'),
        ('--product', 'chat'),
        ('--trace-type', 'conversation'),
        ('--agent', 'agent-a'),
        ('--tool', 'search'),
        ('--tokens-min', '1'),
        ('--tokens-max', '10'),
        ('--duration-ms-min', '1'),
        ('--duration-ms-max', '10'),
    ],
)
def test_finder_export_rejects_population_options(option: str, value: str) -> None:
    result = CliRunner().invoke(_app(), ['insights', '--from-finder', 'finder.json', option, value])

    assert result.exit_code == 2, result.output
    assert f'--from-finder cannot be combined with population options: {option}' in unstyle(result.output)


def test_invalid_finder_exports_are_usage_errors_without_running_pipeline(tmp_path: Path, monkeypatch: Any) -> None:
    missing_path = tmp_path / 'missing.json'
    malformed_path = tmp_path / 'malformed.json'
    malformed_path.write_text('{not json', encoding='utf-8')
    invalid_path = tmp_path / 'invalid-schema.json'
    invalid_path.write_text('{"schema_version": 2}', encoding='utf-8')
    invoked: list[bool] = []

    async def fake_insights(population: Any, **kwargs: Any) -> Any:
        invoked.append(True)
        raise AssertionError('pipeline must not run for invalid finder input')

    monkeypatch.setattr(cli_module, 'insights', fake_insights)

    for path in (missing_path, malformed_path, invalid_path):
        result = CliRunner().invoke(_app(), ['insights', '--from-finder', str(path)])

        assert result.exit_code == 2, result.output
        assert f'could not read a valid finder export from {path}' in result.output
    assert invoked == []


def test_valid_finder_export_reaches_pipeline(tmp_path: Path, monkeypatch: Any, minimal_run: Any) -> None:
    export = RunExport(
        query='refund requests',
        task=ExportTask(kind='choice', instructions='classify', state={}, noul_threshold=0.5),
        selection=ExportValuesSelection(kind='values', values=('refunds',)),
        generated_filters=ExportFilters(),
        filters=ExportFilters(),
        generated_numeric=ExportNumericFilters(),
        numeric=ExportNumericFilters(),
        limit=500,
        parallelism=100,
        times=ExportTimes(elapsed=0, rate=0),
        counts=ExportCounts(total=0, completed=0, failed=0, matched=0, active=0, queued=0, percent=0),
        traces=(),
        matched_trace_ids=[],
    )
    path = tmp_path / 'valid.json'
    path.write_text(export.model_dump_json(), encoding='utf-8')
    captured: dict[str, Any] = {}

    async def fake_insights(population: Any, **kwargs: Any) -> Any:
        captured['population'] = population
        captured.update(kwargs)
        return minimal_run

    monkeypatch.setattr(cli_module, 'insights', fake_insights)

    result = CliRunner().invoke(_app(), ['insights', '--from-finder', str(path)])

    assert result.exit_code == 0, result.output
    assert captured['population'].finder_export == path
    assert captured['_finder_export_source'] == path
