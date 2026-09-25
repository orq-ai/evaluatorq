"""CLI tests for ``eq insights``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from click import unstyle
from typer.testing import CliRunner

from evaluatorq import cli as cli_root
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


def _app() -> typer.Typer:
    app = typer.Typer()
    cli_root._register_subapps(app)
    return app


def test_help_lists_population_and_clustering_options() -> None:
    result = CliRunner().invoke(_app(), ['insights', '--help'], env={'COLUMNS': '120'})

    assert result.exit_code == 0, result.output
    help_text = unstyle(result.output)
    for option in ('--query', '--label', '--dimension', '--from-finder', '--max-clusters'):
        assert option in help_text


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
    monkeypatch.setattr(cli_module, '_stored_run_path', lambda run: None)

    result = CliRunner().invoke(_app(), ['insights', '--label', 'sentiment', '--label', str(label_path)])

    assert result.exit_code == 0, result.output
    assert [spec.name for spec in captured['labels']] == ['sentiment', 'tier']


def test_error_run_exits_one(monkeypatch: Any, minimal_run: Any) -> None:
    async def fake_insights(population: Any, **kwargs: Any) -> Any:
        return minimal_run.model_copy(update={'status': 'error'})

    monkeypatch.setattr(cli_module, 'insights', fake_insights)
    monkeypatch.setattr(cli_module, '_stored_run_path', lambda run: None)

    result = CliRunner().invoke(_app(), ['insights'])

    assert result.exit_code == 1, result.output


def test_finder_export_and_query_are_usage_error() -> None:
    result = CliRunner().invoke(_app(), ['insights', '--from-finder', 'finder.json', '--query', 'refunds'])

    assert result.exit_code == 2, result.output
    assert '--from-finder cannot be combined with --query' in result.output


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
        return minimal_run

    monkeypatch.setattr(cli_module, 'insights', fake_insights)
    monkeypatch.setattr(cli_module, '_stored_run_path', lambda run: None)

    result = CliRunner().invoke(_app(), ['insights', '--from-finder', str(path)])

    assert result.exit_code == 0, result.output
    assert captured['population'].finder_export == path
