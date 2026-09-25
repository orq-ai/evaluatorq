"""CLI tests for ``eq insights``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from typer.testing import CliRunner

from evaluatorq import cli as cli_root
from evaluatorq.insights import cli as cli_module
from evaluatorq.insights.models import LabelSpec


def _app() -> typer.Typer:
    app = typer.Typer()
    cli_root._register_subapps(app)
    return app


def test_help_lists_population_and_clustering_options() -> None:
    result = CliRunner().invoke(_app(), ['insights', '--help'])

    assert result.exit_code == 0, result.output
    for option in ('--query', '--label', '--dimension', '--from-finder', '--max-clusters'):
        assert option in result.output


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
