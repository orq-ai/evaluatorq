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
from evaluatorq.insights import models as models_module
from evaluatorq.insights.models import LabelResult, LabelSpec
from evaluatorq.insights.presets import CONCERNING, USER_FRUSTRATION
from evaluatorq.trace_finder.export import (
    ExportCounts,
    ExportFilters,
    ExportDimension,
    ExportNumericFilters,
    ExportTask,
    ExportTimes,
    ExportValuesSelection,
    RunExport,
)
from evaluatorq.trace_finder import cli as finder_cli
from evaluatorq.trace_finder.settings import DashboardSettings, save_settings
from evaluatorq.trace_finder.models import Snapshot
from tests.insights.test_population import make_trace
from tests.local_sessions.conftest import claude_projects, codex_home


def _app() -> typer.Typer:
    app = typer.Typer()
    cli_root._register_subapps(app)
    return app


def test_help_lists_population_session_and_clustering_options() -> None:
    result = CliRunner().invoke(_app(), ['insights', '--help'], env={'COLUMNS': '120'})

    assert result.exit_code == 0, result.output
    help_text = unstyle(result.output)
    for option in (
        '--query',
        '--profile',
        '--label',
        '--dimension',
        '--from-finder',
        '--from-snapshot',
        '--preview-input',
        '--sessions',
        '--source',
        '--from',
        '--to',
        '--project-dir',
        '--text',
        '--limit',
        '--max-clusters',
        '--classifier-model',
    ):
        assert option in help_text


@pytest.mark.parametrize('spec', [CONCERNING, USER_FRUSTRATION])
def test_score_preset_display_does_not_repeat_level_index(minimal_run: Any, capsys: pytest.CaptureFixture[str], spec: LabelSpec) -> None:
    result = LabelResult(spec=spec, counts={'0': 1}, mean_confidence=None, n_low_confidence=0, n_failed=0)
    run = minimal_run.model_copy(update={'labels': {spec.name: result}})

    cli_module._print_run(run, None)

    output = unstyle(capsys.readouterr().out)
    assert isinstance(spec.criteria, list)
    description = spec.criteria[0].split(':', 1)[0].removeprefix('0 ')
    assert f'0 · {description}' in output
    assert '0 · 0 ' not in output


@pytest.mark.parametrize(
    ('options', 'message'),
    [
        (['--sessions', '--query', 'refunds'], '--sessions cannot be combined with --query'),
        (['--sessions', '--from-snapshot', 'snapshot.json'], '--sessions cannot be combined with --from-snapshot'),
        (['--sessions', '--from-finder', 'finder.json'], '--sessions cannot be combined with --from-finder'),
        (['--sessions', '--window-days', '2'], '--sessions cannot be combined with population options: --window-days'),
        (['--sessions', '--project', 'demo'], '--sessions cannot be combined with population options: --project'),
        (['--sessions', '--source', 'claude'], "unknown --source 'claude'; choose from"),
        (['--project-dir', '.'], '--project-dir requires --sessions'),
        (['--source', 'codex'], '--source requires --sessions'),
        (['--from', '2026-10-01'], '--from requires --sessions'),
    ],
)
def test_sessions_source_conflicts_are_usage_errors(options: list[str], message: str) -> None:
    result = CliRunner().invoke(_app(), ['insights', *options])

    assert result.exit_code == 2
    assert message in result.output


def test_sessions_preview_is_offline_and_does_not_start_pipeline(
    monkeypatch: pytest.MonkeyPatch, claude_projects: Path
) -> None:
    from tests.local_sessions.test_search import _claude

    _claude(claude_projects, 'session-preview')
    monkeypatch.setattr(cli_module, 'resolve_cli_profile', lambda *_args: pytest.fail('preview must not resolve credentials'))
    monkeypatch.setattr(cli_module, 'insights', lambda *_args, **_kwargs: pytest.fail('preview must not run pipeline'))

    result = CliRunner().invoke(
        _app(), ['insights', '--sessions', '--preview-input'], env={'COLUMNS': '120', 'ORQ_API_KEY': ''}
    )

    assert result.exit_code == 0, result.output
    assert 'Model input projection:' in result.output
    assert 'A run would send' in result.output


def test_sessions_freeze_search_results_and_pass_local_content_to_pipeline(
    monkeypatch: pytest.MonkeyPatch, claude_projects: Path, minimal_run: Any
) -> None:
    from tests.local_sessions.test_search import _claude

    _claude(claude_projects, 'session-content', text='the session prompt appears')
    received: dict[str, Any] = {}
    snapshot_paths: list[Path] = []

    async def fake_insights(population: Any, **kwargs: Any) -> Any:
        from evaluatorq.insights.population import resolve_population

        assert population.snapshot_path is not None
        snapshot_paths.append(population.snapshot_path)
        resolved = await resolve_population(
            population,
            orq=None,
            client=None,
            compiler_model='unused',
            classifier_model='unused',
        )
        received['documents'] = resolved.traces
        return minimal_run

    monkeypatch.setattr(cli_module, 'insights', fake_insights)
    result = CliRunner().invoke(
        _app(), ['insights', '--sessions', '--text', 'session prompt', '--limit', '1']
    )

    assert result.exit_code == 0, result.output
    assert len(received['documents']) == 1
    assert 'the session prompt appears' in str(received['documents'][0].trajectory)
    assert len(snapshot_paths) == 1
    assert not snapshot_paths[0].exists()
    assert not snapshot_paths[0].parent.exists()


def test_sessions_source_and_inclusive_local_day_filter_reach_the_pipeline(
    monkeypatch: pytest.MonkeyPatch, claude_projects: Path, codex_home: Path, minimal_run: Any
) -> None:
    from tests.local_sessions.test_search import _claude, _codex, _utc

    _claude(claude_projects, 'matching-claude', text='shared text', start=_utc(2))
    _codex(codex_home, 'matching-codex', text='shared text', start=_utc(2))
    received: dict[str, Any] = {}

    async def fake_insights(population: Any, **kwargs: Any) -> Any:
        from evaluatorq.insights.population import resolve_population

        resolved = await resolve_population(
            population,
            orq=None,
            client=None,
            compiler_model='unused',
            classifier_model='unused',
        )
        received['documents'] = resolved.traces
        return minimal_run

    monkeypatch.setattr(cli_module, 'insights', fake_insights)
    local_day = _utc(2).astimezone().date().isoformat()
    result = CliRunner().invoke(
        _app(),
        [
            'insights',
            '--sessions',
            '--source',
            'codex',
            '--from',
            local_day,
            '--to',
            local_day,
            '--text',
            'shared text',
            '--limit',
            '1',
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(received['documents']) == 1
    assert received['documents'][0].metadata.project == '/work/codex'
    assert 'shared text' in str(received['documents'][0].trajectory)


def test_sessions_snapshot_is_removed_when_pipeline_fails(
    monkeypatch: pytest.MonkeyPatch, claude_projects: Path
) -> None:
    from evaluatorq.insights.models import InsightsPopulation
    from tests.local_sessions.test_search import _claude

    _claude(claude_projects, 'session-error')
    snapshot_paths: list[Path] = []

    async def fail_pipeline(population: Any, **kwargs: Any) -> Any:
        assert isinstance(population, InsightsPopulation)
        assert population.snapshot_path is not None
        snapshot_paths.append(population.snapshot_path)
        raise RuntimeError('expected pipeline failure')

    monkeypatch.setattr(cli_module, 'insights', fail_pipeline)
    result = CliRunner().invoke(_app(), ['insights', '--sessions'])

    assert result.exit_code == 1, result.output
    assert len(snapshot_paths) == 1
    assert not snapshot_paths[0].exists()
    assert not snapshot_paths[0].parent.exists()


def test_snapshot_preview_reports_truncation_without_credentials_or_model_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / 'traces.json'
    trace = make_trace('long').model_copy(
        update={'messages': ({'role': 'user', 'content': 'a' * 600_000}, {'role': 'assistant', 'content': 'done'})}
    )
    path.write_text(Snapshot(traces=(trace,)).model_dump_json(), encoding='utf-8')
    monkeypatch.setattr(cli_module, 'resolve_cli_profile', lambda *_args: pytest.fail('preview must not resolve credentials'))
    monkeypatch.setattr(cli_module, 'insights', lambda *_args, **_kwargs: pytest.fail('preview must not run models'))

    result = CliRunner().invoke(_app(), ['insights', '--from-snapshot', str(path), '--preview-input'])

    assert result.exit_code == 0, result.output
    assert '1 of 1 trace exceeds' in result.output
    assert '1 of 2 whole messages omitted (50.0%)' in result.output


def test_snapshot_run_reaches_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, minimal_run: Any) -> None:
    path = tmp_path / 'traces.json'
    path.write_text(Snapshot(traces=(make_trace('local'),)).model_dump_json(), encoding='utf-8')
    settings_path = tmp_path / 'settings.json'
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(settings_path))
    save_settings(DashboardSettings.model_validate({'orq_profile': 'stale'}), settings_path)
    captured: dict[str, Any] = {}

    async def fake_insights(population: Any, **_kwargs: Any) -> Any:
        captured['population'] = population
        return minimal_run

    monkeypatch.setattr(
        cli_module,
        'resolve_cli_profile',
        lambda *_args: pytest.fail('a local snapshot must not resolve the saved Orq profile'),
    )
    monkeypatch.setattr(cli_module, 'insights', fake_insights)
    result = CliRunner().invoke(_app(), ['insights', '--from-snapshot', str(path), '--dimension', 'intent'])

    assert result.exit_code == 0, result.output
    assert captured['population'].snapshot_path == path
    assert result.output.count('Model input projection:') == 1


def test_snapshot_run_discloses_what_is_sent_to_which_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, minimal_run: Any
) -> None:
    from tests.insights.test_population import make_document

    path = tmp_path / 'sessions.json'
    path.write_text(Snapshot(traces=(), documents=(make_document('s1'),)).model_dump_json(), encoding='utf-8')
    roles = {'insights.summary': 'p/summary', 'insights.labels': 'p/classifier', 'insights.embedding': 'p/embed'}
    monkeypatch.setattr(models_module, 'role_model', lambda _role, *, task: roles[task])
    monkeypatch.setattr(cli_module, 'insights', AsyncMock(return_value=minimal_run))

    result = CliRunner().invoke(
        _app(), ['insights', '--from-snapshot', str(path), '--dimension', 'intent', '--classifier-model', 'x/mine']
    )

    assert result.exit_code == 0, result.output
    assert (
        f'Sending 1 trace ({path.stat().st_size / 1024:.1f} KB) from sessions.json to models: '
        'summary p/summary, classifier x/mine, embedding p/embed.'
    ) in result.output

    preview = CliRunner().invoke(_app(), ['insights', '--from-snapshot', str(path), '--preview-input'])
    assert preview.exit_code == 0, preview.output
    assert 'Sending' not in preview.output
    assert 'A run would send 1 trace (' in preview.output
    insights_mock = cli_module.insights
    assert isinstance(insights_mock, AsyncMock)
    assert insights_mock.await_count == 1


def test_local_send_reports_an_unreadable_snapshot_instead_of_a_traceback(tmp_path: Path) -> None:
    with pytest.raises(typer.Exit) as raised:
        cli_module._print_local_send(
            tmp_path / 'missing.json', n_traces=1, summary_model=None, classifier_model=None, embedding_model=None
        )

    assert raised.value.exit_code == 2


def test_snapshot_cli_rejects_empty_and_conflicting_sources(tmp_path: Path) -> None:
    empty_path = tmp_path / 'empty.json'
    empty_path.write_text(Snapshot(traces=()).model_dump_json(), encoding='utf-8')
    empty = CliRunner().invoke(_app(), ['insights', '--from-snapshot', str(empty_path), '--preview-input'])
    assert empty.exit_code == 2
    assert 'contains no traces' in empty.output

    conflict = CliRunner().invoke(_app(), ['insights', '--from-snapshot', 'a.json', '--from-finder', 'b.json'])
    assert conflict.exit_code == 2
    assert 'cannot be combined' in conflict.output

    missing_source = CliRunner().invoke(_app(), ['insights', '--preview-input'])
    assert missing_source.exit_code == 2
    assert 'requires --from-snapshot' in missing_source.output


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


def test_priority_dimension_must_be_in_selected_dimensions(monkeypatch: Any) -> None:
    invoked: list[bool] = []

    async def fake_insights(population: Any, **kwargs: Any) -> Any:
        invoked.append(True)
        raise AssertionError('pipeline must not run for a missing priority dimension')

    monkeypatch.setattr(cli_module, 'insights', fake_insights)
    result = CliRunner().invoke(
        _app(), ['insights', '--dimension', 'failure', '--priority-dimension', 'intent']
    )

    assert result.exit_code == 2, result.output
    assert "--priority-dimension 'intent' must also be included with --dimension" in unstyle(result.output)
    assert invoked == []


def test_selected_priority_dimension_reaches_pipeline(monkeypatch: Any, minimal_run: Any) -> None:
    captured: dict[str, Any] = {}

    async def fake_run(population: Any, profile: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return minimal_run

    monkeypatch.setattr(cli_module, '_run_insights_with_profile', fake_run)
    result = CliRunner().invoke(
        _app(), ['insights', '--dimension', 'failure', '--priority-dimension', 'failure']
    )

    assert result.exit_code == 0, result.output
    assert captured['dimensions'] == ('failure',)
    assert captured['priority_dimension'] == 'failure'


def test_priority_defaults_to_a_selected_dimension(monkeypatch: Any, minimal_run: Any) -> None:
    captured: dict[str, Any] = {}

    async def fake_run(population: Any, profile: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return minimal_run

    monkeypatch.setattr(cli_module, '_run_insights_with_profile', fake_run)
    result = CliRunner().invoke(_app(), ['insights', '--dimension', 'failure'])

    assert result.exit_code == 0, result.output
    assert captured['dimensions'] == ('failure',)
    assert captured['priority_dimension'] == 'failure'


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
        dimensions=(ExportDimension(name='intent', task=ExportTask(kind='choice', instructions='classify', state={}, noul_threshold=0.5), selection=ExportValuesSelection(kind='values', values=('refunds',))),),
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
    path = tmp_path / 'finder-exports' / 'valid.json'
    path.parent.mkdir()
    path.write_text(export.model_dump_json(), encoding='utf-8')
    captured: dict[str, Any] = {}
    validated: list[RunExport] = []
    validate_export = cli_module._validate_finder_export

    def track_validation(path: Path) -> RunExport:
        export = validate_export(path)
        validated.append(export)
        return export

    async def fake_insights(population: Any, **kwargs: Any) -> Any:
        captured['population'] = population
        captured.update(kwargs)
        return minimal_run

    monkeypatch.setattr(cli_module, 'insights', fake_insights)
    monkeypatch.setattr(cli_module, '_validate_finder_export', track_validation)

    result = CliRunner().invoke(_app(), ['insights', '--from-finder', str(path)])

    assert result.exit_code == 0, result.output
    assert captured['population'].finder_export == path
    assert captured['population'].finder_export_snapshot() == export
    assert len(validated) == 1
    assert captured['population'].finder_export_snapshot() is validated[0]
    assert captured['_finder_export_source'] == path

    monkeypatch.chdir(tmp_path)
    relative_result = CliRunner().invoke(_app(), ['insights', '--from-finder', 'finder-exports/valid.json'])

    assert relative_result.exit_code == 0, relative_result.output
    assert len(validated) == 2
    assert captured['_finder_export_source'] == path


@pytest.mark.asyncio
async def test_cleanup_failure_does_not_replace_the_insights_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.insights.models import InsightsPopulation

    async def failing_insights(*args: Any, **kwargs: Any) -> None:
        raise ValueError('the real failure')

    async def failing_close(_client: Any) -> None:
        raise RuntimeError('close blew up')

    resolved = SimpleNamespace(client=SimpleNamespace(close=AsyncMock(side_effect=RuntimeError('llm close'))), owned=True)
    monkeypatch.setattr(cli_module, 'insights', failing_insights)
    monkeypatch.setattr(cli_module, 'resolve_orq_client', lambda *a, **k: object())
    monkeypatch.setattr(cli_module, 'resolve_llm_client', lambda *a, **k: resolved)
    monkeypatch.setattr(cli_module, 'close_orq_client', failing_close)

    profile = OrqProfile('p', 'k', None, False)
    with pytest.raises(ValueError, match='the real failure'):
        await cli_module._run_insights_with_profile(InsightsPopulation(), profile)
    resolved.client.close.assert_awaited_once()
