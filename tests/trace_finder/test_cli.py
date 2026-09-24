from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import click
import pytest
import typer
from rich.console import Console
from typer.main import get_command
from typer.testing import CliRunner

from evaluatorq import cli as cli_module
from evaluatorq.common.judge import ClassifyQuestion
from evaluatorq.common.orq_client import OrqProfile
from evaluatorq.trace_finder import (
    CompiledQuery,
    RunSnapshot,
    TraceClassification,
    TraceRecord,
    ValueSelection,
)


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv('EVALUATORQ_DASHBOARD_SETTINGS', str(tmp_path / 'settings.json'))


def _app() -> typer.Typer:
    app = typer.Typer()
    cli_module._register_subapps(app)
    return app


def test_find_help_describes_profile_option() -> None:
    result = CliRunner().invoke(_app(), ['find', '--help'])

    assert result.exit_code == 0, result.output
    command = get_command(_app())
    assert isinstance(command, click.Group)
    option = next((param for param in command.commands['find'].params if '--profile' in param.opts), None)
    assert isinstance(option, click.Option)
    assert option.help is not None
    assert 'ORQ_API_KEY and ORQ_BASE_URL' in option.help


def _trace() -> TraceRecord:
    return TraceRecord(
        schema_version=1,
        trace_id='trace-cli-1',
        span_id='span-cli-1',
        timestamp=datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc),
        messages=({'role': 'user', 'content': 'I need a refund.'},),
        project='support-agent',
        model='gpt-5.6-luna',
        provider='openai',
        status='ok',
        product='production',
        trace_type='llm',
    )


class FakeStore:
    def __init__(self) -> None:
        self.request: Any | None = None

    async def close(self) -> None:
        return None

    async def compile(self, request: Any, *, wait: bool = True) -> RunSnapshot:
        assert wait is False
        self.request = request
        trace = _trace()
        compiled = CompiledQuery(
            task=ClassifyQuestion(
                kind='choice',
                instructions='Does the trace mention a refund?',
                criteria={'yes': 'It mentions a refund.', 'no': 'It does not.'},
                state={},
            ),
            selection=ValueSelection(kind='values', values=('yes',)),
        )
        result = TraceClassification(
            trace_id=trace.trace_id,
            span_id=trace.span_id,
            value='yes',
            confidence=0.94,
            matched=True,
            raw_result={'value': 'yes'},
        )
        return RunSnapshot(
            state='completed',
            request=request,
            compiled=compiled,
            trace_ids=(trace.trace_id,),
            traces=(trace,),
            results={trace.trace_id: result},
            total=1,
            completed=1,
            matched=1,
        )


def test_find_exits_nonzero_when_classifications_failed(monkeypatch: Any, tmp_path: Path) -> None:
    from evaluatorq.trace_finder import cli as find_cli

    class FailedStore(FakeStore):
        async def compile(self, request: Any, *, wait: bool = True) -> RunSnapshot:
            snapshot = await super().compile(request, wait=wait)
            return replace(snapshot, failed=1)

    monkeypatch.setattr(find_cli, 'resolve_orq_client', lambda: object())
    monkeypatch.setattr(find_cli, 'resolve_llm_client', lambda **_: SimpleNamespace(client=object()))
    monkeypatch.setattr(find_cli, 'build_run_store', lambda *args, **kwargs: FailedStore())
    output = tmp_path / 'partial.json'

    result = CliRunner().invoke(_app(), ['find', 'refund requests', '--json', str(output)])

    assert result.exit_code == 1
    assert '1 failed classifications' in result.output
    assert not output.exists()


def test_find_missing_orq_key_exits_two(monkeypatch: Any) -> None:
    from evaluatorq.trace_finder import cli as find_cli

    def missing_key() -> Any:
        raise ValueError('ORQ_API_KEY environment variable must be set to reach the Orq API.')

    monkeypatch.setattr(find_cli, 'resolve_orq_client', missing_key)
    result = CliRunner().invoke(_app(), ['find', 'refund requests'])

    assert result.exit_code == 2
    assert 'ORQ_API_KEY' in result.output


def test_find_missing_orq_sdk_exits_two(monkeypatch: Any) -> None:
    from evaluatorq.trace_finder import cli as find_cli

    def missing_sdk() -> Any:
        raise ImportError('Install evaluatorq[orq] to run live trace search.')

    monkeypatch.setattr(find_cli, 'resolve_orq_client', missing_sdk)
    result = CliRunner().invoke(_app(), ['find', 'refund requests'])

    assert result.exit_code == 2
    assert 'Install evaluatorq[orq]' in result.output


def test_find_preserves_unexpected_programming_errors(monkeypatch: Any) -> None:
    from evaluatorq.trace_finder import cli as find_cli

    monkeypatch.setattr(find_cli, 'resolve_orq_client', lambda: object())
    monkeypatch.setattr(find_cli, 'resolve_llm_client', lambda **_: SimpleNamespace(client=object()))
    monkeypatch.setattr(find_cli, 'build_run_store', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('bug')))

    result = CliRunner().invoke(_app(), ['find', 'refund requests'])

    assert isinstance(result.exception, RuntimeError)


@pytest.mark.asyncio
async def test_find_polling_timeout_cancels_store(monkeypatch: Any) -> None:
    from evaluatorq.trace_finder import cli as find_cli

    class StalledStore:
        cancelled = False

        async def compile(self, request: Any, *, wait: bool = True) -> RunSnapshot:
            return RunSnapshot(state='compiling')

        async def snapshot(self) -> RunSnapshot:
            return RunSnapshot(state='compiling')

        async def cancel(self) -> RunSnapshot:
            self.cancelled = True
            return RunSnapshot(state='cancelled')

    store = StalledStore()
    monkeypatch.setattr(find_cli, 'MAX_FIND_WAIT_SECONDS', 0.01)

    with pytest.raises(TimeoutError, match='wait limit'):
        await find_cli._run(store, cast(Any, object()), Console())

    assert store.cancelled


def test_find_writes_json_and_prints_fake_trace(monkeypatch: Any, tmp_path: Path) -> None:
    from evaluatorq.trace_finder import cli as find_cli

    store = FakeStore()
    monkeypatch.setattr(find_cli, 'resolve_orq_client', lambda: object())
    monkeypatch.setattr(find_cli, 'resolve_llm_client', lambda **_: SimpleNamespace(client=object()))
    monkeypatch.setattr(find_cli, 'build_run_store', lambda *args, **kwargs: store)
    output = tmp_path / 'finder.json'

    result = CliRunner().invoke(
        _app(),
        [
            'find',
            'refund requests',
            '--json',
            str(output),
            '--project',
            'support-agent',
            '--tokens-min',
            '100',
        ],
    )

    assert result.exit_code == 0, result.output
    assert 'Matched traces' in result.output
    assert output.exists()
    assert 'trace-cli-1' in output.read_text()
    assert store.request is not None
    assert store.request.population.facets.project == frozenset({'support-agent'})
    assert store.request.population.numeric.tokens_min == 100


@pytest.mark.parametrize('use_flag', [True, False])
def test_find_profile_overrides_environment_for_both_clients_without_mutating_it(
    monkeypatch: Any, tmp_path: Path, use_flag: bool
) -> None:
    from evaluatorq.trace_finder import cli as find_cli
    from evaluatorq.trace_finder.settings import DashboardSettings, credential_fingerprint, save_settings

    monkeypatch.setenv('ORQ_API_KEY', 'environment-key')
    monkeypatch.setenv('ORQ_BASE_URL', 'https://environment.example')
    monkeypatch.setattr(
        find_cli, 'list_orq_profiles', lambda: (OrqProfile('research', 'profile-key', 'https://profile.example', False),)
    )
    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def orq_client(*args: Any, **kwargs: Any) -> object:
        calls.append(('orq', args, kwargs))
        return object()

    def llm_client(**kwargs: Any) -> SimpleNamespace:
        calls.append(('llm', (), kwargs))
        return SimpleNamespace(client=object())

    monkeypatch.setattr(find_cli, 'resolve_orq_client', orq_client)
    monkeypatch.setattr(find_cli, 'resolve_llm_client', llm_client)
    store = FakeStore()
    monkeypatch.setattr(find_cli, 'build_run_store', lambda *args, **kwargs: store)

    if not use_flag:
        save_settings(
            DashboardSettings.model_validate({
                'orq_profile': 'research',
                'orq_project_id': 'project-research',
                'orq_credential_fingerprint': credential_fingerprint('profile-key', 'https://profile.example'),
            }),
            tmp_path / 'settings.json',
        )

    args = ['find', 'refund requests', '--limit', '1']
    if use_flag:
        args.extend(['--profile', 'research'])
    result = CliRunner().invoke(_app(), args)

    assert result.exit_code == 0, result.output
    assert calls == [
        ('orq', ('profile-key',), {'base_url': 'https://profile.example'}),
        (
            'llm',
            (),
            {'extra_api_key': 'profile-key', 'orq_host': 'https://profile.example', 'require_orq': True, 'max_retries': 0},
        ),
    ]
    assert os.environ['ORQ_API_KEY'] == 'environment-key'
    assert os.environ['ORQ_BASE_URL'] == 'https://environment.example'
    assert store.request is not None
    assert store.request.population.facets.project_id == ('project-research' if not use_flag else None)


@pytest.mark.parametrize(
    ('key', 'host', 'expected_project'),
    [
        ('saved-key', 'https://saved.example', 'saved-project'),
        ('different-key', 'https://saved.example', None),
        ('saved-key', 'https://other.example', None),
    ],
)
def test_find_only_reuses_saved_project_for_matching_environment_credentials(
    monkeypatch: Any, tmp_path: Path, key: str, host: str, expected_project: str | None
) -> None:
    from evaluatorq.trace_finder import cli as find_cli
    from evaluatorq.trace_finder.settings import DashboardSettings, credential_fingerprint, save_settings

    save_settings(
        DashboardSettings.model_validate({
            'orq_project_id': 'saved-project',
            'orq_credential_fingerprint': credential_fingerprint('saved-key', 'https://saved.example'),
        }),
        tmp_path / 'settings.json',
    )
    monkeypatch.setenv('ORQ_API_KEY', key)
    monkeypatch.setenv('ORQ_BASE_URL', host)
    monkeypatch.setattr(find_cli, 'resolve_orq_client', lambda: object())
    monkeypatch.setattr(find_cli, 'resolve_llm_client', lambda **_: SimpleNamespace(client=object()))
    store = FakeStore()
    monkeypatch.setattr(find_cli, 'build_run_store', lambda *args, **kwargs: store)

    result = CliRunner().invoke(_app(), ['find', 'refund requests'])

    assert result.exit_code == 0, result.output
    assert store.request is not None
    assert store.request.population.facets.project_id == expected_project


@pytest.mark.parametrize(
    ('profiles', 'expected_error'),
    [
        ((), 'unavailable'),
        ((OrqProfile('research', 'masked***', None, False),), 'masked key'),
    ],
)
def test_find_invalid_explicit_profile_does_not_fall_back_to_environment(
    monkeypatch: Any, profiles: tuple[OrqProfile, ...], expected_error: str
) -> None:
    from evaluatorq.trace_finder import cli as find_cli

    monkeypatch.setenv('ORQ_API_KEY', 'environment-key')
    monkeypatch.setattr(find_cli, 'list_orq_profiles', lambda: profiles)

    def unexpected_client(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError('invalid profile must not fall back to environment credentials')

    monkeypatch.setattr(find_cli, 'resolve_orq_client', unexpected_client)
    result = CliRunner().invoke(_app(), ['find', 'refund requests', '--profile', 'research'])

    assert result.exit_code == 2
    assert expected_error in result.output


def test_dashboard_flags_are_handed_to_reload_worker_environment(monkeypatch: Any, tmp_path: Path) -> None:
    from evaluatorq.dashboard import launch

    names = (
        'EVALUATORQ_COMPILER_MODEL',
        'EVALUATORQ_CLASSIFIER_MODEL',
        'EVALUATORQ_FINDER_WINDOW_DAYS',
        'EVALUATORQ_FINDER_LIMIT',
        'EVALUATORQ_FINDER_PARALLELISM',
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(launch, 'serve', lambda *args, **kwargs: None)

    try:
        result = CliRunner().invoke(
            cli_module.app,
            [
                'dashboard',
                str(tmp_path),
                '--compiler-model',
                'compiler/model',
                '--classifier-model',
                'classifier/model',
                '--window-days',
                '14',
                '--limit',
                '5000',
                '--parallelism',
                '12',
            ],
        )

        assert result.exit_code == 0, result.output
        assert os.environ['EVALUATORQ_COMPILER_MODEL'] == 'compiler/model'
        assert os.environ['EVALUATORQ_CLASSIFIER_MODEL'] == 'classifier/model'
        assert os.environ['EVALUATORQ_FINDER_WINDOW_DAYS'] == '14'
        assert os.environ['EVALUATORQ_FINDER_LIMIT'] == '5000'
        assert os.environ['EVALUATORQ_FINDER_PARALLELISM'] == '12'
    finally:
        for name in names:
            os.environ.pop(name, None)
