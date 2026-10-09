from __future__ import annotations

import io
import json
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
    DimensionAnswer,
    FacetSelection,
    NumericFilters,
    RunSnapshot,
    TraceClassification,
    ThresholdSelection,
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
            name='Refund',
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
            answers=(DimensionAnswer(value='yes', confidence=0.94, matched=True),),
            matched=True,
            raw_result={'value': 'yes'},
        )
        return RunSnapshot(
            state='completed',
            request=request,
            dimensions=(compiled,),
            trace_ids=(trace.trace_id,),
            traces=(trace,),
            results={trace.trace_id: result},
            total=1,
            completed=1,
            matched=1,
        )


class ReviewStore(FakeStore):
    """Stops at the review gate like the real store, then classifies on start."""

    def __init__(self, **review: Any) -> None:
        super().__init__()
        self.review = review
        self.reviewed: RunSnapshot | None = None
        self.compiles = 0
        self.started: list[tuple[Any, tuple[CompiledQuery, ...]]] = []

    async def compile(self, request: Any, *, wait: bool = True) -> RunSnapshot:
        self.compiles += 1
        completed = await super().compile(request, wait=wait)
        facets = request.population.facets.model_copy(update={'model': frozenset({'gpt-5.6-luna'})})
        merged = request.model_copy(update={'population': request.population.model_copy(update={'facets': facets})})
        review: dict[str, Any] = {
            'state': 'awaiting_review',
            'request': merged,
            'results': {},
            'completed': 0,
            'matched': 0,
            'explicit_filters': FacetSelection(project=frozenset({'support-agent'})),
            'explicit_numeric': NumericFilters(tokens_min=100),
            'generated_filters': FacetSelection(model=frozenset({'gpt-5.6-luna'})),
        }
        self.reviewed = replace(completed, **{**review, **self.review})
        return self.reviewed

    async def start(self, request: Any, dimensions: Any, *, wait: bool = True) -> RunSnapshot:
        assert wait is False
        self.started.append((request, tuple(dimensions)))
        return await FakeStore.compile(self, request, wait=False)


def _use_review_store(monkeypatch: Any, store: ReviewStore, *, tty: bool) -> None:
    from evaluatorq.trace_finder import cli as find_cli

    monkeypatch.setattr(find_cli, 'resolve_orq_client', lambda: object())
    monkeypatch.setattr(find_cli, 'resolve_llm_client', lambda **_: SimpleNamespace(client=object()))
    monkeypatch.setattr(find_cli, 'build_run_store', lambda *args, **kwargs: store)
    if tty:
        monkeypatch.setattr(find_cli, 'should_skip_confirm', lambda yes: yes)


def test_find_shows_plan_and_classifies_without_prompt_when_not_a_tty(monkeypatch: Any) -> None:
    store = ReviewStore()
    _use_review_store(monkeypatch, store, tty=False)

    result = CliRunner().invoke(
        _app(), ['find', 'refund requests', '--project', 'support-agent', '--tokens-min', '100']
    )

    assert result.exit_code == 0, result.output
    assert store.request is not None
    assert store.request.mode == 'review'
    assert 'Find plan' in result.output
    assert 'project: support-agent; tokens min: 100' in result.output
    assert 'model: gpt-5.6-luna' in result.output
    assert 'Does the trace mention a refund?' in result.output
    assert 'Included when: Verdict yes' in result.output
    assert 'Classify 1 trace?' not in result.output
    assert 'Matched traces' in result.output


def test_find_starts_the_reviewed_plan_without_recompiling(monkeypatch: Any) -> None:
    store = ReviewStore()
    _use_review_store(monkeypatch, store, tty=False)

    result = CliRunner().invoke(_app(), ['find', 'refund requests'])

    assert result.exit_code == 0, result.output
    assert store.compiles == 1
    assert store.reviewed is not None
    assert len(store.started) == 1
    request, dimensions = store.started[0]
    assert request == store.reviewed.request
    assert request.population.facets.model == frozenset({'gpt-5.6-luna'})
    assert [dimension.name for dimension in dimensions] == ['Refund']


def test_find_classifies_after_confirming_the_plan(monkeypatch: Any) -> None:
    store = ReviewStore()
    _use_review_store(monkeypatch, store, tty=True)

    result = CliRunner().invoke(_app(), ['find', 'refund requests'], input='y\n')

    assert result.exit_code == 0, result.output
    assert 'Classify 1 trace?' in result.output
    assert len(store.started) == 1
    assert 'Matched traces' in result.output


def test_find_declining_the_plan_classifies_nothing(monkeypatch: Any, tmp_path: Path) -> None:
    store = ReviewStore()
    _use_review_store(monkeypatch, store, tty=True)
    output = tmp_path / 'declined.json'

    result = CliRunner().invoke(_app(), ['find', 'refund requests', '--json', str(output)], input='n\n')

    assert result.exit_code == 0, result.output
    assert store.started == []
    assert 'no traces were classified' in result.output
    assert not output.exists()


def test_find_yes_skips_the_prompt(monkeypatch: Any) -> None:
    store = ReviewStore()
    _use_review_store(monkeypatch, store, tty=True)

    result = CliRunner().invoke(_app(), ['find', 'refund requests', '-y'])

    assert result.exit_code == 0, result.output
    assert 'Find plan' in result.output
    assert 'Classify 1 trace?' not in result.output
    assert len(store.started) == 1


def test_find_filter_only_plan_skips_the_prompt(monkeypatch: Any) -> None:
    store = ReviewStore(dimensions=())
    _use_review_store(monkeypatch, store, tty=True)

    result = CliRunner().invoke(_app(), ['find', 'gpt-5.6-luna traces'])

    assert result.exit_code == 0, result.output
    assert 'Classify 1 trace?' not in result.output
    assert len(store.started) == 1
    assert store.started[0][1] == ()


def test_find_plan_shows_window_threshold_rule_and_reached_limit(monkeypatch: Any) -> None:
    scored = CompiledQuery(
        name='Frustration',
        task=ClassifyQuestion(
            kind='score',
            instructions='How frustrated is the user?',
            criteria=['Calm.', 'Annoyed.', 'Very frustrated.'],
            state={},
        ),
        selection=ThresholdSelection(kind='threshold', operator='gte', value=0.7),
    )
    store = ReviewStore(dimensions=(scored,), total=1)
    _use_review_store(monkeypatch, store, tty=False)

    result = CliRunner().invoke(_app(), ['find', 'frustrated users', '--window-days', '3', '--limit', '1'])

    assert result.exit_code == 0, result.output
    assert 'Window' in result.output
    assert ' UTC' in result.output
    assert 'Included when: Score at least 0.7' in result.output
    assert '(the trace limit; more traces may match)' in result.output


def test_find_plan_shows_warnings_and_escapes_markup(monkeypatch: Any) -> None:
    store = ReviewStore(
        filter_selection_error='Facet lookup failed [/]',
        plan_warning='Ranking needs aggregation.',
    )
    _use_review_store(monkeypatch, store, tty=False)

    result = CliRunner().invoke(_app(), ['find', 'refund [v2] requests'])

    assert result.exit_code == 0, result.output
    assert 'Facet lookup failed [/]' in result.output
    assert 'Ranking needs aggregation.' in result.output
    assert 'refund [v2] requests' in result.output


def test_find_failed_planning_never_starts(monkeypatch: Any) -> None:
    store = ReviewStore(state='failed', error='live Orq trace loading failed')
    _use_review_store(monkeypatch, store, tty=True)

    result = CliRunner().invoke(_app(), ['find', 'refund requests'])

    assert result.exit_code == 1
    assert 'live Orq trace loading failed' in result.output
    assert 'Find plan' not in result.output
    assert store.started == []


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


def test_find_prefers_failed_count_over_plan_warning_on_completed_run(monkeypatch: Any, tmp_path: Path) -> None:
    from evaluatorq.trace_finder import cli as find_cli

    class WarnedStore(FakeStore):
        async def compile(self, request: Any, *, wait: bool = True) -> RunSnapshot:
            snapshot = await super().compile(request, wait=wait)
            return replace(snapshot, state='completed', error=None, failed=2, plan_warning='Words not covered: foo')

    monkeypatch.setattr(find_cli, 'resolve_orq_client', lambda: object())
    monkeypatch.setattr(find_cli, 'resolve_llm_client', lambda **_: SimpleNamespace(client=object()))
    monkeypatch.setattr(find_cli, 'build_run_store', lambda *args, **kwargs: WarnedStore())

    result = CliRunner().invoke(_app(), ['find', 'refund requests'])

    assert result.exit_code == 1
    assert '2 failed classifications' in result.output
    assert 'Words not covered' not in result.output


def test_find_prints_plan_warning_when_ask_ai_cannot_answer(monkeypatch: Any, tmp_path: Path) -> None:
    from evaluatorq.trace_finder import cli as find_cli

    reason = "Ask AI finds traces; it can't compute totals, averages or rankings. Ranking needs aggregation."

    class CancelledStore(FakeStore):
        async def compile(self, request: Any, *, wait: bool = True) -> RunSnapshot:
            snapshot = await super().compile(request, wait=wait)
            return replace(snapshot, state='cancelled', error=None, plan_warning=reason)

    monkeypatch.setattr(find_cli, 'resolve_orq_client', lambda: object())
    monkeypatch.setattr(find_cli, 'resolve_llm_client', lambda **_: SimpleNamespace(client=object()))
    monkeypatch.setattr(find_cli, 'build_run_store', lambda *args, **kwargs: CancelledStore())

    result = CliRunner().invoke(_app(), ['find', 'which model costs most?'])

    assert result.exit_code == 1
    assert "can't compute totals" in result.output
    assert 'ended in cancelled' not in result.output


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
    assert 'judged' not in result.output
    assert output.exists()
    assert 'trace-cli-1' in output.read_text()
    assert store.request is not None
    assert store.request.population.facets.project == frozenset({'support-agent'})
    assert store.request.population.numeric.tokens_min == 100


def test_find_positive_only_filters_json_and_keeps_summary(monkeypatch: Any, tmp_path: Path) -> None:
    from evaluatorq.trace_finder import cli as find_cli

    class MixedStore(FakeStore):
        async def compile(self, request: Any, *, wait: bool = True) -> RunSnapshot:
            snapshot = await super().compile(request, wait=wait)
            negative = _trace().model_copy(update={'trace_id': 'trace-cli-negative'})
            return replace(
                snapshot,
                trace_ids=(*snapshot.trace_ids, negative.trace_id),
                traces=(*snapshot.traces, negative),
                results={
                    **snapshot.results,
                    negative.trace_id: TraceClassification(
                        trace_id=negative.trace_id,
                        span_id=negative.span_id,
                        answers=(DimensionAnswer(value='no', matched=False),),
                        matched=False,
                        raw_result={'value': 'no'},
                    ),
                },
                total=2,
                completed=2,
            )

    monkeypatch.setattr(find_cli, 'resolve_orq_client', lambda: object())
    monkeypatch.setattr(find_cli, 'resolve_llm_client', lambda **_: SimpleNamespace(client=object()))
    monkeypatch.setattr(find_cli, 'build_run_store', lambda *args, **kwargs: MixedStore())
    output = tmp_path / 'positive.json'

    result = CliRunner().invoke(_app(), ['find', 'refund requests', '--positive-only', '--json', str(output)])

    assert result.exit_code == 0, result.output
    assert 'trace-cli-1' in result.output
    assert 'trace-cli-negative' not in result.output
    assert 'Summary: 1 matched of 2; 0 failed.' in result.output
    exported = json.loads(output.read_text())
    assert [trace['trace_id'] for trace in exported['traces']] == ['trace-cli-1']
    assert exported['counts']['total'] == 2
    assert exported['counts']['matched'] == 1


def test_find_debug_flag_shows_progress(monkeypatch: Any) -> None:
    from evaluatorq.trace_finder import cli as find_cli

    monkeypatch.setattr(find_cli, 'resolve_orq_client', lambda: object())
    monkeypatch.setattr(find_cli, 'resolve_llm_client', lambda **_: SimpleNamespace(client=object()))
    monkeypatch.setattr(find_cli, 'build_run_store', lambda *args, **kwargs: FakeStore())

    result = CliRunner().invoke(_app(), ['find', 'refund requests', '--debug'])

    assert result.exit_code == 0, result.output
    assert result.output.count('1/1 judged') == 1


@pytest.mark.asyncio
@pytest.mark.slow
async def test_find_debug_progress_skips_unchanged_polls() -> None:
    from evaluatorq.trace_finder import cli as find_cli
    from evaluatorq.trace_finder.debug import cli_debug

    class PollStore:
        polls = 0

        async def compile(self, request: Any, *, wait: bool = True) -> RunSnapshot:
            return RunSnapshot(state='compiling')

        async def snapshot(self) -> RunSnapshot:
            self.polls += 1
            return RunSnapshot(state='completed', total=1, completed=1) if self.polls == 3 else RunSnapshot(state='compiling')

    output = io.StringIO()
    with cli_debug(active=True):
        await find_cli._run(PollStore(), cast(Any, object()), Console(file=output))

    assert output.getvalue().count('COMPILING') == 1
    assert output.getvalue().count('COMPLETED') == 1


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
    assert store.request.population.facets.project_id is None


@pytest.mark.parametrize(
    ('key', 'host'),
    [
        ('saved-key', 'https://saved.example'),
        ('different-key', 'https://saved.example'),
        ('saved-key', 'https://other.example'),
    ],
)
def test_find_ignores_legacy_saved_project_for_environment_credentials(
    monkeypatch: Any, tmp_path: Path, key: str, host: str
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
    assert store.request.population.facets.project_id is None


def test_find_ignores_dashboard_oauth_project_with_environment_credentials(
    monkeypatch: Any, tmp_path: Path
) -> None:
    from evaluatorq.trace_finder import cli as find_cli
    from evaluatorq.trace_finder.settings import DashboardSettings, save_settings

    save_settings(DashboardSettings.model_validate({
        'orq_auth_method': 'cli_oauth', 'orq_oauth_server': 'https://oauth.example',
        'orq_workspace': 'oauth-workspace', 'orq_project_id': 'oauth-project',
    }), tmp_path / 'settings.json')
    monkeypatch.setenv('ORQ_API_KEY', 'environment-key')
    monkeypatch.setattr(find_cli, 'resolve_orq_client', lambda: object())
    monkeypatch.setattr(find_cli, 'resolve_llm_client', lambda **_: SimpleNamespace(client=object()))
    store = FakeStore()
    monkeypatch.setattr(find_cli, 'build_run_store', lambda *args, **kwargs: store)

    result = CliRunner().invoke(_app(), ['find', 'refund requests'])

    assert result.exit_code == 0, result.output
    assert store.request is not None
    assert store.request.population.facets.project_id is None


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
        'EVALUATORQ_MODEL_OVERRIDES',
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
        assert json.loads(os.environ['EVALUATORQ_MODEL_OVERRIDES']) == {
            'roles': {'classifier': 'classifier/model'},
            'overrides': {'finder.compiler': 'compiler/model'},
        }
        assert 'EVALUATORQ_COMPILER_MODEL' not in os.environ
        assert os.environ['EVALUATORQ_FINDER_WINDOW_DAYS'] == '14'
        assert os.environ['EVALUATORQ_FINDER_LIMIT'] == '5000'
        assert os.environ['EVALUATORQ_FINDER_PARALLELISM'] == '12'
    finally:
        for name in names:
            os.environ.pop(name, None)
