"""With only ``ANTHROPIC_API_KEY`` set, the CLIs pass their credential pre-flight and build an Anthropic client (RES-1875)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from evaluatorq.anthropic import AnthropicLLMClient
from evaluatorq.common.llm_client import LLM_CREDENTIAL_KEYS, MissingLLMCredentialsError, ModelClientMismatchError
from evaluatorq.simulation.cli import _provider_label, _resolve_target

_REPLY = {
    'id': 'msg_1',
    'type': 'message',
    'role': 'assistant',
    'model': 'claude-sonnet-5-5',
    'content': [{'type': 'text', 'text': 'ok'}],
    'stop_reason': 'end_turn',
    'stop_sequence': None,
    'usage': {'input_tokens': 1, 'output_tokens': 1},
}


def _set_claude_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    for role in ('SMART', 'FAST', 'CLASSIFIER'):
        monkeypatch.setenv(f'EVALUATORQ_{role}_MODEL', 'claude-sonnet-5-5')


@pytest.fixture(autouse=True)
def _anthropic_only(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in LLM_CREDENTIAL_KEYS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-test')


@pytest.mark.parametrize(
    ('keys', 'label'),
    [
        ({'ORQ_API_KEY': 'o'}, 'Orq router'),
        ({'OPENAI_API_KEY': 'o'}, 'OpenAI-compatible'),
        ({'ANTHROPIC_API_KEY': 'a'}, 'Anthropic'),
        ({}, 'unresolved provider'),
    ],
)
def test_the_simulation_provider_label_follows_the_resolved_provider(
    monkeypatch: pytest.MonkeyPatch, keys: dict[str, str], label: str
) -> None:
    for name in LLM_CREDENTIAL_KEYS:
        monkeypatch.delenv(name, raising=False)
    for name, value in keys.items():
        monkeypatch.setenv(name, value)
    assert _provider_label() == label


def test_the_simulation_cli_accepts_a_model_target_with_only_the_anthropic_key() -> None:
    from evaluatorq.redteam.backends.openai import OpenAIModelTarget

    target = _resolve_target(target=None, vercel_url=None, openai_model='claude-sonnet-5-5')
    assert isinstance(target, OpenAIModelTarget)


def test_openai_model_without_any_key_names_all_three_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('ANTHROPIC_API_KEY')
    with pytest.raises(typer.BadParameter) as info:
        _resolve_target(target=None, vercel_url=None, openai_model='claude-sonnet-5-5')
    for key in LLM_CREDENTIAL_KEYS:
        assert key in str(info.value)


_STRUCTURED_ANSWERS: dict[str, dict[str, Any]] = {
    'PersonaListResponse': {
        'personas': [
            {
                'name': 'Pat',
                'patience': 0.5,
                'assertiveness': 0.5,
                'politeness': 0.5,
                'technical_level': 0.5,
                'communication_style': 'casual',
                'background': 'A customer with a late order.',
            }
        ]
    },
    'ScenarioListResponse': {
        'scenarios': [
            {'name': 'Refund', 'goal': 'Get a refund', 'criteria': [{'description': 'Agent helps', 'type': 'must_happen'}]}
        ]
    },
    'JudgeDecision': {
        'reason': 'The agent helped.',
        'should_terminate': True,
        'goal_achieved': True,
        'goal_completion_score': 1.0,
        'criteria_verdicts': [{'criterion_id': 'criteria_0', 'occurred': True, 'evidence': 'ok'}],
    },
}


def _messages_reply(request: Any) -> dict[str, Any]:
    """A Messages reply that fits the request: the schema-shaped answer for a structured call, plain text otherwise."""
    import json

    body = json.loads(request.content)
    schema = ((body.get('output_config') or {}).get('format') or {}).get('schema') or {}
    answer = _STRUCTURED_ANSWERS.get(schema.get('title', ''))
    text = json.dumps(answer) if answer is not None else 'Happy to help with that.'
    return {**_REPLY, 'content': [{'type': 'text', 'text': text}]}


def test_the_simulation_cli_calls_the_messages_api_with_the_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """`eq sim run` builds its client from ANTHROPIC_API_KEY and completes a run whose every call is a Messages call with that key."""
    import json

    import anthropic
    import httpx

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_messages_reply(request))

    real = anthropic.AsyncAnthropic

    def with_mock_transport(**kwargs: Any) -> anthropic.AsyncAnthropic:
        return real(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(anthropic, 'AsyncAnthropic', with_mock_transport)
    _set_claude_roles(monkeypatch)
    from evaluatorq.simulation import cli as sim_cli

    result = CliRunner().invoke(
        sim_cli.app,
        [
            'run',
            '--agent-description',
            'a support bot',
            '--openai-model',
            'claude-opus-5-5',
            '--num-personas',
            '1',
            '--num-scenarios',
            '1',
            '--max-turns',
            '1',
            '--no-recommendations',
            '--no-executive-summary',
            '-y',
            '--no-save',
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen, f'no HTTP request reached the Anthropic transport: {result.output}'
    assert {request.url.path for request in seen} == {'/v1/messages'}
    assert {request.headers['x-api-key'] for request in seen} == {'sk-ant-test'}
    # The roles run on claude-sonnet-5-5, so a claude-opus-5-5 request is the target's.
    assert any(json.loads(request.content)['model'] == 'claude-opus-5-5' for request in seen), 'the target was never called'


def _run_sim_with_mock_messages(monkeypatch: pytest.MonkeyPatch, *args: str) -> tuple[Any, list[Any]]:
    import anthropic
    import httpx

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_messages_reply(request))

    real = anthropic.AsyncAnthropic

    def with_mock_transport(**kwargs: Any) -> anthropic.AsyncAnthropic:
        return real(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(anthropic, 'AsyncAnthropic', with_mock_transport)
    from evaluatorq.simulation import cli as sim_cli

    base = ['run', '--agent-description', 'a support bot', '--num-personas', '1', '--num-scenarios', '1']
    tail = ['--max-turns', '1', '--no-recommendations', '--no-executive-summary', '-y', '--no-save']
    return CliRunner().invoke(sim_cli.app, [*base, *args, *tail]), seen


def test_an_explicit_sim_model_satisfies_the_pre_flight_without_any_role_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The judge runs on --sim-model at runtime, so the pre-flight must not demand EVALUATORQ_SMART_MODEL as well."""
    for role in ('SMART', 'FAST', 'CLASSIFIER'):
        monkeypatch.delenv(f'EVALUATORQ_{role}_MODEL', raising=False)
    result, seen = _run_sim_with_mock_messages(
        monkeypatch, '--openai-model', 'claude-sonnet-5-5', '--sim-model', 'claude-sonnet-5-5'
    )
    assert result.exit_code == 0, result.output
    assert seen


def test_a_target_model_the_anthropic_key_cannot_serve_stops_the_simulation_cli_before_any_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_claude_roles(monkeypatch)
    result, seen = _run_sim_with_mock_messages(monkeypatch, '--openai-model', 'openai/gpt-5-mini')
    assert result.exit_code == 1, result.output
    assert isinstance(result.exception, SystemExit)
    assert '--openai-model=openai/gpt-5-mini' in result.output
    assert not seen


def test_a_stray_anthropic_key_with_the_default_roles_stops_the_simulation_cli_before_any_work() -> None:
    from evaluatorq.simulation import cli as sim_cli

    result = CliRunner().invoke(
        sim_cli.app,
        ['run', '--agent-description', 'a support bot', '--openai-model', 'claude-sonnet-5-5', '-y', '--no-save'],
    )
    assert result.exit_code == 1, result.output
    assert 'Using for generations' not in result.output
    assert 'ANTHROPIC_API_KEY is the only LLM key set' in result.output
    assert 'EVALUATORQ_FAST_MODEL' in result.output
    assert 'EVALUATORQ_SMART_MODEL' in result.output
    assert isinstance(result.exception, SystemExit)


def test_a_stray_anthropic_key_with_the_default_roles_stops_the_generate_command(tmp_path: Path) -> None:
    from evaluatorq.simulation import cli as sim_cli

    result = CliRunner().invoke(sim_cli.app, ['generate', '-d', str(tmp_path / 'dp.jsonl'), '--agent-description', 'a support bot']
    )
    assert result.exit_code == 1, result.output
    assert 'ANTHROPIC_API_KEY is the only LLM key set' in result.output
    assert 'Generate Plan' not in result.output


def test_the_red_team_pre_flight_passes_with_only_the_anthropic_key_and_claude_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluatorq.redteam.contracts import LLMConfig, Pipeline
    from evaluatorq.redteam.runner import _validate_credentials_and_deps

    _set_claude_roles(monkeypatch)
    _validate_credentials_and_deps(
        config=LLMConfig(), llm_client=None, replay=None, resolved_mode=Pipeline.DYNAMIC, dataset=None
    )


def test_the_red_team_pre_flight_names_the_roles_a_stray_anthropic_key_cannot_serve() -> None:
    from evaluatorq.redteam.contracts import LLMConfig, Pipeline
    from evaluatorq.redteam.runner import _validate_credentials_and_deps

    with pytest.raises(ModelClientMismatchError) as info:
        _validate_credentials_and_deps(
            config=LLMConfig(), llm_client=None, replay=None, resolved_mode=Pipeline.DYNAMIC, dataset=None
        )
    assert 'ANTHROPIC_API_KEY is the only LLM key set' in str(info.value)
    assert 'EVALUATORQ_SMART_MODEL (unset, built-in default) (redteam.attacker)' in str(info.value)
    assert 'EVALUATORQ_SMART_MODEL (unset, built-in default) (redteam.evaluator)' in str(info.value)
    # max_backtracks is 0 by default, so the refusal judge never runs and its model is not checked.
    assert 'CLASSIFIER' not in str(info.value)
    assert 'refusal_judge' not in str(info.value)


def _red_team_pre_flight(config: Any, *, mode: str = 'dynamic', llm_client: Any = None) -> None:
    from evaluatorq.redteam.contracts import Pipeline
    from evaluatorq.redteam.runner import _validate_credentials_and_deps

    _validate_credentials_and_deps(
        config=config, llm_client=llm_client, replay=None, resolved_mode=Pipeline(mode), dataset=None
    )


def _smart_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('EVALUATORQ_SMART_MODEL', 'claude-sonnet-5-5')


def test_the_red_team_pre_flight_checks_the_fast_model_the_refusal_judge_falls_back_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Off the Orq router the classify-only classifier default is replaced by the fast role, so that is the model to check."""
    from evaluatorq.redteam.contracts import LLMConfig

    _smart_claude(monkeypatch)
    with pytest.raises(ModelClientMismatchError) as info:
        _red_team_pre_flight(LLMConfig(max_backtracks=2))
    message = str(info.value)
    assert 'EVALUATORQ_FAST_MODEL (unset, built-in default) (redteam.refusal_judge' in message
    assert 'EVALUATORQ_CLASSIFIER_MODEL' not in message

    monkeypatch.setenv('EVALUATORQ_FAST_MODEL', 'claude-haiku-5-5')
    _red_team_pre_flight(LLMConfig(max_backtracks=2))


def test_the_red_team_pre_flight_skips_the_refusal_judge_on_a_static_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.redteam.contracts import LLMConfig

    _smart_claude(monkeypatch)
    _red_team_pre_flight(LLMConfig(max_backtracks=2), mode='static')


def test_the_red_team_pre_flight_checks_replacement_judges(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.redteam.contracts import EvaluatorConfig, LLMConfig

    _smart_claude(monkeypatch)
    config = LLMConfig(evaluator=EvaluatorConfig(replacement_judges=['openai/gpt-6-sol']))
    with pytest.raises(ModelClientMismatchError, match=r'LLMConfig\.evaluator\.replacement_judges\[0\]=openai/gpt-6-sol'):
        _red_team_pre_flight(config)


def test_the_red_team_pre_flight_checks_the_role_without_a_client_when_the_other_has_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openai import AsyncOpenAI

    from evaluatorq.redteam.contracts import EvaluatorConfig, LLMCallConfig, LLMConfig

    client = AsyncOpenAI(api_key='sk-test', max_retries=0)
    with_attacker_client = LLMConfig(attacker=LLMCallConfig(model='openai/gpt-6-sol', client=client))
    with pytest.raises(ModelClientMismatchError) as info:
        _red_team_pre_flight(with_attacker_client)
    assert '(redteam.evaluator)' in str(info.value)
    assert 'attacker' not in str(info.value)

    with_evaluator_client = LLMConfig(evaluator=EvaluatorConfig(client=client))
    with pytest.raises(ModelClientMismatchError) as info:
        _red_team_pre_flight(with_evaluator_client)
    assert '(redteam.attacker)' in str(info.value)
    assert 'evaluator' not in str(info.value)


def test_the_red_team_pre_flight_names_where_each_model_came_from(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.common import model_roles
    from evaluatorq.redteam.contracts import LLMCallConfig, LLMConfig

    model_roles.set_cli_models(overrides={'redteam.evaluator': 'openai/gpt-6-luna'})
    config = LLMConfig(attacker=LLMCallConfig(model='openai/gpt-5-mini'))
    with pytest.raises(ModelClientMismatchError) as info:
        _red_team_pre_flight(config)
    message = str(info.value)
    assert 'LLMConfig.attacker.model (--attack-model)=openai/gpt-5-mini' in message
    assert '--model-override redteam.evaluator (redteam.evaluator)=openai/gpt-6-luna' in message
    assert 'EVALUATORQ_SMART_MODEL' not in message


def test_the_red_team_pre_flight_leaves_an_injected_client_alone() -> None:
    """An injected client is the caller's choice: the pre-flight neither demands a key nor checks the default roles."""
    from openai import AsyncOpenAI

    from evaluatorq.redteam.contracts import LLMConfig

    _red_team_pre_flight(LLMConfig(), llm_client=AsyncOpenAI(api_key='sk-test', max_retries=0))


def test_the_red_team_pre_flight_raises_credential_error_with_no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.redteam.contracts import LLMConfig, Pipeline
    from evaluatorq.redteam.exceptions import CredentialError
    from evaluatorq.redteam.runner import _validate_credentials_and_deps

    monkeypatch.delenv('ANTHROPIC_API_KEY')
    with pytest.raises(CredentialError, match='ANTHROPIC_API_KEY'):
        _validate_credentials_and_deps(
            config=LLMConfig(), llm_client=None, replay=None, resolved_mode=Pipeline.DYNAMIC, dataset=None
        )


def test_the_red_team_cli_path_resolves_an_anthropic_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no injected client, `eq redteam run` leaves the attacker client to the runner, which builds the adapter."""
    from evaluatorq.redteam import cli as redteam_cli
    from evaluatorq.redteam.exceptions import CancelledError
    from evaluatorq.redteam.runner import _resolve_attacker_llm_client

    resolved: list[object] = []

    async def fake_red_team(**kwargs: Any) -> None:
        resolved.append(
            _resolve_attacker_llm_client(
                kwargs.get('llm_client'),
                kwargs.get('pipeline_config'),
                create_if_missing=True,
            )
        )
        raise CancelledError('stop')

    monkeypatch.setattr('evaluatorq.redteam.red_team', fake_red_team)
    result = CliRunner().invoke(redteam_cli.app, ['run', '-t', 'agent:x', '--mode', 'dynamic', '-y'])
    assert result.exit_code == 0, result.output
    assert len(resolved) == 1
    assert isinstance(resolved[0], AnthropicLLMClient)


def test_a_missing_anthropic_extra_on_the_red_team_path_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.redteam.backends.registry import create_async_llm_client
    from evaluatorq.redteam.exceptions import CredentialError

    monkeypatch.setitem(sys.modules, 'anthropic', None)
    with pytest.raises(CredentialError) as info:
        create_async_llm_client()
    assert 'evaluatorq[anthropic]' in str(info.value)
    assert isinstance(info.value.__cause__, MissingLLMCredentialsError)
    assert isinstance(info.value.__cause__.__cause__, ImportError)


def test_a_missing_anthropic_extra_on_the_simulation_cli_is_one_line_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evaluatorq.simulation import cli as sim_cli

    monkeypatch.setitem(sys.modules, 'anthropic', None)
    _set_claude_roles(monkeypatch)
    result = CliRunner().invoke(
        sim_cli.app,
        ['run', '--agent-description', 'a support bot', '--openai-model', 'claude-sonnet-5-5', '-y', '--no-save'],
    )
    assert result.exit_code != 0
    assert 'evaluatorq[anthropic]' in result.output
    assert isinstance(result.exception, SystemExit)
