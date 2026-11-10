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


def test_the_simulation_cli_calls_the_messages_api_with_the_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """`eq sim run` builds its client from ANTHROPIC_API_KEY: the first HTTP request is a Messages call carrying that key."""
    import anthropic
    import httpx

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_REPLY)

    real = anthropic.AsyncAnthropic

    def with_mock_transport(**kwargs: Any) -> anthropic.AsyncAnthropic:
        return real(**kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(anthropic, 'AsyncAnthropic', with_mock_transport)
    _set_claude_roles(monkeypatch)
    from evaluatorq.simulation import cli as sim_cli

    result = CliRunner().invoke(
        sim_cli.app,
        ['run', '--agent-description', 'a support bot', '--openai-model', 'claude-sonnet-5-5', '-y', '--no-save'],
    )
    assert seen, f'no HTTP request reached the Anthropic transport: {result.output}'
    assert seen[0].url.path == '/v1/messages'
    assert seen[0].headers['x-api-key'] == 'sk-ant-test'
    assert not isinstance(result.exception, (ModelClientMismatchError, ImportError)), result.output


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
    assert 'Traceback' not in result.output


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
    assert 'EVALUATORQ_SMART_MODEL' in str(info.value)
    assert 'EVALUATORQ_CLASSIFIER_MODEL' in str(info.value)


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
    assert 'Traceback' not in result.output
