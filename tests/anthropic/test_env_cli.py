"""With only ``ANTHROPIC_API_KEY`` set, the CLI passes its credential pre-flight and builds an Anthropic client (RES-1875)."""

from __future__ import annotations

import sys
from typing import Any

import pytest
from typer.testing import CliRunner

from evaluatorq.anthropic import AnthropicLLMClient
from evaluatorq.common.llm_client import resolve_llm_client
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


@pytest.fixture(autouse=True)
def _anthropic_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('ORQ_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'sk-ant-test')


def test_simulation_provider_label_names_anthropic() -> None:
    assert _provider_label() == 'Anthropic'


def test_the_simulation_cli_accepts_a_model_target_with_only_the_anthropic_key() -> None:
    from evaluatorq.redteam.backends.openai import OpenAIModelTarget

    target = _resolve_target(target=None, vercel_url=None, openai_model='claude-sonnet-5-5')
    assert isinstance(target, OpenAIModelTarget)


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
    for role in ('SMART', 'FAST', 'CLASSIFIER'):
        monkeypatch.setenv(f'EVALUATORQ_{role}_MODEL', 'claude-sonnet-5-5')
    from evaluatorq.simulation import cli as sim_cli

    CliRunner().invoke(
        sim_cli.app,
        ['run', '--agent-description', 'a support bot', '--openai-model', 'claude-sonnet-5-5', '-y', '--no-save'],
    )
    assert seen, 'no HTTP request reached the Anthropic transport'
    assert seen[0].url.path == '/v1/messages'
    assert seen[0].headers['x-api-key'] == 'sk-ant-test'


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
    assert 'Missing LLM credentials' not in result.output
    assert len(resolved) == 1
    assert isinstance(resolved[0], AnthropicLLMClient)


def test_a_missing_anthropic_extra_names_the_extra_not_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.redteam.backends.registry import create_async_llm_client
    from evaluatorq.redteam.exceptions import BackendError

    monkeypatch.setitem(sys.modules, 'anthropic', None)
    with pytest.raises(BackendError) as info:
        create_async_llm_client()
    assert 'evaluatorq[anthropic]' in str(info.value)
    assert 'uv add openai' not in str(info.value)


def test_the_resolved_client_is_the_adapter() -> None:
    assert isinstance(resolve_llm_client().client, AnthropicLLMClient)
