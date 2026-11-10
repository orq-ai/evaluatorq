"""With only ``ANTHROPIC_API_KEY`` set, the CLI passes its credential pre-flight and builds an Anthropic client (RES-1875)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from evaluatorq.anthropic import AnthropicLLMClient
from evaluatorq.common.llm_client import resolve_llm_client
from evaluatorq.simulation.cli import _provider_label, _resolve_target


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


def test_the_red_team_preflight_accepts_only_the_anthropic_key() -> None:
    from evaluatorq.redteam import cli as redteam_cli

    result = CliRunner().invoke(redteam_cli.app, ['run', '-t', 'agent:x', '--mode', 'dynamic', '-y'])
    assert 'Missing LLM credentials' not in result.output


def test_the_resolved_client_is_the_adapter() -> None:
    assert isinstance(resolve_llm_client().client, AnthropicLLMClient)
