"""BaseAgent must actually read LLMCallConfig, not just .model/.api/.client/.retry_count.

Regression coverage for the silent drop: temperature/max_tokens/timeout_ms/
reasoning_effort/extra_kwargs set on an agent's `LLMCallConfig` used to never
reach the wire. These tests assert each field reaches the actual call params,
and that an explicit config value beats both the per-call-site literal
default and the env-var fallback.
"""

from __future__ import annotations

# ruff: noqa: S101, SLF001
from unittest.mock import AsyncMock, MagicMock

import pytest

from evaluatorq.contracts import LLMCallConfig
from evaluatorq.simulation.agents.base import AgentConfig, BaseAgent
from evaluatorq.simulation.agents.judge import JudgeAgentConfig
from evaluatorq.simulation.agents.user_simulator import UserSimulatorAgentConfig
from evaluatorq.simulation.types import Message


class _ConcreteAgent(BaseAgent):
    @property
    def name(self) -> str:
        return "TestAgent"

    @property
    def system_prompt(self) -> str:
        return "You are a test agent."


def _make_client() -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock()
    return client


def _chat_response(content: str | None = "hi") -> MagicMock:
    mock_message = MagicMock()
    mock_message.content = content
    mock_message.tool_calls = None
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = None
    return mock_response


def _responses_response() -> MagicMock:
    ok = MagicMock()
    ok.output = []
    ok.usage = None
    return ok


def _messages() -> list[Message]:
    return [Message(role="user", content="hello")]


@pytest.fixture(autouse=True)
def _clean_reasoning_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolate from a developer's real env and from the module default ("medium").
    monkeypatch.delenv("EVALUATORQ_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("EVALUATORQ_LLM_MAX_TOKENS", raising=False)
    monkeypatch.delenv("EVALUATORQ_LLM_TIMEOUT_S", raising=False)


class TestChatCompletionsConfigPropagation:
    @pytest.mark.asyncio
    async def test_config_temperature_beats_call_site_literal(self):
        client = _make_client()
        client.chat.completions.create.return_value = _chat_response()
        config = LLMCallConfig(model="gpt-4o", api="chat_completions", client=client, temperature=0.2)
        agent = _ConcreteAgent(config)

        # Call-site literal of 0.0 (mirrors JudgeAgent.evaluate) must lose to config.
        await agent._call_llm(_messages(), temperature=0.0)

        kwargs = client.chat.completions.create.await_args.kwargs
        assert kwargs["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_config_max_tokens_reaches_call_params(self):
        client = _make_client()
        client.chat.completions.create.return_value = _chat_response()
        config = LLMCallConfig(model="gpt-4o", api="chat_completions", client=client, max_tokens=20_000)
        agent = _ConcreteAgent(config)

        await agent._call_llm(_messages())

        kwargs = client.chat.completions.create.await_args.kwargs
        assert kwargs["max_tokens"] == 20_000

    @pytest.mark.asyncio
    async def test_config_reasoning_effort_reaches_call_params(self):
        client = _make_client()
        client.chat.completions.create.return_value = _chat_response()
        config = LLMCallConfig(model="gpt-4o", api="chat_completions", client=client, reasoning_effort="high")
        agent = _ConcreteAgent(config)

        await agent._call_llm(_messages())

        kwargs = client.chat.completions.create.await_args.kwargs
        assert kwargs["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_config_extra_kwargs_reach_call_params(self):
        client = _make_client()
        client.chat.completions.create.return_value = _chat_response()
        config = LLMCallConfig(
            model="gpt-4o", api="chat_completions", client=client, extra_kwargs={"top_p": 0.5}
        )
        agent = _ConcreteAgent(config)

        await agent._call_llm(_messages())

        kwargs = client.chat.completions.create.await_args.kwargs
        assert kwargs["top_p"] == 0.5

    @pytest.mark.asyncio
    async def test_call_site_literal_wins_when_config_unset(self):
        """No explicit config.temperature -> the call-site literal (e.g. judge's 0.0) applies."""
        client = _make_client()
        client.chat.completions.create.return_value = _chat_response()
        config = LLMCallConfig(model="gpt-4o", api="chat_completions", client=client)
        agent = _ConcreteAgent(config)

        await agent._call_llm(_messages(), temperature=0.0)

        kwargs = client.chat.completions.create.await_args.kwargs
        assert kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_env_reasoning_effort_applies_when_config_and_call_site_are_silent(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("EVALUATORQ_REASONING_EFFORT", "low")
        client = _make_client()
        client.chat.completions.create.return_value = _chat_response()
        config = LLMCallConfig(model="gpt-4o", api="chat_completions", client=client)
        agent = _ConcreteAgent(config)

        await agent._call_llm(_messages())

        kwargs = client.chat.completions.create.await_args.kwargs
        assert kwargs["reasoning_effort"] == "low"

    @pytest.mark.asyncio
    async def test_config_reasoning_effort_none_opts_out_of_env_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("EVALUATORQ_REASONING_EFFORT", "medium")
        client = _make_client()
        client.chat.completions.create.return_value = _chat_response()
        # Explicitly set to None -- must win over the env fallback.
        config = LLMCallConfig(model="gpt-4o", api="chat_completions", client=client, reasoning_effort=None)
        agent = _ConcreteAgent(config)

        await agent._call_llm(_messages())

        kwargs = client.chat.completions.create.await_args.kwargs
        assert "reasoning_effort" not in kwargs


class TestResponsesConfigPropagation:
    @pytest.mark.asyncio
    async def test_config_max_tokens_reaches_call_params(self):
        client = _make_client()
        client.responses.create.return_value = _responses_response()
        config = LLMCallConfig(model="gpt-4o", api="responses", client=client, max_tokens=20_000)
        agent = _ConcreteAgent(config)

        await agent._call_llm(_messages())

        kwargs = client.responses.create.await_args.kwargs
        assert kwargs["max_output_tokens"] == 20_000

    @pytest.mark.asyncio
    async def test_config_timeout_ms_reaches_effective_timeout(self):
        client = _make_client()
        client.responses.create.return_value = _responses_response()
        config = LLMCallConfig(model="gpt-4o", api="responses", client=client, timeout_ms=300_000)
        agent = _ConcreteAgent(config)

        assert agent._resolved_timeout_s(None) == 300.0

    @pytest.mark.asyncio
    async def test_config_reasoning_effort_reaches_call_params(self):
        client = _make_client()
        client.responses.create.return_value = _responses_response()
        config = LLMCallConfig(model="gpt-4o", api="responses", client=client, reasoning_effort="high")
        agent = _ConcreteAgent(config)

        await agent._call_llm(_messages())

        kwargs = client.responses.create.await_args.kwargs
        assert kwargs["reasoning"] == {"effort": "high"}

    @pytest.mark.asyncio
    async def test_config_extra_kwargs_override_reasoning(self):
        client = _make_client()
        client.responses.create.return_value = _responses_response()
        config = LLMCallConfig(
            model="gpt-4o",
            api="responses",
            client=client,
            reasoning_effort="high",
            extra_kwargs={"reasoning": {"effort": "low"}},
        )
        agent = _ConcreteAgent(config)

        await agent._call_llm(_messages())

        kwargs = client.responses.create.await_args.kwargs
        assert kwargs["reasoning"] == {"effort": "low"}


class TestLegacyAgentConfigThreading:
    """AgentConfig subclasses (JudgeAgentConfig, UserSimulatorAgentConfig) must
    also be able to express temperature/max_tokens/timeout_ms/reasoning_effort —
    previously only model/client/api_key/api reached the built LLMCallConfig.
    """

    @pytest.mark.asyncio
    async def test_judge_agent_config_threads_max_tokens(self):
        client = _make_client()
        client.responses.create.return_value = _responses_response()
        config = JudgeAgentConfig(client=client, max_tokens=15_000)

        assert config.max_tokens == 15_000
        from evaluatorq.simulation.agents.judge import JudgeAgent

        agent = JudgeAgent(config)
        await agent._call_llm(_messages())

        kwargs = client.responses.create.await_args.kwargs
        assert kwargs["max_output_tokens"] == 15_000

    def test_user_simulator_agent_config_threads_temperature(self):
        client = _make_client()
        config = UserSimulatorAgentConfig(client=client, temperature=0.3)
        assert config.temperature == 0.3

    def test_plain_agent_config_unset_fields_do_not_shadow_defaults(self):
        """A field left untouched on AgentConfig must not leak LLMCallConfig's
        own pydantic default into model_fields_set."""
        from evaluatorq.simulation.agents.base import _config_from_agent_config

        cfg, _ = _config_from_agent_config(AgentConfig())
        assert "temperature" not in cfg.model_fields_set
        assert "max_tokens" not in cfg.model_fields_set
        assert "timeout_ms" not in cfg.model_fields_set
        assert "reasoning_effort" not in cfg.model_fields_set
