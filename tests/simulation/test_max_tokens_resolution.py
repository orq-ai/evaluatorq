"""Truncation messages must name the knob that actually governs the call.

Regression coverage for a truncation message that always pointed at
``EVALUATORQ_LLM_MAX_TOKENS`` even when an agent's `LLMCallConfig.max_tokens`
was pinned — in that case the env var is never consulted (see
`BaseAgent._resolved_max_tokens`), so raising it does nothing and the message
gave the user no way to know why.

The assertions run through the **real message sites** — the two
``finish_reason=length`` / ``stop_reason=length`` ``RuntimeError`` branches and
the empty-Responses warning — rather than calling `_max_tokens_advice` and
comparing against the literal in its own body. Asserting the helper's return
value proves only that the helper is self-consistent: an edit that stops calling
it at one site (the original defect's exact shape) leaves such a test green.
"""

from __future__ import annotations

# ruff: noqa: S101, SLF001
import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from evaluatorq.contracts import DEFAULT_TARGET_MAX_TOKENS, LLMCallConfig
from evaluatorq.simulation.agents import base as base_module
from evaluatorq.simulation.agents.base import BaseAgent
from evaluatorq.simulation.types import Message

CONFIG_ADVICE = "raise max_tokens on this agent's LLMCallConfig"
CALL_ADVICE = 'raise the max_tokens argument passed to this call'
ENV_ADVICE = 'raise the budget via EVALUATORQ_LLM_MAX_TOKENS'


class _ConcreteAgent(BaseAgent):
    @property
    def name(self) -> str:
        return 'TestAgent'

    @property
    def system_prompt(self) -> str:
        return 'You are a test agent.'


def _make_agent(**config_kwargs: Any) -> _ConcreteAgent:
    client = MagicMock()
    client.base_url = 'https://api.openai.com/v1'
    # retry_count=0: one attempt, so a raising branch surfaces its message once
    # instead of being retried by `with_retry`.
    return _ConcreteAgent(config=LLMCallConfig(model='gpt-4o', client=client, retry_count=0, **config_kwargs))


def _truncated_chat_response() -> SimpleNamespace:
    """A chat completion that hit the budget before emitting text or a tool call."""
    message = SimpleNamespace(content=None, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason='length')])


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('EVALUATORQ_LLM_MAX_TOKENS', raising=False)


@pytest.fixture
def _fake_chat(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub `execute_chat_completion`, recording the kwargs it was called with."""
    seen: dict[str, Any] = {}

    async def _fake(**kwargs: Any) -> tuple[Any, None]:
        seen.update(kwargs)
        return _truncated_chat_response(), None

    monkeypatch.setattr(base_module, 'execute_chat_completion', _fake)
    return seen


@pytest.fixture
def _fake_responses(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub `execute_response`; the fake's shape is set per test via ``seen['response']``."""
    seen: dict[str, Any] = {'response': SimpleNamespace(output=[], incomplete_details=None, status='completed')}

    async def _fake(**kwargs: Any) -> tuple[Any, None]:
        seen.update({k: v for k, v in kwargs.items() if k != 'response'})
        return seen['response'], None

    monkeypatch.setattr(base_module, 'execute_response', _fake)
    return seen


# ---------------------------------------------------------------------------
# Chat completions: finish_reason=length
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_truncation_message_names_config_knob_when_pinned(_fake_chat: dict[str, Any]):
    """A pinned `LLMCallConfig.max_tokens` means the env var is never consulted."""
    agent = _make_agent(max_tokens=256)

    with pytest.raises(RuntimeError) as excinfo:
        await agent._call_chat_completions([Message(role='user', content='hi')])

    message = str(excinfo.value)
    assert CONFIG_ADVICE in message
    assert 'EVALUATORQ_LLM_MAX_TOKENS' not in message
    assert 'max_tokens=256' in message
    assert _fake_chat['max_tokens'] == 256


@pytest.mark.asyncio
async def test_chat_truncation_message_names_env_var_when_nothing_set(_fake_chat: dict[str, Any]):
    """Config unset and no per-call value: the env fallback is what governed."""
    agent = _make_agent()

    with pytest.raises(RuntimeError) as excinfo:
        await agent._call_chat_completions([Message(role='user', content='hi')])

    message = str(excinfo.value)
    assert ENV_ADVICE in message
    assert f'max_tokens={DEFAULT_TARGET_MAX_TOKENS}' in message
    assert _fake_chat['max_tokens'] == DEFAULT_TARGET_MAX_TOKENS


@pytest.mark.asyncio
async def test_chat_truncation_message_names_the_call_argument(_fake_chat: dict[str, Any]):
    """The third tier: config unset, caller passed ``max_tokens=``.

    Latent today (no in-repo caller passes it), but the env var is not what
    governed this call, so naming it would send the user to a knob with no effect.
    """
    agent = _make_agent()

    with pytest.raises(RuntimeError) as excinfo:
        await agent._call_chat_completions([Message(role='user', content='hi')], max_tokens=64)

    message = str(excinfo.value)
    assert CALL_ADVICE in message
    assert 'EVALUATORQ_LLM_MAX_TOKENS' not in message
    assert 'max_tokens=64' in message
    assert _fake_chat['max_tokens'] == 64


@pytest.mark.asyncio
async def test_chat_config_beats_the_call_argument(_fake_chat: dict[str, Any]):
    """Config wins over a per-call value, and the message says so."""
    agent = _make_agent(max_tokens=256)

    with pytest.raises(RuntimeError) as excinfo:
        await agent._call_chat_completions([Message(role='user', content='hi')], max_tokens=64)

    message = str(excinfo.value)
    assert CONFIG_ADVICE in message
    assert 'max_tokens=256' in message


@pytest.mark.asyncio
async def test_chat_env_var_governs_when_config_and_call_are_unset(
    _fake_chat: dict[str, Any], monkeypatch: pytest.MonkeyPatch
):
    """The env fallback is resolved at call time, and the message reports its value."""
    monkeypatch.setenv('EVALUATORQ_LLM_MAX_TOKENS', '128')
    agent = _make_agent()

    with pytest.raises(RuntimeError) as excinfo:
        await agent._call_chat_completions([Message(role='user', content='hi')])

    message = str(excinfo.value)
    assert ENV_ADVICE in message
    assert 'max_tokens=128' in message


# ---------------------------------------------------------------------------
# Responses: stop_reason=length
# ---------------------------------------------------------------------------


def _truncated_responses_payload() -> SimpleNamespace:
    return SimpleNamespace(output=[], incomplete_details=SimpleNamespace(reason='max_output_tokens'), status='incomplete')


@pytest.mark.asyncio
async def test_responses_truncation_message_names_config_knob_when_pinned(_fake_responses: dict[str, Any]):
    _fake_responses['response'] = _truncated_responses_payload()
    agent = _make_agent(max_tokens=256)

    with pytest.raises(RuntimeError) as excinfo:
        await agent._call_responses([Message(role='user', content='hi')])

    message = str(excinfo.value)
    assert CONFIG_ADVICE in message
    assert 'EVALUATORQ_LLM_MAX_TOKENS' not in message
    assert 'max_output_tokens=256' in message


@pytest.mark.asyncio
async def test_responses_truncation_message_names_env_var_when_nothing_set(_fake_responses: dict[str, Any]):
    _fake_responses['response'] = _truncated_responses_payload()
    agent = _make_agent()

    with pytest.raises(RuntimeError) as excinfo:
        await agent._call_responses([Message(role='user', content='hi')])

    assert ENV_ADVICE in str(excinfo.value)


@pytest.mark.asyncio
async def test_responses_truncation_message_names_the_call_argument(_fake_responses: dict[str, Any]):
    _fake_responses['response'] = _truncated_responses_payload()
    agent = _make_agent()

    with pytest.raises(RuntimeError) as excinfo:
        await agent._call_responses([Message(role='user', content='hi')], max_tokens=64)

    message = str(excinfo.value)
    assert CALL_ADVICE in message
    assert 'EVALUATORQ_LLM_MAX_TOKENS' not in message
    assert 'max_output_tokens=64' in message


# ---------------------------------------------------------------------------
# Responses: empty output (warning, not a raise)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_responses_empty_warning_names_config_knob_when_pinned(
    _fake_responses: dict[str, Any], caplog: pytest.LogCaptureFixture
):
    """The third message site is a warning; it must name the same knob."""
    agent = _make_agent(max_tokens=256)

    with caplog.at_level(logging.WARNING, logger=base_module.__name__):
        result = await agent._call_responses([Message(role='user', content='hi')])

    assert result.content == ''
    assert CONFIG_ADVICE in caplog.text
    assert 'EVALUATORQ_LLM_MAX_TOKENS' not in caplog.text


@pytest.mark.asyncio
async def test_responses_empty_warning_names_the_call_argument(
    _fake_responses: dict[str, Any], caplog: pytest.LogCaptureFixture
):
    agent = _make_agent()

    with caplog.at_level(logging.WARNING, logger=base_module.__name__):
        await agent._call_responses([Message(role='user', content='hi')], max_tokens=64)

    assert CALL_ADVICE in caplog.text
    assert 'EVALUATORQ_LLM_MAX_TOKENS' not in caplog.text


@pytest.mark.asyncio
async def test_responses_empty_warning_names_env_var_when_nothing_set(
    _fake_responses: dict[str, Any], caplog: pytest.LogCaptureFixture
):
    agent = _make_agent()

    with caplog.at_level(logging.WARNING, logger=base_module.__name__):
        await agent._call_responses([Message(role='user', content='hi')])

    assert ENV_ADVICE in caplog.text
