"""Tests for FirstMessageGenerator error / fallback paths.

Covers:
- 401/403 APIStatusError re-raised (auth errors are not silently masked)
- non-auth APIStatusError returns generic fallback
- empty content returns generic fallback
- leading/trailing quote stripping on returned message
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIStatusError

from evaluatorq.simulation.generators.first_message_generator import FirstMessageGenerator
from evaluatorq.simulation.types import CommunicationStyle, Persona, Scenario


@pytest.fixture(autouse=True)
def _mock_retry_sleep():
    """Strip real sleeps from the retry helper so 5xx tests don't burn 30s."""
    with patch("evaluatorq.common.retry.asyncio.sleep", new=AsyncMock()):
        yield


def _persona() -> Persona:
    return Persona(
        name="Test User",
        patience=0.5,
        assertiveness=0.5,
        politeness=0.5,
        technical_level=0.5,
        communication_style=CommunicationStyle.casual,
        background="bg",
    )


def _scenario(goal: str = "fix my bug") -> Scenario:
    return Scenario(name="S", goal=goal)


def _api_error(status: int) -> APIStatusError:
    request = httpx.Request("POST", "https://api.test/v1/chat/completions")
    response = httpx.Response(status_code=status, request=request)
    return APIStatusError(message=f"http {status}", response=response, body=None)


def _client_with_response(message_content: str | None) -> MagicMock:
    msg = MagicMock()
    msg.content = message_content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=resp)
    return client


def _client_raising(exc: Exception) -> MagicMock:
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=exc)
    return client


@pytest.mark.asyncio
class TestFirstMessageGeneratorErrors:
    async def test_401_is_reraised_not_swallowed(self):
        client = _client_raising(_api_error(401))
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        with pytest.raises(APIStatusError) as exc_info:
            await gen.generate(_persona(), _scenario())
        assert exc_info.value.status_code == 401

    async def test_403_is_reraised_not_swallowed(self):
        client = _client_raising(_api_error(403))
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        with pytest.raises(APIStatusError) as exc_info:
            await gen.generate(_persona(), _scenario())
        assert exc_info.value.status_code == 403

    async def test_500_falls_back_to_generic_message(self):
        client = _client_raising(_api_error(500))
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        result = await gen.generate(_persona(), _scenario("reset my pw"))
        assert result == "Hi, I need help with: reset my pw"

    async def test_400_non_auth_falls_back(self):
        client = _client_raising(_api_error(400))
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        result = await gen.generate(_persona(), _scenario("xyz"))
        assert result == "Hi, I need help with: xyz"

    async def test_empty_content_falls_back_to_generic(self):
        client = _client_with_response("")
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        result = await gen.generate(_persona(), _scenario("login issue"))
        assert result == "Hi, I need help with: login issue"

    async def test_none_content_falls_back_to_generic(self):
        client = _client_with_response(None)
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        result = await gen.generate(_persona(), _scenario("login"))
        assert result == "Hi, I need help with: login"

    async def test_leading_and_trailing_double_quotes_stripped(self):
        client = _client_with_response('"hello there"')
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        result = await gen.generate(_persona(), _scenario())
        assert result == "hello there"

    async def test_leading_and_trailing_single_quotes_stripped(self):
        client = _client_with_response("'hello'")
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        result = await gen.generate(_persona(), _scenario())
        assert result == "hello"

    async def test_missing_api_key_raises_helpful_value_error(self, monkeypatch):
        monkeypatch.delenv("ORQ_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ORQ_API_KEY"):
            FirstMessageGenerator(model="gpt-4o")
