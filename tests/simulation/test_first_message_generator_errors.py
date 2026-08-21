"""Tests for FirstMessageGenerator error / fallback paths.

Covers:
- 4xx APIStatusError re-raised (auth + client errors are not silently masked)
- 5xx / 429 APIStatusError returns generic fallback (keeps a long run alive)
- empty content returns generic fallback
- leading/trailing quote stripping on returned message
"""

from __future__ import annotations

# ruff: noqa: S101
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


def _response(output_text: str | None, *, stop_reason: str | None = None, refusal: str | None = None) -> MagicMock:
    resp = MagicMock()
    resp.output_text = output_text
    resp.stop_reason = stop_reason
    resp.incomplete_details = None
    resp.output = [] if refusal is None else [MagicMock(content=[MagicMock(type='refusal', refusal=refusal)])]
    resp.usage = MagicMock(input_tokens=1, output_tokens=1, total_tokens=2)
    return resp


def _client_with_response(message_content: str | None, **response_kwargs: str | None) -> MagicMock:
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock(return_value=_response(message_content, **response_kwargs))
    return client


def _client_with_responses(*message_contents: str | None) -> MagicMock:
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock(side_effect=[_response(c) for c in message_contents])
    return client


def _client_raising(exc: Exception) -> MagicMock:
    client = MagicMock()
    client.responses = MagicMock()
    client.responses.create = AsyncMock(side_effect=exc)
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
        # Persistent server error (survived with_retry) — fall back to keep a
        # long run alive rather than abort it.
        client = _client_raising(_api_error(500))
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        result = await gen.generate(_persona(), _scenario("reset my pw"))
        assert result == "Hi, I need help with: reset my pw"

    async def test_429_falls_back_to_generic_message(self):
        client = _client_raising(_api_error(429))
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        result = await gen.generate(_persona(), _scenario("rate limited"))
        assert result == "Hi, I need help with: rate limited"

    async def test_400_is_reraised_not_masked(self):
        # A 4xx client error (bad request / model-not-found) is a real
        # misconfiguration — surface it, don't hide it behind a canned message.
        client = _client_raising(_api_error(400))
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        with pytest.raises(APIStatusError) as exc_info:
            await gen.generate(_persona(), _scenario("xyz"))
        assert exc_info.value.status_code == 400

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

    async def test_length_stop_reason_falls_back_without_retry(self):
        client = _client_with_response("", stop_reason="length")
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        result = await gen.generate(_persona(), _scenario("truncated"))
        assert result == "Hi, I need help with: truncated"
        assert client.responses.create.await_count == 1

    @pytest.mark.parametrize('stop_reason', ['length', 'max_output_tokens'])
    async def test_partial_length_response_falls_back_without_retry(self, stop_reason):
        client = _client_with_response('partial opening', stop_reason=stop_reason)
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        result = await gen.generate(_persona(), _scenario("partial"))
        assert result == "Hi, I need help with: partial"
        assert client.responses.create.await_count == 1

    async def test_refusal_falls_back_without_retry(self):
        client = _client_with_response("", refusal="not allowed")
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        result = await gen.generate(_persona(), _scenario("refused"))
        assert result == "Hi, I need help with: refused"
        assert client.responses.create.await_count == 1

    async def test_empty_content_retries_before_fallback(self):
        client = _client_with_responses("", "I need help logging in")
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        result = await gen.generate(_persona(), _scenario("login"))
        assert result == "I need help logging in"
        assert client.responses.create.await_count == 2

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
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ORQ_API_KEY"):
            FirstMessageGenerator(model="gpt-4o")
