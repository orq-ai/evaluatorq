"""Tests for FirstMessageGenerator error / fallback paths.

Covers:
- 4xx APIStatusError re-raised (auth + client errors are not silently masked)
- 5xx / 429 APIStatusError surviving retries is raised, never replaced by a canned message
- empty or refused content is retried, then raises FirstMessageGenerationError
- truncation raises without retry
- leading/trailing quote stripping on returned message
"""

from __future__ import annotations

# ruff: noqa: S101
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import APIStatusError

from evaluatorq.simulation.generators.first_message_generator import (
    FirstMessageGenerationError,
    FirstMessageGenerator,
)
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
    async def test_no_temperature_is_sent(self):
        """RES-#168: this generator is where the weekly example run died.

        It sent a hardcoded ``temperature=0.8``; the default pipeline model
        answers 400 to the parameter, which failed every persona x scenario pair
        and turned ``simulate()`` into a RuntimeError.
        """
        client = _client_with_response("hello there")
        gen = FirstMessageGenerator(model="gpt-4o", client=client)

        await gen.generate(_persona(), _scenario())

        assert "temperature" not in client.responses.create.await_args.kwargs

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

    @pytest.mark.parametrize("status", [500, 429])
    async def test_persistent_server_error_is_raised_not_masked(self, status):
        # Survived with_retry: the datapoint fails rather than being simulated
        # on a canned opening line.
        client = _client_raising(_api_error(status))
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        with pytest.raises(APIStatusError) as exc_info:
            await gen.generate(_persona(), _scenario("reset my pw"))
        assert exc_info.value.status_code == status
        assert client.responses.create.await_count > 1  # retried before giving up

    async def test_400_is_reraised_not_masked(self):
        # A 4xx client error (bad request / model-not-found) is a real
        # misconfiguration — surface it, don't hide it behind a canned message.
        client = _client_raising(_api_error(400))
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        with pytest.raises(APIStatusError) as exc_info:
            await gen.generate(_persona(), _scenario("xyz"))
        assert exc_info.value.status_code == 400

    @pytest.mark.parametrize("content", ["", None])
    async def test_empty_content_retries_then_raises(self, content):
        client = _client_with_response(content)
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        with pytest.raises(FirstMessageGenerationError, match="empty content"):
            await gen.generate(_persona(), _scenario("login issue"))
        assert client.responses.create.await_count == 3

    @pytest.mark.parametrize("stop_reason", ["length", "max_output_tokens"])
    @pytest.mark.parametrize("content", ["", "partial opening"])
    async def test_truncation_raises_without_retry(self, stop_reason, content):
        client = _client_with_response(content, stop_reason=stop_reason)
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        with pytest.raises(FirstMessageGenerationError, match="truncated"):
            await gen.generate(_persona(), _scenario("truncated"))
        assert client.responses.create.await_count == 1

    async def test_refusal_retries_then_raises(self):
        client = _client_with_response("", refusal="not allowed")
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        with pytest.raises(FirstMessageGenerationError, match="refused"):
            await gen.generate(_persona(), _scenario("refused"))
        assert client.responses.create.await_count == 3

    async def test_empty_content_retries_until_a_message_arrives(self):
        client = _client_with_responses("", "", "I need help logging in")
        gen = FirstMessageGenerator(model="gpt-4o", client=client)
        result = await gen.generate(_persona(), _scenario("login"))
        assert result == "I need help logging in"
        assert client.responses.create.await_count == 3

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


@pytest.mark.asyncio
async def test_datapoint_generator_drops_only_the_failed_pair(caplog):
    """One failed opening fails its datapoint, not the batch, and the drop count is logged."""
    import logging

    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.simulation.generators import DatapointGenerator

    gen = DatapointGenerator(config=LLMCallConfig(model="gpt-4o", client=MagicMock()))

    async def fake_generate(persona: Persona, scenario: Scenario) -> str:
        if scenario.name == "bad":
            raise FirstMessageGenerationError("empty content")
        return f"hello from {scenario.name}"

    gen._first_message_generator.generate = fake_generate  # type: ignore[method-assign]
    scenarios = [Scenario(name="good", goal="g"), Scenario(name="bad", goal="b")]

    with caplog.at_level(logging.WARNING):
        datapoints = await gen.generate_from_combinations([_persona()], scenarios)

    assert [dp.first_message for dp in datapoints] == ["hello from good"]
    assert "Generated 1 of 2 datapoints" in caplog.text


@pytest.mark.asyncio
async def test_datapoint_generator_raises_when_every_pair_fails():
    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.simulation.generators import DatapointGenerator

    gen = DatapointGenerator(config=LLMCallConfig(model="gpt-4o", client=MagicMock()))
    gen._first_message_generator.generate = AsyncMock(side_effect=FirstMessageGenerationError("x"))  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="failed for all 1"):
        await gen.generate_from_combinations([_persona()], [_scenario()])


@pytest.mark.asyncio
@pytest.mark.parametrize('status', [401, 400])
async def test_datapoint_generator_propagates_provider_configuration_errors(status: int) -> None:
    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.simulation.generators import DatapointGenerator

    gen = DatapointGenerator(config=LLMCallConfig(model='gpt-4o', client=MagicMock()))
    gen._first_message_generator.generate = AsyncMock(side_effect=_api_error(status))  # type: ignore[method-assign]

    with pytest.raises(APIStatusError) as exc_info:
        await gen.generate_from_combinations([_persona()], [_scenario()])
    assert exc_info.value.status_code == status


@pytest.mark.asyncio
async def test_datapoint_generator_drops_transient_provider_failure() -> None:
    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.simulation.generators import DatapointGenerator

    gen = DatapointGenerator(config=LLMCallConfig(model='gpt-4o', client=MagicMock()))

    async def fake_generate(persona: Persona, scenario: Scenario) -> str:
        del persona
        if scenario.name == 'bad':
            raise _api_error(429)
        return 'hello'

    gen._first_message_generator.generate = fake_generate  # type: ignore[method-assign]
    datapoints = await gen.generate_from_combinations([_persona()], [Scenario(name='bad', goal='b'), _scenario()])
    assert [datapoint.first_message for datapoint in datapoints] == ['hello']


@pytest.mark.asyncio
async def test_datapoint_generator_preserves_empty_input() -> None:
    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.simulation.generators import DatapointGenerator

    gen = DatapointGenerator(config=LLMCallConfig(model="gpt-4o", client=MagicMock()))
    assert await gen.generate_from_combinations([], [_scenario()]) == []


@pytest.mark.asyncio
async def test_datapoint_construction_error_is_not_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.simulation.generators import DatapointGenerator

    gen = DatapointGenerator(config=LLMCallConfig(model="gpt-4o", client=MagicMock()))
    gen._first_message_generator.generate = AsyncMock(return_value="hello")  # type: ignore[method-assign]

    def fail_construction(*_args: object) -> None:
        raise ValueError('malformed datapoint')

    monkeypatch.setattr('evaluatorq.simulation.generators.datapoint_generator.generate_datapoint', fail_construction)
    with pytest.raises(ValueError, match='malformed datapoint'):
        await gen.generate_from_combinations([_persona()], [_scenario()])


@pytest.mark.asyncio
async def test_simulation_datapoint_construction_error_is_not_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.simulation.api import _resolve_or_generate_datapoints

    async def fake_generate(*_args: object) -> str:
        return 'hello'

    def fail_construction(*_args: object) -> None:
        raise ValueError('malformed datapoint')

    monkeypatch.setattr(FirstMessageGenerator, 'generate', fake_generate)
    monkeypatch.setattr('evaluatorq.simulation.utils.prompt_builders.generate_datapoint', fail_construction)
    with pytest.raises(ValueError, match='malformed datapoint'):
        await _resolve_or_generate_datapoints(
            caller='simulate',
            datapoints=None,
            personas=[_persona()],
            scenarios=[_scenario()],
            dataset_id=None,
            llm_config=LLMCallConfig(model='test'),
            generation_client=MagicMock(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('limit', [2, -1])
async def test_simulation_limits_pending_generation_tasks(monkeypatch: pytest.MonkeyPatch, limit: int) -> None:
    from evaluatorq.common.llm_limit import llm_concurrency_limit
    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.simulation.api import _resolve_or_generate_datapoints

    live = 0
    peak = 0

    async def fake_generate(*_args: object) -> str:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.to_thread(time.sleep, 0.01)
        live -= 1
        return 'hello'

    monkeypatch.setattr(FirstMessageGenerator, 'generate', fake_generate)
    async with llm_concurrency_limit(limit):
        datapoints = await _resolve_or_generate_datapoints(
            caller='simulate',
            datapoints=None,
            personas=[_persona()],
            scenarios=[_scenario(f'goal-{i}') for i in range(30)],
            dataset_id=None,
            llm_config=LLMCallConfig(model='test'),
            generation_client=MagicMock(),
        )
    assert len(datapoints) == 30
    if limit == -1:
        assert peak == 30
    else:
        assert peak <= 4


@pytest.mark.asyncio
async def test_direct_datapoint_generator_limits_pending_tasks() -> None:
    from evaluatorq.common.llm_limit import llm_concurrency_limit
    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.simulation.generators import DatapointGenerator

    gen = DatapointGenerator(config=LLMCallConfig(model='gpt-4o', client=MagicMock()))
    live = 0
    peak = 0

    async def fake_generate(persona: Persona, scenario: Scenario) -> str:
        nonlocal live, peak
        del persona, scenario
        live += 1
        peak = max(peak, live)
        await asyncio.to_thread(time.sleep, 0.01)
        live -= 1
        return 'hello'

    gen._first_message_generator.generate = fake_generate  # type: ignore[method-assign]
    async with llm_concurrency_limit(2):
        datapoints = await gen.generate_from_combinations([_persona()], [_scenario(f'goal-{i}') for i in range(30)])
    assert len(datapoints) == 30
    assert peak <= 4


@pytest.mark.asyncio
async def test_simulation_propagates_first_message_configuration_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.contracts import LLMCallConfig
    from evaluatorq.simulation.api import _resolve_or_generate_datapoints

    async def fail_generate(*_args: object) -> str:
        raise _api_error(401)

    monkeypatch.setattr(FirstMessageGenerator, 'generate', fail_generate)
    with pytest.raises(APIStatusError) as exc_info:
        await _resolve_or_generate_datapoints(
            caller='simulate',
            datapoints=None,
            personas=[_persona()],
            scenarios=[_scenario()],
            dataset_id=None,
            llm_config=LLMCallConfig(model='test'),
            generation_client=MagicMock(),
        )
    assert exc_info.value.status_code == 401
