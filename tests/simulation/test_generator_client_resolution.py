import asyncio
from typing import Any, cast
from weakref import WeakKeyDictionary

import pytest

from evaluatorq.simulation.generators.first_message_generator import FirstMessageGenerator
from evaluatorq.simulation.generators.persona_generator import PersonaGenerator
from evaluatorq.simulation.generators.scenario_generator import ScenarioGenerator

GEN_CLASSES = [PersonaGenerator, ScenarioGenerator, FirstMessageGenerator]


@pytest.mark.parametrize("gen_cls", GEN_CLASSES)
def test_openai_key_only_uses_openai_base_url(gen_cls, monkeypatch):
    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    gen = gen_cls()
    assert "api.openai.com" in str(gen._client.base_url)
    assert "/v3/router" not in str(gen._client.base_url)


@pytest.mark.parametrize("gen_cls", GEN_CLASSES)
def test_auto_built_generator_client_disables_sdk_retries(gen_cls, monkeypatch):
    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    gen = gen_cls()

    assert gen._client.max_retries == 0


@pytest.mark.parametrize("gen_cls", GEN_CLASSES)
def test_orq_key_wins_when_both_set(gen_cls, monkeypatch):
    monkeypatch.setenv("ORQ_API_KEY", "orq-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    gen = gen_cls()
    assert str(gen._client.base_url).rstrip("/").endswith("/v3/router")


@pytest.mark.parametrize("gen_cls", GEN_CLASSES)
def test_no_keys_raises(gen_cls, monkeypatch):
    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Missing LLM credentials"):
        gen_cls()


@pytest.mark.parametrize("gen_cls", GEN_CLASSES)
def test_injected_client_used_as_is(gen_cls, monkeypatch):
    from openai import AsyncOpenAI

    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    injected = AsyncOpenAI(api_key="sk-x", base_url="https://example.test/v1")
    gen = gen_cls(client=injected)
    assert gen._client is injected
    assert gen._client_owned is False  # caller owns the lifecycle; generator must not close it


def test_datapoint_generator_openai_key_only(monkeypatch):
    from evaluatorq.simulation.generators.datapoint_generator import DatapointGenerator

    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    gen = DatapointGenerator()
    assert "api.openai.com" in str(gen._shared_client.base_url)
    assert gen._client_owned is True


def test_datapoint_generator_no_keys_raises(monkeypatch):
    from evaluatorq.simulation.generators.datapoint_generator import DatapointGenerator

    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Missing LLM credentials"):
        DatapointGenerator()


def test_datapoint_generator_can_cross_event_loop_boundaries(monkeypatch):
    from evaluatorq.simulation.generators import datapoint_generator as dpg_mod
    from evaluatorq.simulation.generators.datapoint_generator import DatapointGenerator

    class FakeFirstMessageGenerator:
        async def generate(self, persona, scenario):
            await asyncio.sleep(0)
            return 'opening message'

    generator = cast(Any, object.__new__(DatapointGenerator))
    generator._rate_limit_delay = 0.0
    generator._max_concurrent_calls = 1
    generator._semaphores = WeakKeyDictionary()
    generator._first_message_generator = FakeFirstMessageGenerator()
    monkeypatch.setattr(dpg_mod, 'generate_datapoint', lambda *args: object())

    async def generate() -> list[Any]:
        return await generator.generate_from_combinations([object(), object()], [object()])

    assert len(asyncio.run(generate())) == 2
    assert len(asyncio.run(generate())) == 2


def test_datapoint_generator_shares_the_cap_across_concurrent_calls(monkeypatch):
    from evaluatorq.simulation.generators import datapoint_generator as dpg_mod
    from evaluatorq.simulation.generators.datapoint_generator import DatapointGenerator

    active = 0
    peak = 0

    class FakeFirstMessageGenerator:
        async def generate(self, persona, scenario):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.01)
                return 'opening message'
            finally:
                active -= 1

    generator = cast(Any, object.__new__(DatapointGenerator))
    generator._rate_limit_delay = 0.0
    generator._max_concurrent_calls = 2
    generator._semaphores = WeakKeyDictionary()
    generator._first_message_generator = FakeFirstMessageGenerator()
    monkeypatch.setattr(dpg_mod, 'generate_datapoint', lambda *args: object())

    async def generate_concurrently() -> None:
        combinations = ([object(), object()], [object(), object(), object()])
        await asyncio.gather(*[
            generator.generate_from_combinations(personas, [object()])
            for personas in combinations
        ])

    asyncio.run(generate_concurrently())

    assert peak == 2
