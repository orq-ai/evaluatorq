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
    assert "/v2/router" not in str(gen._client.base_url)


@pytest.mark.parametrize("gen_cls", GEN_CLASSES)
def test_orq_key_wins_when_both_set(gen_cls, monkeypatch):
    monkeypatch.setenv("ORQ_API_KEY", "orq-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    gen = gen_cls()
    assert str(gen._client.base_url).rstrip("/").endswith("/v2/router")


@pytest.mark.parametrize("gen_cls", GEN_CLASSES)
def test_no_keys_raises(gen_cls, monkeypatch):
    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key found"):
        gen_cls()


@pytest.mark.parametrize("gen_cls", GEN_CLASSES)
def test_injected_client_used_as_is(gen_cls, monkeypatch):
    from openai import AsyncOpenAI

    monkeypatch.delenv("ORQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    injected = AsyncOpenAI(api_key="sk-x", base_url="https://example.test/v1")
    gen = gen_cls(client=injected)
    assert gen._client is injected


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
    with pytest.raises(ValueError, match="No API key found"):
        DatapointGenerator()
