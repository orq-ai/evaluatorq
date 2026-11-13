"""A raw Anthropic client handed to each public entry point reaches the inner call wrapped (RES-1874)."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import anthropic
import pytest

from evaluatorq.anthropic import AnthropicLLMClient
from evaluatorq.contracts import LLMCallConfig

from .test_wiring import _one_tool_trajectory, scripted, text


def _real_client() -> Any:
    return anthropic.AsyncAnthropic(api_key='test')


def test_door_classification_sends_its_questions_through_a_raw_anthropic_client() -> None:
    from evaluatorq.signals.config import SignalsConfig
    from evaluatorq.signals.doors import classify_tool_doors

    fake = scripted([text('{"value": "benign", "explanation": "reads only"}')])
    results = asyncio.run(
        classify_tool_doors(
            _one_tool_trajectory(), SignalsConfig(classifier={'model': 'claude-sonnet-5-5'}), client=fake
        )
    )
    assert fake.messages.calls, 'the Anthropic client was never reached'
    assert results['one_way_call_count'].value == 0
    assert results['unknown_call_count'].value == 0


@pytest.mark.parametrize('entry', ['generate_personas', 'generate_scenarios'])
def test_generation_entry_points_wrap_a_raw_anthropic_client(entry: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.simulation import api, generators
    from evaluatorq.simulation.exceptions import SimulationError

    seen: list[Any] = []

    class Recorder:
        def __init__(self, *, client: Any, config: Any) -> None:
            seen.append(client)

        async def generate(self, **kwargs: Any) -> list[Any]:
            return []

        async def close(self) -> None:
            return None

    monkeypatch.setattr(generators, 'PersonaGenerator', Recorder)
    monkeypatch.setattr(generators, 'ScenarioGenerator', Recorder)
    with pytest.raises(SimulationError):
        asyncio.run(getattr(api, entry)(['angry customer'], generation_client=_real_client()))
    assert len(seen) == 1
    assert isinstance(seen[0], AnthropicLLMClient)


def test_the_simulation_runner_wraps_a_raw_anthropic_llm_client() -> None:
    from evaluatorq.simulation.runner.simulation import SimulationRunner

    runner = SimulationRunner(target=lambda messages: 'ok', llm_client=_real_client())
    assert isinstance(runner._get_shared_client(), AnthropicLLMClient)


def test_the_simulation_runner_wraps_the_client_in_llm_config() -> None:
    from evaluatorq.simulation.runner.simulation import SimulationRunner

    runner = SimulationRunner(
        target=lambda messages: 'ok',
        llm_config=LLMCallConfig(model='claude-sonnet-5-5', client=_real_client()),
    )
    assert isinstance(runner._llm_config.client, AnthropicLLMClient)


@pytest.mark.asyncio
async def test_the_trace_finder_run_store_wraps_a_raw_anthropic_client(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluatorq.trace_finder import pipeline
    from evaluatorq.trace_finder.models import PopulationRequest
    from evaluatorq.trace_finder.settings import DashboardSettings

    seen: list[Any] = []

    async def fake_plan(query: str, population: Any, **kwargs: Any) -> object:
        seen.append(kwargs['client'])
        return object()

    monkeypatch.setattr(pipeline, 'plan_query', fake_plan)
    store = pipeline.build_run_store(DashboardSettings(), client=_real_client(), orq=cast('Any', object()))
    planner = store._planner
    assert planner is not None
    await planner('refunds', PopulationRequest(), within_results=False)
    assert len(seen) == 1
    assert isinstance(seen[0], AnthropicLLMClient)


def test_the_merge_call_of_apply_recommendations_takes_a_raw_anthropic_client() -> None:
    # apply_recommendations never wraps its llm_client; the executor in common.llm_call does.
    from evaluatorq.common.apply import _merge_call

    fake = scripted([text('{"instructions": "Be kind."}')])
    reply = asyncio.run(
        _merge_call(fake, 'claude-sonnet-5-5', 'system', 'user', max_tokens=100, cfg=None, temperature=None)
    )
    assert reply == '{"instructions": "Be kind."}'
    assert fake.messages.calls, 'the Anthropic client was never reached'
    assert fake.messages.calls[0]['messages'] == [{'role': 'user', 'content': [{'type': 'text', 'text': 'user'}]}]


def test_the_same_raw_client_always_gets_the_same_adapter() -> None:
    from evaluatorq.common.llm_client import as_llm_client, is_anthropic_client

    raw = _real_client()
    assert is_anthropic_client(raw)
    wrapped = as_llm_client(raw)
    assert as_llm_client(raw) is wrapped
    assert as_llm_client(_real_client()) is not wrapped
