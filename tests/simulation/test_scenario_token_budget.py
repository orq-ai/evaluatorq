"""A batched structured call sizes its token budget from the count it was given.

A flat cap truncates once the count grows, and truncated structured output is
unrecoverable on both legs of ``generate_structured`` — it raises rather than
retrying at the same budget. So the arithmetic is not the interesting part: the
wiring is. Every generator method that takes a caller-controlled count is
exercised here, because a method that keeps a flat literal reintroduces the
defect while the helper's own unit test stays green.
"""

from __future__ import annotations

import json
# ruff: noqa: S101
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import AsyncOpenAI

from evaluatorq.simulation.generators.persona_generator import PersonaGenerator, _persona_token_budget
from evaluatorq.simulation.generators.scenario_generator import ScenarioGenerator, _scenario_token_budget


def test_budget_scales_past_the_flat_cap():
    assert _scenario_token_budget(5) == 6000
    assert _scenario_token_budget(30) == 15000
    # The crossover: below it the floor wins, above it the per-item cost does.
    assert _scenario_token_budget(12) == 6000
    assert _scenario_token_budget(13) == 6500


def test_persona_budget_scales_past_its_own_flat_cap():
    assert _persona_token_budget(5) == 4000
    assert _persona_token_budget(30) == 18000


class _CapturingResponses:
    """Answers the raw Responses leg with an empty result and records the kwargs."""

    def __init__(self, sink: dict[str, Any]) -> None:
        self._sink = sink

    async def create(self, **kwargs: Any) -> Any:
        self._sink.update(kwargs)
        schema = kwargs['text']['format']['schema']
        output_text = json.dumps({name: [] for name in schema['properties']})
        content = SimpleNamespace(type='output_text', text=output_text, annotations=[])
        content.to_dict = lambda: {'type': content.type, 'text': content.text, 'annotations': content.annotations}
        output = SimpleNamespace(type='message', role='assistant', content=[content], status='completed')
        output.to_dict = lambda: {
            'type': output.type,
            'role': output.role,
            'content': [content.to_dict()],
            'status': output.status,
        }
        response = SimpleNamespace(
            output=[output],
            output_text=output_text,
            stop_reason='stop',
            incomplete_details=None,
            usage=None,
        )
        response.to_dict = lambda: {
            'output': [output.to_dict()],
            'output_text': response.output_text,
            'stop_reason': response.stop_reason,
            'incomplete_details': response.incomplete_details,
        }
        return response


class _CapturingClient:
    def __init__(self, sink: dict[str, Any]) -> None:
        self.responses = _CapturingResponses(sink)
        self.base_url = 'https://my.orq.ai/v3/router'

    async def close(self) -> None:  # pragma: no cover - the generator doesn't own us
        pass


def _client(sink: dict[str, Any]) -> AsyncOpenAI:
    return cast(AsyncOpenAI, cast(object, _CapturingClient(sink)))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'method',
    ['generate', 'generate_with_coverage', 'generate_edge_cases', 'generate_boundary_scenarios', 'generate_security_scenarios'],
)
async def test_every_scenario_generator_method_sizes_its_budget(method: str) -> None:
    sink: dict[str, Any] = {}
    generator = ScenarioGenerator(client=_client(sink))
    count_kwarg = 'num_edge_cases' if method == 'generate_edge_cases' else 'num_scenarios'

    await getattr(generator, method)(agent_description='support bot', **{count_kwarg: 30})

    assert sink['max_output_tokens'] == _scenario_token_budget(30)


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['generate', 'generate_with_coverage'])
async def test_every_persona_generator_method_sizes_its_budget(method: str) -> None:
    sink: dict[str, Any] = {}
    generator = PersonaGenerator(client=_client(sink))

    await getattr(generator, method)(agent_description='support bot', num_personas=30)

    assert sink['max_output_tokens'] == _persona_token_budget(30)
