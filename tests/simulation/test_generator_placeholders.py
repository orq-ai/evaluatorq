"""The generators refuse the stub some models return under a JSON-schema format, and ask again another way."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from evaluatorq.common.structured_output import StructuredResult
from evaluatorq.simulation.generators._placeholders import expecting, is_stub_text, too_few
from evaluatorq.simulation.generators.persona_generator import PersonaGenerator, PersonaListResponse
from evaluatorq.simulation.generators.scenario_generator import ScenarioGenerator, ScenarioListResponse, _parse_scenarios


def _persona(name: str = 'Anxious First-Time Buyer', background: str = 'Ordered a gift that has not arrived.') -> dict[str, Any]:
    return {
        'name': name,
        'patience': 0.5,
        'assertiveness': 0.5,
        'politeness': 0.5,
        'technical_level': 0.5,
        'communication_style': 'casual',
        'background': background,
    }


def _scenario(name: str = 'Late delivery', goal: str = 'Get a delivery date or a refund.') -> dict[str, Any]:
    return {'name': name, 'goal': goal}


@pytest.mark.parametrize(
    ('returned', 'expected', 'verdict'),
    [(1, 5, True), (2, 5, True), (3, 5, False), (4, 5, False), (5, 5, False), (0, 5, False), (1, 2, False), (0, 2, False), (1, 1, False), (0, 1, False), (0, 0, False), (1, 0, False), (4, 10, True), (5, 10, False)],
)
def test_some_items_but_fewer_than_half_of_those_asked_for_is_too_few(returned: int, expected: int, verdict: bool) -> None:
    assert too_few(returned, expected) is verdict


@pytest.mark.parametrize(('text', 'verdict'), [('', True), ('   ', True), ('x', True), (' ab ', True), ('abc', False), ('Buyer', False)])
def test_text_under_three_characters_is_a_stub(text: str, verdict: bool) -> None:
    assert is_stub_text('A real name', text) is verdict
    assert is_stub_text(text, 'A real background') is verdict


@pytest.mark.parametrize('model', [PersonaListResponse, ScenarioListResponse])
def test_expecting_changes_the_count_and_nothing_the_provider_sees(model: Any) -> None:
    five = expecting(model, 5)

    assert (five.expected, model.expected) == (5, 0)
    assert five.__name__ == model.__name__
    assert five.model_json_schema() == model.model_json_schema()


@pytest.mark.parametrize(
    ('personas', 'message'),
    [
        ([_persona(name='', background='')], '1 persona\\(s\\) returned for 5 asked'),
        ([_persona(name='placeholder', background='placeholder')], '1 persona\\(s\\) returned for 5 asked'),
        ([_persona(), _persona(), _persona(name='x')], 'index \\[2\\] have a placeholder name or background'),
        ([_persona(), _persona(), _persona(background=' ')], 'index \\[2\\] have a placeholder name or background'),
    ],
)
def test_a_stub_persona_answer_is_refused(personas: list[dict[str, Any]], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        expecting(PersonaListResponse, 5).model_validate_json(json.dumps({'personas': personas}))


@pytest.mark.parametrize(
    ('scenarios', 'message'),
    [
        ([_scenario(name='x', goal='x')], '1 scenario\\(s\\) returned for 5 asked'),
        ([_scenario(), _scenario(), _scenario(goal='')], 'index \\[2\\] have a placeholder name or goal'),
    ],
)
def test_a_stub_scenario_answer_is_refused(scenarios: list[dict[str, Any]], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        expecting(ScenarioListResponse, 5).model_validate_json(json.dumps({'scenarios': scenarios}))


def test_an_honest_answer_that_is_one_short_is_kept() -> None:
    four = expecting(PersonaListResponse, 5).model_validate({'personas': [_persona()] * 4})
    assert len(four.personas) == 4
    assert len(expecting(ScenarioListResponse, 5).model_validate({'scenarios': [_scenario()] * 4}).scenarios) == 4
    # Without a stated count only the text is checked.
    assert len(PersonaListResponse.model_validate({'personas': [_persona(name='placeholder')]}).personas) == 1


def test_the_salvage_path_drops_stubs_too() -> None:
    kept = PersonaGenerator._parse_personas(json.dumps([_persona(name=''), _persona(), _persona(background='x')]))
    assert [persona.name for persona in kept] == ['Anxious First-Time Buyer']
    assert [s.name for s in _parse_scenarios([_scenario(name='x'), _scenario(), _scenario(goal='')])] == ['Late delivery']


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    async def generate(_client: object, **kwargs: Any) -> StructuredResult[Any]:
        calls.append(kwargs)
        model = kwargs['response_format']
        if model.__name__ == 'PersonaListResponse':
            return StructuredResult(model(personas=[_persona()] * model.expected), '', None)
        return StructuredResult(model(scenarios=[_scenario()] * model.expected), '', None)

    monkeypatch.setattr('evaluatorq.openresponses.client.build_simulation_client', lambda _c=None, **_kw: (object(), False))
    monkeypatch.setattr('evaluatorq.simulation.generators.persona_generator.generate_structured', generate)
    monkeypatch.setattr('evaluatorq.simulation.generators.scenario_generator.generate_structured', generate)
    return calls


@pytest.mark.asyncio
async def test_every_generator_call_states_its_count_and_turns_the_tool_fallback_on(recorded: list[dict[str, Any]]) -> None:
    personas = PersonaGenerator(model='openai/gpt-6-luna')
    scenarios = ScenarioGenerator(model='openai/gpt-6-luna')

    assert len(await personas.generate(agent_description='A bookshop agent', num_personas=4)) == 4
    assert len(await scenarios.generate(agent_description='A bookshop agent', num_scenarios=6)) == 6

    assert [(call['response_format'].__name__, call['response_format'].expected) for call in recorded] == [
        ('PersonaListResponse', 4),
        ('ScenarioListResponse', 6),
    ]
    assert [call['tool_fallback_on_invalid'] for call in recorded] == [True, True]


@pytest.mark.parametrize('module', ['persona_generator', 'scenario_generator'])
def test_no_generator_call_site_is_left_without_the_guard(module: str) -> None:
    """Seven call sites share one pattern; a new one written without the count or the fallback would bring the bug back."""
    from pathlib import Path

    import evaluatorq.simulation.generators as generators

    source = (Path(generators.__file__).parent / f'{module}.py').read_text(encoding='utf-8')
    calls = source.count('await generate_structured(')
    assert calls >= 2
    assert source.count('tool_fallback_on_invalid=True') == calls
    assert source.count('response_format=expecting(') == calls
