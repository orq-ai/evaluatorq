"""The generators spot the stub some models return under a JSON-schema format, ask again, and never return it."""

from __future__ import annotations

import json
from typing import Any

import pytest

from evaluatorq.common.structured_output import StructuredResult
from evaluatorq.simulation.generators._placeholders import is_stub_item, too_few
from evaluatorq.simulation.generators.persona_generator import (
    PersonaGenerator,
    PersonaListResponse,
    _accepts_personas,
)
from evaluatorq.simulation.generators.scenario_generator import (
    ScenarioGenerator,
    ScenarioListResponse,
    _accepts_scenarios,
)


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


def _personas(*items: dict[str, Any]) -> PersonaListResponse:
    return PersonaListResponse.model_validate({'personas': list(items)})


def _scenarios(*items: dict[str, Any]) -> ScenarioListResponse:
    return ScenarioListResponse.model_validate({'scenarios': list(items)})


@pytest.mark.parametrize(
    ('returned', 'expected', 'verdict'),
    [(1, 5, True), (2, 5, True), (3, 5, False), (4, 5, False), (5, 5, False), (0, 5, False), (1, 2, False), (0, 2, False), (1, 1, False), (0, 1, False), (0, 0, False), (1, 0, False), (4, 10, True), (5, 10, False)],
)
def test_some_items_but_fewer_than_half_of_those_asked_for_is_too_few(returned: int, expected: int, verdict: bool) -> None:
    assert too_few(returned, expected) is verdict


@pytest.mark.parametrize(
    ('name', 'body', 'verdict'),
    [
        ('', 'A real background', True),
        ('   ', 'A real background', True),
        ('A real name', '', True),
        ('x', 'x', True),
        ('ab', ' y ', True),
        ('Li', 'A retired teacher ordering a gift.', False),
        ('QA', 'Check the refund flow end to end.', False),
        ('A real name', 'ok', False),
        ('placeholder', 'placeholder', False),
    ],
)
def test_an_item_is_a_stub_when_a_text_is_blank_or_both_are_too_short(name: str, body: str, verdict: bool) -> None:
    assert is_stub_item(name, body) is verdict


@pytest.mark.parametrize(
    ('response', 'verdict'),
    [
        (_personas(_persona(name='', background='')), False),
        (_personas(_persona(name='placeholder', background='placeholder')), False),  # one item for five
        (_personas(_persona(), _persona(), _persona(name='x', background='x')), False),
        (_personas(_persona(), _persona(), _persona(background=' ')), False),
        (_personas(_persona(), _persona(), _persona(), _persona()), True),  # one short is a miscount
        (_personas(_persona(name='Li'), _persona(), _persona()), True),
        (_personas(), True),  # an empty list is not the stub; the generator warns about it as before
    ],
)
def test_which_persona_answers_are_accepted_for_five_asked(response: PersonaListResponse, verdict: bool) -> None:
    assert _accepts_personas(5)(response) is verdict


@pytest.mark.parametrize(
    ('response', 'verdict'),
    [
        (_scenarios(_scenario(name='x', goal='x')), False),
        (_scenarios(_scenario(), _scenario()), False),  # two for five
        (_scenarios(_scenario(), _scenario(), _scenario(goal='')), False),
        (_scenarios(_scenario(name='QA'), _scenario(), _scenario()), True),
    ],
)
def test_which_scenario_answers_are_accepted_for_five_asked(response: ScenarioListResponse, verdict: bool) -> None:
    assert _accepts_scenarios(5)(response) is verdict


def test_a_request_for_one_item_is_judged_on_its_text_alone() -> None:
    assert _accepts_personas(1)(_personas(_persona())) is True
    assert _accepts_personas(1)(_personas(_persona(name=''))) is False
    assert _accepts_scenarios(0)(_scenarios(_scenario())) is True


class _Fake:
    """Stands in for ``generate_structured`` in both generator modules and records how it was called."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, _client: object, **kwargs: Any) -> StructuredResult[Any]:
        self.calls.append(kwargs)
        return self.answer(kwargs['response_format'])


def _install(monkeypatch: pytest.MonkeyPatch, answer: Any) -> _Fake:
    fake = _Fake(answer)
    monkeypatch.setattr('evaluatorq.openresponses.client.build_simulation_client', lambda _c=None, **_kw: (object(), False))
    monkeypatch.setattr('evaluatorq.simulation.generators.persona_generator.generate_structured', fake)
    monkeypatch.setattr('evaluatorq.simulation.generators.scenario_generator.generate_structured', fake)
    return fake


@pytest.mark.asyncio
async def test_every_generator_method_hands_over_a_content_check_for_its_own_count(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(monkeypatch, lambda model: StructuredResult(model.model_validate({next(iter(model.model_fields)): []}), '', None))
    personas = PersonaGenerator(model='openai/gpt-6-luna')
    scenarios = ScenarioGenerator(model='openai/gpt-6-luna')

    await personas.generate(agent_description='A bookshop agent', num_personas=6)
    await personas.generate_with_coverage(agent_description='A bookshop agent', num_personas=6)
    await scenarios.generate(agent_description='A bookshop agent', num_scenarios=6)
    await scenarios.generate_with_coverage(agent_description='A bookshop agent', num_scenarios=6)
    await scenarios.generate_edge_cases(agent_description='A bookshop agent', num_edge_cases=6)
    await scenarios.generate_boundary_scenarios(agent_description='A bookshop agent', num_scenarios=6)
    await scenarios.generate_security_scenarios(agent_description='A bookshop agent', num_scenarios=6)

    assert len(fake.calls) == 7
    for call in fake.calls:
        accept, model = call['accept'], call['response_format']
        key = next(iter(model.model_fields))
        item = _persona() if key == 'personas' else _scenario()
        # Asked for six: two real items are refused as too few, three are accepted.
        assert accept(model.model_validate({key: [item] * 2})) is False
        assert accept(model.model_validate({key: [item] * 3})) is True


@pytest.mark.asyncio
async def test_a_stub_that_still_comes_back_is_dropped_not_returned(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    stub = _persona(name='', background='')
    _install(monkeypatch, lambda model: StructuredResult(model.model_validate({'personas': [stub, _persona()]}), '', None))

    with caplog.at_level('WARNING'):
        personas = await PersonaGenerator(model='openai/gpt-6-luna').generate(agent_description='A bookshop agent', num_personas=5)

    assert [persona.name for persona in personas] == ['Anxious First-Time Buyer']
    assert 'dropped 1 placeholder persona(s)' in caplog.text
    assert 'requested 5 personas but only 1 were successfully parsed' in caplog.text


@pytest.mark.asyncio
async def test_the_salvage_path_drops_stubs_too(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = json.dumps({'scenarios': [_scenario(name='x', goal='x'), _scenario(), _scenario(goal='')]})
    _install(monkeypatch, lambda _model: StructuredResult(None, raw, None))

    scenarios = await ScenarioGenerator(model='openai/gpt-6-luna').generate(agent_description='A bookshop agent', num_scenarios=3)

    assert [scenario.name for scenario in scenarios] == ['Late delivery']


@pytest.mark.asyncio
async def test_a_short_real_name_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, lambda model: StructuredResult(model.model_validate({'personas': [_persona(name='Li'), _persona(), _persona()]}), '', None))

    personas = await PersonaGenerator(model='openai/gpt-6-luna').generate(agent_description='A bookshop agent', num_personas=3)

    assert [persona.name for persona in personas][0] == 'Li'
    assert len(personas) == 3


@pytest.mark.asyncio
async def test_real_items_in_the_second_runs_text_replace_a_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """What Opus does: a stub from the schema rung, then a bare array of real personas the wrapper model cannot parse."""
    raw = json.dumps([_persona(name=f'Persona {index}') for index in range(5)])
    _install(monkeypatch, lambda model: StructuredResult(model.model_validate({'personas': [_persona(name='', background='')]}), raw, None))

    personas = await PersonaGenerator(model='openai/gpt-6-luna').generate(agent_description='A bookshop agent', num_personas=5)

    assert [persona.name for persona in personas] == [f'Persona {index}' for index in range(5)]


@pytest.mark.asyncio
@pytest.mark.parametrize('raw', ['', 'not json at all', '[]', json.dumps([_scenario(name='Only one')])])
async def test_an_honest_short_list_is_never_made_shorter(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """Two real items for five asked are re-asked; whatever the second run left, the two are still returned."""
    two = [_persona(name='First real'), _persona(name='Second real')]
    _install(monkeypatch, lambda model: StructuredResult(model.model_validate({'personas': two}), raw, None))
    personas = await PersonaGenerator(model='openai/gpt-6-luna').generate(agent_description='A bookshop agent', num_personas=5)
    assert [persona.name for persona in personas] == ['First real', 'Second real']

    pair = [_scenario(name='First real'), _scenario(name='Second real')]
    _install(monkeypatch, lambda model: StructuredResult(model.model_validate({'scenarios': pair}), raw, None))
    scenarios = await ScenarioGenerator(model='openai/gpt-6-luna').generate(agent_description='A bookshop agent', num_scenarios=5)
    assert [scenario.name for scenario in scenarios] == ['First real', 'Second real']


@pytest.mark.asyncio
async def test_real_scenarios_in_the_second_runs_text_replace_a_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = json.dumps({'scenarios': [_scenario(name=f'Scenario {index}') for index in range(4)]})
    _install(monkeypatch, lambda model: StructuredResult(model.model_validate({'scenarios': [_scenario(name='x', goal='x')]}), raw, None))

    scenarios = await ScenarioGenerator(model='openai/gpt-6-luna').generate_security_scenarios(agent_description='A bookshop agent', num_scenarios=4)

    assert [scenario.name for scenario in scenarios] == [f'Scenario {index}' for index in range(4)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('method', 'count_kwarg'),
    [
        ('generate_edge_cases', 'num_edge_cases'),
        ('generate_boundary_scenarios', 'num_scenarios'),
        ('generate_security_scenarios', 'num_scenarios'),
    ],
)
async def test_scenarios_taken_from_the_second_runs_text_are_still_marked_as_edge_cases(
    monkeypatch: pytest.MonkeyPatch, method: str, count_kwarg: str
) -> None:
    raw = json.dumps([{**_scenario(name=f'Scenario {index}'), 'is_edge_case': False} for index in range(4)])
    _install(monkeypatch, lambda model: StructuredResult(model.model_validate({'scenarios': [_scenario(name='x', goal='x')]}), raw, None))

    scenarios = await getattr(ScenarioGenerator(model='openai/gpt-6-luna'), method)(agent_description='A bookshop agent', **{count_kwarg: 4})

    assert [scenario.name for scenario in scenarios] == [f'Scenario {index}' for index in range(4)]
    assert [scenario.is_edge_case for scenario in scenarios] == [True] * 4


@pytest.mark.asyncio
async def test_scenarios_taken_from_the_second_runs_text_still_go_through_coverage_and_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = json.dumps([_scenario(name=f'Scenario {index}') for index in range(6)])
    _install(monkeypatch, lambda model: StructuredResult(model.model_validate({'scenarios': [_scenario(name='x', goal='x')]}), raw, None))
    generator = ScenarioGenerator(model='openai/gpt-6-luna')

    from_text = await generator.generate_with_coverage(agent_description='A bookshop agent', num_scenarios=3)

    # The same six items arriving as a parsed answer give the same result: the text took the method's usual path.
    six = [_scenario(name=f'Scenario {index}') for index in range(6)]
    _install(monkeypatch, lambda model: StructuredResult(model.model_validate({'scenarios': six}), '', None))
    from_parsed = await ScenarioGenerator(model='openai/gpt-6-luna').generate_with_coverage(agent_description='A bookshop agent', num_scenarios=3)

    assert from_text
    assert [scenario.model_dump() for scenario in from_text] == [scenario.model_dump() for scenario in from_parsed]
