"""The generators spot the stub some models return under a JSON-schema format, ask again, and never return it."""

from __future__ import annotations

import json
from typing import Any

import pytest

from evaluatorq.common.structured_output import StructuredGenerationError, StructuredResult
from evaluatorq.contracts import TokenUsage
from evaluatorq.simulation.generators._placeholders import generate_real_items, is_stub_item, too_few
from evaluatorq.simulation.generators.persona_generator import PersonaGenerator
from evaluatorq.simulation.generators.scenario_generator import ScenarioGenerator


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


def _usage(tokens: int) -> TokenUsage:
    return TokenUsage(input_tokens=tokens, output_tokens=1, total_tokens=tokens + 1, calls=1)


def _parsed(items: list[dict[str, Any]], raw: str = '', usage: TokenUsage | None = None) -> Any:
    """An answer that validates against whichever list model the generator asked for."""
    return lambda model: StructuredResult(model.model_validate({next(iter(model.model_fields)): items}), raw, usage)


def _unparsed(raw: str, usage: TokenUsage | None = None) -> Any:
    return lambda _model: StructuredResult(None, raw, usage)


class _Fake:
    """Stands in for ``generate_structured``: ``first`` answers the normal call, ``second`` the re-ask."""

    def __init__(self, first: Any, second: Any = None) -> None:
        self.first, self.second = first, second if second is not None else first
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, _client: object, **kwargs: Any) -> StructuredResult[Any]:
        self.calls.append(kwargs)
        answer = self.second if kwargs['skip_schema_rungs'] else self.first
        if isinstance(answer, BaseException):
            raise answer
        return answer(kwargs['response_format'])


def _install(monkeypatch: pytest.MonkeyPatch, first: Any, second: Any = None) -> _Fake:
    fake = _Fake(first, second)
    monkeypatch.setattr('evaluatorq.openresponses.client.build_simulation_client', lambda _c=None, **_kw: (object(), False))
    monkeypatch.setattr('evaluatorq.simulation.generators.persona_generator.generate_structured', fake)
    monkeypatch.setattr('evaluatorq.simulation.generators.scenario_generator.generate_structured', fake)
    return fake


def _personas() -> PersonaGenerator:
    return PersonaGenerator(model='openai/gpt-6-luna')


def _scenarios() -> ScenarioGenerator:
    return ScenarioGenerator(model='openai/gpt-6-luna')


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('items', 'reasks'),
    [
        ([_persona(name='', background='')], True),
        ([_persona(name='placeholder', background='placeholder')], True),  # one item for five
        ([_persona(), _persona(), _persona(name='x', background='x')], True),
        ([_persona(), _persona(), _persona(background=' ')], True),
        ([_persona(), _persona(), _persona(), _persona()], False),  # one short is a miscount
        ([_persona(name='Li'), _persona(), _persona()], False),
        ([], False),  # an empty list is not the stub; the generator warns about it as before
    ],
)
async def test_which_persona_answers_are_asked_again_for_five_asked(monkeypatch: pytest.MonkeyPatch, items: list[dict[str, Any]], reasks: bool) -> None:
    fake = _install(monkeypatch, _parsed(items))
    await _personas().generate(agent_description='A bookshop agent', num_personas=5)
    assert [call['skip_schema_rungs'] for call in fake.calls] == ([False, True] if reasks else [False])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('items', 'reasks'),
    [
        ([_scenario(name='x', goal='x')], True),
        ([_scenario(), _scenario()], True),  # two for five
        ([_scenario(), _scenario(), _scenario(goal='')], True),
        ([_scenario(name='QA'), _scenario(), _scenario()], False),
    ],
)
async def test_which_scenario_answers_are_asked_again_for_five_asked(monkeypatch: pytest.MonkeyPatch, items: list[dict[str, Any]], reasks: bool) -> None:
    fake = _install(monkeypatch, _parsed(items))
    await _scenarios().generate(agent_description='A bookshop agent', num_scenarios=5)
    assert len(fake.calls) == (2 if reasks else 1)


@pytest.mark.asyncio
async def test_a_request_for_one_item_is_judged_on_its_text_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(monkeypatch, _parsed([_persona()]))
    await _personas().generate(agent_description='A bookshop agent', num_personas=1)
    assert len(fake.calls) == 1

    fake = _install(monkeypatch, _parsed([_persona(name='')]))
    await _personas().generate(agent_description='A bookshop agent', num_personas=1)
    assert len(fake.calls) == 2

    fake = _install(monkeypatch, _parsed([_scenario()]))
    await _scenarios().generate(agent_description='A bookshop agent', num_scenarios=0)
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_every_generator_method_re_asks_for_its_own_count_through_the_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    methods = [
        (_personas, 'generate', 'num_personas', _persona),
        (_personas, 'generate_with_coverage', 'num_personas', _persona),
        (_scenarios, 'generate', 'num_scenarios', _scenario),
        (_scenarios, 'generate_with_coverage', 'num_scenarios', _scenario),
        (_scenarios, 'generate_edge_cases', 'num_edge_cases', _scenario),
        (_scenarios, 'generate_boundary_scenarios', 'num_scenarios', _scenario),
        (_scenarios, 'generate_security_scenarios', 'num_scenarios', _scenario),
    ]
    for build, method, count_kwarg, item in methods:
        # Asked for six: two real items are re-asked as too few, three are kept.
        for count, calls in ((2, [False, True]), (3, [False])):
            fake = _install(monkeypatch, _parsed([item()] * count))
            await getattr(build(), method)(agent_description='A bookshop agent', **{count_kwarg: 6})
            assert [call['skip_schema_rungs'] for call in fake.calls] == calls, (method, count)


@pytest.mark.asyncio
async def test_a_stub_that_still_comes_back_is_dropped_not_returned(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    _install(monkeypatch, _parsed([_persona(name='', background=''), _persona()]))

    with caplog.at_level('WARNING'):
        personas = await _personas().generate(agent_description='A bookshop agent', num_personas=5)

    assert [persona.name for persona in personas] == ['Anxious First-Time Buyer']
    assert 'PersonaGenerator.generate: dropped 1 placeholder item(s)' in caplog.text
    assert 'requested 5 personas but only 1 were successfully parsed' in caplog.text


@pytest.mark.asyncio
async def test_the_salvage_path_drops_stubs_too_without_asking_again(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = json.dumps({'scenarios': [_scenario(name='x', goal='x'), _scenario(), _scenario(goal='')]})
    fake = _install(monkeypatch, _unparsed(raw))

    scenarios = await _scenarios().generate(agent_description='A bookshop agent', num_scenarios=3)

    assert [scenario.name for scenario in scenarios] == ['Late delivery']
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_bad_json_with_nothing_parsed_still_counts_its_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _unparsed('not json at all', _usage(100)))
    generator = _scenarios()

    assert await generator.generate(agent_description='A bookshop agent', num_scenarios=3) == []
    assert generator.get_usage().input_tokens == 100


@pytest.mark.asyncio
async def test_a_short_real_name_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, _parsed([_persona(name='Li'), _persona(), _persona()]))

    personas = await _personas().generate(agent_description='A bookshop agent', num_personas=3)

    assert [persona.name for persona in personas][0] == 'Li'
    assert len(personas) == 3


@pytest.mark.asyncio
async def test_real_items_in_the_second_runs_text_replace_a_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """What Opus does: a stub from the schema rung, then a bare array of real personas the wrapper model cannot parse."""
    raw = json.dumps([_persona(name=f'Persona {index}') for index in range(5)])
    _install(monkeypatch, _parsed([_persona(name='', background='')]), _unparsed(raw))

    personas = await _personas().generate(agent_description='A bookshop agent', num_personas=5)

    assert [persona.name for persona in personas] == [f'Persona {index}' for index in range(5)]


@pytest.mark.asyncio
@pytest.mark.parametrize('raw', ['', 'not json at all', '[]', json.dumps([_scenario(name='Only one')])])
async def test_an_honest_short_list_is_never_made_shorter(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    """Two real items for five asked are re-asked; whatever the second run left, the two are still returned."""
    _install(monkeypatch, _parsed([_persona(name='First real'), _persona(name='Second real')]), _unparsed(raw))
    personas = await _personas().generate(agent_description='A bookshop agent', num_personas=5)
    assert [persona.name for persona in personas] == ['First real', 'Second real']

    _install(monkeypatch, _parsed([_scenario(name='First real'), _scenario(name='Second real')]), _unparsed(raw))
    scenarios = await _scenarios().generate(agent_description='A bookshop agent', num_scenarios=5)
    assert [scenario.name for scenario in scenarios] == ['First real', 'Second real']


@pytest.mark.asyncio
async def test_real_scenarios_in_the_second_runs_text_replace_a_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = json.dumps({'scenarios': [_scenario(name=f'Scenario {index}') for index in range(4)]})
    _install(monkeypatch, _parsed([_scenario(name='x', goal='x')]), _unparsed(raw))

    scenarios = await _scenarios().generate_security_scenarios(agent_description='A bookshop agent', num_scenarios=4)

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
    _install(monkeypatch, _parsed([_scenario(name='x', goal='x')]), _unparsed(raw))

    scenarios = await getattr(_scenarios(), method)(agent_description='A bookshop agent', **{count_kwarg: 4})

    assert [scenario.name for scenario in scenarios] == [f'Scenario {index}' for index in range(4)]
    assert [scenario.is_edge_case for scenario in scenarios] == [True] * 4


@pytest.mark.asyncio
async def test_scenarios_taken_from_the_second_runs_text_still_go_through_coverage_and_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    six = [_scenario(name=f'Scenario {index}') for index in range(6)]
    _install(monkeypatch, _parsed([_scenario(name='x', goal='x')]), _unparsed(json.dumps(six)))
    from_text = await _scenarios().generate_with_coverage(agent_description='A bookshop agent', num_scenarios=3)

    # The same six items arriving as a parsed answer give the same result: the text took the method's usual path.
    _install(monkeypatch, _parsed(six))
    from_parsed = await _scenarios().generate_with_coverage(agent_description='A bookshop agent', num_scenarios=3)

    assert from_text
    assert [scenario.model_dump() for scenario in from_text] == [scenario.model_dump() for scenario in from_parsed]


@pytest.mark.asyncio
async def test_a_clean_second_answer_does_not_cost_real_items_the_first_one_had(monkeypatch: pytest.MonkeyPatch) -> None:
    """Five asked: the first answer had four real personas and a stub, the second has three. Four come back."""
    four_and_a_stub = [*[_persona(name=f'First {index}') for index in range(4)], _persona(name='', background='')]
    _install(monkeypatch, _parsed(four_and_a_stub), _parsed([_persona(name=f'Second {index}') for index in range(3)]))

    personas = await _personas().generate(agent_description='A bookshop agent', num_personas=5)

    assert [persona.name for persona in personas] == [f'First {index}' for index in range(4)]


@pytest.mark.asyncio
async def test_the_scenario_methods_also_take_the_answer_with_more_real_items(monkeypatch: pytest.MonkeyPatch) -> None:
    four_and_a_stub = [*[_scenario(name=f'First {index}') for index in range(4)], _scenario(name='x', goal='x')]
    _install(monkeypatch, _parsed(four_and_a_stub), _parsed([_scenario(name=f'Second {index}') for index in range(3)]))
    scenarios = await _scenarios().generate(agent_description='A bookshop agent', num_scenarios=5)
    assert [scenario.name for scenario in scenarios] == [f'First {index}' for index in range(4)]

    _install(monkeypatch, _parsed([_scenario(name='First 0')]), _parsed([_scenario(name=f'Second {index}') for index in range(5)]))
    better = await _scenarios().generate(agent_description='A bookshop agent', num_scenarios=5)
    assert [scenario.name for scenario in better] == [f'Second {index}' for index in range(5)]


@pytest.mark.asyncio
async def test_a_tie_keeps_the_first_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    three_and_a_stub = [*[_persona(name=f'First {index}') for index in range(3)], _persona(name='', background='')]
    _install(monkeypatch, _parsed(three_and_a_stub), _parsed([_persona(name=f'Second {index}') for index in range(3)]))
    personas = await _personas().generate(agent_description='A bookshop agent', num_personas=5)
    assert [persona.name for persona in personas] == [f'First {index}' for index in range(3)]

    pair = [_scenario(name='First 0'), _scenario(name='First 1')]
    _install(monkeypatch, _parsed(pair), _parsed([_scenario(name='Second 0'), _scenario(name='Second 1')]))
    scenarios = await _scenarios().generate(agent_description='A bookshop agent', num_scenarios=5)
    assert [scenario.name for scenario in scenarios] == ['First 0', 'First 1']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('method', 'count_kwarg'),
    [
        ('generate', 'num_scenarios'),
        ('generate_with_coverage', 'num_scenarios'),
        ('generate_edge_cases', 'num_edge_cases'),
        ('generate_boundary_scenarios', 'num_scenarios'),
        ('generate_security_scenarios', 'num_scenarios'),
    ],
)
async def test_a_stub_inside_a_parsed_scenario_answer_is_dropped_by_every_method(
    monkeypatch: pytest.MonkeyPatch, method: str, count_kwarg: str
) -> None:
    _install(monkeypatch, _parsed([_scenario(name='x', goal='x'), _scenario()]))

    scenarios = await getattr(_scenarios(), method)(agent_description='A bookshop agent', **{count_kwarg: 2})

    assert 'x' not in [scenario.name for scenario in scenarios]
    assert 'Late delivery' in [scenario.name for scenario in scenarios]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('second', 'tokens'),
    [
        (_parsed([_persona(name='Second')], usage=_usage(200)), 300),
        (_unparsed('', _usage(250)), 350),
        (StructuredGenerationError('model refused to generate', usage=_usage(200)), 300),
        (RuntimeError('connection reset'), 100),
    ],
)
async def test_the_usage_of_both_runs_is_counted_and_a_failed_second_run_keeps_the_first_answer(
    monkeypatch: pytest.MonkeyPatch, second: Any, tokens: int
) -> None:
    _install(monkeypatch, _parsed([_persona(name='First'), _persona(name='', background='')], usage=_usage(100)), second)
    generator = _personas()

    personas = await generator.generate(agent_description='A bookshop agent', num_personas=2)

    assert [persona.name for persona in personas] == ['First']
    assert generator.get_usage().input_tokens == tokens


@pytest.mark.asyncio
async def test_a_first_run_that_raises_is_not_asked_again_and_its_usage_is_counted() -> None:
    calls: list[bool] = []
    recorded: list[TokenUsage | None] = []

    async def call(*, skip_schema_rungs: bool) -> StructuredResult[Any]:
        calls.append(skip_schema_rungs)
        raise StructuredGenerationError('model refused to generate', usage=_usage(150))

    with pytest.raises(StructuredGenerationError):
        await generate_real_items(
            call,
            items_of=lambda answer: answer,
            texts_of=lambda item: item,
            salvage=lambda _raw: [],
            expected=5,
            label='Sample.generate',
            record_usage=recorded.append,
        )

    assert calls == [False]
    assert [usage.input_tokens for usage in recorded if usage is not None] == [150]
