"""`MustHappenCriterion` / `MustNotHappenCriterion` are `Criterion`s with the type fixed."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evaluatorq.simulation import Criterion, MustHappenCriterion, MustNotHappenCriterion, Scenario
from evaluatorq.simulation.utils.prompt_builders import build_scenario_user_context


def test_type_is_fixed_and_description_is_positional() -> None:
    must = MustHappenCriterion('Agent asks for order details')
    must_not = MustNotHappenCriterion('Agent blames the customer')

    assert (must.type, must.description, must.evaluator) == ('must_happen', 'Agent asks for order details', None)
    assert (must_not.type, must_not.description) == ('must_not_happen', 'Agent blames the customer')
    assert isinstance(must, Criterion)


def test_description_also_binds_as_a_keyword() -> None:
    assert MustHappenCriterion(description='Agent refunds').description == 'Agent refunds'
    assert MustNotHappenCriterion(description='Agent blames').type == 'must_not_happen'


def test_evaluator_is_still_settable() -> None:
    assert MustHappenCriterion('Agent refunds', evaluator='task_achieved').evaluator == 'task_achieved'


def test_the_other_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        MustHappenCriterion('x', type='must_not_happen')
    with pytest.raises(ValidationError):
        MustNotHappenCriterion('x', type='must_happen')


def test_dump_reloads_as_a_plain_criterion() -> None:
    dumped = MustNotHappenCriterion('Agent blames the customer').model_dump()

    assert dumped == {'description': 'Agent blames the customer', 'type': 'must_not_happen', 'evaluator': None}
    assert Criterion.model_validate(dumped).type == 'must_not_happen'


def test_scenario_mixes_both_and_round_trips() -> None:
    scenario = Scenario(
        name='S',
        goal='g',
        criteria=[MustHappenCriterion('Agent asks for order details'), MustNotHappenCriterion('Agent blames the customer')],
    )

    reloaded = Scenario.model_validate_json(scenario.model_dump_json())
    assert [(c.type, c.description) for c in reloaded.criteria or []] == [
        ('must_happen', 'Agent asks for order details'),
        ('must_not_happen', 'Agent blames the customer'),
    ]
    context = build_scenario_user_context(scenario)
    assert 'Agent asks for order details' in context
    assert 'Agent blames the customer' in context
