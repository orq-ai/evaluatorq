from __future__ import annotations

from evaluatorq.simulation.runner.simulation import _build_criteria_meta
from evaluatorq.simulation.types import Criterion, Judgment, Scenario


def _scenario() -> Scenario:
    return Scenario(
        name='billing',
        goal='g',
        context='c',
        criteria=[
            Criterion(description='explain charge', type='must_happen'),
            Criterion(description='no rudeness', type='must_not_happen'),
        ],
    )


def test_build_criteria_meta_is_id_keyed_with_type_and_passed():
    judgment = Judgment(
        should_terminate=True,
        reason='r',
        goal_achieved=False,
        rules_broken=['criteria_0'],
        goal_completion_score=0.0,
    )
    meta = _build_criteria_meta(_scenario(), judgment)
    assert meta == [
        {'id': 'criteria_0', 'description': 'explain charge', 'type': 'must_happen', 'passed': False},
        {'id': 'criteria_1', 'description': 'no rudeness', 'type': 'must_not_happen', 'passed': True},
    ]


def test_build_criteria_meta_survives_duplicate_descriptions():
    scenario = Scenario(
        name='s',
        goal='g',
        context='c',
        criteria=[
            Criterion(description='same text', type='must_happen'),
            Criterion(description='same text', type='must_happen'),
        ],
    )
    judgment = Judgment(
        should_terminate=True,
        reason='r',
        goal_achieved=False,
        rules_broken=['criteria_1'],
        goal_completion_score=0.0,
    )
    meta = _build_criteria_meta(scenario, judgment)
    # both criteria preserved despite identical descriptions
    assert len(meta) == 2
    assert meta[0]['passed'] is True and meta[1]['passed'] is False
