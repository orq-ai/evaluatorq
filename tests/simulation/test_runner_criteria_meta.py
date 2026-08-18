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
    meta = _build_criteria_meta(
        _scenario(),
        judgment,
        frozenset({'criteria_0', 'criteria_1'}),
        {'criteria_1': 'thanks for your patience'},
    )
    assert [entry.model_dump(mode='json') for entry in meta] == [
        {
            'id': 'criteria_0',
            'description': 'explain charge',
            'type': 'must_happen',
            'passed': False,
            'audited': True,
            'evidence': '',
        },
        {
            'id': 'criteria_1',
            'description': 'no rudeness',
            'type': 'must_not_happen',
            'passed': True,
            'audited': True,
            'evidence': 'thanks for your patience',
        },
    ]


def test_build_criteria_meta_separates_a_confirmed_failure_from_an_unaudited_one():
    """Both rows render red, but only one is a verdict the judge actually gave.

    Without `audited` a must_happen the judge confirmed never occurred and one it
    silently skipped are indistinguishable in the report.
    """
    judgment = Judgment(
        should_terminate=True,
        reason='r',
        goal_achieved=False,
        rules_broken=['criteria_0', 'criteria_1'],
        goal_completion_score=0.0,
    )
    meta = _build_criteria_meta(_scenario(), judgment, frozenset({'criteria_0'}))
    assert [m.passed for m in meta] == [False, False]
    assert [m.audited for m in meta] == [True, False]


def test_build_criteria_meta_without_a_tracker_reports_audited_unknown():
    judgment = Judgment(
        should_terminate=True,
        reason='r',
        goal_achieved=False,
        rules_broken=[],
        goal_completion_score=0.0,
    )
    assert [m.audited for m in _build_criteria_meta(_scenario(), judgment)] == [None, None]


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
    assert meta[0].passed is True and meta[1].passed is False


def test_build_criteria_meta_without_evidence_reports_it_unknown():
    """No tracker means no evidence map; an empty string would claim the judge
    quoted nothing, which is a different statement from "we do not know"."""
    judgment = Judgment(
        should_terminate=True,
        reason='r',
        goal_achieved=False,
        rules_broken=[],
        goal_completion_score=0.0,
    )
    assert [m.evidence for m in _build_criteria_meta(_scenario(), judgment)] == [None, None]
