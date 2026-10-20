"""Tests for the model-free persona/scenario generation validation pass (RES-1086)."""

from __future__ import annotations

from evaluatorq.simulation.contracts import CommunicationStyle, Criterion, Persona, Scenario
from evaluatorq.simulation.generators.validation import (
    trait_spread,
    trait_variance,
    validate_personas,
    validate_scenarios,
)


def _persona(
    name: str, background: str, *, patience=0.5, assertiveness=0.5, politeness=0.5, technical_level=0.5
) -> Persona:
    return Persona(
        name=name,
        patience=patience,
        assertiveness=assertiveness,
        politeness=politeness,
        technical_level=technical_level,
        communication_style=CommunicationStyle.terse,
        background=background,
    )


_GOOD_PERSONAS = [
    _persona(
        'Overwhelmed Working Parent',
        'Sarah is a working mom with two kids under five. She ordered a tablet two weeks ago that has not arrived and the party is in three days.',
        patience=0.3,
        assertiveness=0.6,
        politeness=0.7,
        technical_level=0.4,
    ),
    _persona(
        'Retired Engineer Seeking Help',
        'A meticulous former aerospace engineer who reads every manual and wants precise, technically-accurate answers about his new router.',
        patience=0.9,
        assertiveness=0.7,
        politeness=0.9,
        technical_level=0.95,
    ),
    _persona(
        'Anxious First-Time Buyer',
        'A college student making their first big online purchase, nervous about being scammed and asking many clarifying questions before committing.',
        patience=0.6,
        assertiveness=0.2,
        politeness=0.3,
        technical_level=0.3,
    ),
]


def test_good_personas_pass_with_high_scores():
    report = validate_personas(_GOOD_PERSONAS)
    assert report.passed
    assert report.overall_score >= 0.9
    assert all(not item.issues for item in report.items)


def test_generic_and_thin_personas_are_flagged():
    bad = [
        _persona('Angry Customer', 'A customer who is angry about something'),
        _persona('Customer', 'Has a problem'),
    ]
    report = validate_personas(bad)
    assert not report.passed
    codes = {issue.code for item in report.items for issue in item.issues}
    assert 'generic_name' in codes
    assert {'thin_background', 'generic_background'} & codes
    assert report.overall_score < 0.7
    assert all(item.issues for item in report.items)


def test_all_similar_traits_flag_low_diversity():
    clones = [
        _persona(
            f'Persona {i}', 'A detailed, specific background that is long enough to be considered specific indeed.'
        )
        for i in range(3)
    ]
    report = validate_personas(clones)
    codes = {issue.code for issue in report.batch_issues}
    assert 'low_diversity' in codes  # every numeric trait is 0.5 -> zero spread
    assert 'duplicate' in codes  # identical backgrounds


def test_seed_adherence_is_checked():
    off_seed = [
        _persona(
            'Cheerful Shopper',
            'A delighted repeat customer praising the fast delivery and friendly chat support team today.',
        )
    ]
    report = validate_personas(off_seed, seed='angry refund')
    assert any(issue.code == 'seed_not_reflected' for issue in report.items[0].issues)
    on_seed = [
        _persona(
            'Angry Refund Seeker',
            'An irate customer demanding an immediate refund after a second broken replacement arrived damaged again.',
        )
    ]
    assert all(
        issue.code != 'seed_not_reflected' for issue in validate_personas(on_seed, seed='angry refund').items[0].issues
    )


def test_trait_spread_and_variance():
    spread = trait_spread(_GOOD_PERSONAS)
    assert spread['technical_level'] > 0.5  # 0.95 - 0.3
    assert trait_variance(_GOOD_PERSONAS)['patience'] > 0.0
    assert trait_spread([]) == {'patience': 0.0, 'assertiveness': 0.0, 'politeness': 0.0, 'technical_level': 0.0}


def test_good_scenarios_pass():
    scenarios = [
        Scenario(
            name='Wrong Item Refund',
            goal='Get a full refund for the phone case received instead of the ordered headphones',
            context='Customer ordered headphones but received a phone case',
            criteria=[Criterion(description='Agent asks for order details', type='must_happen')],
        ),
    ]
    report = validate_scenarios(scenarios)
    assert report.passed
    assert report.overall_score >= 0.9


def test_thin_scenario_without_criteria_is_flagged():
    scenarios = [Scenario(name='Refund', goal='refund please')]
    report = validate_scenarios(scenarios)
    assert not report.passed
    codes = {issue.code for item in report.items for issue in item.issues}
    assert 'thin_goal' in codes
    assert 'no_criteria' in codes
    assert 'no_context' in codes


def test_duplicate_scenario_goals_are_flagged():
    goal = 'Get a full refund for the damaged item that arrived late and broken'
    scenarios = [
        Scenario(name='A', goal=goal, context='x', criteria=[Criterion(description='c', type='must_happen')]),
        Scenario(name='B', goal=goal, context='y', criteria=[Criterion(description='c', type='must_happen')]),
    ]
    report = validate_scenarios(scenarios)
    assert any(issue.code == 'duplicate' for issue in report.batch_issues)
