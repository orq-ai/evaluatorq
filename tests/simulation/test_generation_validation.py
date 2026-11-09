"""Tests for the model-free persona/scenario generation validation pass."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evaluatorq.simulation.contracts import CommunicationStyle, Criterion, Persona, Scenario
from evaluatorq.simulation.generators.validation import (
    ValidationReport,
    Issue,
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
    assert 'thin_background' in codes
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


_SPECIFIC = 'A meticulous former aerospace engineer who reads every manual and wants precise answers about his router.'
_CRITERIA = [Criterion(description='c', type='must_happen')]


def test_an_item_level_fail_fails_the_batch():
    # thin_background alone scores 0.7, exactly the threshold, so only the severity can fail it.
    report = validate_personas([_persona('Angry Shopper', 'Wants a refund today')])
    assert report.overall_score == 0.7
    assert not report.passed


def test_an_item_level_warn_does_not_fail_the_batch():
    report = validate_personas([_persona('Shopper', _SPECIFIC)])
    assert report.items[0].issues[0].code == 'generic_name'
    assert report.passed


def test_generic_filler_in_long_text_is_flagged():
    filler = 'This is simply a customer with a problem who wants some kind of help from the agent today.'
    report = validate_personas([_persona('Some Shopper', filler)])
    assert {i.code for i in report.items[0].issues} == {'generic_background'}

    goal = 'Help a customer with a problem get an answer from support'
    scenario = Scenario(name='Vague', goal=goal, context='x', criteria=_CRITERIA)
    assert {i.code for i in validate_scenarios([scenario]).items[0].issues} == {'generic_goal'}


def test_specific_text_is_not_read_as_generic():
    for text in (
        'A customer who ordered a router and needs a refund for the damaged unit that arrived.',
        'Refills a generic blood pressure medication every month and needs the order moved up by a week.',
    ):
        assert not validate_personas([_persona('Real Person', text)]).items[0].issues, text
    goal = 'Find out why something I never ordered was charged to my card'
    scenario = Scenario(name='Charge', goal=goal, context='x', criteria=_CRITERIA)
    assert not validate_scenarios([scenario]).items[0].issues


def test_a_marker_inside_a_longer_word_does_not_match():
    for text in (
        'Their brother-in-law has a problematic router and asks the agent to replace it under the warranty.',
        'A subcustomer with a problem on their reseller invoice asks the agent to correct the billed amount today.',
    ):
        assert not validate_personas([_persona('Real Person', text)]).items[0].issues, text


def test_a_seed_too_short_to_check_is_reported_as_unchecked():
    persona = validate_personas(_GOOD_PERSONAS, seed='AI')
    scenario = validate_scenarios(
        [Scenario(name='A', goal='Get a refund for the broken phone', context='x', criteria=_CRITERIA)], seed='UX'
    )
    for report in (persona, scenario):
        assert [i.code for i in report.batch_issues] == ['seed_unchecked']
        assert report.batch_issues[0].severity == 'info'
        assert report.passed
        assert report.overall_score == 1.0  # nothing was measured, so nothing is deducted
        assert report.checks_skipped == ['seed_adherence']
        assert not report.items[0].issues


def test_issue_codes_are_a_closed_set():
    with pytest.raises(ValidationError):
        Issue.model_validate({'code': 'thin_backgrond', 'message': 'a typo in the code'})


@pytest.mark.parametrize(
    ('name', 'seed'),
    [
        ('Careful Retired Librarian', 'first-time elderly shoppers'),
        ('Deadline-Driven Office Procurement Lead', 'impatient business buyers'),
    ],
)
def test_a_paraphrased_seed_is_reported_and_does_not_fail_the_batch(name: str, seed: str):
    # Real generator output: both fit their seed and share no word with it.
    report = validate_personas([_persona(name, _SPECIFIC)], seed=seed)
    assert report.items[0].issues[0].code == 'seed_not_reflected'
    assert report.items[0].issues[0].severity == 'info'
    assert report.overall_score == 1.0
    assert report.passed


def test_seed_words_ignore_punctuation():
    report = validate_personas(
        [_persona('Real Person', 'Wants a refund for the broken phone and is losing patience fast.')], seed='refund,'
    )
    assert not report.items[0].issues


def test_a_fully_checked_batch_skips_nothing():
    assert validate_personas(_GOOD_PERSONAS).checks_skipped == []


def test_a_batch_too_small_for_the_trait_check_lists_it_as_skipped():
    report = validate_personas(_GOOD_PERSONAS[:2])
    assert [i.code for i in report.batch_issues] == ['diversity_unchecked']
    assert report.batch_issues[0].severity == 'info'
    assert report.checks_skipped == ['trait_diversity']
    assert report.overall_score == 1.0
    assert report.passed


def test_a_paraphrased_seed_on_a_scenario_is_reported_and_does_not_fail_the_batch():
    scenario = Scenario(
        name='Bulk Order Deadline',
        goal='Confirm the forty office chairs arrive before Monday',
        context='The company moves offices next week.',
        criteria=_CRITERIA,
    )
    report = validate_scenarios([scenario], seed='impatient business buyers')
    issue = report.items[0].issues[0]
    assert (issue.code, issue.severity) == ('seed_not_reflected', 'info')
    assert report.overall_score == 1.0
    assert report.passed


def test_a_hyphenated_seed_word_stays_whole():
    # `first-time` must not split into `time`, which would match `sometimes`.
    persona = _persona('Real Person', 'Sometimes orders gifts online and wants the parcel tracked closely every day.')
    report = validate_personas([persona], seed='first-time shoppers')
    assert [i.code for i in report.items[0].issues] == ['seed_not_reflected']


@pytest.mark.parametrize('partial', [{'overall_score': 1.0}, {'passed': True}])
def test_a_report_nothing_populated_cannot_read_as_a_pass(partial: dict[str, float | bool]):
    with pytest.raises(ValidationError):
        ValidationReport.model_validate(partial)


@pytest.mark.parametrize('validate', [validate_personas, validate_scenarios])
def test_an_empty_batch_does_not_pass(validate):
    report = validate([])
    assert report.overall_score == 0.0
    assert report.passed is False


def test_blank_criteria_and_context_count_as_missing():
    scenario = Scenario(
        name='A',
        goal='Get a refund for the broken phone',
        context='   ',
        criteria=[Criterion(description='   ', type='must_happen')],
    )
    codes = {i.code for i in validate_scenarios([scenario]).items[0].issues}
    assert {'no_criteria', 'no_context'} <= codes
