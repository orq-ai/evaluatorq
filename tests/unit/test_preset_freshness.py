"""Jury presets rot even when nobody edits them (RES-1171, ported from RES-1346).

A preset is immutable; the garden and the price list are not. Four ways a
published preset becomes a lie without a single line changing:

  1. a judge is retired, so the preset names a model nobody can call
  2. a provider reprices, so the published $/1k is wrong
  3. a newer generation ships and the preset quietly stays a generation behind
  4. nobody looks for long enough that none of the above is noticed

These assert against the committed snapshot rather than the live gateway, on
purpose. Live calls would make CI depend on a gateway being up and on our own
workspace's provider billing: at capture time our Moonshot account was suspended
(429) while `moonshotai/kimi-k2.6` was still listed and current. A preset ships
to customers on their own contracts, so our billing state must not be able to
rewrite it. Membership in the garden is the model fact; whether our key can call
it today is not, and that half is
`tests/integration/test_preset_pricing_drift.py`.

The snapshot is refreshed in the research repo
(`projects/hitl-evaluators/scripts/refresh_model_garden_snapshot.py`) and copied
here with the preset file, so the two cannot describe different gardens.
"""

import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from evaluatorq.common.model_frontier import (
    SNAPSHOT_PATH,
    ScoredModel,
    in_family_upgrades,
    load_garden,
    seatable,
)
from evaluatorq.jury_presets import (
    AGED_SEATS,
    MAX_SEAT_AGE_DAYS,
    PRESETS,
    REVIEWED_SUCCESSORS,
    all_router_ids,
    judge_family,
    model_identity,
)

# Long enough not to nag on a quiet month, short enough that a preset cannot
# ship a year-old price. Model generations have been turning over in ~8 weeks.
MAX_SNAPSHOT_AGE = timedelta(days=60)


@pytest.fixture(scope='module')
def snapshot() -> dict[str, Any]:
    return json.loads(SNAPSHOT_PATH.read_text())


def test_snapshot_is_not_stale(snapshot):
    """Nobody looking is itself a failure mode, so age is a test and not a note."""
    captured = datetime.fromisoformat(snapshot['captured_at'])
    age = datetime.now(timezone.utc) - captured

    assert age < MAX_SNAPSHOT_AGE, (
        f'model garden snapshot is {age.days} days old. Re-run the research repo capture '
        '(projects/hitl-evaluators/scripts/refresh_model_garden_snapshot.py), review the diff, '
        'and copy it in with any preset changes it implies.'
    )


@pytest.mark.parametrize('router_id', sorted(all_router_ids()))
def test_every_shipped_judge_is_still_served(snapshot, router_id):
    """A retired judge is dead weight the user picks without being able to see."""
    assert router_id in snapshot['served_model_ids'], (
        f'{router_id} is named by a preset but no longer served by the garden. '
        'Promote a reserve rather than deleting the seat.'
    )


def test_captured_price_is_the_blend_of_the_captured_rates(snapshot):
    """Nothing else couples the selection axis to the billing rates.

    `price` on a scored card is what seating decisions rank on;
    `preset_judge_pricing` rates are what costs are computed from, and three rows
    had drifted apart before this test existed: the EU luna and Vertex Sonnet
    corrections, then gemini-3-flash-preview shipping at the vendor list price
    while the platform bills double. So the coupling is asserted: every model in
    both stores must satisfy price == (3*input + output) / 4.
    """
    scored = snapshot['scored_models']
    for router_id, rates in snapshot['preset_judge_pricing'].items():
        if router_id not in scored:
            continue
        blend = (3 * rates['input_rate'] + rates['output_rate']) / 4
        assert blend == pytest.approx(scored[router_id]['price']), (
            f'{router_id}: captured rates blend to {blend} but the card price is '
            f'{scored[router_id]["price"]}. One of the two stores is stale.'
        )


CONTRADICTED_CARDS: dict[str, str] = {
    'moonshotai/kimi-k2.6': (
        'Reads a headline of 44.2 taken at effort `none` beside a `none` rung '
        'of 34.6, so the card scores the same operating point twice and gives '
        'two answers. One of them is wrong and the card does not say which, so '
        'nothing is ranked on it: `seatable` refuses it and it lost the '
        'Open-Weight / Portable middle seat and the Cheap Aggregate reserve on '
        '2026-09-03. It is also the one card here that has never been probed, '
        'since our Moonshot account is suspended, so the index is the only '
        'evidence there is about it. Drop this entry when the card agrees with '
        'itself, or when the garden publishes the rung the headline came from.'
    ),
}
"""Cards whose own ladder contradicts their own headline, keyed by router ID.

Narrower than "the headline must appear on the ladder" on purpose. Seven
Anthropic-lineage cards are called at `high` and publish rungs at `none`, `low`
or `max` without listing `high`, so their headline is a real measurement at a
rung the ladder omits; `ScoredModel.ladder` folds those back in rather than
refusing them. A contradiction is the narrower thing: the card scores the rung
its headline came from, and disagrees with itself about it.
"""


def _contradicted(card: dict[str, Any]) -> str | None:
    """Why a card disagrees with itself about one rung, if it does."""
    rung = card.get('scored_effort') or card['reasoning_effort'] or 'none'
    published = card['effort_indices'].get(rung)
    if published is None or published == pytest.approx(card['intelligence_index']):
        return None
    return f'headline {card["intelligence_index"]} was taken at {rung}, where the card publishes {published}'


def test_no_card_scores_its_own_rung_twice(snapshot):
    """A card that gives two numbers for one rung is not a measurement to seat on.

    Folding cannot repair this one, because there is nothing to fold, only two
    scores for a rung measured once. On the 2026-08-25 garden exactly one card
    does it.
    """
    contradicted = {
        router_id: reason
        for router_id, card in sorted(snapshot['scored_models'].items())
        if router_id not in CONTRADICTED_CARDS and (reason := _contradicted(card))
    }

    assert not contradicted, (
        f'these cards disagree with themselves about one rung: {contradicted}. '
        f'Re-capture, or record the card in CONTRADICTED_CARDS. Nothing may be '
        f'seated on a number the card contradicts.'
    )


def test_every_headline_is_a_published_rung(snapshot):
    """The headline index must be one of the card's own rungs.

    AA puts the headline index and its rung in fields separate from
    `reasoning_efforts`, and the Anthropic lineage does not repeat it there, so
    seven cards read as scoring above every rung they publish. The capture writes
    that rung back into the ladder, which is what makes this rule assertable on
    real cards rather than only on OpenAI-shaped ones. What is left over after
    folding is a card contradicting itself, which is the test above.
    """
    off_ladder = {
        router_id: f'headline {card["intelligence_index"]} against rungs {card["effort_indices"]}'
        for router_id, card in sorted(snapshot['scored_models'].items())
        if router_id not in CONTRADICTED_CARDS
        and card['effort_indices']
        and not any(index == pytest.approx(card['intelligence_index']) for index in card['effort_indices'].values())
    }

    assert not off_ladder, (
        f'these cards score at a rung they do not publish: {off_ladder}. '
        f'Re-capture: `_ladder` in refresh_model_garden_snapshot.py writes the '
        f'headline back into the ladder at the rung it was measured at.'
    )


def test_a_contradicted_card_holds_no_seat():
    """The register records a bad card; it does not license seating one."""
    seats = {j for preset in PRESETS.values() for j in (*preset.judges, *preset.reserve_judges)}
    seated = sorted(seats & set(CONTRADICTED_CARDS))

    assert not seated, f'{seated} are seated on a card that contradicts itself. Reseat.'


def test_contradicted_entries_are_dropped_once_the_card_agrees_with_itself(snapshot):
    """An entry whose card no longer contradicts itself is stale."""
    resolved = sorted(
        router_id
        for router_id, card in snapshot['scored_models'].items()
        if router_id in CONTRADICTED_CARDS and _contradicted(card) is None
    )

    assert not resolved, f'{resolved} now agree with themselves. Drop the entries.'


def test_captured_index_is_the_score_at_the_captured_effort(snapshot):
    """The coupling that cost us a seat.

    The snapshot this work first shipped against predated the effort fields and
    stored the *maximum*-effort score beside a default-effort price:
    `openai/gpt-5.4` was seated in two panels, and defended in review, on 51.4,
    which is its `xhigh` figure, while its own map gave 27.7 at the effort it
    billed at. `scored_effort` is the effort the headline was measured at, so
    this asks the capture to agree with itself.
    """
    for router_id, card in snapshot['scored_models'].items():
        effort = card.get('scored_effort')
        if effort is None:
            continue
        expected = card['effort_indices'].get(effort)
        assert card['intelligence_index'] == pytest.approx(expected), (
            f'{router_id}: card index {card["intelligence_index"]} is not the '
            f'{effort} score {expected} its own slug says it was measured at. '
            f'Available: {card["effort_indices"]}. Re-capture rather than '
            f'hand-editing.'
        )


def test_every_seat_states_the_effort_it_is_ranked_at():
    """Ranking at the ceiling is only honest if the caller is told which rung it is.

    A preset names judges and a caller sends the call, so a seat ranked at `max`
    and called at the provider default is costed and chosen at one operating
    point and run at another. Every seat publishes the rung it was ranked at, and
    it has to be a rung that card actually measures.
    """
    garden = load_garden()
    seats = {j: p.name for p in PRESETS.values() for j in p.judges}

    for router_id, owner in sorted(seats.items()):
        model = garden.get(router_id)
        if model is None:
            continue
        effort = model.ceiling_effort
        assert effort is None or effort in model.ladder, f'{owner}: {router_id} is ranked at an unmeasured rung'
        rung = model.intelligence_index if effort is None else model.ladder[effort]
        assert model.ceiling_index == pytest.approx(rung), (
            f'{owner}: {router_id} publishes a ceiling that is not the score at the rung it names'
        )


@pytest.mark.parametrize('preset', PRESETS.values(), ids=lambda p: p.name)
def test_every_jury_publishes_an_effort_for_every_judge(preset):
    """Not one entry per ladder-bearing judge: one per judge, `None` included."""
    assert set(preset.seated_efforts()) == set(preset.judges)


class TestEffortAssertionsFire:
    """Each effort rule with a model built to break it.

    None of them had one in the first draft. They were verified by reading the
    garden they were written against, which is how the rule they replaced passed
    for a week while skipping the card it existed to catch: a test nobody has
    watched fail is a test nobody has watched.
    """

    def test_the_gpt_5_4_pairing_is_refused(self):
        """A stored index that is the max-effort score against a default-effort price."""
        snapshot = {
            'scored_models': {
                'openai/overstated': {
                    'intelligence_index': 51.4,
                    'reasoning_effort': 'none',
                    'scored_effort': 'none',
                    'effort_indices': {'none': 27.7, 'xhigh': 51.4},
                }
            }
        }

        with pytest.raises(AssertionError, match='not the none score'):
            test_captured_index_is_the_score_at_the_captured_effort(snapshot)

    def test_a_card_that_scores_one_rung_twice_is_refused(self):
        """The kimi-k2.6 shape, on a card with no register entry."""
        snapshot = {
            'scored_models': {
                'moonshotai/unregistered': {
                    'intelligence_index': 44.2,
                    'reasoning_effort': None,
                    'scored_effort': None,
                    'effort_indices': {'none': 34.6},
                }
            }
        }

        with pytest.raises(AssertionError, match='disagree with themselves'):
            test_no_card_scores_its_own_rung_twice(snapshot)

    def test_a_ladder_that_omits_the_headline_rung_is_left_alone(self):
        """The claude-sonnet-5 shape: called at high, publishes only `none`."""
        snapshot = {
            'scored_models': {
                'anthropic/sonnet-shaped': {
                    'intelligence_index': 53.4,
                    'reasoning_effort': 'high',
                    'scored_effort': None,
                    'effort_indices': {'none': 41.7},
                }
            }
        }

        test_no_card_scores_its_own_rung_twice(snapshot)

    def test_a_headline_the_ladder_never_publishes_is_refused(self):
        """The sonnet shape as AA ships it, before the capture folds it in.

        This is the case that made the rule as written look too strong. It is a
        capture defect, so the test names the repair rather than a register.
        """
        snapshot = {
            'scored_models': {
                'anthropic/unfolded': {
                    'intelligence_index': 53.4,
                    'reasoning_effort': 'high',
                    'scored_effort': None,
                    'effort_indices': {'none': 41.7},
                }
            }
        }

        with pytest.raises(AssertionError, match='rung they do not publish'):
            test_every_headline_is_a_published_rung(snapshot)

    def test_the_same_card_passes_once_its_headline_rung_is_folded_in(self):
        """What the capture writes for that card, which is what the garden holds."""
        snapshot = {
            'scored_models': {
                'anthropic/folded': {
                    'intelligence_index': 53.4,
                    'reasoning_effort': 'high',
                    'scored_effort': None,
                    'effort_indices': {'high': 53.4, 'none': 41.7},
                }
            }
        }

        test_every_headline_is_a_published_rung(snapshot)
        test_no_card_scores_its_own_rung_twice(snapshot)

    def test_a_card_publishing_one_number_is_left_alone(self):
        """Most of the garden. An empty ladder claims no breakdown to contradict."""
        snapshot = {
            'scored_models': {
                'anthropic/opus-shaped': {
                    'intelligence_index': 60.7,
                    'reasoning_effort': 'high',
                    'scored_effort': None,
                    'effort_indices': {},
                }
            }
        }

        test_no_card_scores_its_own_rung_twice(snapshot)


@pytest.mark.parametrize('preset', PRESETS.values(), ids=lambda p: p.name)
def test_every_preset_names_at_least_one_reserve(preset):
    """A preset with no reserve has no answer to a retirement."""
    assert preset.reserve_judges


def test_every_judge_and_reserve_is_seatable(snapshot):
    """The index floor and the NEVER_SEAT denylist bind the juries too: a family
    the platform will not route to is not one it asks for judgment either."""
    scored = snapshot['scored_models']
    for preset in PRESETS.values():
        for router_id in (*preset.judges, *preset.reserve_judges):
            if router_id not in scored:
                continue
            model = ScoredModel(router_id=router_id, **scored[router_id])
            assert seatable(model), f'{preset.name} recommends unseatable {router_id}'


def _unreviewed_successors(snapshot: dict[str, Any]) -> dict[str, str]:
    """Garden models newer than everything a preset seats from their own lineage.

    Compared against the newest seat per lineage rather than per judge: a panel
    that already seats GPT-5.6 does not need to answer for GPT-5.1 carrying a
    later catalog timestamp than an older seat. Identity-normalised, so the same
    weights on a second host are not a second decision.
    """
    created = snapshot['model_created_at']
    seated_identities = {model_identity(r) for r in all_router_ids()}

    newest_seated: dict[str, int] = {}
    for router_id in all_router_ids():
        stamp = created.get(router_id)
        if stamp is None:
            continue
        family = judge_family(router_id)
        newest_seated[family] = max(newest_seated.get(family, 0), stamp)

    candidates: dict[str, str] = {}
    for model_id, stamp in created.items():
        # Research-workspace endpoints are not generally served, and image
        # models are not judges.
        if model_id.startswith('orq-research@') or 'image' in model_id:
            continue
        identity = model_identity(model_id)
        if identity in seated_identities:
            continue
        try:
            family = judge_family(model_id)
        except ValueError:
            continue  # unclassifiable lineage cannot be a successor to ours
        if family in newest_seated and stamp > newest_seated[family]:
            candidates[identity] = model_id

    return candidates


def test_no_newer_model_is_silently_ignored(snapshot):
    """A newer sibling must be seated or dismissed in writing, never just missed.

    This is the failure the other tests cannot see: every judge still served,
    every price still correct, and the panel quietly a generation behind.
    """
    unreviewed = {
        identity: model_id
        for identity, model_id in _unreviewed_successors(snapshot).items()
        if identity not in REVIEWED_SUCCESSORS
    }

    assert not unreviewed, (
        'the garden has newer models in a lineage a preset seats: '
        f'{sorted(unreviewed.values())}. Seat one, or add its identity to '
        'REVIEWED_SUCCESSORS with the reason it stays out.'
    )


def test_reviewed_successors_are_not_carried_after_they_are_seated():
    """A dismissal that outlives its subject is stale documentation."""
    seated_identities = {model_identity(r) for r in all_router_ids()}

    still_out = seated_identities & set(REVIEWED_SUCCESSORS)

    assert not still_out, f'{sorted(still_out)} are seated but still listed as reviewed-and-rejected'


DOMINANCE_EXEMPTIONS: dict[str, str] = {}
"""Seats knowingly kept despite a computed in-family upgrade.

Same disposition-register contract as REVIEWED_SUCCESSORS: seat it or write down
why not. An empty register means the arithmetic and the shipped presets agree.
"""


class TestFrontier:
    """The upgrade rule, as arithmetic rather than taste (RES-1347)."""

    def test_no_seat_has_an_unreviewed_in_family_upgrade(self):
        """A judge must be the best buy available within its own lineage.

        This is the whole auto-update mechanism. Every refresh re-runs it, so a
        cheaper-and-stronger same-family model reaching the garden fails CI the
        month it lands instead of sitting unnoticed until someone reads the
        table. Scoped to the lineage on purpose: a jury buys family diversity,
        and a global frontier check would push all three seats onto whichever
        vendor is cheapest, recreating the correlated errors the panel exists to
        cancel out.
        """
        models = list(load_garden().values())
        by_id = {m.router_id: m for m in models}

        stale = {}
        for preset in PRESETS.values():
            for judge in preset.judges:
                incumbent = by_id.get(judge)
                if incumbent is None:
                    continue
                upgrades = in_family_upgrades(incumbent, models, judge_family, model_identity)
                if upgrades and judge not in DOMINANCE_EXEMPTIONS:
                    stale[judge] = upgrades[0].router_id

        assert not stale, (
            f'These seats are beaten on both index and price by a same-family, '
            f'same-region model: {stale}. Either reseat, or record the reason in '
            f'DOMINANCE_EXEMPTIONS.'
        )

    def test_exemptions_are_dropped_once_they_stop_applying(self):
        """An exemption whose upgrade has gone away is stale reasoning."""
        models = list(load_garden().values())
        by_id = {m.router_id: m for m in models}

        pointless = [
            judge
            for judge in DOMINANCE_EXEMPTIONS
            if judge in by_id and not in_family_upgrades(by_id[judge], models, judge_family, model_identity)
        ]

        assert not pointless, (
            f'{pointless} are exempted from a dominance check that no longer flags them. Drop the entries.'
        )

    def test_every_seat_is_scored(self):
        """An unscored seat is one the upgrade check silently skips."""
        unscored = all_router_ids() - set(load_garden())

        assert not unscored, (
            f'{unscored} carry no autorouter intelligence/price pair, so no upgrade check runs against them.'
        )


def _first_listed(snapshot: dict[str, Any]) -> dict[str, datetime]:
    """Earliest garden listing per model identity, across every host of it.

    A second host relisting the same weights carries its own `created` stamp,
    which is younger than the model: the EU Bedrock haiku listing is 125 days
    newer than the Anthropic one for a model that shipped once. Taking the
    minimum across the identity measures the model rather than the listing.
    """
    earliest: dict[str, datetime] = {}
    for router_id, stamp in snapshot['model_created_at'].items():
        if stamp is None:
            continue
        # Some cards stamp milliseconds, some seconds.
        listed = datetime.fromtimestamp(stamp / 1000 if stamp > 1e11 else stamp, timezone.utc)
        identity = model_identity(router_id)
        if identity not in earliest or listed < earliest[identity]:
            earliest[identity] = listed
    return earliest


def _all_seats() -> dict[str, list[str]]:
    """Every jury seat in the library, with the panel it sits on."""
    seats: dict[str, list[str]] = {}
    for preset in PRESETS.values():
        for judge in preset.judges:
            seats.setdefault(judge, []).append(f'jury {preset.name}')
    return seats


def test_no_seat_ages_out_unargued(snapshot):
    """A lineage that stops shipping is the blind spot in every check above.

    The upgrade and successor tests both need a newer sibling to exist before
    they can say anything. If a vendor ships nothing, they stay silent forever
    and the seat keeps getting older while the suite stays green. This is the
    only test that fires on absence.
    """
    earliest = _first_listed(snapshot)
    now = datetime.now(timezone.utc)

    aged = {}
    for router_id, held in _all_seats().items():
        listed = earliest.get(model_identity(router_id))
        if listed is None:
            continue
        days = (now - listed).days
        if days > MAX_SEAT_AGE_DAYS and model_identity(router_id) not in AGED_SEATS:
            aged[router_id] = f'{days}d, held by {", ".join(held)}'

    assert not aged, (
        f'these seats were listed more than {MAX_SEAT_AGE_DAYS} days ago and no '
        f'successor has been flagged, so nothing else will ever ask about them: '
        f'{aged}. Reseat, or record the reason in AGED_SEATS.'
    )


def test_reviewed_successors_are_dropped_once_they_stop_being_successors(snapshot):
    """A dismissal survives the seat it was written about, and then reads as fact.

    The test above catches an entry whose model got seated. This catches the
    other direction, which is the one that bit: reseating Cheap Aggregate off
    qwen left three entries explaining why various qwen models were not worth
    taking the seated 35b-a3b's place, when no qwen holds a seat at all. Every
    word of that was still true and none of it was about anything.
    """
    stale = sorted(set(REVIEWED_SUCCESSORS) - set(_unreviewed_successors(snapshot)))

    assert not stale, (
        f'{stale} are recorded as reviewed-and-rejected successors but no seat '
        f'in their lineage makes them a successor to anything. Drop the entries.'
    )


def test_aged_seat_entries_are_dropped_once_the_seat_is_young_again(snapshot):
    """A vendor shipping again should retire the excuse, not leave it standing."""
    earliest = _first_listed(snapshot)
    now = datetime.now(timezone.utc)
    seated = {model_identity(r) for r in _all_seats()}

    pointless = sorted(
        identity
        for identity in AGED_SEATS
        if identity not in seated or (identity in earliest and (now - earliest[identity]).days <= MAX_SEAT_AGE_DAYS)
    )

    assert not pointless, (
        f'{pointless} are excused for age but are either no longer seated or no longer old. Drop the entries.'
    )
